"""OpenAI-compatible FastAPI server wrapping a Miniforge model.

Exposes:
    GET  /health              liveness probe
    GET  /v1/models           list of loaded model aliases
    POST /v1/chat/completions chat completion (streaming + non-streaming)
    POST /v1/completions      legacy text completion
    GET  /metrics             Prometheus exposition

Environment variables (CLI flags take precedence):
    MINIFORGE_MODEL_ID         HF model id or local path (default: MiniMaxAI/MiniMax-M2.7)
    MINIFORGE_QUANT            quantization (default from M7Config)
    MINIFORGE_DOWNLOAD_DIR     override for GGUF cache (default: M7Config.download_dir or $MODEL_DIR)
    MINIFORGE_CONFIG           YAML config path (default: platform config dir)
    MINIFORGE_BACKEND          llama_cpp | transformers | airllm (default: llama_cpp)
    MINIFORGE_HOST             bind host (default: 127.0.0.1)
    MINIFORGE_PORT             bind port (default: 8003)
    MINIFORGE_ALIAS            public model name in /v1/models (default: mirrors MODEL_ID)

Run with:
    python -m miniforge.server.openai_server
    miniforge-server --model 0xSero/MiniMax-M2.7-161B-REAP-GGUF --quant Q2_K --port 8003
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from miniforge import Miniforge
from miniforge.server.metrics import (
    ENGINE_INFO,
    REQUESTS_TOTAL,
    TOKENS_TOTAL,
    render_metrics,
    track_generation,
)
from miniforge.utils.config import M7Config, load_config

logger = logging.getLogger("miniforge.server")


# ---------- request/response schemas (OpenAI-compatible subset) ----------


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    # None = use M7Config.default_max_tokens (20K). 512 was the old pydantic
    # default and silently truncated chat replies when the UI omitted the field.
    max_tokens: int | None = None
    temperature: float | None = 1.0
    top_p: float | None = 0.95
    top_k: int | None = 40
    stream: bool = False
    stop: list[str] | None = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int | None = None
    temperature: float | None = 1.0
    top_p: float | None = 0.95
    top_k: int | None = 40
    stream: bool = False
    stop: list[str] | None = None


class ChoiceDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: ChoiceDelta
    finish_reason: str | None = None


class StreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


# ---------- state ----------


class ServerState:
    model: Miniforge | None = None
    alias: str = "miniforge"
    quant: str = "Q4_K_M"
    model_id: str = "MiniMaxAI/MiniMax-M2.7"


STATE = ServerState()


# ---------- app ----------


def _text_from_message(msg: ChatMessage) -> str:
    """Flatten OpenAI multipart content into plain text."""
    if isinstance(msg.content, str):
        return msg.content
    parts: list[str] = []
    for chunk in msg.content:
        if isinstance(chunk, dict):
            if chunk.get("type") == "text":
                parts.append(str(chunk.get("text", "")))
    return "".join(parts)


def _build_settings_from_env(
    args: argparse.Namespace,
) -> tuple[M7Config, str, str, str | None, str]:
    """Build (config, model_id, quantization, download_dir, backend) from args + env."""
    config_path = args.config or os.environ.get("MINIFORGE_CONFIG")
    config = load_config(config_path) if config_path else load_config()

    model_id = (
        args.model
        or os.environ.get("MINIFORGE_MODEL_ID")
        or config.model_id
        or "MiniMaxAI/MiniMax-M2.7"
    )
    quant = (
        args.quant
        or os.environ.get("MINIFORGE_QUANT")
        or config.quantization
        or "Q4_K_M"
    )
    download_dir = (
        args.download_dir
        or os.environ.get("MINIFORGE_DOWNLOAD_DIR")
        or os.environ.get("MODEL_DIR")
        or config.download_dir
    )
    backend = (
        args.backend
        or os.environ.get("MINIFORGE_BACKEND")
        or config.backend
        or "llama_cpp"
    )
    return config, model_id, quant, download_dir, backend


async def _load_model(args: argparse.Namespace) -> None:
    config, model_id, quant, download_dir, backend = _build_settings_from_env(args)
    # Keep the dataclass in sync so downstream code that reads config.backend
    # (e.g. Miniforge.__init__ fallback) sees the resolved value.
    config.backend = backend
    logger.info(
        "Loading miniforge model id=%s quant=%s backend=%s download_dir=%s",
        model_id,
        quant,
        backend,
        download_dir,
    )
    model = await Miniforge.from_pretrained(
        model_id,
        quantization=quant,
        config=config,
        backend=backend,
        download_dir=download_dir,
    )
    STATE.model = model
    STATE.model_id = model_id
    STATE.quant = quant
    STATE.alias = args.alias or os.environ.get("MINIFORGE_ALIAS") or model_id
    ENGINE_INFO.labels(engine="miniforge", model=STATE.alias, quantization=quant).set(1)
    logger.info("miniforge-server ready as '%s' (backend=%s)", STATE.alias, backend)


def make_app(args: argparse.Namespace) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await _load_model(args)
        try:
            yield
        finally:
            if STATE.model is not None:
                await STATE.model.cleanup()
                STATE.model = None

    app = FastAPI(title="miniforge OpenAI server", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        REQUESTS_TOTAL.labels(endpoint="/health", status="200").inc()
        return {
            "status": "ok" if STATE.model else "loading",
            "model": STATE.alias,
            "quantization": STATE.quant,
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        REQUESTS_TOTAL.labels(endpoint="/v1/models", status="200").inc()
        return {
            "object": "list",
            "data": [
                {
                    "id": STATE.alias,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "miniforge",
                    "metadata": {
                        "quantization": STATE.quant,
                        "source_model": STATE.model_id,
                    },
                }
            ],
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        body, content_type = render_metrics()
        return Response(content=body, media_type=content_type)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, http_request: Request):
        if STATE.model is None:
            REQUESTS_TOTAL.labels(endpoint="/v1/chat/completions", status="503").inc()
            raise HTTPException(503, "Model still loading")

        model = STATE.model
        messages = [m.dict() for m in req.messages]
        system_prompt = None
        history: list[dict[str, str]] = []
        user_message = ""

        for m in messages:
            text = _text_from_message(ChatMessage(**m))
            if m["role"] == "system":
                system_prompt = text
            elif m["role"] == "user":
                # last user message becomes the new prompt
                if user_message:
                    history.append({"role": "user", "content": user_message})
                user_message = text
            elif m["role"] == "assistant":
                history.append({"role": "assistant", "content": text})

        # approximate prompt tokens for metrics
        prompt_tokens = sum(len((m.get("content") or "").split()) for m in messages)
        TOKENS_TOTAL.labels(direction="prompt").inc(prompt_tokens)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        if not req.stream:
            with track_generation() as tracker:
                tracker.first_token()
                response_text = await model.chat(
                    message=user_message,
                    history=history,
                    system_prompt=system_prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    stream=False,
                )
                n_tokens = len(response_text.split()) if isinstance(response_text, str) else 0
                tracker.record(n_tokens)

            REQUESTS_TOTAL.labels(endpoint="/v1/chat/completions", status="200").inc()
            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": STATE.alias,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": response_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": n_tokens,
                        "total_tokens": prompt_tokens + n_tokens,
                    },
                }
            )

        async def event_stream() -> AsyncIterator[bytes]:
            with track_generation() as tracker:
                # Role preamble chunk (OpenAI convention)
                first_chunk = StreamChunk(
                    id=completion_id,
                    created=created,
                    model=STATE.alias,
                    choices=[StreamChoice(delta=ChoiceDelta(role="assistant"))],
                )
                yield f"data: {first_chunk.model_dump_json()}\n\n".encode()

                n_tokens = 0
                token_stream = await model.chat(
                    message=user_message,
                    history=history,
                    system_prompt=system_prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    stream=True,
                )
                async for token in token_stream:
                    if n_tokens == 0:
                        tracker.first_token()
                    n_tokens += 1
                    chunk = StreamChunk(
                        id=completion_id,
                        created=created,
                        model=STATE.alias,
                        choices=[StreamChoice(delta=ChoiceDelta(content=token))],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n".encode()

                    if await http_request.is_disconnected():
                        break

                final_chunk = StreamChunk(
                    id=completion_id,
                    created=created,
                    model=STATE.alias,
                    choices=[
                        StreamChoice(delta=ChoiceDelta(), finish_reason="stop")
                    ],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n".encode()
                yield b"data: [DONE]\n\n"

                tracker.record(n_tokens)
            REQUESTS_TOTAL.labels(endpoint="/v1/chat/completions", status="200").inc()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, http_request: Request):
        if STATE.model is None:
            REQUESTS_TOTAL.labels(endpoint="/v1/completions", status="503").inc()
            raise HTTPException(503, "Model still loading")

        completion_id = f"cmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        prompt_tokens = len(req.prompt.split())
        TOKENS_TOTAL.labels(direction="prompt").inc(prompt_tokens)

        if not req.stream:
            with track_generation() as tracker:
                tracker.first_token()
                text = await STATE.model.generate(
                    prompt=req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    stream=False,
                )
                n_tokens = len(text.split()) if isinstance(text, str) else 0
                tracker.record(n_tokens)

            REQUESTS_TOTAL.labels(endpoint="/v1/completions", status="200").inc()
            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": STATE.alias,
                    "choices": [
                        {
                            "index": 0,
                            "text": text,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": n_tokens,
                        "total_tokens": prompt_tokens + n_tokens,
                    },
                }
            )

        async def event_stream() -> AsyncIterator[bytes]:
            with track_generation() as tracker:
                n_tokens = 0
                stream = await STATE.model.generate(
                    prompt=req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    stream=True,
                )
                async for token in stream:
                    if n_tokens == 0:
                        tracker.first_token()
                    n_tokens += 1
                    payload = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": STATE.alias,
                        "choices": [{"index": 0, "text": token, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n".encode()
                    if await http_request.is_disconnected():
                        break

                payload = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": STATE.alias,
                    "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(payload)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                tracker.record(n_tokens)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


# ---------- entry ----------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Miniforge OpenAI-compatible server")
    p.add_argument("--model", "-m", default=None, help="HF model id or local path")
    p.add_argument("--quant", "-q", default=None, help="Quantization (e.g. Q2_K, Q4_K_M, UD-IQ2_XXS)")
    p.add_argument("--config", "-c", default=None, help="M7Config YAML path")
    p.add_argument("--download-dir", "-d", default=None, help="GGUF cache directory")
    p.add_argument(
        "--backend",
        "-b",
        default=None,
        help="Inference backend: llama_cpp, transformers, or airllm",
    )
    p.add_argument("--alias", "-a", default=None, help="Public model alias in /v1/models")
    p.add_argument("--host", default=os.environ.get("MINIFORGE_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("MINIFORGE_PORT", "8003")))
    p.add_argument("--log-level", default=os.environ.get("MINIFORGE_LOG_LEVEL", "info"))
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    app = make_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
