"""engine_manager — one-engine-at-a-time process orchestrator.

Listens on :9292 and presents an OpenAI-compatible facade. Incoming requests
name a route alias (e.g. `minimax-reap-ikllama`); the manager resolves that to
(engine, model), ensures the right engine subprocess is running on its local
port, and streams the response back to the caller.

This replaces docker-compose + llama-swap for our native-Windows target.

Endpoints:
    GET  /health                       liveness + current engine
    GET  /v1/models                    all route aliases from config
    POST /v1/chat/completions          routed by `model` field
    POST /v1/completions               routed by `model` field
    GET  /metrics                      Prometheus
    GET  /engine                       current engine (plain)
    POST /engine/swap {"alias": ...}   preload a route without sending a request

Config:
    ENGINES_CONFIG env var              default: ./config/engines.yaml
    MANAGER_HOST / MANAGER_PORT         default: 127.0.0.1 / 9292
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from miniforge.launcher.engines import EngineConfig, EngineSpec, ModelSpec, Route
from miniforge.server.metrics import (
    ENGINE_INFO,
    REQUESTS_TOTAL,
    render_metrics,
)

logger = logging.getLogger("miniforge.launcher")


class SwapRequest(BaseModel):
    alias: str


class ManagerState:
    config: EngineConfig | None = None
    current_alias: str | None = None
    current_engine: EngineSpec | None = None
    current_model: ModelSpec | None = None
    current_route: Route | None = None
    process: subprocess.Popen | None = None
    last_used_ts: float = 0.0
    lock: asyncio.Lock = asyncio.Lock()
    unload_task: asyncio.Task | None = None
    http: httpx.AsyncClient | None = None


STATE = ManagerState()


def _normalize_route_alias(alias: str, routes: dict[str, Route]) -> str:
    """Map LiteLLM/OpenAI-style ids (e.g. openai/minimax-m27-full-miniforge) to routes."""
    if alias in routes:
        return alias
    if "/" in alias:
        rest = alias.split("/", 1)[1]
        if rest in routes:
            return rest
    return alias


# ---------- subprocess lifecycle ----------


def _build_command(engine: EngineSpec, model: ModelSpec) -> tuple[list[str], dict[str, str]]:
    args, env = STATE.config.expand_engine_args(engine, model)  # type: ignore[union-attr]
    if engine.kind == "binary":
        if not engine.binary:
            raise ValueError(f"Engine '{engine.name}' is kind=binary but has no 'binary' path")
        bin_path = Path(engine.binary)
        if not bin_path.is_absolute():
            bin_path = Path.cwd() / bin_path
        if not bin_path.exists():
            raise FileNotFoundError(
                f"Engine binary not found: {bin_path}. Did you run the install script?"
            )
        cmd = [str(bin_path), *args]
    elif engine.kind == "python":
        if not engine.module:
            raise ValueError(f"Engine '{engine.name}' is kind=python but has no 'module'")
        cmd = [sys.executable, "-u", "-m", engine.module, *args]
    else:
        raise ValueError(f"Unknown engine kind: {engine.kind}")
    return cmd, env


async def _wait_for_ready(engine: EngineSpec) -> None:
    url = f"http://127.0.0.1:{engine.port}{engine.health_path}"
    deadline = time.perf_counter() + engine.ready_timeout_s
    assert STATE.http is not None
    last_err: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            r = await STATE.http.get(url, timeout=2.0)
            if r.status_code < 500:
                return
        except Exception as e:  # noqa: BLE001 - any exception counts as not-ready
            last_err = e
        # Abort early if process has already died.
        if STATE.process and STATE.process.poll() is not None:
            raise RuntimeError(
                f"Engine '{engine.name}' exited during startup (code={STATE.process.returncode}). "
                "Check its stderr for details."
            )
        await asyncio.sleep(1.0)
    raise TimeoutError(
        f"Engine '{engine.name}' did not respond on {url} within {engine.ready_timeout_s}s "
        f"(last error: {last_err})"
    )


def _stop_process() -> None:
    if STATE.process is None:
        return
    proc = STATE.process
    if proc.poll() is not None:
        STATE.process = None
        return
    logger.info("Stopping engine '%s' (pid=%s)...", STATE.current_alias, proc.pid)
    try:
        proc.terminate()
        try:
            proc.wait(timeout=(STATE.current_engine.kill_timeout_s if STATE.current_engine else 30))
        except subprocess.TimeoutExpired:
            logger.warning("Engine did not exit after terminate(), killing.")
            proc.kill()
            proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to stop engine, trying hard kill")
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    STATE.process = None


async def _start_route(alias: str) -> None:
    assert STATE.config is not None
    engine, model, route = STATE.config.resolve_route(alias)

    if STATE.current_alias == alias and STATE.process and STATE.process.poll() is None:
        STATE.last_used_ts = time.perf_counter()
        return

    _stop_process()

    cmd, extra_env = _build_command(engine, model)
    env = os.environ.copy()
    env.update(extra_env)

    logger.info("Starting engine '%s' for alias '%s': %s", engine.name, alias, cmd)
    # CREATE_NEW_PROCESS_GROUP so Ctrl-C propagates cleanly on Windows.
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    STATE.process = subprocess.Popen(  # noqa: S603 - paths come from trusted config
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        creationflags=creationflags,
    )
    STATE.current_alias = alias
    STATE.current_engine = engine
    STATE.current_model = model
    STATE.current_route = route
    STATE.last_used_ts = time.perf_counter()
    ENGINE_INFO.labels(engine=engine.name, model=alias, quantization="").set(1)

    try:
        await _wait_for_ready(engine)
    except Exception:
        _stop_process()
        STATE.current_alias = None
        raise


async def ensure_engine(alias: str) -> EngineSpec:
    async with STATE.lock:
        await _start_route(alias)
        assert STATE.current_engine is not None
        return STATE.current_engine


# ---------- TTL unloader ----------


async def _idle_watcher() -> None:
    while True:
        await asyncio.sleep(30)
        try:
            async with STATE.lock:
                if STATE.process is None or STATE.current_route is None:
                    continue
                idle = time.perf_counter() - STATE.last_used_ts
                if idle > STATE.current_route.ttl_s:
                    logger.info(
                        "Engine '%s' idle for %.0fs > ttl=%ds; unloading.",
                        STATE.current_alias,
                        idle,
                        STATE.current_route.ttl_s,
                    )
                    _stop_process()
                    STATE.current_alias = None
                    STATE.current_engine = None
                    STATE.current_model = None
                    STATE.current_route = None
        except Exception:  # noqa: BLE001
            logger.exception("idle watcher iteration failed")


# ---------- FastAPI app ----------


def _upstream_base(engine: EngineSpec) -> str:
    return f"http://127.0.0.1:{engine.port}"


def make_app(config_path: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        STATE.config = EngineConfig.load(config_path)
        STATE.http = httpx.AsyncClient(timeout=None)
        STATE.unload_task = asyncio.create_task(_idle_watcher())
        logger.info(
            "engine_manager ready. %d routes, %d engines, %d models.",
            len(STATE.config.routes),
            len(STATE.config.engines),
            len(STATE.config.models),
        )
        try:
            yield
        finally:
            if STATE.unload_task:
                STATE.unload_task.cancel()
            _stop_process()
            if STATE.http:
                await STATE.http.aclose()

    app = FastAPI(title="miniforge engine_manager", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        REQUESTS_TOTAL.labels(endpoint="/health", status="200").inc()
        running = STATE.process is not None and STATE.process.poll() is None
        return {
            "status": "ok",
            "current": STATE.current_alias,
            "running": running,
            "pid": STATE.process.pid if running and STATE.process else None,
        }

    @app.get("/engine")
    async def engine() -> dict[str, Any]:
        REQUESTS_TOTAL.labels(endpoint="/engine", status="200").inc()
        return {"alias": STATE.current_alias}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        REQUESTS_TOTAL.labels(endpoint="/v1/models", status="200").inc()
        assert STATE.config is not None
        return {
            "object": "list",
            "data": [
                {
                    "id": alias,
                    "object": "model",
                    "owned_by": "miniforge",
                    "created": int(time.time()),
                    "metadata": {"engine": route.engine, "model": route.model},
                }
                for alias, route in STATE.config.routes.items()
            ],
        }

    @app.post("/engine/swap")
    async def swap(req: SwapRequest) -> dict[str, Any]:
        try:
            assert STATE.config is not None
            alias = _normalize_route_alias(req.alias.strip(), STATE.config.routes)
            await ensure_engine(alias)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except (FileNotFoundError, TimeoutError, RuntimeError, ValueError) as e:
            raise HTTPException(500, str(e)) from e
        REQUESTS_TOTAL.labels(endpoint="/engine/swap", status="200").inc()
        return {"alias": STATE.current_alias}

    @app.get("/metrics")
    async def metrics() -> Response:
        body, ct = render_metrics()
        return Response(content=body, media_type=ct)

    async def _proxy_openai(path: str, req: Request) -> Response:
        # Parse body once to read .model, then forward verbatim.
        raw = await req.body()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid JSON: {e}") from e

        raw_model = payload.get("model")
        if not raw_model:
            raise HTTPException(400, "Request must include a 'model' field matching a route alias.")

        assert STATE.config is not None
        alias = _normalize_route_alias(str(raw_model).strip(), STATE.config.routes)

        try:
            engine = await ensure_engine(alias)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except (FileNotFoundError, TimeoutError, RuntimeError, ValueError) as e:
            raise HTTPException(500, str(e)) from e

        STATE.last_used_ts = time.perf_counter()

        stream = bool(payload.get("stream"))
        upstream_url = f"{_upstream_base(engine)}{path}"
        headers = {
            k: v for k, v in req.headers.items()
            if k.lower() not in {"host", "content-length"}
        }
        # Many llama.cpp / ik_llama servers ignore "model" or expect their own ids.
        # Forward the resolved route id (not openai/... from LiteLLM).
        payload["model"] = alias

        if stream:
            async def forward_stream() -> AsyncIterator[bytes]:
                assert STATE.http is not None
                async with STATE.http.stream(
                    "POST",
                    upstream_url,
                    json=payload,
                    headers=headers,
                    timeout=None,
                ) as resp:
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield chunk

            REQUESTS_TOTAL.labels(endpoint=path, status="200").inc()
            return StreamingResponse(forward_stream(), media_type="text/event-stream")

        assert STATE.http is not None
        resp = await STATE.http.post(upstream_url, json=payload, headers=headers, timeout=None)
        REQUESTS_TOTAL.labels(endpoint=path, status=str(resp.status_code)).inc()
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={k: v for k, v in resp.headers.items() if k.lower() != "content-encoding"},
            media_type=resp.headers.get("content-type", "application/json"),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        return await _proxy_openai("/v1/chat/completions", req)

    @app.post("/v1/completions")
    async def completions(req: Request):
        return await _proxy_openai("/v1/completions", req)

    return app


# ---------- entry ----------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Miniforge engine_manager")
    p.add_argument(
        "--config",
        "-c",
        default=os.environ.get("ENGINES_CONFIG", "config/engines.yaml"),
    )
    p.add_argument("--host", default=os.environ.get("MANAGER_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("MANAGER_PORT", "9292")))
    p.add_argument("--log-level", default=os.environ.get("MANAGER_LOG_LEVEL", "info"))
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    # Route Ctrl-C cleanly so we kill the child.
    def _graceful(_signum, _frame):
        _stop_process()
        sys.exit(0)

    if not sys.platform.startswith("win"):
        signal.signal(signal.SIGTERM, _graceful)

    app = make_app(config_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
