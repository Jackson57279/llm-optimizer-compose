"""AirLLM backend for layer-streamed inference of large models.

AirLLM loads HuggingFace safetensors one transformer block at a time and
discards them between layers, trading throughput (~0.3 t/s on a 161B-class
model) for the ability to run models that do not fit in RAM.

Only accepts HF repo IDs or local HF-format directories — NOT GGUF files.
"""

from typing import Optional, Dict, Any, List, AsyncIterator, Union
from pathlib import Path
import asyncio
import logging
import os

from miniforge.core.backends.transformers import resolve_pretrained_source

logger = logging.getLogger(__name__)


class AirLLMBackend:
    """Backend that streams model layers from disk via the `airllm` package."""

    def __init__(self, model_path: Union[str, Path], config: Dict[str, Any]):
        self.model_path = resolve_pretrained_source(model_path)
        self.config = config
        self._model = None
        self._tokenizer = None
        self._max_ctx: int = int(
            config.get("airllm_max_ctx")
            or os.environ.get("AIRLLM_MAX_CTX")
            or 1024
        )
        self._compression: Optional[str] = (
            config.get("airllm_compression")
            or os.environ.get("AIRLLM_COMPRESSION")
            or "4bit"
        ) or None
        self._shards_dir: Optional[str] = (
            config.get("airllm_shards_dir")
            or os.environ.get("AIRLLM_SHARDS_DIR")
        )

    async def initialize(self) -> None:
        try:
            from airllm import AutoModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "airllm is not installed. Install with: uv pip install -e '.[airllm]'"
            ) from e

        kwargs: Dict[str, Any] = {
            "compression": self._compression,
            "profiling_mode": False,
        }
        if self._shards_dir:
            kwargs["layer_shards_saving_path"] = self._shards_dir
        if os.environ.get("HF_TOKEN"):
            kwargs["hf_token"] = os.environ["HF_TOKEN"]

        logger.info(
            "AirLLM loading %s (compression=%s, max_ctx=%d)...",
            self.model_path,
            self._compression,
            self._max_ctx,
        )

        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None, lambda: AutoModel.from_pretrained(self.model_path, **kwargs)
        )
        self._tokenizer = self._model.tokenizer
        logger.info("AirLLM model loaded")

    def _tokenize(self, prompt: str):
        inputs = self._tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=self._max_ctx,
            padding=False,
        )
        input_ids = inputs["input_ids"]
        try:
            input_ids = input_ids.cuda()
        except Exception:
            pass
        return input_ids

    def _run_generate(self, input_ids, max_tokens: int) -> str:
        output = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            use_cache=True,
            return_dict_in_generate=True,
        )
        tokens = output.sequences[0].tolist()
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
    ) -> str:
        if not self._model or not self._tokenizer:
            raise RuntimeError("Backend not initialized")

        # AirLLM ignores temperature/top_p/top_k sampling knobs — it runs greedy
        # decoding in its generate(). We accept them for interface parity.
        input_ids = self._tokenize(prompt)
        effective_max = max_tokens if max_tokens and max_tokens > 0 else 256

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None, self._run_generate, input_ids, effective_max
        )

        if text.startswith(prompt):
            text = text[len(prompt):]

        if stop:
            for s in stop:
                if s and s in text:
                    text = text[: text.index(s)]
                    break

        return text

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        # AirLLM has no token-level streaming hook. Run to completion, then
        # emit the full text as a single chunk so the OpenAI streaming contract
        # is preserved.
        text = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )
        if text:
            yield text

    async def get_info(self) -> Dict[str, Any]:
        if not self._model:
            return {"status": "not_initialized"}
        return {
            "backend": "airllm",
            "model_path": str(self.model_path),
            "compression": self._compression or "fp16",
            "max_ctx": self._max_ctx,
            "shards_dir": self._shards_dir,
        }

    async def cleanup(self) -> None:
        if self._model:
            del self._model
            self._model = None
        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None
        logger.info("AirLLM model cleaned up")
