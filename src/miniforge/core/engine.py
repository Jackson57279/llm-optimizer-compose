"""Core engine for model inference."""

from typing import Optional, Dict, Any, List, AsyncIterator, Union
from pathlib import Path
import asyncio
import logging

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Unified inference engine supporting multiple backends.

    Primary: llama-cpp (fastest CPU inference with GGUF)
    Fallback: transformers (native HF models)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        backend: str = "llama_cpp",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if backend == "llama_cpp":
            self.model_path = Path(model_path)
        else:
            self.model_path = model_path
        self.backend_name = backend
        self.config = config or {}
        self._backend = None
        self._initialized = False

    async def initialize(self) -> None:
        """Lazy initialization of backend."""
        if self._initialized:
            return

        if self.backend_name == "llama_cpp":
            from miniforge.core.backends.llama_cpp import LlamaCppBackend

            self._backend = LlamaCppBackend(self.model_path, self.config)
        elif self.backend_name == "transformers":
            from miniforge.core.backends.transformers import TransformersBackend

            self._backend = TransformersBackend(self.model_path, self.config)
        elif self.backend_name == "airllm":
            from miniforge.core.backends.airllm import AirLLMBackend

            self._backend = AirLLMBackend(self.model_path, self.config)
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")

        await self._backend.initialize()
        self._initialized = True
        logger.info(f"InferenceEngine initialized with {self.backend_name} backend")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """Generate text from prompt."""
        if not self._initialized:
            await self.initialize()

        if stream:
            return self._generate_stream(prompt, max_tokens, temperature, top_p, top_k, stop)
        else:
            return await self._generate_sync(prompt, max_tokens, temperature, top_p, top_k, stop)

    async def _generate_sync(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop: Optional[List[str]],
    ) -> str:
        """Synchronous generation."""
        result = await self._backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )
        return result

    async def _generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop: Optional[List[str]],
    ) -> AsyncIterator[str]:
        """Streaming generation."""
        async for token in self._backend.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        ):
            yield token

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """Chat completion with message history."""
        prompt = self._format_chat_prompt(messages)
        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=stream,
        )

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into prompt string."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        formatted.append("Assistant:")
        return "\n\n".join(formatted)

    async def get_info(self) -> Dict[str, Any]:
        """Get model info."""
        if not self._initialized:
            await self.initialize()
        return await self._backend.get_info()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._backend:
            await self._backend.cleanup()
            self._initialized = False
