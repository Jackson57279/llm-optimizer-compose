"""Transformers backend for native HuggingFace models."""

from typing import Optional, Dict, Any, List, AsyncIterator, Union
from pathlib import Path
import asyncio
import logging

logger = logging.getLogger(__name__)


def resolve_pretrained_source(model_path: Union[str, Path]) -> Union[str, Path]:
    """
    Path('org/model') uses backslashes on Windows, which breaks HF hub repo ids.
    Use the original string or as_posix() when the path is not a real filesystem location.
    """
    if isinstance(model_path, str):
        p = Path(model_path).expanduser()
        try:
            if p.exists():
                return p
        except OSError:
            pass
        return model_path
    p = model_path.expanduser()
    try:
        if p.exists():
            return p
    except OSError:
        pass
    return p.as_posix()


class TransformersBackend:
    """
    Backend using HuggingFace Transformers.

    Fallback when GGUF is not available.
    Supports quantization via BitsAndBytes.
    """

    def __init__(self, model_path: Union[str, Path], config: Dict[str, Any]):
        self.model_path = resolve_pretrained_source(model_path)
        self.config = config
        self._model = None
        self._tokenizer = None
        self._device = None

    async def initialize(self) -> None:
        """Initialize the Transformers model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
        except ImportError:
            raise ImportError(
                "transformers or torch not installed. "
                "Install with: uv pip install transformers torch"
            )

        logger.info(f"Loading model from {self.model_path}")

        # Determine device
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        logger.info(f"Using device: {self._device}")

        # Check for quantization config
        load_in_8bit = self.config.get("load_in_8bit", False)
        load_in_4bit = self.config.get("load_in_4bit", True)

        quantization_config = None
        if load_in_4bit and self._device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using 4-bit quantization")
        elif load_in_8bit and self._device == "cuda":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Using 8-bit quantization")

        # Load in thread pool
        loop = asyncio.get_event_loop()

        self._tokenizer = await loop.run_in_executor(
            None,
            lambda: AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            ),
        )

        # Set pad token if not present
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self._device in ["cuda", "mps"] else torch.float32,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"

        self._model = await loop.run_in_executor(
            None,
            lambda: AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs,
            ),
        )

        if not quantization_config:
            self._model = self._model.to(self._device)

        logger.info("Transformers model loaded successfully")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate text synchronously."""
        if not self._model or not self._tokenizer:
            raise RuntimeError("Backend not initialized")

        import torch

        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._device != "auto":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        # Add stop sequences
        if stop:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class StopSequenceCriteria(StoppingCriteria):
                def __init__(self, stops, tokenizer):
                    self.stops = [tokenizer.encode(s, add_special_tokens=False) for s in stops]
                    self.tokenizer = tokenizer

                def __call__(self, input_ids, scores, **kwargs):
                    for stop_ids in self.stops:
                        if len(input_ids[0]) >= len(stop_ids):
                            if input_ids[0][-len(stop_ids) :].tolist() == stop_ids:
                                return True
                    return False

            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [StopSequenceCriteria(stop, self._tokenizer)]
            )

        # Generate in thread pool
        loop = asyncio.get_event_loop()

        with torch.no_grad():
            outputs = await loop.run_in_executor(
                None, lambda: self._model.generate(inputs["input_ids"], **gen_kwargs)
            )

        # Decode
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        result = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Trim at stop sequence if present
        if stop:
            for s in stop:
                if s in result:
                    result = result[: result.index(s)]

        return result

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """Generate text with streaming."""
        if not self._model or not self._tokenizer:
            raise RuntimeError("Backend not initialized")

        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread

        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._device != "auto":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Setup streamer
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generation kwargs
        gen_kwargs = {
            "input_ids": inputs["input_ids"],
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }

        # Run generation in separate thread
        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        # Stream tokens
        generated_text = ""
        for text in streamer:
            generated_text += text

            # Check for stop sequences
            if stop:
                for s in stop:
                    if s in generated_text:
                        # Return only up to stop sequence
                        idx = generated_text.index(s)
                        if idx > 0:
                            yield generated_text[:idx]
                        thread.join()
                        return

            yield text
            await asyncio.sleep(0)

        thread.join()

    async def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self._model:
            return {"status": "not_initialized"}

        return {
            "backend": "transformers",
            "device": self._device,
            "model_path": str(self.model_path),
        }

    async def cleanup(self) -> None:
        """Cleanup model resources."""
        if self._model:
            del self._model
            self._model = None
        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Transformers model cleaned up")
