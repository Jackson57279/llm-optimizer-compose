"""Test script optimized for full 192K context on GMKtech M7 with AMD Ryzen 7 PRO 6850H.

Performance optimizations:
- Full 192K context window (194,560 tokens)
- Larger batch sizes for better throughput (n_batch=2048)
- TurboQuant 3-bit KV cache compression
- Flash Attention enabled
- Optimized for AMD Ryzen 8-core CPU
"""

import asyncio
import logging
from miniforge import Miniforge
from miniforge.utils.config import M7Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def main():
    # Optimized configuration for full 192K context
    config = M7Config(
        quantization="UD-IQ2_XXS",
        cache_type_k="turbo3",
        cache_type_v="turbo3",
        n_ctx=194_560,
        n_threads=8,
        n_batch=2048,
        n_ubatch=512,
        n_gpu_layers=0,
        flash_attn=True,
        use_mmap=True,
        verbose=True,
    )

    print("Loading MiniMax M2.7 with 192K context optimization...")
    print(f"Configuration: ctx={config.n_ctx}, batch={config.n_batch}, threads={config.n_threads}")

    # Load model from unsloth GGUF repository
    model = await Miniforge.from_pretrained(
        "unsloth/MiniMax-M2.7-GGUF",
        quantization=config.quantization,
        config=config,
        cache_dir=".",
    )

    print("Model loaded!")

    # Simple chat
    response = await model.chat(
        "Explain quantum computing",
        system_prompt="You are a helpful assistant.",
    )
    print(response)

    # Streaming
    print("\nStreaming response:")
    stream = await model.chat("Tell me a story", stream=True)
    async for token in stream:
        print(token, end="", flush=True)
    print()

    # Cleanup
    await model.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
