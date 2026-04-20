"""Test script for Kimi K2.5 on GMKtech M7 (28 GB RAM) via MoE mmap.

How it works:
- Kimi K2.5 has 1T total params but only 8/384 experts activate per token (2%).
- llama.cpp mmap pages only the active expert weights from disk each forward pass.
- Effective RAM footprint is far smaller than the full model file.

Quantization options (pick based on free disk space on D:/AI):
  UD-TQ1_0  — 240 GB  (smallest, recommended)
  Q2_K      — 373 GB  (bakosh abliterated repo)
  Q3_K      — ~550 GB

Performance expectations on NVMe:
  ~0.5–2 tok/s (expert cache warms up after the first few tokens)
"""

import asyncio
import logging
from miniforge import Miniforge
from miniforge.utils.config import M7Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# --- Configuration -------------------------------------------------------
# n_ctx=8192 keeps the KV cache comfortably inside 28 GB RAM.
# Raise to 16384 if you have headroom; 32768+ will likely OOM.
# download_dir: where the GGUF shards live (or will be downloaded to).
DOWNLOAD_DIR = "D:/AI"
QUANTIZATION = "UD-TQ1_0"   # Change to "Q2_K" for the bakosh abliterated repo

# To use the uncensored bakosh version instead, swap model_id below:
#   MODEL_ID = "bakosh/Huihui-Kimi-K2.5-BF16-abliterated-GGUF"
#   QUANTIZATION = "Q2_K"
MODEL_ID = "unsloth/Kimi-K2.5-GGUF"
# -------------------------------------------------------------------------


async def main():
    config = M7Config(
        quantization=QUANTIZATION,
        cache_type_k="turbo3",
        cache_type_v="turbo3",
        n_ctx=8_192,       # Safe default for 28 GB — MoE KV cache grows fast
        n_threads=8,
        n_batch=512,       # Smaller batches work better when experts page from disk
        n_ubatch=128,
        n_gpu_layers=0,    # CPU-only; set >0 to offload attention to Radeon 680M
        flash_attn=True,
        use_mmap=True,     # Critical — this is what makes MoE viable on 28 GB
        use_mlock=False,   # Never lock 240+ GB in RAM
        verbose=True,
    )

    print(f"Loading {MODEL_ID} ({QUANTIZATION}) from {DOWNLOAD_DIR}...")
    print(f"Config: ctx={config.n_ctx}, batch={config.n_batch}, threads={config.n_threads}")
    print("Note: first token may be slow while expert weights are paged from disk.\n")

    model = await Miniforge.from_pretrained(
        MODEL_ID,
        quantization=QUANTIZATION,
        config=config,
        download_dir=DOWNLOAD_DIR,
    )

    print("Model loaded!\n")

    # --- Basic chat -------------------------------------------------------
    response = await model.chat(
        "Explain quantum computing in simple terms.",
        system_prompt="You are a helpful assistant.",
        max_tokens=256,
    )
    print(f"Assistant: {response}\n")

    # --- Streaming --------------------------------------------------------
    print("Streaming response:")
    stream = await model.chat(
        "Write a short poem about the stars.",
        system_prompt="You are a creative writer.",
        max_tokens=128,
        stream=True,
    )
    async for token in stream:
        print(token, end="", flush=True)
    print("\n")

    # --- Memory stats -----------------------------------------------------
    stats = model.get_memory_stats()
    print(f"Memory stats: {stats}")

    await model.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
