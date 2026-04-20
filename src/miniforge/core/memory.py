"""Memory management for GMKtech M7 28GB constraint.

AirLLM-inspired: dynamically probe available RAM and size context/cache
to fit, rather than relying on static defaults that cause swap thrashing.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import psutil
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    model_memory_gb: float = 0.0
    kv_cache_gb: float = 0.0


class MemoryManager:
    """
    Manages memory allocation for the GMKtech M7 constraint (28GB total).

    Key constraints:
    - 28GB total system RAM (4GB allocated to VRAM)
    - 4GB reserved for OS and WSL2 overhead
    - 24GB available for model inference
    """

    # Hardware limits
    TOTAL_RAM_GB = 28.0
    RESERVE_OS_GB = 4.0
    MAX_AVAILABLE_GB = TOTAL_RAM_GB - RESERVE_OS_GB  # 24GB

    # Safety margins
    SAFETY_FACTOR = 0.85  # Use only 85% of available to avoid OOM
    MAX_USABLE_GB = MAX_AVAILABLE_GB * SAFETY_FACTOR  # ~20.4GB

    def __init__(self, target_utilization: Optional[float] = None):
        """
        Initialize memory manager.

        Args:
            target_utilization: Override default safety factor (0.0-1.0)
        """
        if target_utilization:
            self.max_usable_gb = self.MAX_AVAILABLE_GB * target_utilization
        else:
            self.max_usable_gb = self.MAX_USABLE_GB

        self.current_model_memory = 0.0
        self.current_kv_memory = 0.0

        logger.info(f"MemoryManager initialized: max_usable={self.max_usable_gb:.1f}GB")

    def get_stats(self) -> MemoryStats:
        """Get current system memory statistics."""
        mem = psutil.virtual_memory()
        return MemoryStats(
            total_gb=mem.total / (1024**3),
            available_gb=mem.available / (1024**3),
            used_gb=mem.used / (1024**3),
            percent_used=mem.percent,
            model_memory_gb=self.current_model_memory,
            kv_cache_gb=self.current_kv_memory,
        )

    def select_quantization(self, model_params_billions: float) -> str:
        """
        Auto-select optimal quantization based on model size.

        Args:
            model_params_billions: Model size in billions of parameters

        Returns:
            Quantization type string (Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0)
        """
        # Estimate FP16 size (2 bytes per param)
        fp16_size_gb = model_params_billions * 2

        logger.info(
            f"Selecting quantization for {model_params_billions}B model "
            f"(FP16: {fp16_size_gb:.1f}GB)"
        )

        # Quantization ratios relative to FP16
        quant_sizes = {
            "Q8_0": 1.0,  # ~same as FP16 for llama.cpp
            "Q6_K": 0.75,  # 75% of FP16
            "Q5_K_M": 0.625,  # 62.5% of FP16
            "Q4_K_M": 0.5,  # 50% of FP16
            "Q3_K_M": 0.375,  # 37.5% of FP16
            "Q2_K": 0.25,  # 25% of FP16 (aggressive)
        }

        # KV cache overhead with TurboQuant for full 192K context
        # turbo3 = 3-bit = ~14KB per token for 2.7B model
        # For 194,560 tokens (192K usable): ~2.6 GB KV cache
        kv_per_token_kb = 14  # turbo3 (~14 KiB / token for ~2.7B, rough)
        context_tokens = 194_560  # Full 192K context window
        kv_overhead_gb = (kv_per_token_kb * context_tokens) / (1024**2)

        working_memory_gb = 2.0

        preference = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "Q3_K_M", "Q2_K"]
        for quant in preference:
            ratio = quant_sizes[quant]
            model_size = fp16_size_gb * ratio
            total_needed = model_size + kv_overhead_gb + working_memory_gb

            logger.debug(
                f"  {quant}: model={model_size:.1f}GB + kv={kv_overhead_gb:.2f}GB + "
                f"working={working_memory_gb:.1f}GB = {total_needed:.1f}GB"
            )

            if total_needed <= self.max_usable_gb:
                logger.info(
                    f"Selected {quant}: total={total_needed:.1f}GB (model={model_size:.1f}GB)"
                )
                return quant

        logger.warning("Model may not fit in memory, using Q2_K")
        return "Q2_K"

    def calculate_max_context(
        self,
        model_quantized_gb: float,
        kv_cache_type: str = "turbo3",
    ) -> int:
        """
        Calculate maximum safe context window.

        Args:
            model_quantized_gb: Size of quantized model in GB
            kv_cache_type: KV cache quantization type (turbo3, turbo4, q8_0, etc.)

        Returns:
            Maximum context window in tokens
        """
        # KV cache bytes per token depends on model size and quantization
        # Rough estimates for 2.7B model:
        kv_bytes_per_token = {
            "f16": 64 * 1024,  # ~64KB per token
            "q8_0": 32 * 1024,  # ~32KB per token
            "q4_0": 16 * 1024,  # ~16KB per token
            "turbo4": 18 * 1024,  # ~18KB per token (4-bit)
            "turbo3": 14 * 1024,  # ~14KB per token (3-bit)
        }

        bytes_per_token = kv_bytes_per_token.get(kv_cache_type, 14 * 1024)

        # Available memory after model load
        available_gb = self.max_usable_gb - model_quantized_gb
        # Reserve working memory
        available_gb -= 2.0

        if available_gb <= 0:
            logger.warning("No memory available for KV cache!")
            return 512  # Minimum safe context

        # Calculate max tokens
        max_tokens = int((available_gb * 1024**3) / bytes_per_token)

        # Round to common context sizes including full 192K
        context_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 194_560, 200_000]
        safe_context = 512
        for size in context_sizes:
            if size <= max_tokens:
                safe_context = size
            else:
                break

        logger.info(
            f"Max context: {safe_context} tokens (available: {available_gb:.1f}GB, "
            f"kv_type={kv_cache_type})"
        )

        return safe_context

    def check_memory_available(self, required_gb: float) -> bool:
        """Check if required memory is available."""
        stats = self.get_stats()
        available = stats.available_gb - self.current_model_memory - self.current_kv_memory

        logger.debug(f"Memory check: required={required_gb:.1f}GB, available={available:.1f}GB")

        return available >= required_gb

    def register_model_memory(self, model_gb: float) -> None:
        """Register model memory usage."""
        self.current_model_memory = model_gb
        logger.info(f"Registered model memory: {model_gb:.1f}GB")

    def register_kv_memory(self, context_tokens: int, kv_type: str = "turbo3") -> None:
        """Register KV cache memory usage."""
        kv_bytes_per_token = {
            "f16": 64 * 1024,
            "q8_0": 32 * 1024,
            "q4_0": 16 * 1024,
            "turbo4": 18 * 1024,
            "turbo3": 14 * 1024,
        }

        bytes_per_token = kv_bytes_per_token.get(kv_type, 14 * 1024)
        kv_gb = (context_tokens * bytes_per_token) / (1024**3)

        self.current_kv_memory = kv_gb
        logger.info(f"Registered KV cache: {context_tokens} tokens = {kv_gb:.2f}GB")

    def compute_moe_context(
        self,
        model_disk_gb: float,
        n_layers: int = 62,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        kv_cache_type: str = "q8_0",
        is_moe: bool = True,
    ) -> int:
        """
        AirLLM-inspired: compute the largest context window that fits in RAM.

        For MoE models, the weights live on disk via mmap. Only a fraction
        (active experts + routing tables) is resident at any time.  The KV
        cache, however, must be fully resident.  So the formula is:

            available_for_kv = physical_RAM - OS_reserve - mmap_resident_estimate
            max_ctx = available_for_kv / kv_bytes_per_token

        Args:
            model_disk_gb: Total GGUF size on disk (e.g. 61 for IQ2_XXS).
            n_layers: Number of transformer layers.
            n_kv_heads: Number of KV attention heads.
            head_dim: Dimension per head.
            kv_cache_type: Cache quantization (q4_0, q8_0, f16).
            is_moe: Whether this is a Mixture-of-Experts model.

        Returns:
            Recommended n_ctx that fits in RAM without swap thrashing.
        """
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)

        # Reserve for OS + other processes
        os_reserve_gb = max(4.0, total_gb * 0.15)

        if is_moe:
            # MoE models: only ~5-15% of weights are resident via mmap at any time
            # (active experts + routing tables + embeddings).
            # On 28GB with 61GB model: ~3-6GB resident, rest paged from SSD.
            mmap_resident_gb = min(model_disk_gb * 0.10, total_gb * 0.25)
        else:
            # Dense models: most weights will be resident
            mmap_resident_gb = min(model_disk_gb, total_gb * 0.7)

        # Working memory for compute buffers
        working_gb = 1.0

        budget_gb = total_gb - os_reserve_gb - mmap_resident_gb - working_gb
        budget_gb = max(budget_gb, 0.5)  # Floor

        # KV cache bytes per token: 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
        dtype_bytes = {
            "f16": 2.0,
            "q8_0": 1.0,
            "q4_0": 0.5,
            "q4_1": 0.5625,
        }
        bpt = dtype_bytes.get(kv_cache_type, 1.0)
        kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * bpt

        max_tokens = int((budget_gb * 1024**3) / kv_bytes_per_token)

        # Snap to standard sizes
        standard_sizes = [
            1024, 2048, 4096, 8192, 16384, 32768,
            65536, 131072, 194_560,
        ]
        best_ctx = 2048  # minimum useful size
        for size in standard_sizes:
            if size <= max_tokens:
                best_ctx = size
            else:
                break

        logger.info(
            "AirLLM context sizing: RAM=%.1fGB, budget=%.1fGB (os=%.1f, mmap_resident=%.1f, "
            "working=%.1f), kv_per_token=%dB, max_tokens=%d -> n_ctx=%d",
            total_gb,
            budget_gb,
            os_reserve_gb,
            mmap_resident_gb,
            working_gb,
            int(kv_bytes_per_token),
            max_tokens,
            best_ctx,
        )

        return best_ctx

    def get_optimal_settings(self, model_params_billions: float) -> Dict[str, Any]:
        """
        Get optimal settings for a model on this hardware.

        Returns dict with recommended settings:
        - quantization
        - n_ctx
        - cache_type_k
        - cache_type_v
        - n_threads
        - n_batch
        """
        quant = self.select_quantization(model_params_billions)

        # Get model size for this quantization
        fp16_size = model_params_billions * 2
        quant_ratios = {
            "Q8_0": 1.0,
            "Q6_K": 0.75,
            "Q5_K_M": 0.625,
            "Q4_K_M": 0.5,
            "Q3_K_M": 0.375,
            "Q2_K": 0.25,
        }
        model_size = fp16_size * quant_ratios.get(quant, 0.5)

        # Use turbo3 for KV cache (best compression, minimal quality loss)
        kv_type = "turbo3"
        max_ctx = self.calculate_max_context(model_size, kv_type)

        # CPU threads - use physical cores (8 for Ryzen 7 PRO 6850H)
        cpu_count = psutil.cpu_count(logical=False) or 4

        # Batch size optimized for context window (larger = better throughput)
        if max_ctx >= 131072:  # Full 192K context
            batch_size = 2048
            ubatch_size = 512
        elif max_ctx >= 8192:
            batch_size = 1024
            ubatch_size = 512
        elif max_ctx >= 4096:
            batch_size = 512
            ubatch_size = 256
        else:
            batch_size = 256
            ubatch_size = 128

        settings = {
            "quantization": quant,
            "n_ctx": max_ctx,
            "cache_type_k": kv_type,
            "cache_type_v": kv_type,
            "n_threads": cpu_count,
            "n_batch": batch_size,
            "n_ubatch": ubatch_size,
            "flash_attn": True,
            "use_mmap": True,
            "use_mlock": False,  # WSL2 doesn't support mlock well
        }

        logger.info(f"Optimal settings for {model_params_billions}B model: {settings}")

        return settings
