"""Configuration management for Miniforge."""

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import os
import sys
import yaml
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "MiniMaxAI/MiniMax-M2.7"


@dataclass
class M7Config:
    """
    Configuration optimized for GMKtech M7 hardware.

    Defaults tuned for:
    - AMD Ryzen 7 PRO 6850H (8 cores)
    - 28GB total RAM (4GB to VRAM)
    - CPU-only inference with llama.cpp
    """

    # Hardware limits
    max_memory_gb: float = 24.0  # Leave 4GB for OS
    reserve_memory_gb: float = 4.0

    # CPU settings
    n_threads: int = 8  # Physical cores - Ryzen 7 PRO 6850H
    n_batch: int = 2048  # Increased for better throughput with 192K context
    n_ubatch: int = 512  # Micro-batch size for processing
    cpu_mask: str = "0-7"  # CPU affinity mask (e.g., "0-7" for cores 0-7)
    cpu_strict: bool = False  # Strict CPU pinning (may improve cache locality)
    priority: str = "normal"  # Thread priority: "normal", "realtime", "high"

    # CPU ISA extensions (auto-detected if None)
    use_avx: Optional[bool] = None
    use_avx2: Optional[bool] = None
    use_avx512: Optional[bool] = None
    use_fma: Optional[bool] = None
    use_f16c: Optional[bool] = None

    # Model settings
    model_id: str = DEFAULT_MODEL_ID
    model_weights_path: Optional[str] = None
    n_ctx: int = 194_560  # Full 192K context minus safety headroom (196608 - 2048)
    quantization: str = "UD-IQ2_XXS"  # Default for MiniMax M2.7 228B MoE on 28GB

    # KV cache compression
    # q4_0 halves KV memory vs q8_0, widely supported in llama-cpp.
    # AirLLM-style context sizing will auto-calculate max context for available RAM.
    cache_type_k: str = "q4_0"
    cache_type_v: str = "q4_0"

    # Advanced KV cache settings
    cache_quantization_group: int = 128  # KV cache quantization group size
    cache_mixed_precision: bool = True  # Allow per-layer KV cache quantization

    # Speculative decoding (draft model acceleration)
    speculative_n_max: int = 16  # Max draft tokens to speculatively decode
    speculative_n_min: int = 5   # Min draft tokens for speculative mode
    speculative_p_min: float = 0.8  # Min acceptance probability to continue drafting

    # GPU offloading (for AMD Radeon 680M iGPU)
    n_gpu_layers: int = 0  # Set to 15-20 to offload some layers to 4GB VRAM
    main_gpu: int = 0  # Primary GPU device
    tensor_split: Optional[list[float]] = None  # For multi-GPU setups

    # Performance features
    flash_attn: bool = True
    use_mmap: bool = True
    use_mlock: bool = False  # WSL2 doesn't support mlock
    rope_scaling_type: Optional[str] = "linear"  # For long context scaling
    rope_freq_base: float = 10000.0  # Default RoPE base frequency
    rope_freq_scale: float = 1.0  # RoPE scaling factor (1.0 = no scaling)

    # Backend selection
    backend: str = "llama_cpp"  # llama_cpp, transformers, or airllm

    # Generation defaults
    # 20K-token cap prevents runaway generations on long/open-ended prompts
    # while still leaving plenty of room for code, reasoning traces, and
    # long-form answers. The old 512 silently truncated chat replies.
    default_max_tokens: int = 20480
    default_temperature: float = 1.0
    default_top_p: float = 0.95
    default_top_k: int = 40

    # Feature flags
    enable_tools: bool = True
    enable_vision: bool = True
    enable_streaming: bool = True

    # Paths
    cache_dir: Optional[str] = None
    download_dir: Optional[str] = None  # Override where GGUF shards are downloaded (e.g. "D:/AI")
    llama_cpp_path: Optional[str] = None

    # Per-model overrides (set automatically from registry, or manually)
    max_model_ctx: Optional[int] = None  # Model's trained context ceiling (e.g. 262144 for Kimi K2.5)
    is_moe: bool = False  # Mixture-of-Experts: forces mmap=True, mlock=False, caps default n_ctx

    # Logging
    verbose: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate configuration."""
        if self.n_ctx < 512:
            logger.warning(f"Context window {self.n_ctx} very small, minimum 512 recommended")

        valid_quants = [
            # Standard GGUF quants
            "Q2_K", "Q3_K", "Q3_K_S", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16",
            # Unsloth UD-* quants (1-bit through 8-bit)
            "UD-TQ1_0", "UD-IQ1_S", "UD-IQ1_M",
            "UD-IQ2_XXS", "UD-IQ2_M", "UD-Q2_K_XL",
            "UD-IQ3_XXS", "UD-IQ3_S", "UD-IQ3_K_S", "UD-Q3_K_M", "UD-Q3_K_XL",
            "UD-IQ4_XS", "UD-Q4_K_S", "MXFP4_MOE", "UD-Q4_NL", "UD-Q4_K_M", "UD-Q4_K_XL",
            "UD-Q5_K_S", "UD-Q5_K_M", "UD-Q5_K_XL",
            "UD-Q6_K", "UD-Q6_K_XL", "UD-Q8_K_XL", "BF16",
        ]
        if self.quantization not in valid_quants:
            logger.warning(f"Unknown quantization {self.quantization}, using Q4_K_M")
            self.quantization = "Q4_K_M"

        valid_backends = ["llama_cpp", "transformers", "airllm"]
        if self.backend not in valid_backends:
            logger.warning(f"Unknown backend {self.backend}, using llama_cpp")
            self.backend = "llama_cpp"

    @classmethod
    def performance_preset(cls, preset: str = "balanced") -> "M7Config":
        """Create a config optimized for a specific performance profile.

        Presets:
            - "speed": Maximum throughput, higher memory use
            - "balanced": Good speed with reasonable memory (default)
            - "memory": Minimize RAM usage, may be slower
            - "quality": Best output quality, no compression
            - "moe": Optimized for Mixture-of-Experts models
        """
        config = cls()

        if preset == "speed":
            config.n_threads = 16
            config.n_batch = 4096
            config.n_ubatch = 1024
            config.cache_type_k = "f16"
            config.cache_type_v = "f16"
            config.flash_attn = True
            config.priority = "high"
        elif preset == "balanced":
            config.n_threads = 12
            config.n_batch = 2048
            config.n_ubatch = 512
            config.cache_type_k = "q8_0"
            config.cache_type_v = "q8_0"
            config.flash_attn = True
        elif preset == "memory":
            config.n_threads = 8
            config.n_batch = 512
            config.n_ubatch = 256
            config.cache_type_k = "q4_0"
            config.cache_type_v = "q4_0"
            config.flash_attn = True
            config.quantization = "UD-IQ2_XXS"
        elif preset == "quality":
            config.n_threads = 8
            config.n_batch = 1024
            config.cache_type_k = "f16"
            config.cache_type_v = "f16"
            config.flash_attn = False  # May affect precision slightly
            config.quantization = "Q8_0"
        elif preset == "moe":
            config.n_threads = 16
            config.n_batch = 2048
            config.n_ubatch = 512
            config.cache_type_k = "q4_0"
            config.cache_type_v = "q4_0"
            config.flash_attn = True
            config.use_mmap = True
            config.use_mlock = False
            config.quantization = "UD-IQ2_XXS"

        return config

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "M7Config":
        """Load configuration from YAML file."""
        path = Path(path)

        if not path.exists():
            # Return default config
            logger.info(f"Config file not found at {path}, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f)

        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

        logger.info(f"Configuration saved to {path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def get_backend_config(self) -> Dict[str, Any]:
        """Get backend-specific configuration."""
        config = {
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_batch": self.n_batch,
            "n_ubatch": self.n_ubatch,
            "cache_type_k": self.cache_type_k,
            "cache_type_v": self.cache_type_v,
            "flash_attn": self.flash_attn,
            "use_mmap": self.use_mmap,
            "use_mlock": self.use_mlock,
            "verbose": self.verbose,
            "n_gpu_layers": self.n_gpu_layers,
            "main_gpu": self.main_gpu,
            # Performance optimizations
            "cpu_mask": self.cpu_mask,
            "cpu_strict": self.cpu_strict,
            "priority": self.priority,
        }
        # CPU ISA extensions (only if explicitly set)
        if self.use_avx is not None:
            config["use_avx"] = self.use_avx
        if self.use_avx2 is not None:
            config["use_avx2"] = self.use_avx2
        if self.use_avx512 is not None:
            config["use_avx512"] = self.use_avx512
        if self.use_fma is not None:
            config["use_fma"] = self.use_fma
        if self.use_f16c is not None:
            config["use_f16c"] = self.use_f16c
        if self.tensor_split is not None:
            config["tensor_split"] = self.tensor_split
        if self.rope_scaling_type is not None:
            config["rope_scaling_type"] = self.rope_scaling_type
        config["rope_freq_base"] = self.rope_freq_base
        config["rope_freq_scale"] = self.rope_freq_scale
        if self.max_model_ctx is not None:
            config["max_model_ctx"] = self.max_model_ctx
        if self.is_moe:
            config["is_moe"] = True
        return config

    def get_generation_defaults(self) -> Dict[str, Any]:
        """Get default generation parameters."""
        return {
            "max_tokens": self.default_max_tokens,
            "temperature": self.default_temperature,
            "top_p": self.default_top_p,
            "top_k": self.default_top_k,
        }

    def resolved_model_weights_dir(self) -> Optional[Path]:
        """If set, weights are stored and loaded only from this folder (no duplicate hub cache)."""
        if not self.model_weights_path:
            return None
        return Path(self.model_weights_path).expanduser()


def _default_config_dir() -> Path:
    """
    Return the platform-appropriate config directory for miniforge.

    - Windows: %APPDATA%\\miniforge  (e.g. C:\\Users\\you\\AppData\\Roaming\\miniforge)
    - Other:   ~/.config/miniforge
    """
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "miniforge"
    return Path.home() / ".config" / "miniforge"


def load_config(config_path: Optional[Union[str, Path]] = None) -> M7Config:
    """
    Load configuration from file or return defaults.

    Searches in order:
    1. Provided path
    2. Platform config dir (Windows: %APPDATA%\\miniforge\\config.yaml, other: ~/.config/miniforge/config.yaml)
    3. ~/.config/miniforge/config.yaml  (always checked as fallback on all platforms)
    4. ./miniforge.yaml
    5. Default configuration
    """
    if config_path:
        return M7Config.from_yaml(config_path)

    # Search paths — platform dir first so Windows users find it at a sensible location
    search_paths = [
        _default_config_dir() / "config.yaml",
        Path.home() / ".config" / "miniforge" / "config.yaml",
        Path("miniforge.yaml"),
        Path("config.yaml"),
    ]

    for path in search_paths:
        if path.exists():
            logger.info("Loaded config from %s", path)
            return M7Config.from_yaml(path)

    return M7Config()


def create_default_config_file(path: Optional[Union[str, Path]] = None) -> Path:
    """Create a default configuration file at the platform-appropriate location."""
    if path is None:
        path = _default_config_dir() / "config.yaml"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = M7Config()
    config.to_yaml(path)

    return path
