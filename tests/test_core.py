"""Test core functionality."""

import pytest
from pathlib import Path

from miniforge.core.memory import MemoryManager, MemoryStats
from miniforge.utils.config import M7Config


def test_memory_manager_initialization():
    """Test memory manager initialization."""
    mem = MemoryManager()

    # Should have correct limits for M7
    assert mem.TOTAL_RAM_GB == 28.0
    assert mem.MAX_AVAILABLE_GB == 24.0
    assert mem.max_usable_gb > 0


def test_memory_manager_select_quantization():
    """Test auto quantization selection."""
    mem = MemoryManager()

    # 2.7B model should fit with Q4_K_M
    quant = mem.select_quantization(2.7)
    assert quant in ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"]

    # 7B model might need more aggressive quantization
    quant_7b = mem.select_quantization(7.0)
    assert quant_7b in ["Q3_K_M", "Q4_K_M"]


def test_memory_manager_calculate_max_context():
    """Test context window calculation."""
    mem = MemoryManager()

    # With 3.1GB model, should get good context
    max_ctx = mem.calculate_max_context(3.1, "turbo3")
    assert max_ctx >= 4096
    assert max_ctx <= 200_000


def test_memory_stats():
    """Test memory stats dataclass."""
    stats = MemoryStats(
        total_gb=28.0,
        available_gb=20.0,
        used_gb=8.0,
        percent_used=28.5,
    )

    assert stats.total_gb == 28.0
    assert stats.available_gb == 20.0


def test_m7_config_defaults():
    """Test default configuration."""
    config = M7Config()

    assert config.n_threads == 8
    # 192K usable context (196608 trained - 2048 safety headroom)
    assert config.n_ctx == 194_560
    # Default targets 228B MoE, not small dense models
    assert config.quantization == "UD-IQ2_XXS"
    assert config.cache_type_k == "q4_0"
    assert config.max_memory_gb == 24.0


def test_m7_config_validation():
    """Test configuration validation."""
    # Invalid quantization should be corrected
    config = M7Config(quantization="INVALID")
    assert config.quantization == "Q4_K_M"

    # Invalid backend should be corrected
    config2 = M7Config(backend="invalid")
    assert config2.backend == "llama_cpp"


def test_config_to_dict():
    """Test config serialization."""
    config = M7Config()
    data = config.to_dict()

    assert isinstance(data, dict)
    assert "n_ctx" in data
    assert "quantization" in data


def test_backend_config():
    """Test backend config extraction."""
    config = M7Config()
    backend_config = config.get_backend_config()

    assert "n_ctx" in backend_config
    assert "cache_type_k" in backend_config
    assert "flash_attn" in backend_config


def test_auto_detect_config():
    """Test auto-detected configuration values."""
    config = M7Config.auto_detect()

    assert config.n_threads >= 1
    assert config.max_memory_gb > 0
    assert config.reserve_memory_gb >= 4.0
    assert config.cpu_mask.startswith("0")


def test_generation_defaults():
    """Test generation defaults."""
    config = M7Config(
        default_max_tokens=1024,
        default_temperature=0.7,
    )

    defaults = config.get_generation_defaults()
    assert defaults["max_tokens"] == 1024
    assert defaults["temperature"] == 0.7


@pytest.mark.asyncio
async def test_engine_initialization_mock():
    """Mock test for engine initialization."""
    # This would require actual model files to test fully
    # For now, just test the config passes through
    from miniforge.core.engine import InferenceEngine

    # Can't test without model, but can verify structure
    pass
