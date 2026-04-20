"""Utility functions and helpers."""

from typing import Optional
from pathlib import Path
import logging
import sys

from miniforge.utils.config import M7Config
from miniforge.utils.monitoring import PerformanceMonitor, MemoryMonitor, GenerationMetrics

__all__ = [
    "M7Config",
    "PerformanceMonitor",
    "MemoryMonitor",
    "GenerationMetrics",
]


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_cache_dir() -> Path:
    """Get default cache directory."""
    return Path.home() / ".cache" / "miniforge"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path
