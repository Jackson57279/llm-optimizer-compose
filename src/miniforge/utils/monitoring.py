"""Performance monitoring and metrics."""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Metrics for a single generation."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    prompt_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def prompt_tps(self) -> float:
        """Tokens per second for prompt processing."""
        if self.prompt_time_ms > 0:
            return (self.prompt_tokens / self.prompt_time_ms) * 1000
        return 0.0

    @property
    def generation_tps(self) -> float:
        """Tokens per second for generation."""
        if self.generation_time_ms > 0:
            return (self.generated_tokens / self.generation_time_ms) * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "prompt_tps": round(self.prompt_tps, 2),
            "generation_tps": round(self.generation_tps, 2),
            "total_time_ms": round(self.total_time_ms, 2),
        }


class PerformanceMonitor:
    """Monitor generation performance."""

    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self._current_metrics: Optional[GenerationMetrics] = None
        self._start_time: Optional[float] = None

    def start_generation(self) -> None:
        """Start timing a generation."""
        self._start_time = time.time()
        self._current_metrics = GenerationMetrics()

    def record_prompt(self, tokens: int) -> None:
        """Record prompt processing completion."""
        if self._current_metrics and self._start_time:
            elapsed = (time.time() - self._start_time) * 1000
            self._current_metrics.prompt_tokens = tokens
            self._current_metrics.prompt_time_ms = elapsed

    def record_token(self) -> None:
        """Record a generated token."""
        if self._current_metrics:
            self._current_metrics.generated_tokens += 1

    def end_generation(self) -> GenerationMetrics:
        """End timing and return metrics."""
        if self._current_metrics and self._start_time:
            total_elapsed = (time.time() - self._start_time) * 1000
            self._current_metrics.total_time_ms = total_elapsed

            # Estimate generation time if not recorded separately
            if self._current_metrics.generation_time_ms == 0:
                self._current_metrics.generation_time_ms = (
                    total_elapsed - self._current_metrics.prompt_time_ms
                )

            self.metrics_history.append(self._current_metrics)

            metrics = self._current_metrics
            self._current_metrics = None
            self._start_time = None

            return metrics

        return GenerationMetrics()

    def get_average_tps(self, window: Optional[int] = None) -> Dict[str, float]:
        """Get average tokens per second over history."""
        history = list(self.metrics_history)
        if window:
            history = history[-window:]

        if not history:
            return {"prompt_tps": 0.0, "generation_tps": 0.0}

        avg_prompt = sum(m.prompt_tps for m in history) / len(history)
        avg_gen = sum(m.generation_tps for m in history) / len(history)

        return {
            "prompt_tps": round(avg_prompt, 2),
            "generation_tps": round(avg_gen, 2),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.metrics_history:
            return {"generations": 0}

        history = list(self.metrics_history)

        return {
            "generations": len(history),
            "average_metrics": self.get_average_tps(),
            "last_generation": history[-1].to_dict() if history else None,
        }


class MemoryMonitor:
    """Monitor memory usage during inference."""

    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self._monitoring = False
        self._peak_usage_gb = 0.0

    def start(self) -> None:
        """Start memory monitoring."""
        self._monitoring = True
        self._peak_usage_gb = 0.0

        import threading

        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> None:
        """Stop memory monitoring."""
        self._monitoring = False

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        import psutil
        import time

        while self._monitoring:
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024**3)

            if used_gb > self._peak_usage_gb:
                self._peak_usage_gb = used_gb

            time.sleep(self.check_interval)

    def get_peak_usage(self) -> float:
        """Get peak memory usage in GB."""
        return self._peak_usage_gb

    def reset_peak(self) -> None:
        """Reset peak memory tracking."""
        self._peak_usage_gb = 0.0
