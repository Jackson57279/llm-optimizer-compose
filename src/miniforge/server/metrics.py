"""Prometheus metrics shared across miniforge servers."""

from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry()

REQUESTS_TOTAL = Counter(
    "miniforge_requests_total",
    "Total HTTP requests grouped by endpoint and status.",
    labelnames=("endpoint", "status"),
    registry=REGISTRY,
)

TOKENS_TOTAL = Counter(
    "miniforge_tokens_total",
    "Total tokens processed.",
    labelnames=("direction",),  # prompt | completion
    registry=REGISTRY,
)

TTFT_SECONDS = Histogram(
    "miniforge_ttft_seconds",
    "Time to first token in seconds.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
    registry=REGISTRY,
)

GEN_SECONDS = Histogram(
    "miniforge_generation_seconds",
    "Total wall time of a single generation.",
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
    registry=REGISTRY,
)

TPS = Gauge(
    "miniforge_tps",
    "Tokens/sec of the most recent generation.",
    registry=REGISTRY,
)

ACTIVE_GENERATIONS = Gauge(
    "miniforge_active_generations",
    "Number of generations currently in flight.",
    registry=REGISTRY,
)

ENGINE_INFO = Gauge(
    "miniforge_engine_info",
    "Engine metadata (labels carry the info; value is always 1).",
    labelnames=("engine", "model", "quantization"),
    registry=REGISTRY,
)


@contextmanager
def track_generation():
    """Context manager that updates TTFT/TPS/active counters around one generation.

    Usage:
        with track_generation() as tracker:
            tracker.first_token()   # call as soon as the first token lands
            ...
            tracker.record(n_tokens)  # call with final completion token count
    """
    ACTIVE_GENERATIONS.inc()
    started = time.perf_counter()
    first_token_ts: list[float] = []

    class _Tracker:
        def first_token(self) -> None:
            if not first_token_ts:
                first_token_ts.append(time.perf_counter())
                TTFT_SECONDS.observe(first_token_ts[0] - started)

        def record(self, n_tokens: int) -> None:
            elapsed = time.perf_counter() - started
            GEN_SECONDS.observe(elapsed)
            if n_tokens > 0 and elapsed > 0:
                TPS.set(n_tokens / elapsed)
            TOKENS_TOTAL.labels(direction="completion").inc(max(n_tokens, 0))

    try:
        yield _Tracker()
    finally:
        ACTIVE_GENERATIONS.dec()


def render_metrics() -> tuple[bytes, str]:
    """Return (body, content_type) for a /metrics response."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
