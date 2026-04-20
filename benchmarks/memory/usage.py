"""Memory benchmarks for Miniforge."""

import asyncio
import json
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class MemoryMeasurement:
    """Single memory measurement."""

    timestamp: float
    rss_mb: float
    vms_mb: float
    available_mb: float
    percent_used: float


class MemoryBenchmark:
    """Benchmark memory usage patterns."""

    def __init__(self, model, memory_manager=None, output_dir: Path = Path("benchmarks/results")):
        self.model = model
        self.memory_manager = memory_manager
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.process = psutil.Process()

    def _get_memory_info(self) -> MemoryMeasurement:
        """Get current memory information."""
        mem = psutil.virtual_memory()
        proc_mem = self.process.memory_info()

        return MemoryMeasurement(
            timestamp=time.time(),
            rss_mb=proc_mem.rss / (1024**2),
            vms_mb=proc_mem.vms / (1024**2),
            available_mb=mem.available / (1024**2),
            percent_used=mem.percent,
        )

    async def benchmark_model_loading_memory(
        self,
        quantizations: List[str] = None,
    ) -> Dict[str, Any]:
        """Measure memory usage for different quantization levels."""

        if quantizations is None:
            quantizations = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]

        results = []

        print("  Testing memory usage by quantization...")

        for quant in quantizations:
            print(f"    Testing {quant}...")

            # Measure baseline
            gc.collect()
            baseline = self._get_memory_info()

            # This would actually load different quantized models
            # For now, simulate or use model info
            if self.memory_manager:
                fp16_size = 2.7 * 2  # 2.7B params * 2 bytes
                ratios = {
                    "Q8_0": 1.0,
                    "Q6_K": 0.75,
                    "Q5_K_M": 0.625,
                    "Q4_K_M": 0.5,
                    "Q3_K_M": 0.375,
                    "Q2_K": 0.25,
                }
                expected_model_gb = fp16_size * ratios.get(quant, 0.5)

                # Estimate KV cache for full context
                kv_gb = (194560 * 14 * 1024) / (1024**3)  # turbo3, full context
                expected_total = expected_model_gb + kv_gb + 2.0  # working memory
            else:
                expected_model_gb = 1.5
                expected_total = 4.0

            post_load = self._get_memory_info()
            actual_increase = post_load.rss_mb - baseline.rss_mb

            results.append(
                {
                    "quantization": quant,
                    "expected_model_gb": expected_model_gb,
                    "expected_total_gb": expected_total,
                    "measured_increase_mb": actual_increase,
                    "baseline_mb": baseline.rss_mb,
                    "post_load_mb": post_load.rss_mb,
                }
            )

        return {
            "benchmark": "model_loading_memory",
            "quantizations_tested": quantizations,
            "results": results,
        }

    async def benchmark_context_memory_scaling(
        self,
        context_sizes: List[int] = None,
    ) -> Dict[str, Any]:
        """Measure how memory scales with context window size."""

        if context_sizes is None:
            context_sizes = [512, 2048, 8192, 32768, 65536, 131072]

        base_text = "Document text content. " * 50
        results = []

        print("  Testing memory scaling with context...")

        # Baseline
        gc.collect()
        baseline = self._get_memory_info()

        for ctx_size in context_sizes:
            # Build context
            repeats = (ctx_size // len(base_text.split())) + 1
            prompt = (base_text * repeats)[: ctx_size * 5]

            # Process context
            pre = self._get_memory_info()
            _ = await self.model.generate(prompt, max_tokens=1)
            post = self._get_memory_info()

            memory_increase_mb = post.rss_mb - pre.rss_mb

            results.append(
                {
                    "context_size": ctx_size,
                    "memory_increase_mb": memory_increase_mb,
                    "pre_memory_mb": pre.rss_mb,
                    "post_memory_mb": post.rss_mb,
                }
            )

            print(f"    {ctx_size} tokens: +{memory_increase_mb:.1f} MB")

        # Calculate scaling factor
        if len(results) >= 2:
            first = results[0]
            last = results[-1]

            delta_tokens = last["context_size"] - first["context_size"]
            delta_memory = last["memory_increase_mb"] - first["memory_increase_mb"]

            bytes_per_token = (delta_memory * 1024 * 1024) / delta_tokens if delta_tokens > 0 else 0
        else:
            bytes_per_token = 0

        return {
            "benchmark": "context_memory_scaling",
            "context_sizes": context_sizes,
            "bytes_per_token": bytes_per_token,
            "scaling_factor_kb_per_token": bytes_per_token / 1024,
            "results": results,
        }

    async def benchmark_generation_memory_stability(
        self,
        prompt: str = "Write a 500 word essay about artificial intelligence.",
        max_tokens: int = 512,
        sample_interval_ms: float = 100,
    ) -> Dict[str, Any]:
        """Monitor memory stability during generation."""

        measurements = []

        print("  Testing memory stability during generation...")

        # Start generation in background
        async def monitor():
            start = time.time()
            while monitoring:
                measurements.append(self._get_memory_info())
                await asyncio.sleep(sample_interval_ms / 1000)

        monitoring = True
        monitor_task = asyncio.create_task(monitor())

        # Generate
        start_mem = self._get_memory_info()
        result = await self.model.generate(prompt, max_tokens=max_tokens)
        end_mem = self._get_memory_info()

        monitoring = False
        await monitor_task

        # Analyze
        rss_values = [m.rss_mb for m in measurements]

        return {
            "benchmark": "generation_memory_stability",
            "num_samples": len(measurements),
            "mean_rss_mb": sum(rss_values) / len(rss_values),
            "max_rss_mb": max(rss_values),
            "min_rss_mb": min(rss_values),
            "std_rss_mb": self._std(rss_values),
            "rss_increase_mb": end_mem.rss_mb - start_mem.rss_mb,
            "measurements": [
                {"timestamp": m.timestamp, "rss_mb": m.rss_mb}
                for m in measurements[::10]  # Subsample for output
            ],
        }

    async def benchmark_memory_efficiency_score(
        self,
    ) -> Dict[str, Any]:
        """Calculate overall memory efficiency score."""

        print("  Calculating memory efficiency score...")

        # Test parameters
        test_context = 8192
        test_generation = 256

        # Measure
        gc.collect()
        baseline = self._get_memory_info()

        prompt = "Test content. " * (test_context // 2)
        _ = await self.model.generate(prompt, max_tokens=test_generation)

        peak = self._get_memory_info()
        gc.collect()
        post_gc = self._get_memory_info()

        # Calculate metrics
        total_available_mb = baseline.available_mb + baseline.rss_mb
        peak_usage_percent = (peak.rss_mb / total_available_mb) * 100

        # Efficiency score (0-100)
        # Factors: peak usage, garbage collection recovery, relative to theoretical minimum
        theoretical_min_mb = 2000  # ~2GB for model + context

        usage_score = max(0, 100 - (peak.rss_mb - theoretical_min_mb) / 50)
        recovery_score = (
            ((peak.rss_mb - post_gc.rss_mb) / peak.rss_mb) * 100 if peak.rss_mb > 0 else 100
        )

        efficiency_score = usage_score * 0.6 + recovery_score * 0.4

        return {
            "benchmark": "memory_efficiency_score",
            "efficiency_score": min(100, max(0, efficiency_score)),
            "peak_usage_percent": peak_usage_percent,
            "peak_rss_mb": peak.rss_mb,
            "post_gc_rss_mb": post_gc.rss_mb,
            "memory_recovered_mb": peak.rss_mb - post_gc.rss_mb,
            "baseline_mb": baseline.rss_mb,
        }

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    async def run_all(self) -> Dict[str, Any]:
        """Run all memory benchmarks."""

        print("\n=== Memory Benchmarks ===")

        all_results = {
            "benchmark_type": "memory",
            "timestamp": time.time(),
            "system_info": {
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            },
            "results": [],
        }

        # Model loading memory
        print("\nBenchmarking model loading memory...")
        result = await self.benchmark_model_loading_memory()
        all_results["results"].append(result)

        # Context scaling
        print("\nBenchmarking context memory scaling...")
        result = await self.benchmark_context_memory_scaling()
        all_results["results"].append(result)

        # Memory stability
        print("\nBenchmarking generation memory stability...")
        result = await self.benchmark_generation_memory_stability()
        all_results["results"].append(result)

        # Efficiency score
        print("\nBenchmarking memory efficiency...")
        result = await self.benchmark_memory_efficiency_score()
        all_results["results"].append(result)

        # Save results
        self._save_results(all_results, "memory_benchmarks.json")

        return all_results

    def _save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save results to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filepath}")


async def run_memory_benchmarks(model, memory_manager=None):
    """Convenience function to run all memory benchmarks."""
    benchmark = MemoryBenchmark(model, memory_manager)
    return await benchmark.run_all()


if __name__ == "__main__":
    print("Miniforge Memory Benchmarks")
    print("Import and use with a model instance")
