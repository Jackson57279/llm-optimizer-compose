"""Performance benchmarks for Miniforge inference."""

import asyncio
import json
import time
import statistics
import psutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ThroughputMetrics:
    """Metrics for throughput benchmarking."""

    tokens_per_second: float
    prompt_tokens: int
    generated_tokens: int
    total_time_ms: float
    time_to_first_token_ms: float
    inter_token_latency_ms: float


class ThroughputBenchmark:
    """Benchmark token generation throughput."""

    PROMPT_TEMPLATES = {
        "code": """Write a Python function to implement quicksort.

The function should:
1. Take a list of integers as input
2. Return a sorted list
3. Use the quicksort algorithm
4. Include type hints

Provide the complete implementation:""",
        "creative": "Write a short story about a robot discovering emotions. The story should be engaging and approximately 300 words.",
        "analytical": "Analyze the pros and cons of renewable energy sources. Consider solar, wind, and hydroelectric power. Provide a balanced assessment.",
        "qa": "What are the main differences between Python and JavaScript? Compare syntax, use cases, and ecosystem.",
        "summarization": """Summarize the following text in 3 bullet points:

Artificial intelligence has revolutionized numerous industries, from healthcare to transportation. 
Machine learning algorithms can now detect diseases earlier than human doctors, optimize traffic 
flow in smart cities, and even create art and music. However, these advancements also raise 
important ethical questions about privacy, job displacement, and the concentration of power 
in the hands of a few tech companies. As AI continues to evolve, society must grapple with 
balancing innovation against these potential risks.""",
    }

    def __init__(self, model, output_dir: Path = Path("benchmarks/results")):
        self.model = model
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def benchmark_generation_throughput(
        self,
        prompt_key: str = "code",
        max_tokens: int = 512,
        iterations: int = 5,
        warmup: int = 2,
    ) -> Dict[str, Any]:
        """Measure text generation throughput."""

        prompt = self.PROMPT_TEMPLATES[prompt_key]

        # Warmup runs
        print(f"  Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            _ = await self.model.generate(prompt, max_tokens=64)

        # Benchmark runs
        print(f"  Running {iterations} iterations...")
        results = []

        for i in range(iterations):
            start_time = time.perf_counter()

            result = await self.model.generate(prompt, max_tokens=max_tokens)

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Approximate token count (words / 0.75 is rough estimate for tokens)
            output_tokens = int(len(result.split()) / 0.75)
            input_tokens = int(len(prompt.split()) / 0.75)

            throughput = output_tokens / (duration_ms / 1000)

            results.append(
                {
                    "duration_ms": duration_ms,
                    "output_tokens": output_tokens,
                    "throughput_tok_s": throughput,
                    "iteration": i + 1,
                }
            )

            print(f"    Iteration {i + 1}: {throughput:.2f} tok/s ({duration_ms:.0f}ms)")

        # Calculate statistics
        throughputs = [r["throughput_tok_s"] for r in results]
        durations = [r["duration_ms"] for r in results]

        return {
            "benchmark": "generation_throughput",
            "prompt_type": prompt_key,
            "max_tokens": max_tokens,
            "iterations": iterations,
            "mean_throughput_tok_s": statistics.mean(throughputs),
            "std_throughput_tok_s": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
            "min_throughput_tok_s": min(throughputs),
            "max_throughput_tok_s": max(throughputs),
            "p95_throughput_tok_s": np.percentile(throughputs, 95),
            "mean_duration_ms": statistics.mean(durations),
            "raw_results": results,
        }

    async def benchmark_prompt_processing(
        self,
        context_sizes: List[int] = None,
    ) -> Dict[str, Any]:
        """Measure prompt processing speed at different context lengths."""

        if context_sizes is None:
            context_sizes = [512, 1024, 2048, 4096, 8192, 16384]

        base_text = "The quick brown fox jumps over the lazy dog. "
        results = []

        print("  Testing prompt processing at different context sizes...")

        for ctx_size in context_sizes:
            # Generate prompt of appropriate length
            repeats = (ctx_size // len(base_text.split())) + 1
            prompt = (base_text * repeats)[: ctx_size * 5]  # Approximate char per token

            # Time processing (generate just 1 token)
            start = time.perf_counter()
            _ = await self.model.generate(prompt, max_tokens=1)
            end = time.perf_counter()

            duration_ms = (end - start) * 1000
            processing_speed = ctx_size / (duration_ms / 1000)  # tokens/second

            results.append(
                {
                    "context_size": ctx_size,
                    "duration_ms": duration_ms,
                    "processing_speed_tok_s": processing_speed,
                }
            )

            print(f"    {ctx_size} tokens: {processing_speed:.2f} tok/s ({duration_ms:.0f}ms)")

        speeds = [r["processing_speed_tok_s"] for r in results]

        return {
            "benchmark": "prompt_processing",
            "context_sizes": context_sizes,
            "mean_speed_tok_s": statistics.mean(speeds),
            "results_by_size": results,
        }

    async def benchmark_streaming_latency(
        self,
        prompt_key: str = "creative",
        max_tokens: int = 256,
        iterations: int = 5,
    ) -> Dict[str, Any]:
        """Measure streaming token delivery latency."""

        prompt = self.PROMPT_TEMPLATES[prompt_key]
        results = []

        print("  Testing streaming latency...")

        for i in range(iterations):
            token_times = []
            first_token_time = None
            start_time = time.perf_counter()

            stream = await self.model.generate(prompt, max_tokens=max_tokens, stream=True)

            async for token in stream:
                current_time = time.perf_counter()

                if first_token_time is None:
                    first_token_time = current_time
                    ttft_ms = (first_token_time - start_time) * 1000
                else:
                    token_times.append(current_time)

            end_time = time.perf_counter()

            # Calculate inter-token latencies
            inter_token_latencies = []
            for j in range(1, len(token_times)):
                latency_ms = (token_times[j] - token_times[j - 1]) * 1000
                inter_token_latencies.append(latency_ms)

            num_tokens = len(token_times)
            total_time_ms = (end_time - start_time) * 1000

            result = {
                "iteration": i + 1,
                "num_tokens": num_tokens,
                "ttft_ms": ttft_ms if first_token_time else None,
                "total_time_ms": total_time_ms,
                "mean_inter_token_ms": statistics.mean(inter_token_latencies)
                if inter_token_latencies
                else 0,
                "p95_inter_token_ms": np.percentile(inter_token_latencies, 95)
                if inter_token_latencies
                else 0,
            }
            results.append(result)

            print(
                f"    Iteration {i + 1}: TTFT={result['ttft_ms']:.1f}ms, "
                f"Mean ITL={result['mean_inter_token_ms']:.1f}ms"
            )

        ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"]]
        itls = [r["mean_inter_token_ms"] for r in results]

        return {
            "benchmark": "streaming_latency",
            "prompt_type": prompt_key,
            "iterations": iterations,
            "mean_ttft_ms": statistics.mean(ttfts) if ttfts else 0,
            "mean_inter_token_ms": statistics.mean(itls),
            "p95_inter_token_ms": np.percentile(itls, 95),
            "raw_results": results,
        }

    async def run_all(self) -> Dict[str, Any]:
        """Run all throughput benchmarks."""

        print("\n=== Throughput Benchmarks ===")

        all_results = {
            "benchmark_type": "performance",
            "timestamp": time.time(),
            "system_info": self._get_system_info(),
            "results": [],
        }

        # Generation throughput for different prompt types
        for prompt_type in ["code", "qa", "summarization"]:
            print(f"\nBenchmarking: {prompt_type}")
            result = await self.benchmark_generation_throughput(prompt_type)
            all_results["results"].append(result)

        # Prompt processing at scale
        print("\nBenchmarking prompt processing...")
        result = await self.benchmark_prompt_processing()
        all_results["results"].append(result)

        # Streaming latency
        print("\nBenchmarking streaming latency...")
        result = await self.benchmark_streaming_latency()
        all_results["results"].append(result)

        # Save results
        self._save_results(all_results, "performance_benchmarks.json")

        return all_results

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        mem = psutil.virtual_memory()
        cpu_freq = psutil.cpu_freq()

        return {
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "cpu_freq_mhz": cpu_freq.current if cpu_freq else None,
            "total_memory_gb": mem.total / (1024**3),
            "platform": "linux",
        }

    def _save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save results to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filepath}")


async def run_performance_benchmarks(model):
    """Convenience function to run all performance benchmarks."""
    benchmark = ThroughputBenchmark(model)
    return await benchmark.run_all()


if __name__ == "__main__":
    print("Miniforge Performance Benchmarks")
    print("Import and use with a model instance")
