"""Comprehensive benchmarking suite for Miniforge.

This module provides a complete benchmarking framework for evaluating
MiniMax model performance across multiple dimensions:
- Inference throughput (tokens/second)
- Memory utilization
- Context window scaling
- Quantization quality impact
- Tool calling latency
- Vision processing performance
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union
from contextlib import asynccontextmanager
import psutil
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    metric: str
    value: float
    unit: str
    iterations: int
    samples: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def mean(self) -> float:
        """Calculate mean of samples."""
        return statistics.mean(self.samples) if self.samples else self.value

    @property
    def std(self) -> float:
        """Calculate standard deviation."""
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    @property
    def min_val(self) -> float:
        """Get minimum value."""
        return min(self.samples) if self.samples else self.value

    @property
    def max_val(self) -> float:
        """Get maximum value."""
        return max(self.samples) if self.samples else self.value

    @property
    def p95(self) -> float:
        """Get 95th percentile."""
        if not self.samples:
            return self.value
        return np.percentile(self.samples, 95)

    @property
    def p99(self) -> float:
        """Get 99th percentile."""
        if not self.samples:
            return self.value
        return np.percentile(self.samples, 99)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "iterations": self.iterations,
            "statistics": {
                "mean": self.mean,
                "std": self.std,
                "min": self.min_val,
                "max": self.max_val,
                "p95": self.p95,
                "p99": self.p99,
            },
            "samples": self.samples,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        return f"{self.name}: {self.mean:.2f} {self.unit} (±{self.std:.2f})"


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    system_info: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a result to the suite."""
        self.results.append(result)

    def finish(self) -> None:
        """Mark suite as complete."""
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        """Get total duration."""
        end = self.end_time or time.time()
        return end - self.start_time

    def get_result(self, name: str) -> Optional[BenchmarkResult]:
        """Get result by name."""
        for r in self.results:
            if r.name == name:
                return r
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "duration_seconds": self.duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "system_info": self.system_info,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save benchmark results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Benchmark results saved to {path}")


class PerformanceBenchmark:
    """Benchmark inference performance metrics."""

    # Standard test prompts of varying lengths
    TEST_PROMPTS = {
        "short": "Explain quantum computing in simple terms.",
        "medium": "Write a detailed analysis of the impacts of artificial intelligence "
        "on modern software engineering practices, including code generation, "
        "testing, and architectural decisions. Discuss both benefits and challenges.",
        "long": "Artificial intelligence has become an integral part of modern technology "
        "infrastructure. From machine learning models that power recommendation "
        "systems to large language models that can generate human-like text, AI "
        "systems are transforming how we interact with computers. In this essay, "
        "we will explore the technical foundations of modern AI systems, including "
        "neural network architectures, training methodologies, and inference "
        "optimization techniques. We will also examine the societal implications "
        "of widespread AI adoption, considering issues of privacy, bias, and "
        "economic disruption. Furthermore, we will analyze the current state of "
        "AI research, identifying key open problems and promising directions "
        "for future investigation. The goal is to provide a comprehensive "
        "overview that balances technical depth with accessibility." * 10,
    }

    def __init__(self, model: Any):
        self.model = model
        self.results: List[BenchmarkResult] = []

    async def benchmark_token_throughput(
        self,
        prompt_type: str = "medium",
        max_tokens: int = 512,
        iterations: int = 5,
        warmup: int = 1,
    ) -> BenchmarkResult:
        """Measure generation throughput in tokens/second."""

        prompt = self.TEST_PROMPTS[prompt_type]
        samples = []

        # Warmup
        for _ in range(warmup):
            _ = await self.model.generate(prompt, max_tokens=64)

        # Benchmark
        for i in range(iterations):
            start_time = time.perf_counter()
            start_tokens = 0  # Would track from model if available

            result = await self.model.generate(prompt, max_tokens=max_tokens)

            end_time = time.perf_counter()
            actual_tokens = len(result.split())  # Approximate

            duration = end_time - start_time
            throughput = actual_tokens / duration if duration > 0 else 0
            samples.append(throughput)

            logger.info(f"Iteration {i + 1}/{iterations}: {throughput:.2f} tok/s")

        return BenchmarkResult(
            name=f"token_throughput_{prompt_type}",
            metric="throughput",
            value=statistics.mean(samples),
            unit="tokens/second",
            iterations=iterations,
            samples=samples,
            metadata={
                "prompt_type": prompt_type,
                "max_tokens": max_tokens,
                "prompt_length": len(prompt),
            },
        )

    async def benchmark_prompt_processing(
        self,
        context_lengths: Optional[List[int]] = None,
        iterations: int = 3,
    ) -> List[BenchmarkResult]:
        """Measure prompt processing speed at different context lengths."""

        if context_lengths is None:
            context_lengths = [512, 1024, 2048, 4096, 8192, 16384]

        results = []
        base_text = "The quick brown fox jumps over the lazy dog. "

        for ctx_len in context_lengths:
            # Generate text of appropriate length
            repeats = (ctx_len // len(base_text.split())) + 1
            prompt = (base_text * repeats)[:ctx_len]

            samples = []
            for i in range(iterations):
                start = time.perf_counter()
                _ = await self.model.generate(prompt, max_tokens=1)
                end = time.perf_counter()

                duration = end - start
                tokens_per_sec = ctx_len / duration if duration > 0 else 0
                samples.append(tokens_per_sec)

            result = BenchmarkResult(
                name=f"prompt_processing_{ctx_len}",
                metric="prompt_processing",
                value=statistics.mean(samples),
                unit="tokens/second",
                iterations=iterations,
                samples=samples,
                metadata={
                    "context_length": ctx_len,
                },
            )
            results.append(result)
            logger.info(f"Context {ctx_len}: {result.mean:.2f} tok/s")

        return results

    async def benchmark_latency(
        self,
        prompt_type: str = "short",
        max_tokens: int = 256,
        iterations: int = 10,
    ) -> BenchmarkResult:
        """Measure time-to-first-token (TTFT) and total latency."""

        prompt = self.TEST_PROMPTS[prompt_type]
        ttft_samples = []
        total_samples = []

        for i in range(iterations):
            # Measure TTFT using streaming
            start = time.perf_counter()
            first_token_time = None
            end_time = None

            stream = await self.model.generate(prompt, max_tokens=max_tokens, stream=True)
            async for token in stream:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                end_time = time.perf_counter()

            if first_token_time and end_time:
                ttft_samples.append((first_token_time - start) * 1000)  # ms
                total_samples.append((end_time - start) * 1000)  # ms

        ttft_result = BenchmarkResult(
            name=f"ttft_{prompt_type}",
            metric="time_to_first_token",
            value=statistics.mean(ttft_samples),
            unit="milliseconds",
            iterations=iterations,
            samples=ttft_samples,
        )

        total_result = BenchmarkResult(
            name=f"total_latency_{prompt_type}",
            metric="total_latency",
            value=statistics.mean(total_samples),
            unit="milliseconds",
            iterations=iterations,
            samples=total_samples,
        )

        return ttft_result, total_result

    async def benchmark_streaming_performance(
        self,
        prompt_type: str = "medium",
        max_tokens: int = 512,
        iterations: int = 5,
    ) -> BenchmarkResult:
        """Measure streaming token delivery performance."""

        prompt = self.TEST_PROMPTS[prompt_type]
        inter_token_delays = []

        for _ in range(iterations):
            prev_time = None
            token_count = 0

            stream = await self.model.generate(prompt, max_tokens=max_tokens, stream=True)
            async for token in stream:
                current_time = time.perf_counter()
                if prev_time is not None:
                    inter_token_delays.append((current_time - prev_time) * 1000)
                prev_time = current_time
                token_count += 1

        avg_delay = statistics.mean(inter_token_delays) if inter_token_delays else 0

        return BenchmarkResult(
            name=f"streaming_latency_{prompt_type}",
            metric="inter_token_latency",
            value=avg_delay,
            unit="milliseconds",
            iterations=iterations,
            samples=inter_token_delays[:100],  # Limit stored samples
            metadata={
                "total_tokens_measured": len(inter_token_delays),
                "p95_delay_ms": np.percentile(inter_token_delays, 95) if inter_token_delays else 0,
            },
        )

    async def run_all(self) -> List[BenchmarkResult]:
        """Run all performance benchmarks."""
        results = []

        # Token throughput
        for prompt_type in ["short", "medium"]:
            result = await self.benchmark_token_throughput(prompt_type)
            results.append(result)

        # Prompt processing at different lengths
        processing_results = await self.benchmark_prompt_processing()
        results.extend(processing_results)

        # Latency benchmarks
        ttft, total = await self.benchmark_latency()
        results.extend([ttft, total])

        # Streaming
        streaming = await self.benchmark_streaming_performance()
        results.append(streaming)

        self.results = results
        return results


class MemoryBenchmark:
    """Benchmark memory usage and efficiency."""

    def __init__(self, model: Any, memory_manager: Any):
        self.model = model
        self.memory_manager = memory_manager
        self.results: List[BenchmarkResult] = []

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent,
        }

    async def benchmark_memory_overhead(
        self,
        context_lengths: Optional[List[int]] = None,
    ) -> List[BenchmarkResult]:
        """Measure memory overhead at different context lengths."""

        if context_lengths is None:
            context_lengths = [512, 2048, 8192, 32768, 131072]

        results = []
        baseline = self.get_memory_usage()

        for ctx_len in context_lengths:
            # Force context creation
            prompt = "Test " * (ctx_len // 5)

            pre_memory = self.get_memory_usage()
            _ = await self.model.generate(prompt, max_tokens=1)
            post_memory = self.get_memory_usage()

            overhead = post_memory["used_gb"] - pre_memory["used_gb"]

            result = BenchmarkResult(
                name=f"memory_overhead_{ctx_len}",
                metric="memory_overhead",
                value=overhead,
                unit="GB",
                iterations=1,
                metadata={
                    "context_length": ctx_len,
                    "baseline_used_gb": pre_memory["used_gb"],
                    "post_used_gb": post_memory["used_gb"],
                },
            )
            results.append(result)

        return results

    async def benchmark_memory_scaling(
        self,
        max_context: int = 131072,
        step: int = 8192,
    ) -> BenchmarkResult:
        """Measure how memory scales with context window."""

        measurements = []
        base_text = "Word "

        for ctx in range(step, max_context + 1, step):
            prompt = base_text * (ctx // len(base_text))

            pre = self.get_memory_usage()
            _ = await self.model.generate(prompt, max_tokens=1)
            post = self.get_memory_usage()

            delta = post["used_gb"] - pre["used_gb"]
            measurements.append((ctx, delta))

        # Calculate scaling factor (should be roughly linear for KV cache)
        if len(measurements) > 1:
            first_point = measurements[0]
            last_point = measurements[-1]
            scaling = (last_point[1] - first_point[1]) / (last_point[0] - first_point[0])
        else:
            scaling = 0

        return BenchmarkResult(
            name="memory_scaling_factor",
            metric="memory_per_token",
            value=scaling * 1024 * 1024,  # Convert to MB per token
            unit="MB/token",
            iterations=len(measurements),
            metadata={
                "measurements": measurements,
                "scaling_type": "linear" if scaling > 0 else "unknown",
            },
        )


class ContextBenchmark:
    """Benchmark context window capabilities."""

    # Needle-in-haystack test
    NEEDLE_TEXT = "The secret code is 8742."

    def __init__(self, model: Any):
        self.model = model

    async def needle_in_haystack(
        self,
        context_lengths: Optional[List[int]] = None,
        needle_depths: Optional[List[float]] = None,
    ) -> BenchmarkResult:
        """Test retrieval accuracy at different context positions."""

        if context_lengths is None:
            context_lengths = [1024, 4096, 16384, 65536, 131072]

        if needle_depths is None:
            needle_depths = [0.0, 0.25, 0.5, 0.75, 1.0]

        results = []
        base_text = (
            "This is a test document with various information. The weather is nice today. " * 50
        )

        for ctx_len in context_lengths:
            for depth in needle_depths:
                # Build context with needle at specific depth
                position = int(ctx_len * depth)
                prefix_len = min(position, ctx_len - len(self.NEEDLE_TEXT) - 100)

                prefix = base_text * ((prefix_len // len(base_text)) + 1)
                prefix = prefix[:prefix_len]

                suffix_len = ctx_len - len(prefix) - len(self.NEEDLE_TEXT)
                suffix = base_text * ((suffix_len // len(base_text)) + 1)
                suffix = suffix[:suffix_len]

                context = f"{prefix} {self.NEEDLE_TEXT} {suffix}"

                prompt = (
                    f"Context: {context}\n\n"
                    "Question: What is the secret code mentioned in the context? "
                    "Answer with just the number."
                )

                response = await self.model.generate(prompt, max_tokens=20, temperature=0)

                # Check if code is in response
                correct = "8742" in response
                results.append(
                    {
                        "context_length": ctx_len,
                        "depth": depth,
                        "correct": correct,
                        "response": response,
                    }
                )

        accuracy = sum(1 for r in results if r["correct"]) / len(results)

        return BenchmarkResult(
            name="needle_in_haystack",
            metric="retrieval_accuracy",
            value=accuracy * 100,
            unit="percent",
            iterations=len(results),
            metadata={
                "detailed_results": results,
                "test_method": "exact_match",
            },
        )

    async def benchmark_context_consistency(
        self,
        context_length: int = 8192,
        iterations: int = 5,
    ) -> BenchmarkResult:
        """Test consistency of responses across context window."""

        # Create a context with key facts
        facts = [
            "Fact 1: The capital of France is Paris.",
            "Fact 2: Water boils at 100 degrees Celsius at sea level.",
            "Fact 3: The speed of light is approximately 299,792,458 meters per second.",
            "Fact 4: Python was created by Guido van Rossum.",
            "Fact 5: The Earth orbits around the Sun.",
        ]

        # Pad to target context length
        filler = "General information text. " * 1000
        context = filler[: context_length // 2]
        for fact in facts:
            context += f" {fact}"
        context += " " + filler[: context_length // 2]

        responses = []
        questions = [
            "What is the capital of France?",
            "At what temperature does water boil at sea level?",
            "Who created Python?",
        ]

        for _ in range(iterations):
            iteration_responses = []
            for question in questions:
                prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
                response = await self.model.generate(prompt, max_tokens=50, temperature=0)
                iteration_responses.append(response)
            responses.append(iteration_responses)

        # Check consistency across iterations
        consistency_scores = []
        for q_idx in range(len(questions)):
            answers = [responses[i][q_idx] for i in range(iterations)]
            # Simple consistency: all answers should be similar
            unique_answers = set(answers)
            consistency = 1.0 / len(unique_answers) if unique_answers else 0
            consistency_scores.append(consistency)

        avg_consistency = statistics.mean(consistency_scores)

        return BenchmarkResult(
            name="context_consistency",
            metric="response_consistency",
            value=avg_consistency * 100,
            unit="percent",
            iterations=iterations,
            metadata={
                "questions": questions,
                "context_length": context_length,
                "consistency_by_question": consistency_scores,
            },
        )


class ToolBenchmark:
    """Benchmark tool calling performance."""

    def __init__(self, model: Any):
        self.model = model

    async def benchmark_tool_latency(
        self,
        num_tools: List[int] = [1, 5, 10, 20],
        iterations: int = 5,
    ) -> List[BenchmarkResult]:
        """Measure latency with varying numbers of available tools."""

        results = []

        for n_tools in num_tools:
            # Create N simple tools
            tools = []
            for i in range(n_tools):
                tool = {
                    "type": "function",
                    "function": {
                        "name": f"tool_{i}",
                        "description": f"Tool number {i}",
                        "parameters": {
                            "type": "object",
                            "properties": {"input": {"type": "string"}},
                        },
                    },
                }
                tools.append(tool)

            samples = []
            for _ in range(iterations):
                start = time.perf_counter()

                # Simulate tool selection request
                prompt = (
                    "You have access to several tools. "
                    "Use the tool_0 tool with input='test'. "
                    "Respond with the tool call in <tool> tags."
                )

                response = await self.model.generate(prompt, max_tokens=100)

                end = time.perf_counter()
                samples.append((end - start) * 1000)

            result = BenchmarkResult(
                name=f"tool_latency_{n_tools}_tools",
                metric="tool_selection_latency",
                value=statistics.mean(samples),
                unit="milliseconds",
                iterations=iterations,
                samples=samples,
                metadata={"num_tools": n_tools},
            )
            results.append(result)

        return results


class QuantizationBenchmark:
    """Benchmark quantization quality impact."""

    # Standard test tasks
    TEST_TASKS = {
        "summarization": {
            "input": "Artificial intelligence (AI) is intelligence demonstrated by machines, "
            "as opposed to the natural intelligence displayed by animals including humans. "
            "AI research has been defined as the field of study of intelligent agents, "
            "which refers to any system that perceives its environment and takes actions "
            "that maximize its chance of achieving its goals.",
            "instruction": "Summarize the above text in one sentence.",
        },
        "qa": {
            "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, "
            "France. It is named after the engineer Gustave Eiffel, whose company designed "
            "and built the tower. Constructed from 1887 to 1889, it was initially criticized "
            "by some of France's leading artists and intellectuals for its design.",
            "question": "Who designed the Eiffel Tower?",
        },
        "reasoning": {
            "problem": "If a train travels 120 km in 2 hours, what is its average speed?",
        },
    }

    def __init__(self, model_factory: Callable):
        """
        Args:
            model_factory: Function that takes quantization type and returns model
        """
        self.model_factory = model_factory

    async def benchmark_perplexity(
        self,
        quantization_types: List[str],
        test_texts: Optional[List[str]] = None,
    ) -> List[BenchmarkResult]:
        """Measure perplexity across quantization types."""

        if test_texts is None:
            test_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
            ]

        results = []

        for quant in quantization_types:
            logger.info(f"Testing perplexity for {quant}...")

            model = await self.model_factory(quant)

            perplexities = []
            for text in test_texts:
                # Calculate perplexity (simplified)
                # In practice, would use proper log-prob calculation
                tokens = len(text.split())

                start = time.perf_counter()
                _ = await model.generate(text, max_tokens=1)
                duration = time.perf_counter() - start

                # Approximate perplexity from generation confidence
                # This is a simplified metric
                approx_perplexity = duration * 10  # Placeholder
                perplexities.append(approx_perplexity)

            avg_perplexity = statistics.mean(perplexities)

            result = BenchmarkResult(
                name=f"perplexity_{quant}",
                metric="perplexity",
                value=avg_perplexity,
                unit="nats",
                iterations=len(test_texts),
                metadata={"quantization": quant},
            )
            results.append(result)

            await model.cleanup()

        return results

    async def benchmark_task_accuracy(
        self,
        quantization_types: List[str],
    ) -> List[BenchmarkResult]:
        """Measure task accuracy across quantization types."""

        results = []

        for quant in quantization_types:
            logger.info(f"Testing task accuracy for {quant}...")

            model = await self.model_factory(quant)

            task_scores = {}

            # Summarization task
            summarization_prompt = (
                f"Text: {self.TEST_TASKS['summarization']['input']}\n\n"
                f"{self.TEST_TASKS['summarization']['instruction']}"
            )
            summary = await model.generate(summarization_prompt, max_tokens=50)
            # Check if summary contains key concepts
            key_concepts = ["machines", "intelligence", "AI"]
            summary_score = sum(1 for concept in key_concepts if concept.lower() in summary.lower())
            task_scores["summarization"] = summary_score / len(key_concepts)

            # QA task
            qa_prompt = (
                f"Context: {self.TEST_TASKS['qa']['context']}\n\n"
                f"Question: {self.TEST_TASKS['qa']['question']}\nAnswer:"
            )
            answer = await model.generate(qa_prompt, max_tokens=30)
            # Check if answer contains correct info
            qa_score = 1.0 if "gustave" in answer.lower() or "eiffel" in answer.lower() else 0.0
            task_scores["qa"] = qa_score

            # Reasoning task
            reasoning_prompt = f"Problem: {self.TEST_TASKS['reasoning']['problem']}"
            reasoning = await model.generate(reasoning_prompt, max_tokens=50)
            # Check if answer is correct (60 km/h)
            reasoning_score = 1.0 if "60" in reasoning else 0.0
            task_scores["reasoning"] = reasoning_score

            overall_accuracy = statistics.mean(task_scores.values())

            result = BenchmarkResult(
                name=f"task_accuracy_{quant}",
                metric="accuracy",
                value=overall_accuracy * 100,
                unit="percent",
                iterations=len(task_scores),
                metadata={
                    "quantization": quant,
                    "task_scores": task_scores,
                },
            )
            results.append(result)

            await model.cleanup()

        return results


async def run_full_benchmark_suite(
    model: Any,
    output_dir: Union[str, Path] = "benchmarks/results",
) -> BenchmarkSuite:
    """Run complete benchmark suite."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect system information
    system_info = {
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
        },
        "memory": {
            "total_gb": psutil.virtual_memory().total / (1024**3),
        },
        "platform": "linux",  # Could detect more details
    }

    suite = BenchmarkSuite(
        name="Miniforge Full Benchmark Suite",
        system_info=system_info,
    )

    logger.info("Starting performance benchmarks...")
    perf = PerformanceBenchmark(model)
    perf_results = await perf.run_all()
    for r in perf_results:
        suite.add_result(r)

    logger.info("Starting memory benchmarks...")
    # Note: Would need actual memory manager
    # mem = MemoryBenchmark(model, memory_manager)
    # mem_results = await mem.run_all()
    # for r in mem_results:
    #     suite.add_result(r)

    logger.info("Starting context benchmarks...")
    ctx = ContextBenchmark(model)
    needle_result = await ctx.needle_in_haystack()
    suite.add_result(needle_result)

    consistency_result = await ctx.benchmark_context_consistency()
    suite.add_result(consistency_result)

    suite.finish()

    # Save results
    timestamp = int(time.time())
    suite.save(output_dir / f"benchmark_results_{timestamp}.json")

    return suite


if __name__ == "__main__":
    # Example usage
    print("Miniforge Benchmark Suite")
    print("Import this module to run benchmarks with your model instance")
