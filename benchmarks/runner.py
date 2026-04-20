"""Main benchmark runner for Miniforge.

This module provides a unified interface to run all benchmarks
and generate comprehensive reports.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import argparse

# Import benchmark modules
from benchmarks.performance.throughput import ThroughputBenchmark, run_performance_benchmarks
from benchmarks.memory.usage import MemoryBenchmark, run_memory_benchmarks
from benchmarks.context.retrieval import ContextBenchmark, run_context_benchmarks
from benchmarks.quality.evaluation import QualityBenchmark, run_quality_benchmarks


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report."""

    timestamp: float
    miniforge_version: str = "0.1.0"
    results: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "miniforge_version": self.miniforge_version,
            "summary": self.summary,
            "results": self.results,
        }


class BenchmarkRunner:
    """Runner for all Miniforge benchmarks."""

    BENCHMARK_MODULES = {
        "performance": {
            "runner": run_performance_benchmarks,
            "description": "Token throughput, latency, and processing speed",
        },
        "memory": {
            "runner": run_memory_benchmarks,
            "description": "Memory usage, scaling, and efficiency",
        },
        "context": {
            "runner": run_context_benchmarks,
            "description": "Context window and retrieval accuracy",
        },
        "quality": {
            "runner": run_quality_benchmarks,
            "description": "Output quality across tasks",
        },
    }

    def __init__(
        self,
        model,
        memory_manager=None,
        output_dir: Path = Path("benchmarks/results"),
        select_benchmarks: Optional[List[str]] = None,
    ):
        self.model = model
        self.memory_manager = memory_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.select_benchmarks = select_benchmarks or list(self.BENCHMARK_MODULES.keys())

    async def run_benchmarks(self) -> BenchmarkReport:
        """Run all selected benchmarks."""

        report = BenchmarkReport(timestamp=time.time())

        print("=" * 60)
        print("Miniforge Comprehensive Benchmark Suite")
        print("=" * 60)

        start_time = time.time()

        for name in self.select_benchmarks:
            if name not in self.BENCHMARK_MODULES:
                print(f"\nWarning: Unknown benchmark '{name}', skipping...")
                continue

            bench_info = self.BENCHMARK_MODULES[name]
            print(f"\n{'=' * 60}")
            print(f"Running: {name.upper()}")
            print(f"Description: {bench_info['description']}")
            print(f"{'=' * 60}")

            try:
                if name == "memory" and self.memory_manager:
                    results = await bench_info["runner"](self.model, self.memory_manager)
                else:
                    results = await bench_info["runner"](self.model)

                report.results[name] = results
                print(f"\n✓ {name} benchmark completed successfully")

            except Exception as e:
                print(f"\n✗ {name} benchmark failed: {e}")
                report.results[name] = {"error": str(e)}

        total_duration = time.time() - start_time

        # Generate summary
        report.summary = self._generate_summary(report.results)
        report.summary["total_duration_seconds"] = total_duration

        # Save report
        self._save_report(report)

        # Print summary
        self._print_summary(report)

        return report

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""

        summary = {
            "benchmarks_run": list(results.keys()),
            "benchmarks_failed": [k for k, v in results.items() if "error" in v],
            "key_metrics": {},
        }

        # Extract key metrics from each benchmark
        for bench_name, bench_results in results.items():
            if "error" in bench_results:
                continue

            metrics = {}

            if bench_name == "performance":
                for result in bench_results.get("results", []):
                    if result.get("benchmark") == "generation_throughput":
                        metrics["mean_throughput_tok_s"] = result.get("mean_throughput_tok_s", 0)
                    elif result.get("benchmark") == "streaming_latency":
                        metrics["mean_ttft_ms"] = result.get("mean_ttft_ms", 0)

            elif bench_name == "memory":
                for result in bench_results.get("results", []):
                    if result.get("benchmark") == "memory_efficiency_score":
                        metrics["efficiency_score"] = result.get("efficiency_score", 0)
                    elif result.get("benchmark") == "context_memory_scaling":
                        metrics["bytes_per_token"] = result.get("bytes_per_token", 0)

            elif bench_name == "context":
                for result in bench_results.get("results", []):
                    if result.get("benchmark") == "needle_in_haystack":
                        metrics["retrieval_accuracy"] = result.get("overall_accuracy", 0)
                    elif result.get("benchmark") == "max_context_window":
                        metrics["max_context_tokens"] = result.get("max_successful", 0)

            elif bench_name == "quality":
                metrics["overall_quality_score"] = bench_results.get("overall_quality_score", 0)
                for result in bench_results.get("results", []):
                    if result.get("benchmark") == "qa_accuracy":
                        metrics["qa_accuracy"] = result.get("accuracy", 0)
                    elif result.get("benchmark") == "reasoning":
                        metrics["reasoning_solve_rate"] = result.get("solve_rate", 0)

            summary["key_metrics"][bench_name] = metrics

        return summary

    def _save_report(self, report: BenchmarkReport) -> None:
        """Save report to file."""

        timestamp = int(report.timestamp)
        filepath = self.output_dir / f"benchmark_report_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"Report saved to: {filepath}")

    def _print_summary(self, report: BenchmarkReport) -> None:
        """Print formatted summary."""

        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        print(f"\nTotal Duration: {report.summary.get('total_duration_seconds', 0):.1f}s")
        print(f"Benchmarks Run: {len(report.summary.get('benchmarks_run', []))}")
        print(f"Benchmarks Failed: {len(report.summary.get('benchmarks_failed', []))}")

        print("\n--- Key Metrics ---")
        for bench, metrics in report.summary.get("key_metrics", {}).items():
            print(f"\n{bench.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    if "accuracy" in metric or "rate" in metric or "score" in metric:
                        print(f"  {metric}: {value:.1%}")
                    elif "ms" in metric:
                        print(f"  {metric}: {value:.1f}ms")
                    elif "tok" in metric:
                        print(f"  {metric}: {value:.2f}")
                    else:
                        print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value}")

        # Overall score if quality was tested
        if "quality" in report.results:
            score = report.results["quality"].get("overall_quality_score", 0)
            print(f"\n{'=' * 60}")
            print(f"OVERALL QUALITY SCORE: {score:.1%}")
            print("=" * 60)


def generate_html_report(report_path: Path) -> Path:
    """Generate HTML visualization of benchmark results."""

    with open(report_path) as f:
        data = json.load(f)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Miniforge Benchmark Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .benchmark-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 4px;
            min-width: 200px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .success {{
            color: #4CAF50;
        }}
        .warning {{
            color: #FF9800;
        }}
        .error {{
            color: #F44336;
        }}
        pre {{
            background: #f5f5f5;
            padding: 15px;
            overflow-x: auto;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <h1>Miniforge Benchmark Report</h1>
    <p>Generated: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data["timestamp"]))}</p>
    <p>Version: {data["miniforge_version"]}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Duration: {data["summary"].get("total_duration_seconds", 0):.1f} seconds</p>
        <p>Benchmarks: {", ".join(data["summary"].get("benchmarks_run", []))}</p>
    </div>
"""

    # Add sections for each benchmark
    for bench_name, bench_results in data.get("results", {}).items():
        html += f"""
    <div class="benchmark-section">
        <h2>{bench_name.upper()}</h2>
"""
        if "error" in bench_results:
            html += f"""
        <p class="error">Error: {bench_results["error"]}</p>
"""
        else:
            # Display results
            html += "        <pre>" + json.dumps(bench_results, indent=2) + "</pre>"

        html += "    </div>"

    html += """
</body>
</html>
"""

    output_path = report_path.with_suffix(".html")
    with open(output_path, "w") as f:
        f.write(html)

    return output_path


async def main():
    """Main entry point for benchmark runner."""

    parser = argparse.ArgumentParser(
        description="Run Miniforge benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python -m benchmarks.runner
  
  # Run specific benchmarks
  python -m benchmarks.runner --benchmarks performance quality
  
  # Generate HTML report from existing results
  python -m benchmarks.runner --html-only --report results/benchmark_report_1234567890.json
        """,
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["performance", "memory", "context", "quality", "all"],
        default=["all"],
        help="Benchmarks to run (default: all)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory for benchmark results",
    )

    parser.add_argument(
        "--html-only", action="store_true", help="Generate HTML report from existing JSON results"
    )

    parser.add_argument(
        "--report", type=Path, help="Path to existing report JSON file (for --html-only)"
    )

    args = parser.parse_args()

    if args.html_only:
        if not args.report:
            print("Error: --report required with --html-only")
            sys.exit(1)

        html_path = generate_html_report(args.report)
        print(f"HTML report generated: {html_path}")
        return

    # Normal benchmark run
    print("Initializing model for benchmarks...")

    # Import here to avoid dependency issues if just generating HTML
    from miniforge import Miniforge

    model = await Miniforge.from_pretrained(
        "saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10",
    )

    # Determine which benchmarks to run
    benchmarks = args.benchmarks
    if "all" in benchmarks:
        benchmarks = ["performance", "memory", "context", "quality"]

    # Run benchmarks
    runner = BenchmarkRunner(
        model=model,
        output_dir=args.output_dir,
        select_benchmarks=benchmarks,
    )

    report = await runner.run_benchmarks()

    # Generate HTML report
    latest_report = max(
        args.output_dir.glob("benchmark_report_*.json"), key=lambda p: p.stat().st_mtime
    )
    html_path = generate_html_report(latest_report)
    print(f"\nHTML report: {html_path}")

    # Cleanup
    await model.cleanup()

    print("\nBenchmarks complete!")


if __name__ == "__main__":
    asyncio.run(main())
