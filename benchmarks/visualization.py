"""Visualization tools for benchmark results."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class BenchmarkVisualizer:
    """Generate visualizations and charts from benchmark data."""

    def __init__(self, results_path: Path):
        with open(results_path) as f:
            self.data = json.load(f)

    def generate_markdown_report(self) -> str:
        """Generate a markdown report from benchmark results."""

        lines = []

        # Header
        lines.append("# Miniforge Benchmark Report")
        lines.append("")
        lines.append(f"**Generated:** {self._format_timestamp()}")
        lines.append(f"**Version:** {self.data.get('miniforge_version', 'unknown')}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")

        summary = self.data.get("summary", {})
        duration = summary.get("total_duration_seconds", 0)
        lines.append(f"- **Total Duration:** {duration:.1f} seconds")
        lines.append(f"- **Benchmarks Run:** {', '.join(summary.get('benchmarks_run', []))}")

        failed = summary.get("benchmarks_failed", [])
        if failed:
            lines.append(f"- **Failed:** {', '.join(failed)}")
        lines.append("")

        # Key metrics table
        lines.append("### Key Metrics")
        lines.append("")
        lines.append("| Benchmark | Metric | Value |")
        lines.append("|-----------|--------|-------|")

        for bench, metrics in summary.get("key_metrics", {}).items():
            for metric, value in metrics.items():
                formatted_value = self._format_value(metric, value)
                lines.append(f"| {bench} | {metric} | {formatted_value} |")

        lines.append("")

        # Detailed results
        lines.append("## Detailed Results")
        lines.append("")

        for bench_name, bench_results in self.data.get("results", {}).items():
            lines.append(f"### {bench_name.upper()}")
            lines.append("")

            if "error" in bench_results:
                lines.append(f"**Error:** {bench_results['error']}")
            else:
                # Performance section
                if bench_name == "performance":
                    lines.extend(self._format_performance_results(bench_results))
                elif bench_name == "memory":
                    lines.extend(self._format_memory_results(bench_results))
                elif bench_name == "context":
                    lines.extend(self._format_context_results(bench_results))
                elif bench_name == "quality":
                    lines.extend(self._format_quality_results(bench_results))

            lines.append("")

        return "\n".join(lines)

    def _format_performance_results(self, results: Dict) -> List[str]:
        """Format performance benchmark results."""
        lines = []

        for result in results.get("results", []):
            bench_type = result.get("benchmark", "unknown")

            if bench_type == "generation_throughput":
                lines.append(f"**{result.get('prompt_type', 'unknown')} prompt:**")
                lines.append(
                    f"- Mean throughput: {result.get('mean_throughput_tok_s', 0):.2f} tok/s"
                )
                lines.append(f"- P95 throughput: {result.get('p95_throughput_tok_s', 0):.2f} tok/s")
                lines.append(f"- Iterations: {result.get('iterations', 0)}")

            elif bench_type == "streaming_latency":
                lines.append("**Streaming:**")
                lines.append(f"- Mean TTFT: {result.get('mean_ttft_ms', 0):.1f}ms")
                lines.append(f"- Mean inter-token: {result.get('mean_inter_token_ms', 0):.1f}ms")
                lines.append(f"- P95 inter-token: {result.get('p95_inter_token_ms', 0):.1f}ms")

            elif bench_type == "prompt_processing":
                lines.append("**Prompt Processing Speed by Context Size:**")
                lines.append("")
                lines.append("| Context Size | Speed (tok/s) |")
                lines.append("|--------------|---------------|")
                for item in result.get("results_by_size", []):
                    ctx = item.get("context_size", 0)
                    speed = item.get("processing_speed_tok_s", 0)
                    lines.append(f"| {ctx:,} | {speed:.2f} |")

            lines.append("")

        return lines

    def _format_memory_results(self, results: Dict) -> List[str]:
        """Format memory benchmark results."""
        lines = []

        for result in results.get("results", []):
            bench_type = result.get("benchmark", "unknown")

            if bench_type == "memory_efficiency_score":
                score = result.get("efficiency_score", 0)
                lines.append(f"**Memory Efficiency Score:** {score:.1f}/100")
                lines.append(f"- Peak RSS: {result.get('peak_rss_mb', 0):.1f} MB")
                lines.append(f"- Post-GC RSS: {result.get('post_gc_rss_mb', 0):.1f} MB")

            elif bench_type == "context_memory_scaling":
                bpt = result.get("bytes_per_token", 0)
                lines.append(f"**Memory Scaling:** {bpt:.1f} bytes/token")
                lines.append("")
                lines.append("| Context Size | Memory Increase (MB) |")
                lines.append("|--------------|---------------------|")
                for item in result.get("results", []):
                    ctx = item.get("context_size", 0)
                    inc = item.get("memory_increase_mb", 0)
                    lines.append(f"| {ctx:,} | {inc:.1f} |")

            lines.append("")

        return lines

    def _format_context_results(self, results: Dict) -> List[str]:
        """Format context benchmark results."""
        lines = []

        for result in results.get("results", []):
            bench_type = result.get("benchmark", "unknown")

            if bench_type == "needle_in_haystack":
                acc = result.get("overall_accuracy", 0)
                lines.append(f"**Needle in Haystack Accuracy:** {acc:.1%}")
                lines.append(f"- Tests run: {result.get('total_tests', 0)}")
                lines.append("")

                # Accuracy by context length
                lines.append("**Accuracy by Context Length:**")
                lines.append("")
                lines.append("| Context Length | Accuracy |")
                lines.append("|----------------|----------|")
                for ctx, acc in result.get("accuracy_by_context_length", {}).items():
                    lines.append(f"| {int(ctx):,} | {acc:.1%} |")

            elif bench_type == "max_context_window":
                max_ctx = result.get("max_successful", 0)
                lines.append(f"**Maximum Context Window:** {max_ctx:,} tokens")

            lines.append("")

        return lines

    def _format_quality_results(self, results: Dict) -> List[str]:
        """Format quality benchmark results."""
        lines = []

        overall = results.get("overall_quality_score", 0)
        lines.append(f"**Overall Quality Score:** {overall:.1%}")
        lines.append("")

        for result in results.get("results", []):
            bench_type = result.get("benchmark", "unknown")

            if bench_type == "qa_accuracy":
                acc = result.get("accuracy", 0)
                lines.append(f"**QA Accuracy:** {acc:.1%}")
                lines.append(
                    f"- Correct: {result.get('correct_count', 0)}/{result.get('total_questions', 0)}"
                )

            elif bench_type == "reasoning":
                rate = result.get("solve_rate", 0)
                expl = result.get("avg_explanation_quality", 0)
                lines.append(
                    f"**Reasoning:** {rate:.1%} solve rate, {expl:.1%} explanation quality"
                )

            elif bench_type == "summarization":
                cov = result.get("avg_concept_coverage", 0)
                good = result.get("good_summaries", 0)
                total = result.get("total_tasks", 0)
                lines.append(
                    f"**Summarization:** {cov:.1%} concept coverage, {good}/{total} good summaries"
                )

            elif bench_type == "instruction_following":
                rate = result.get("follow_rate", 0)
                lines.append(f"**Instruction Following:** {rate:.1%}")

            elif bench_type == "coherence":
                score = result.get("coherence_score", 0)
                words = result.get("word_count", 0)
                lines.append(f"**Coherence:** {score:.2f} score, {words} words generated")

            lines.append("")

        return lines

    def _format_timestamp(self) -> str:
        """Format timestamp from data."""
        import time

        ts = self.data.get("timestamp", time.time())
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

    def _format_value(self, metric: str, value: Any) -> str:
        """Format a metric value for display."""
        if isinstance(value, float):
            if "accuracy" in metric or "rate" in metric or "score" in metric:
                return f"{value:.1%}"
            elif "ms" in metric:
                return f"{value:.1f}ms"
            elif "tok" in metric or "bytes" in metric:
                return f"{value:.2f}"
            else:
                return f"{value:.2f}"
        elif isinstance(value, int):
            if "tokens" in metric or "length" in metric:
                return f"{value:,}"
            return str(value)
        return str(value)

    def generate_latex_tables(self) -> Dict[str, str]:
        """Generate LaTeX tables for paper inclusion."""

        tables = {}

        # Performance table
        perf_data = self.data.get("results", {}).get("performance", {})
        if perf_data and "error" not in perf_data:
            perf_table = r"""
\begin{table}[h]
\centering
\caption{Inference Performance Benchmarks}
\label{tab:performance}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Unit} \\
\midrule
"""
            for result in perf_data.get("results", []):
                if result.get("benchmark") == "generation_throughput":
                    prompt = result.get("prompt_type", "unknown")
                    mean_val = result.get("mean_throughput_tok_s", 0)
                    p95_val = result.get("p95_throughput_tok_s", 0)
                    perf_table += f"{prompt} throughput (mean) & {mean_val:.2f} & tok/s \\\\\n"
                    perf_table += f"{prompt} throughput (p95) & {p95_val:.2f} & tok/s \\\\\n"

            perf_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
            tables["performance"] = perf_table

        # Memory table
        mem_data = self.data.get("results", {}).get("memory", {})
        if mem_data and "error" not in mem_data:
            mem_table = r"""
\begin{table}[h]
\centering
\caption{Memory Usage Benchmarks}
\label{tab:memory}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
"""
            for result in mem_data.get("results", []):
                if result.get("benchmark") == "memory_efficiency_score":
                    score = result.get("efficiency_score", 0)
                    mem_table += f"Efficiency Score & {score:.1f}/100 \\\\\n"
                elif result.get("benchmark") == "context_memory_scaling":
                    bpt = result.get("bytes_per_token", 0)
                    mem_table += f"Memory per Token & {bpt:.1f} bytes \\\\\n"

            mem_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
            tables["memory"] = mem_table

        # Context table
        ctx_data = self.data.get("results", {}).get("context", {})
        if ctx_data and "error" not in ctx_data:
            ctx_table = r"""
\begin{table}[h]
\centering
\caption{Context Window Benchmarks}
\label{tab:context}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
"""
            for result in ctx_data.get("results", []):
                if result.get("benchmark") == "needle_in_haystack":
                    acc = result.get("overall_accuracy", 0)
                    ctx_table += f"Retrieval Accuracy & {acc:.1%} \\\\\n"
                elif result.get("benchmark") == "max_context_window":
                    max_ctx = result.get("max_successful", 0)
                    ctx_table += f"Maximum Context & {max_ctx:,} tokens \\\\\n"

            ctx_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
            tables["context"] = ctx_table

        # Quality table
        qual_data = self.data.get("results", {}).get("quality", {})
        if qual_data and "error" not in qual_data:
            qual_table = r"""
\begin{table}[h]
\centering
\caption{Quality Benchmarks}
\label{tab:quality}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
"""
            overall = qual_data.get("overall_quality_score", 0)
            qual_table += f"Overall Quality Score & {overall:.1%} \\\\\n"

            for result in qual_data.get("results", []):
                bench_type = result.get("benchmark", "")
                if bench_type == "qa_accuracy":
                    acc = result.get("accuracy", 0)
                    qual_table += f"QA Accuracy & {acc:.1%} \\\\\n"
                elif bench_type == "reasoning":
                    rate = result.get("solve_rate", 0)
                    qual_table += f"Reasoning Solve Rate & {rate:.1%} \\\\\n"
                elif bench_type == "instruction_following":
                    rate = result.get("follow_rate", 0)
                    qual_table += f"Instruction Following & {rate:.1%} \\\\\n"

            qual_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
            tables["quality"] = qual_table

        return tables

    def save_markdown_report(self, output_path: Optional[Path] = None) -> Path:
        """Save markdown report to file."""

        if output_path is None:
            timestamp = int(self.data.get("timestamp", 0))
            output_path = Path(f"benchmarks/results/benchmark_report_{timestamp}.md")

        content = self.generate_markdown_report()

        with open(output_path, "w") as f:
            f.write(content)

        return output_path


def create_comparison_chart(
    results_paths: List[Path],
    output_path: Path = Path("benchmarks/results/comparison.html"),
) -> Path:
    """Create comparison chart of multiple benchmark runs."""

    all_data = []
    for path in results_paths:
        with open(path) as f:
            all_data.append(json.load(f))

    # Generate HTML with comparison
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Miniforge Benchmark Comparison</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .chart-container { margin: 30px 0; }
    </style>
</head>
<body>
    <h1>Benchmark Comparison</h1>
    <div class="chart-container">
        <canvas id="throughputChart"></canvas>
    </div>
    <div class="chart-container">
        <canvas id="qualityChart"></canvas>
    </div>
    <script>
"""

    # Add chart configurations
    html += """
        // Throughput comparison
        new Chart(document.getElementById('throughputChart'), {
            type: 'bar',
            data: {
                labels: ['Code', 'QA', 'Summarization'],
                datasets: [
"""

    # Add datasets for each run
    for i, data in enumerate(all_data):
        label = f"Run {i + 1}"
        perf = data.get("results", {}).get("performance", {})
        throughputs = []
        for result in perf.get("results", []):
            if result.get("benchmark") == "generation_throughput":
                throughputs.append(result.get("mean_throughput_tok_s", 0))

        html += f"""
                    {{
                        label: '{label}',
                        data: {throughputs},
                        backgroundColor: 'rgba(75, 192, 192, 0.{6 - i})',
                    }},"""

    html += """
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Token Throughput Comparison'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Tokens/Second'
                        }
                    }
                }
            }
        });
"""

    html += """
    </script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


if __name__ == "__main__":
    print("Benchmark visualization tools")
    print("Usage: python -m benchmarks.visualization <results.json>")
