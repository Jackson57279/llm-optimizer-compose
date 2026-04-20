"""Quality benchmarks for Miniforge - evaluating model outputs."""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import re


class QualityBenchmark:
    """Benchmark quality of model outputs across various tasks."""

    # Test datasets
    QA_PAIRS = [
        {
            "question": "What is the capital of France?",
            "expected_keywords": ["Paris"],
            "forbidden": ["London", "Berlin", "Rome", "Madrid"],
        },
        {
            "question": "Who wrote 'Romeo and Juliet'?",
            "expected_keywords": ["Shakespeare", "William Shakespeare"],
            "forbidden": ["Hemingway", "Dickens", "Austen"],
        },
        {
            "question": "What is 15 multiplied by 7?",
            "expected_keywords": ["105", "one hundred five"],
            "forbidden": [],
        },
        {
            "question": "What is the largest planet in our solar system?",
            "expected_keywords": ["Jupiter"],
            "forbidden": ["Saturn", "Earth", "Mars", "Venus"],
        },
        {
            "question": "In which year did World War II end?",
            "expected_keywords": ["1945"],
            "forbidden": ["1939", "1944", "1946", "1950"],
        },
    ]

    REASONING_PROBLEMS = [
        {
            "problem": "If a train travels 120 km in 2 hours, what is its average speed?",
            "expected_answer": "60",
            "unit": "km/h",
            "explanation_keywords": ["speed", "distance", "time", "divide", "120", "2", "60"],
        },
        {
            "problem": "A rectangle has length 8 and width 5. What is its area?",
            "expected_answer": "40",
            "unit": "square units",
            "explanation_keywords": ["area", "length", "width", "multiply", "8", "5", "40"],
        },
        {
            "problem": "If 3 workers can complete a job in 4 days, how many days would 6 workers take?",
            "expected_answer": "2",
            "unit": "days",
            "explanation_keywords": ["workers", "days", "inverse", "proportion", "2"],
        },
    ]

    SUMMARIZATION_TASKS = [
        {
            "text": """Artificial intelligence (AI) is intelligence demonstrated by machines, 
            as opposed to the natural intelligence displayed by animals including humans. 
            AI research has been defined as the field of study of intelligent agents, 
            which refers to any system that perceives its environment and takes actions 
            that maximize its chance of achieving its goals. The term "artificial intelligence" 
            had previously been used to describe machines that mimic and display human-like 
            cognitive skills, but this definition has been largely abandoned by AI researchers.""",
            "expected_concepts": ["intelligence", "machines", "research", "agents", "goals"],
            "max_length": 100,
        },
        {
            "text": """The water cycle, also known as the hydrologic cycle, describes the 
            continuous movement of water on, above, and below the surface of the Earth. 
            Water changes state between liquid, vapor, and ice at various places in the 
            water cycle. The water cycle involves the exchange of energy, which leads to 
            temperature changes. When water evaporates, it takes up energy from its surroundings 
            and cools the environment. When it condenses, it releases energy and warms the 
            environment.""",
            "expected_concepts": ["water", "cycle", "evaporation", "condensation", "earth"],
            "max_length": 100,
        },
    ]

    def __init__(self, model, output_dir: Path = Path("benchmarks/results")):
        self.model = model
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def benchmark_qa_accuracy(
        self,
        qa_pairs: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Test question-answering accuracy."""

        if qa_pairs is None:
            qa_pairs = self.QA_PAIRS

        results = []

        print(f"  Testing QA accuracy on {len(qa_pairs)} questions...")

        for item in qa_pairs:
            question = item["question"]
            expected = item["expected_keywords"]
            forbidden = item.get("forbidden", [])

            response = await self.model.generate(
                question,
                max_tokens=50,
                temperature=0,
            )

            response_lower = response.lower()

            # Check expected keywords
            found_expected = any(kw.lower() in response_lower for kw in expected)

            # Check forbidden keywords
            found_forbidden = any(f.lower() in response_lower for f in forbidden)

            correct = found_expected and not found_forbidden

            results.append(
                {
                    "question": question,
                    "response": response,
                    "expected": expected,
                    "correct": correct,
                    "found_expected": found_expected,
                    "found_forbidden": found_forbidden,
                }
            )

            status = "✓" if correct else "✗"
            print(f"    {status} Q: {question[:50]}...")

        accuracy = sum(1 for r in results if r["correct"]) / len(results)

        return {
            "benchmark": "qa_accuracy",
            "total_questions": len(qa_pairs),
            "accuracy": accuracy,
            "correct_count": sum(1 for r in results if r["correct"]),
            "results": results,
        }

    async def benchmark_reasoning(
        self,
        problems: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Test reasoning and mathematical problem-solving."""

        if problems is None:
            problems = self.REASONING_PROBLEMS

        results = []

        print(f"  Testing reasoning on {len(problems)} problems...")

        for problem in problems:
            prompt = (
                f"Problem: {problem['problem']}\n\nSolve step by step and provide the final answer:"
            )

            response = await self.model.generate(
                prompt,
                max_tokens=200,
                temperature=0,
            )

            # Check if answer is in response
            answer_found = problem["expected_answer"] in response

            # Check explanation quality
            explanation_keywords = problem.get("explanation_keywords", [])
            found_explanation = sum(
                1 for kw in explanation_keywords if kw.lower() in response.lower()
            )
            explanation_score = (
                found_explanation / len(explanation_keywords) if explanation_keywords else 0
            )

            results.append(
                {
                    "problem": problem["problem"],
                    "response": response,
                    "expected_answer": problem["expected_answer"],
                    "answer_found": answer_found,
                    "explanation_score": explanation_score,
                    "explanation_keywords_found": found_explanation,
                }
            )

            status = "✓" if answer_found else "✗"
            print(
                f"    {status} Problem solved: {answer_found}, "
                f"Explanation quality: {explanation_score:.0%}"
            )

        solve_rate = sum(1 for r in results if r["answer_found"]) / len(results)
        avg_explanation = sum(r["explanation_score"] for r in results) / len(results)

        return {
            "benchmark": "reasoning",
            "total_problems": len(problems),
            "solve_rate": solve_rate,
            "avg_explanation_quality": avg_explanation,
            "results": results,
        }

    async def benchmark_summarization(
        self,
        tasks: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Test summarization quality."""

        if tasks is None:
            tasks = self.SUMMARIZATION_TASKS

        results = []

        print(f"  Testing summarization on {len(tasks)} texts...")

        for task in tasks:
            prompt = (
                f"Text: {task['text']}\n\n"
                f"Summarize the above text in at most {task['max_length']} characters:"
            )

            summary = await self.model.generate(
                prompt,
                max_tokens=100,
                temperature=0.5,
            )

            # Check concept coverage
            expected_concepts = task["expected_concepts"]
            found_concepts = [c for c in expected_concepts if c.lower() in summary.lower()]
            concept_coverage = len(found_concepts) / len(expected_concepts)

            # Check length
            length_ok = len(summary) <= task["max_length"] * 1.2  # Allow 20% margin

            # Check for copy-paste
            text_words = set(task["text"].lower().split())
            summary_words = set(summary.lower().split())
            overlap = len(text_words & summary_words) / len(summary_words) if summary_words else 0
            is_copy = overlap > 0.8  # High overlap suggests copying

            results.append(
                {
                    "original_length": len(task["text"]),
                    "summary_length": len(summary),
                    "summary": summary,
                    "concept_coverage": concept_coverage,
                    "concepts_found": found_concepts,
                    "length_ok": length_ok,
                    "is_copy_paste": is_copy,
                    "word_overlap": overlap,
                }
            )

            status = "✓" if concept_coverage > 0.6 and length_ok and not is_copy else "✗"
            print(
                f"    {status} Coverage: {concept_coverage:.0%}, "
                f"Length OK: {length_ok}, Original: {not is_copy}"
            )

        avg_coverage = sum(r["concept_coverage"] for r in results) / len(results)
        good_summaries = sum(
            1
            for r in results
            if r["concept_coverage"] > 0.6 and r["length_ok"] and not r["is_copy_paste"]
        )

        return {
            "benchmark": "summarization",
            "total_tasks": len(tasks),
            "avg_concept_coverage": avg_coverage,
            "good_summaries": good_summaries,
            "results": results,
        }

    async def benchmark_instruction_following(
        self,
    ) -> Dict[str, Any]:
        """Test ability to follow specific instructions."""

        instruction_tests = [
            {
                "instruction": "List 3 colors. Answer with only the colors, separated by commas.",
                "check": lambda r: len(r.split(",")) >= 3 and len(r) < 100,
                "description": "List with constraints",
            },
            {
                "instruction": "Say 'hello' in exactly 5 different languages. Number them 1-5.",
                "check": lambda r: r.count("1.") > 0 and r.count("5.") > 0,
                "description": "Numbered list format",
            },
            {
                "instruction": "Write a haiku about nature. A haiku has 3 lines with 5-7-5 syllables.",
                "check": lambda r: len(r.split("\n")) >= 3,
                "description": "Structured output (haiku)",
            },
            {
                "instruction": "Answer with exactly one word: What is the opposite of hot?",
                "check": lambda r: len(r.split()) <= 2 and "cold" in r.lower(),
                "description": "Single word constraint",
            },
            {
                "instruction": "Provide a JSON object with keys 'name' and 'age'.",
                "check": lambda r: '"name"' in r and '"age"' in r,
                "description": "JSON format",
            },
        ]

        results = []

        print(f"  Testing instruction following on {len(instruction_tests)} tasks...")

        for test in instruction_tests:
            response = await self.model.generate(
                test["instruction"],
                max_tokens=100,
                temperature=0,
            )

            followed = test["check"](response)

            results.append(
                {
                    "description": test["description"],
                    "instruction": test["instruction"],
                    "response": response,
                    "followed": followed,
                }
            )

            status = "✓" if followed else "✗"
            print(f"    {status} {test['description']}")

        follow_rate = sum(1 for r in results if r["followed"]) / len(results)

        return {
            "benchmark": "instruction_following",
            "total_tests": len(instruction_tests),
            "follow_rate": follow_rate,
            "results": results,
        }

    async def benchmark_coherence(
        self,
    ) -> Dict[str, Any]:
        """Test coherence over longer generations."""

        prompt = """Write a short story (300-400 words) about a scientist who discovers 
        something unexpected. The story should have a clear beginning, middle, and end.
        
        Story:"""

        print("  Testing coherence in long generation...")

        story = await self.model.generate(
            prompt,
            max_tokens=600,
            temperature=0.7,
        )

        # Coherence checks
        word_count = len(story.split())
        sentence_count = len(re.split(r"[.!?]+", story))
        paragraph_count = len([p for p in story.split("\n\n") if p.strip()])

        # Check for repetition
        words = story.lower().split()
        unique_words = set(words)
        repetition_score = len(unique_words) / len(words) if words else 0

        # Check structure
        has_beginning = word_count > 50
        has_middle = word_count > 150
        has_end = any(
            ending in story.lower()[-100:]
            for ending in ["end", "finally", "conclusion", "last", "finished"]
        )

        coherence_score = (
            (0.3 if has_beginning else 0)
            + (0.3 if has_middle else 0)
            + (0.2 if has_end else 0)
            + (0.2 * repetition_score)
        )

        return {
            "benchmark": "coherence",
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "repetition_score": repetition_score,
            "has_clear_structure": has_beginning and has_middle,
            "coherence_score": coherence_score,
            "story_preview": story[:200] + "..." if len(story) > 200 else story,
        }

    async def run_all(self) -> Dict[str, Any]:
        """Run all quality benchmarks."""

        print("\n=== Quality Benchmarks ===")

        all_results = {
            "benchmark_type": "quality",
            "timestamp": time.time(),
            "results": [],
        }

        # QA
        print("\nBenchmarking QA accuracy...")
        result = await self.benchmark_qa_accuracy()
        all_results["results"].append(result)

        # Reasoning
        print("\nBenchmarking reasoning...")
        result = await self.benchmark_reasoning()
        all_results["results"].append(result)

        # Summarization
        print("\nBenchmarking summarization...")
        result = await self.benchmark_summarization()
        all_results["results"].append(result)

        # Instruction following
        print("\nBenchmarking instruction following...")
        result = await self.benchmark_instruction_following()
        all_results["results"].append(result)

        # Coherence
        print("\nBenchmarking coherence...")
        result = await self.benchmark_coherence()
        all_results["results"].append(result)

        # Calculate overall quality score
        scores = []
        for r in all_results["results"]:
            if "accuracy" in r:
                scores.append(r["accuracy"])
            elif "solve_rate" in r:
                scores.append(r["solve_rate"])
            elif "avg_concept_coverage" in r:
                scores.append(r["avg_concept_coverage"])
            elif "follow_rate" in r:
                scores.append(r["follow_rate"])
            elif "coherence_score" in r:
                scores.append(r["coherence_score"])

        overall_score = sum(scores) / len(scores) if scores else 0
        all_results["overall_quality_score"] = overall_score

        # Save
        self._save_results(all_results, "quality_benchmarks.json")

        print(f"\n=== Overall Quality Score: {overall_score:.1%} ===")

        return all_results

    def _save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save results to JSON."""
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filepath}")


async def run_quality_benchmarks(model):
    """Convenience function to run all quality benchmarks."""
    benchmark = QualityBenchmark(model)
    return await benchmark.run_all()


if __name__ == "__main__":
    print("Miniforge Quality Benchmarks")
    print("Import and use with a model instance")
