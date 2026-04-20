"""Context window and retrieval benchmarks for Miniforge."""

import asyncio
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


class ContextBenchmark:
    """Benchmark context window capabilities and retrieval accuracy."""

    # Standard test text
    HAYSTACK_TEMPLATE = """
{prefix_content}

{needle}

{suffix_content}
"""

    NEEDLE_FACTS = [
        "The secret passcode is 8742.",
        "Dr. Sarah Chen discovered the quantum particle in 2023.",
        "The meeting location is the downtown library at 3pm.",
        "Remember: blueberries are the secret ingredient.",
        "Pi to 10 digits: 3.1415926535",
        "The password for the vault is 'stardust'.",
        "Project codename: PHOENIX RISING",
        "The largest prime under 100 is 97.",
    ]

    FILLER_SENTENCES = [
        "The weather was pleasant and sunny that afternoon.",
        "Researchers have been studying this phenomenon for decades.",
        "Technology continues to evolve at a rapid pace.",
        "The conference attracted attendees from around the world.",
        "New discoveries often challenge existing theories.",
        "The team worked tirelessly to complete the project on time.",
        "Economic factors played a significant role in the decision.",
        "The methodology involved careful data collection and analysis.",
        "Environmental concerns have become increasingly important.",
        "The historical context helps us understand modern developments.",
    ]

    def __init__(self, model, output_dir: Path = Path("benchmarks/results")):
        self.model = model
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_haystack(self, total_tokens: int, needle: str, needle_position: float) -> str:
        """Generate test document with needle at specific position."""

        # Approximate: 4 chars per token
        total_chars = total_tokens * 4
        needle_chars = len(needle)

        # Position in characters
        needle_char_pos = int((total_chars - needle_chars) * needle_position)

        prefix_chars = needle_char_pos
        suffix_chars = total_chars - needle_char_pos - needle_chars

        # Generate content
        prefix = ""
        while len(prefix) < prefix_chars:
            prefix += random.choice(self.FILLER_SENTENCES) + " "
        prefix = prefix[:prefix_chars]

        suffix = ""
        while len(suffix) < suffix_chars:
            suffix += random.choice(self.FILLER_SENTENCES) + " "
        suffix = suffix[:suffix_chars]

        return self.HAYSTACK_TEMPLATE.format(
            prefix_content=prefix,
            needle=needle,
            suffix_content=suffix,
        )

    async def needle_in_haystack_test(
        self,
        context_lengths: List[int] = None,
        needle_depths: List[float] = None,
        num_needles: int = 1,
    ) -> Dict[str, Any]:
        """
        Run needle-in-haystack retrieval test.

        Tests if the model can find specific information at various positions
        in long contexts.
        """

        if context_lengths is None:
            context_lengths = [1024, 4096, 16384, 65536, 131072]

        if needle_depths is None:
            needle_depths = [0.0, 0.25, 0.5, 0.75, 1.0]

        all_results = []

        print(
            f"  Running needle-in-haystack with {len(context_lengths)} lengths "
            f"and {len(needle_depths)} depths..."
        )

        for ctx_len in context_lengths:
            for depth in needle_depths:
                needle = random.choice(self.NEEDLE_FACTS)
                haystack = self._generate_haystack(ctx_len, needle, depth)

                # Extract expected answer
                answer_map = {
                    "secret passcode is 8742": "8742",
                    "Dr. Sarah Chen": "Sarah Chen",
                    "downtown library at 3pm": "downtown library",
                    "blueberries": "blueberries",
                    "Pi to 10 digits": "3.1415926535",
                    "password for the vault": "stardust",
                    "Project codename": "PHOENIX RISING",
                    "largest prime under 100": "97",
                }

                expected_answer = None
                for key, val in answer_map.items():
                    if key in needle:
                        expected_answer = val
                        break

                # Query
                prompt = (
                    f"Document: {haystack}\n\n"
                    f"Question: Based on the document, {self._generate_question(needle)}\n"
                    f"Answer with just the specific information requested:"
                )

                response = await self.model.generate(prompt, max_tokens=50, temperature=0)

                # Check accuracy
                correct = self._check_answer(response, expected_answer, needle)

                result = {
                    "context_length": ctx_len,
                    "needle_depth": depth,
                    "needle": needle,
                    "response": response,
                    "expected": expected_answer,
                    "correct": correct,
                }
                all_results.append(result)

                status = "✓" if correct else "✗"
                print(
                    f"    {status} ctx={ctx_len}, depth={depth:.2f}: "
                    f"got='{response[:30]}...' expected='{expected_answer}'"
                )

        # Calculate accuracy by context length
        accuracy_by_length = {}
        for ctx_len in context_lengths:
            results_for_length = [r for r in all_results if r["context_length"] == ctx_len]
            correct_count = sum(1 for r in results_for_length if r["correct"])
            accuracy_by_length[ctx_len] = correct_count / len(results_for_length)

        # Calculate accuracy by depth
        accuracy_by_depth = {}
        for depth in needle_depths:
            results_for_depth = [r for r in all_results if abs(r["needle_depth"] - depth) < 0.01]
            correct_count = sum(1 for r in results_for_depth if r["correct"])
            accuracy_by_depth[depth] = correct_count / len(results_for_depth)

        overall_accuracy = sum(1 for r in all_results if r["correct"]) / len(all_results)

        return {
            "benchmark": "needle_in_haystack",
            "context_lengths": context_lengths,
            "needle_depths": needle_depths,
            "overall_accuracy": overall_accuracy,
            "accuracy_by_context_length": accuracy_by_length,
            "accuracy_by_depth": accuracy_by_depth,
            "total_tests": len(all_results),
            "detailed_results": all_results,
        }

    def _generate_question(self, needle: str) -> str:
        """Generate a question based on the needle content."""
        if "passcode" in needle or "password" in needle:
            return "what is the password or code mentioned?"
        elif "Dr." in needle or "discovered" in needle:
            return "who made the discovery and when?"
        elif "location" in needle or "library" in needle:
            return "where is the meeting?"
        elif "ingredient" in needle:
            return "what is the secret ingredient?"
        elif "prime" in needle or "Pi" in needle:
            return "what is the numerical value mentioned?"
        elif "codename" in needle:
            return "what is the project codename?"
        return "what specific information is contained in this document?"

    def _check_answer(self, response: str, expected: str, needle: str) -> bool:
        """Check if response contains expected answer."""
        response_lower = response.lower()

        if expected:
            if expected.lower() in response_lower:
                return True

        # Check for needle content
        key_phrases = needle.replace("The ", "").replace("is ", "").split()
        key_phrases = [p.strip(".,") for p in key_phrases if len(p) > 2]

        matches = sum(1 for phrase in key_phrases if phrase.lower() in response_lower)
        return matches >= max(1, len(key_phrases) // 3)

    async def multi_needle_test(
        self,
        context_length: int = 16384,
        num_needles: int = 5,
    ) -> Dict[str, Any]:
        """Test retrieval of multiple facts from context."""

        print(f"  Running multi-needle test with {num_needles} needles...")

        # Generate context with multiple needles at different depths
        needles = random.sample(self.NEEDLE_FACTS, min(num_needles, len(self.NEEDLE_FACTS)))

        # Approximate chars per token
        total_chars = context_length * 4
        chars_per_section = total_chars // (num_needles + 1)

        document_parts = []
        for i, needle in enumerate(needles):
            # Add filler
            filler = ""
            while len(filler) < chars_per_section - len(needle) - 20:
                filler += random.choice(self.FILLER_SENTENCES) + " "
            document_parts.append(filler)
            document_parts.append(needle)

        # Final filler
        final_filler = ""
        while len(final_filler) < chars_per_section:
            final_filler += random.choice(self.FILLER_SENTENCES) + " "
        document_parts.append(final_filler)

        document = " ".join(document_parts)

        # Query for each needle
        results = []
        for i, needle in enumerate(needles):
            question = self._generate_question(needle)
            prompt = f"Document: {document[: context_length * 4]}\n\nQuestion: {question}\nAnswer:"

            response = await self.model.generate(prompt, max_tokens=50, temperature=0)

            # Determine expected answer
            answer_map = {
                "passcode": "8742",
                "Sarah Chen": "Sarah Chen",
                "library": "downtown library",
                "blueberries": "blueberries",
                "Pi": "3.1415926535",
                "vault": "stardust",
                "codename": "PHOENIX",
                "prime": "97",
            }
            expected = None
            for key, val in answer_map.items():
                if key in needle:
                    expected = val
                    break

            correct = self._check_answer(response, expected, needle)

            results.append(
                {
                    "needle_index": i,
                    "needle": needle,
                    "response": response,
                    "correct": correct,
                }
            )

        accuracy = sum(1 for r in results if r["correct"]) / len(results)

        return {
            "benchmark": "multi_needle_retrieval",
            "context_length": context_length,
            "num_needles": num_needles,
            "accuracy": accuracy,
            "results": results,
        }

    async def context_consistency_test(
        self,
        context_length: int = 8192,
        iterations: int = 3,
    ) -> Dict[str, Any]:
        """Test response consistency across multiple queries to same context."""

        print(f"  Testing context consistency...")

        # Create a context with specific facts
        key_facts = [
            ("John Smith is the CEO of TechCorp", "Who is the CEO of TechCorp?", "John Smith"),
            ("The company was founded in 2015", "When was the company founded?", "2015"),
            ("Headquarters are in San Francisco", "Where are the headquarters?", "San Francisco"),
            ("They have 500 employees", "How many employees?", "500"),
        ]

        # Build document
        base_text = "General business information. " * 500
        document = base_text[: context_length * 2]

        # Insert facts at various positions
        for fact, _, _ in key_facts:
            pos = random.randint(0, len(document) - len(fact) - 100)
            document = document[:pos] + fact + " " + document[pos:]

        # Test consistency
        all_responses = {i: [] for i in range(len(key_facts))}

        for iteration in range(iterations):
            for i, (_, question, _) in enumerate(key_facts):
                prompt = (
                    f"Document: {document[: context_length * 4]}\n\nQuestion: {question}\nAnswer:"
                )
                response = await self.model.generate(prompt, max_tokens=30, temperature=0)
                all_responses[i].append(response)

        # Calculate consistency for each question
        consistency_scores = []
        for i, (_, _, expected) in enumerate(key_facts):
            responses = all_responses[i]

            # Check if all responses contain expected answer
            correct_count = sum(1 for r in responses if expected.lower() in r.lower())
            consistency = correct_count / len(responses)

            # Also check similarity between responses
            if len(set(r.strip() for r in responses)) == 1:
                consistency = 1.0

            consistency_scores.append(
                {
                    "question_index": i,
                    "expected": expected,
                    "responses": responses,
                    "consistency": consistency,
                }
            )

        overall_consistency = sum(s["consistency"] for s in consistency_scores) / len(
            consistency_scores
        )

        return {
            "benchmark": "context_consistency",
            "context_length": context_length,
            "iterations": iterations,
            "overall_consistency": overall_consistency,
            "per_question_scores": consistency_scores,
        }

    async def benchmark_context_window_maximum(
        self,
        max_test_length: int = 200000,
        step: int = 16384,
    ) -> Dict[str, Any]:
        """Find maximum effective context window."""

        print(f"  Testing maximum context window...")

        test_lengths = list(range(step, max_test_length + 1, step))
        results = []

        for length in test_lengths:
            # Create a test context
            fact = "The answer is 42."
            base = "Context text. " * (length // 10)

            # Insert fact at the beginning
            context = fact + " " + base
            context = context[: length * 4]  # Approximate char count

            prompt = (
                f"Document: {context}\n\n"
                f"Question: What is the answer?\n"
                f"Answer with just the number:"
            )

            try:
                response = await self.model.generate(prompt, max_tokens=10, temperature=0)
                success = "42" in response
                error = None
            except Exception as e:
                success = False
                error = str(e)

            results.append(
                {
                    "context_length": length,
                    "success": success,
                    "response": response if success else None,
                    "error": error,
                }
            )

            status = "✓" if success else "✗"
            print(f"    {status} {length} tokens")

            if not success:
                break

        max_successful = max((r["context_length"] for r in results if r["success"]), default=0)

        return {
            "benchmark": "max_context_window",
            "tested_lengths": test_lengths,
            "max_successful": max_successful,
            "results": results,
        }

    async def run_all(self) -> Dict[str, Any]:
        """Run all context benchmarks."""

        print("\n=== Context Window Benchmarks ===")

        all_results = {
            "benchmark_type": "context",
            "timestamp": time.time(),
            "results": [],
        }

        # Needle in haystack
        print("\nRunning needle-in-haystack test...")
        result = await self.needle_in_haystack_test()
        all_results["results"].append(result)

        # Multi-needle
        print("\nRunning multi-needle test...")
        result = await self.multi_needle_test()
        all_results["results"].append(result)

        # Consistency
        print("\nRunning context consistency test...")
        result = await self.context_consistency_test()
        all_results["results"].append(result)

        # Maximum window
        print("\nTesting maximum context window...")
        result = await self.benchmark_context_window_maximum()
        all_results["results"].append(result)

        # Save
        self._save_results(all_results, "context_benchmarks.json")

        return all_results

    def _save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save results to JSON."""
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filepath}")


async def run_context_benchmarks(model):
    """Convenience function to run all context benchmarks."""
    benchmark = ContextBenchmark(model)
    return await benchmark.run_all()


if __name__ == "__main__":
    print("Miniforge Context Window Benchmarks")
    print("Import and use with a model instance")
