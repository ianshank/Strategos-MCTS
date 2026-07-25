"""
Verify RAG Evaluation Dataset in LangSmith.

This script verifies that the rag-eval-dataset was created successfully
and displays information about its contents.

Usage:
    python scripts/verify_rag_dataset.py
"""

import os
import sys
import traceback
from pathlib import Path

# Add project root to path before local imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.utils.langsmith_tracing import get_langsmith_client  # noqa: E402


def verify_dataset():
    """Verify the RAG evaluation dataset exists and show its contents."""
    # Check if LangSmith is configured
    if not os.getenv("LANGSMITH_API_KEY"):
        print("[ERROR] LANGSMITH_API_KEY environment variable not set")
        print("        Set it with: export LANGSMITH_API_KEY=your_key_here")
        sys.exit(1)

    print("=" * 70)
    print("Verifying RAG Evaluation Dataset")
    print("=" * 70)
    print()

    try:
        client = get_langsmith_client()

        # Read dataset
        print("Reading dataset from LangSmith...")
        dataset = client.read_dataset(dataset_name="rag-eval-dataset")

        print(f"[OK] Found dataset: {dataset.name}")
        print(f"[OK] Dataset ID: {dataset.id}")
        print(f"[OK] Description: {dataset.description}")
        print()

        # List examples
        print("Counting examples...")
        examples = list(client.list_examples(dataset_name="rag-eval-dataset"))
        total_examples = len(examples)

        print(f"[OK] Total examples: {total_examples}")
        print()

        # Show sample examples
        print("=" * 70)
        print("Sample Examples")
        print("=" * 70)
        print()

        # Show first 3 examples
        for i, example in enumerate(examples[:3], 1):
            question = example.inputs.get("question", "N/A")
            contexts = example.inputs.get("contexts", [])
            ground_truth = example.outputs.get("ground_truth", "N/A")

            print(f"Example {i}:")
            print(f"  Question: {question[:80]}...")
            print(f"  Contexts: {len(contexts)} snippets")
            print(f"  Ground Truth: {ground_truth[:100]}...")
            print()

        # Topic analysis
        print("=" * 70)
        print("Dataset Coverage Analysis")
        print("=" * 70)
        print()

        # Count questions by keyword
        topics = {
            "MCTS basics": ["Monte Carlo Tree Search", "game tree", "four phases", "branching factor"],
            "UCB1/PUCT": ["UCB1", "PUCT", "exploration", "exploitation"],
            "AlphaZero": ["AlphaZero", "self-play", "policy network", "value network"],
            "Advanced techniques": ["RAVE", "progressive widening", "virtual loss", "transposition"],
            "Training": ["training", "replay buffer", "Dirichlet noise", "temperature"],
        }

        topic_counts = dict.fromkeys(topics, 0)

        for example in examples:
            question = example.inputs.get("question", "").lower()
            ground_truth = example.outputs.get("ground_truth", "").lower()
            combined_text = question + " " + ground_truth

            for topic, keywords in topics.items():
                if any(keyword.lower() in combined_text for keyword in keywords):
                    topic_counts[topic] += 1

        for topic, count in topic_counts.items():
            percentage = (count / total_examples) * 100 if total_examples > 0 else 0
            bar_length = int(percentage / 5)  # Scale to 20 chars max
            bar = "#" * bar_length + "-" * (20 - bar_length)
            print(f"{topic:20s} [{bar}] {count:2d} ({percentage:5.1f}%)")

        print()

        # Validation checks
        print("=" * 70)
        print("Validation Checks")
        print("=" * 70)
        print()

        issues = []
        warnings = []

        # Check each example
        for i, example in enumerate(examples, 1):
            question = example.inputs.get("question", "")
            contexts = example.inputs.get("contexts", [])
            ground_truth = example.outputs.get("ground_truth", "")

            # Check for missing fields
            if not question:
                issues.append(f"Example {i}: Missing question")
            if not contexts:
                issues.append(f"Example {i}: Missing contexts")
            if not ground_truth:
                issues.append(f"Example {i}: Missing ground truth")

            # Check context count
            if len(contexts) < 2:
                warnings.append(f"Example {i}: Only {len(contexts)} context(s) (expected 2-4)")

            # Check ground truth length
            if len(ground_truth) < 100:
                warnings.append(f"Example {i}: Short ground truth ({len(ground_truth)} chars)")

        if not issues and not warnings:
            print("[OK] All validation checks passed!")
        else:
            if issues:
                print(f"[ERROR] Found {len(issues)} issue(s):")
                for issue in issues:
                    print(f"  - {issue}")
                print()

            if warnings:
                print(f"[WARNING] Found {len(warnings)} warning(s):")
                for warning in warnings[:5]:  # Show first 5 warnings
                    print(f"  - {warning}")
                if len(warnings) > 5:
                    print(f"  ... and {len(warnings) - 5} more")
                print()

        # Summary statistics
        print("=" * 70)
        print("Summary Statistics")
        print("=" * 70)
        print()

        total_contexts = sum(len(ex.inputs.get("contexts", [])) for ex in examples)
        avg_contexts = total_contexts / total_examples if total_examples > 0 else 0

        total_question_length = sum(len(ex.inputs.get("question", "")) for ex in examples)
        avg_question_length = total_question_length / total_examples if total_examples > 0 else 0

        total_ground_truth_length = sum(len(ex.outputs.get("ground_truth", "")) for ex in examples)
        avg_ground_truth_length = total_ground_truth_length / total_examples if total_examples > 0 else 0

        print(f"Total examples:              {total_examples}")
        print(f"Total contexts:              {total_contexts}")
        print(f"Avg contexts per example:    {avg_contexts:.1f}")
        print(f"Avg question length:         {avg_question_length:.0f} chars")
        print(f"Avg ground truth length:     {avg_ground_truth_length:.0f} chars")
        print()

        print("=" * 70)
        print("[SUCCESS] Dataset verification complete!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. View dataset in LangSmith UI: https://smith.langchain.com/")
        print("  2. Use dataset for RAG evaluation experiments")
        print("  3. Benchmark retrieval and generation quality")
        print()

    except Exception as e:
        print(f"[ERROR] Error verifying dataset: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    verify_dataset()
