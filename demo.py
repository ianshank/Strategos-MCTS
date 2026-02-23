#!/usr/bin/env python3
"""
LangGraph Multi-Agent MCTS - MVP Demo

A lightweight CLI demo that demonstrates the core value proposition:
MCTS-guided multi-strategy LLM reasoning produces better answers than
single-shot prompting.

Zero external dependencies - works with Python 3.10+ standard library only.

Usage:
    # Mock mode (no API key needed - great for demos)
    python demo.py

    # With real LLM calls
    OPENAI_API_KEY=sk-... python demo.py --provider openai
    ANTHROPIC_API_KEY=sk-... python demo.py --provider anthropic

    # Custom query
    python demo.py --query "How should I design a rate limiter?"

    # Adjust MCTS parameters
    python demo.py --iterations 15 --exploration 2.0
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

# Import directly from the module file to avoid triggering the mcts package
# __init__.py which requires numpy. llm_mcts.py has zero external dependencies.
import importlib.util

_mod_path = os.path.join(_PROJECT_ROOT, "src", "framework", "mcts", "llm_mcts.py")
_spec = importlib.util.spec_from_file_location("llm_mcts", _mod_path)
_llm_mcts = importlib.util.module_from_spec(_spec)
sys.modules["llm_mcts"] = _llm_mcts  # Required for dataclass decorator
_spec.loader.exec_module(_llm_mcts)

MultiAgentMCTSPipeline = _llm_mcts.MultiAgentMCTSPipeline
PipelineResult = _llm_mcts.PipelineResult

# ---------------------------------------------------------------------------
# Terminal formatting (no dependencies)
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RED = "\033[31m"
MAGENTA = "\033[35m"


def supports_color() -> bool:
    """Check if the terminal supports ANSI colors."""
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = supports_color()


def c(text: str, color: str) -> str:
    """Apply color if supported."""
    if not _COLOR:
        return text
    return f"{color}{text}{RESET}"


def bar(value: float, width: int = 30) -> str:
    """Render a horizontal bar chart."""
    filled = int(value * width)
    empty = width - filled
    return f"[{'#' * filled}{'.' * empty}]"


def wrap(text: str, indent: int = 4, width: int = 80) -> str:
    """Wrap text with indent."""
    return textwrap.fill(text, width=width, initial_indent=" " * indent, subsequent_indent=" " * indent)


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------


def print_header():
    print()
    print(c("=" * 72, DIM))
    print(c("  LangGraph Multi-Agent MCTS Framework", BOLD))
    print(c("  MVP Demo - MCTS-Guided Multi-Strategy Reasoning", CYAN))
    print(c("=" * 72, DIM))
    print()


def print_query(query: str):
    print(c("QUERY:", BOLD))
    print(wrap(query, indent=2, width=70))
    print()


def print_mcts_progress(result: PipelineResult):
    """Print the MCTS exploration results."""
    mcts = result.mcts_result

    print(c("--- MCTS Exploration ---", BOLD))
    print()
    print(f"  Iterations: {mcts.iterations_run}")
    print(f"  LLM calls:  {len(mcts.llm_calls)}")
    print(f"  Provider:   {result.provider}")
    print()

    # Strategy scores table
    print(c("  Strategy Scores (UCB1-guided exploration):", YELLOW))
    print()

    # Sort by value descending
    sorted_strategies = sorted(
        mcts.all_strategies.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    visits = mcts.tree_stats.get("strategy_visits", {})

    for strategy, score in sorted_strategies:
        v = visits.get(strategy, 0)
        is_best = strategy == mcts.best_strategy
        prefix = c(">>", GREEN) if is_best else "  "
        name = f"{strategy:<15}"
        score_bar = bar(score)
        score_str = f"{score:.3f}"
        visits_str = f"({v} visits)"

        if is_best:
            print(f"  {prefix} {c(name, GREEN)} {score_bar} {c(score_str, GREEN)} {c(visits_str, DIM)}")
        else:
            print(f"  {prefix} {name} {score_bar} {score_str} {c(visits_str, DIM)}")

    print()


def print_best_response(result: PipelineResult):
    """Print the best strategy's response."""
    mcts = result.mcts_result

    print(c(f"--- Best Strategy: {mcts.best_strategy.upper()} (score: {mcts.best_score:.3f}) ---", BOLD))
    print()

    # Print the response with wrapping
    response = mcts.best_response
    for line in response.split("\n"):
        if line.strip():
            print(f"  {line}")
        else:
            print()
    print()


def print_consensus(result: PipelineResult):
    """Print the consensus synthesis."""
    if not result.consensus_response:
        return

    print(c("--- Consensus Synthesis ---", BOLD))
    print(c("  (Combined insights from top strategies)", DIM))
    print()

    for line in result.consensus_response.split("\n"):
        if line.strip():
            print(f"  {line}")
        else:
            print()
    print()


def print_stats(result: PipelineResult):
    """Print performance statistics."""
    mcts = result.mcts_result

    print(c("--- Statistics ---", DIM))
    print(f"  Total time:      {result.total_time_ms:.0f} ms")
    print(f"  MCTS iterations: {mcts.iterations_run}")
    print(f"  LLM calls:       {len(mcts.llm_calls)}")
    print(f"  Total tokens:    {mcts.tree_stats.get('total_tokens', 'N/A')}")

    # Per-call breakdown
    if mcts.llm_calls:
        latencies = [call.latency_ms for call in mcts.llm_calls]
        avg_latency = sum(latencies) / len(latencies)
        print(f"  Avg call time:   {avg_latency:.0f} ms")

    print()


def print_comparison_note():
    """Print a note about what MCTS adds."""
    print(c("--- Why MCTS? ---", MAGENTA))
    print()
    print("  Without MCTS:  Single prompt -> single response (may miss better approaches)")
    print("  With MCTS:     Explores 5 strategies, UCB1 focuses on the best ones,")
    print("                 then synthesizes the strongest answer from top strategies.")
    print()
    print(c("  The tree search ensures we don't just try once - we systematically", DIM))
    print(c("  explore and exploit the best reasoning approaches.", DIM))
    print()


# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "What are the key trade-offs between microservices and monolithic architecture for a team of 5 engineers?",
    "How should I design a distributed rate limiter that handles 100k requests per second?",
    "What is the best strategy for migrating a legacy Python 2 codebase to Python 3?",
    "Compare B-trees vs LSM-trees for write-heavy database workloads.",
    "How can I improve the reliability of a CI/CD pipeline that fails intermittently?",
]


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


def interactive_mode(pipeline: MultiAgentMCTSPipeline):
    """Run an interactive REPL."""
    print(c("Interactive mode. Type a question or 'quit' to exit.", CYAN))
    print(c("Type 'examples' to see example queries.", DIM))
    print()

    while True:
        try:
            query = input(c("Query> ", BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "examples":
            for i, q in enumerate(EXAMPLE_QUERIES, 1):
                print(f"  {i}. {q}")
            print()
            continue
        if query.isdigit() and 1 <= int(query) <= len(EXAMPLE_QUERIES):
            query = EXAMPLE_QUERIES[int(query) - 1]
            print(f"  Using: {query}")
            print()

        result = pipeline.run(query)
        print()
        print_mcts_progress(result)
        print_best_response(result)
        print_consensus(result)
        print_stats(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Multi-Agent MCTS - MVP Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python demo.py                              # Mock mode (no API key)
              python demo.py --provider openai             # Use OpenAI
              python demo.py --provider anthropic          # Use Anthropic
              python demo.py --query "your question here"  # Custom query
              python demo.py --interactive                 # Interactive REPL
        """),
    )
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "openai", "anthropic"],
        help="LLM provider to use (default: mock)",
    )
    parser.add_argument("--api-key", default=None, help="API key (or set env var)")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument(
        "--query",
        default=None,
        help="Query to process (default: uses example query)",
    )
    parser.add_argument("--iterations", type=int, default=10, help="MCTS iterations (default: 10)")
    parser.add_argument("--exploration", type=float, default=1.414, help="UCB1 exploration weight (default: 1.414)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-consensus", action="store_true", help="Skip consensus synthesis step")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive REPL mode")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    # Header
    if not args.json:
        print_header()

    # Create the pipeline
    try:
        pipeline = MultiAgentMCTSPipeline(
            provider=args.provider,
            api_key=args.api_key,
            model=args.model,
            iterations=args.iterations,
            exploration_weight=args.exploration,
            seed=args.seed,
            use_consensus=not args.no_consensus,
        )
    except ValueError as e:
        print(c(f"Error: {e}", RED))
        print()
        print("Tip: Run with --provider mock for a demo without API keys.")
        sys.exit(1)

    if not args.json:
        provider_label = args.provider if args.provider != "mock" else "mock (no API key needed)"
        print(f"  Provider:    {c(provider_label, CYAN)}")
        print(f"  Iterations:  {args.iterations}")
        print(f"  Exploration: {args.exploration}")
        print()

    # Interactive mode
    if args.interactive:
        interactive_mode(pipeline)
        return

    # Single query mode
    query = args.query or EXAMPLE_QUERIES[0]

    if not args.json:
        print_query(query)
        print(c("  Running MCTS exploration...", DIM))
        print()

    result = pipeline.run(query)

    if args.json:
        import json

        output = {
            "query": result.query,
            "best_strategy": result.mcts_result.best_strategy,
            "best_score": result.mcts_result.best_score,
            "best_response": result.mcts_result.best_response,
            "consensus_response": result.consensus_response,
            "all_strategies": result.mcts_result.all_strategies,
            "tree_stats": result.mcts_result.tree_stats,
            "total_time_ms": result.total_time_ms,
            "provider": result.provider,
        }
        print(json.dumps(output, indent=2))
    else:
        print_mcts_progress(result)
        print_best_response(result)
        print_consensus(result)
        print_stats(result)
        print_comparison_note()


if __name__ == "__main__":
    main()
