#!/usr/bin/env python3
"""
LangGraph Multi-Agent MCTS - Chess Demo

Demonstrates the full agent framework applied to chess: HRM for strategic
decomposition, TRM for move refinement, MCTS for multi-strategy exploration,
Meta-Controller for routing, and MCP tools for position analysis.

Requires python-chess for full functionality, but works in mock mode without it.

Usage:
    # Mock mode (no API key or python-chess needed)
    python chess_demo.py

    # Analyse starting position with real LLM
    OPENAI_API_KEY=sk-... python chess_demo.py --provider openai

    # Analyse a specific position
    python chess_demo.py --analyze --fen "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"

    # Human vs engine interactive game
    python chess_demo.py --interactive

    # Engine vs engine self-play
    python chess_demo.py --self-play --depth 4

    # JSON output
    python chess_demo.py --json --analyze
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

# Import the LLM MCTS module directly to avoid numpy dependency chain
import importlib.util  # noqa: E402

_mod_path = os.path.join(_PROJECT_ROOT, "src", "framework", "mcts", "llm_mcts.py")
_spec = importlib.util.spec_from_file_location("llm_mcts", _mod_path)
_llm_mcts = importlib.util.module_from_spec(_spec)
sys.modules["llm_mcts"] = _llm_mcts
_spec.loader.exec_module(_llm_mcts)

MockLLMClient = _llm_mcts.MockLLMClient
StdlibLLMClient = _llm_mcts.StdlibLLMClient

# ---------------------------------------------------------------------------
# Terminal formatting (same palette as demo.py)
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
WHITE_BG = "\033[47m"
BLACK_FG = "\033[30m"

_COLOR = (not os.environ.get("NO_COLOR")) and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if _COLOR else text


def bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return f"[{'#' * filled}{'.' * (width - filled)}]"


# ---------------------------------------------------------------------------
# FEN helpers
# ---------------------------------------------------------------------------

INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

PIECE_SYMBOLS = {
    "K": "\u2654", "Q": "\u2655", "R": "\u2656", "B": "\u2657", "N": "\u2658", "P": "\u2659",
    "k": "\u265a", "q": "\u265b", "r": "\u265c", "b": "\u265d", "n": "\u265e", "p": "\u265f",
}


def fen_to_ascii(fen: str, *, use_unicode: bool = True) -> str:
    """Render a FEN string as an ASCII/Unicode board."""
    board_part = fen.split()[0]
    rows = board_part.split("/")
    lines = []
    lines.append("    a   b   c   d   e   f   g   h")
    lines.append("  +---+---+---+---+---+---+---+---+")
    for rank_idx, row in enumerate(rows):
        rank = 8 - rank_idx
        cells = []
        for ch in row:
            if ch.isdigit():
                cells.extend([" "] * int(ch))
            else:
                cells.append(PIECE_SYMBOLS.get(ch, ch) if use_unicode else ch)
        row_str = " | ".join(cells)
        lines.append(f"  | {row_str} | {rank}")
        lines.append("  +---+---+---+---+---+---+---+---+")
    return "\n".join(lines)


def fen_side_to_move(fen: str) -> str:
    parts = fen.split()
    return "White" if (len(parts) > 1 and parts[1] == "w") else "Black"


def fen_move_number(fen: str) -> int:
    parts = fen.split()
    try:
        return int(parts[5]) if len(parts) >= 6 else 1
    except (ValueError, IndexError):
        return 1


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_header():
    print()
    print(c("=" * 72, DIM))
    print(c("  LangGraph Multi-Agent MCTS Framework", BOLD))
    print(c("  Chess Demo - All Agents, Tools & MCP", CYAN))
    print(c("=" * 72, DIM))
    print()


def print_board(fen: str):
    print(c("--- Board ---", BOLD))
    print()
    print(fen_to_ascii(fen))
    print()
    print(f"  FEN:  {c(fen, DIM)}")
    print(f"  Turn: {c(fen_side_to_move(fen), CYAN)}  Move: {fen_move_number(fen)}")
    print()


def print_routing(routing):
    print(c("--- Meta-Controller Routing ---", BOLD))
    print(f"  Phase:     {c(routing.game_phase.capitalize(), CYAN)}")
    print(f"  Primary:   {c(routing.primary_agent.upper(), GREEN)}")
    print(f"  Reasoning: {routing.reasoning}")
    print(f"  Weights:")
    for agent, weight in sorted(routing.agent_weights.items(), key=lambda x: -x[1]):
        is_primary = agent == routing.primary_agent
        prefix = c(">>", GREEN) if is_primary else "  "
        name = f"{agent.upper():<6}"
        score_bar = bar(weight)
        color = GREEN if is_primary else RESET
        print(f"    {prefix} {c(name, color)} {score_bar} {weight:.2f}")
    print()


def print_agent_results(agent_results: dict):
    print(c("--- Agent Results ---", BOLD))
    for name, result in agent_results.items():
        print(f"  {c(name.upper(), YELLOW)}: move={c(result.move, GREEN)} "
              f"score={result.confidence:.2f}")
        if result.reasoning:
            # Truncate reasoning for display
            short = result.reasoning[:120].replace("\n", " ")
            print(f"    {c(short, DIM)}")
    print()


def print_best_move(analysis):
    print(c("--- Best Move ---", BOLD))
    print(f"  {c(analysis.best_move, GREEN + BOLD)}")
    if analysis.consensus_move:
        print(f"  (Consensus from {len(analysis.agent_results)} agents)")
    if analysis.consensus_reasoning:
        short = analysis.consensus_reasoning[:200].replace("\n", " ")
        print(f"  {c(short, DIM)}")
    print()


def print_stats(analysis):
    print(c("--- Statistics ---", DIM))
    print(f"  Total time: {analysis.total_time_ms:.0f} ms")
    print(f"  Agents:     {', '.join(analysis.metadata.get('agents_used', []))}")
    print()


# ---------------------------------------------------------------------------
# Mock LLM adapter for chess (returns chess-specific responses)
# ---------------------------------------------------------------------------


class ChessMockLLMAdapter:
    """Mock LLM that returns plausible chess analysis for testing."""

    def __init__(self):
        self.call_count = 0

    async def generate(
        self, *, messages=None, prompt=None, temperature=0.7, max_tokens=None, **kwargs
    ):
        self.call_count += 1
        prompt_text = prompt or ""

        # Default move based on position in prompt
        if "e2e4" in prompt_text or "1. " in prompt_text or "opening" in prompt_text.lower():
            move = "d2d4"
        elif "endgame" in prompt_text.lower():
            move = "e1e2"
        else:
            move = "e2e4"

        # Route based on agent type in prompt
        if "hierarchical" in prompt_text.lower() or "hrm" in prompt_text.lower():
            text = (
                "### Sub-problems\n"
                "1. Pawn structure\n   **Analysis:** Central pawns are strong\n"
                "   **Implication:** Support center control\n"
                "2. Piece activity\n   **Analysis:** Knights need development\n"
                "   **Implication:** Develop towards center\n\n"
                "### Synthesis\n"
                f"**Recommended move:** {move}\n"
                "**Reasoning:** Develops centrally and controls key squares.\n"
            )
        elif "refinement" in prompt_text.lower() or "trm" in prompt_text.lower():
            text = (
                "### Initial Candidate\n"
                f"**Move:** {move}\n"
                "**Reasoning:** Controls the center effectively.\n\n"
                "### Critical Review\n"
                "- Weakness: slightly committal\n"
                "- Alternative: d2d4 also strong\n\n"
                "### Refined Recommendation\n"
                f"**Move:** {move}\n"
                "**Score:** 0.75\n"
                "**Reasoning:** Best balance of activity and safety.\n"
            )
        elif "tactical" in prompt_text.lower():
            text = (
                f"**Move:** {move}\n"
                "**Score:** 0.70\n"
                "**Reasoning:** No immediate tactics; this move improves position.\n"
            )
        elif "positional" in prompt_text.lower():
            text = (
                f"**Move:** {move}\n"
                "**Score:** 0.72\n"
                "**Reasoning:** Improves piece placement and central control.\n"
            )
        elif "prophylactic" in prompt_text.lower():
            text = (
                f"**Move:** {move}\n"
                "**Score:** 0.65\n"
                "**Reasoning:** Prevents opponent's plan while improving position.\n"
            )
        elif "consensus" in prompt_text.lower() or "synthesiz" in prompt_text.lower():
            text = (
                f"**Best Move:** {move}\n"
                "**Confidence:** 0.80\n"
                "**Reasoning:** All agents converge on central play.\n"
            )
        else:
            text = (
                f"**Move:** {move}\n"
                "**Score:** 0.65\n"
                "**Reasoning:** Solid developing move.\n"
            )

        # Return mock response compatible with LLMClient protocol
        class _Resp:
            pass

        resp = _Resp()
        resp.text = text
        resp.usage = {"total_tokens": 80, "prompt_tokens": 40, "completion_tokens": 40}
        resp.model = "chess-mock"
        resp.raw_response = None
        resp.finish_reason = "stop"
        resp.total_tokens = 80
        resp.prompt_tokens = 40
        resp.completion_tokens = 40
        return resp


# ---------------------------------------------------------------------------
# Core demo functions
# ---------------------------------------------------------------------------


def _get_engine(args):
    """Create an LLMChessEngine from CLI args."""
    from src.games.chess.llm_chess_engine import LLMChessEngine

    if args.provider == "mock":
        adapter = ChessMockLLMAdapter()
    else:
        adapter = StdlibLLMClient(provider=args.provider, api_key=args.api_key, model=args.model)

    return LLMChessEngine(
        model_adapter=adapter,
        mcts_depth=args.depth,
        temperature=args.temperature,
    )


def analyze_position_cmd(args):
    """Analyse a single position and display results."""
    engine = _get_engine(args)
    fen = args.fen or INITIAL_FEN

    if not args.json:
        print_board(fen)
        print(c("  Analysing position...", DIM))
        print()

    analysis = asyncio.get_event_loop().run_until_complete(engine.analyze_position(fen))

    if args.json:
        output = {
            "fen": fen,
            "best_move": analysis.best_move,
            "consensus_move": analysis.consensus_move,
            "candidates": [
                {"move": cm.move, "score": cm.score, "agent": cm.agent_name}
                for cm in analysis.candidate_moves
            ],
            "routing": {
                "primary": analysis.routing_decision.primary_agent,
                "phase": analysis.routing_decision.game_phase,
                "weights": analysis.routing_decision.agent_weights,
            },
            "total_time_ms": analysis.total_time_ms,
        }
        print(json.dumps(output, indent=2))
    else:
        print_routing(analysis.routing_decision)
        print_agent_results(analysis.agent_results)
        print_best_move(analysis)
        print_stats(analysis)


def self_play_cmd(args):
    """Run an engine-vs-engine game."""
    engine = _get_engine(args)
    fen = args.fen or INITIAL_FEN
    max_moves = args.max_moves

    if not args.json:
        print(c("--- Self-Play Game ---", BOLD))
        print()

    moves_played: list[str] = []
    current_fen = fen

    try:
        import chess as _chess_lib
        board = _chess_lib.Board(current_fen)
        use_chess_lib = True
    except ImportError:
        board = None
        use_chess_lib = False

    for move_idx in range(max_moves):
        if use_chess_lib and board is not None and board.is_game_over():
            break

        if not args.json:
            side = fen_side_to_move(current_fen)
            print(f"  Move {move_idx + 1} ({side})...", end=" ", flush=True)

        analysis = asyncio.get_event_loop().run_until_complete(
            engine.analyze_position(current_fen)
        )
        move = analysis.best_move
        moves_played.append(move)

        if not args.json:
            print(c(move, GREEN))

        # Apply move
        if use_chess_lib and board is not None:
            try:
                board.push_uci(move)
                current_fen = board.fen()
            except (ValueError, _chess_lib.IllegalMoveError):
                if not args.json:
                    print(c(f"  Illegal move {move} — game ended.", RED))
                break
        else:
            # Without python-chess, we can't update the FEN properly
            if not args.json:
                print(c("  (Cannot update FEN without python-chess)", DIM))
            break

    if not args.json:
        print()
        if use_chess_lib and board is not None:
            print_board(board.fen())
            if board.is_game_over():
                outcome = board.outcome()
                if outcome:
                    if outcome.winner is True:
                        print(c("  Result: White wins!", GREEN))
                    elif outcome.winner is False:
                        print(c("  Result: Black wins!", GREEN))
                    else:
                        print(c("  Result: Draw", YELLOW))
                    print(f"  Termination: {outcome.termination.name}")
            else:
                print(f"  Game stopped after {len(moves_played)} moves")
        print(f"  Moves: {' '.join(moves_played)}")
        print()
    else:
        result = {
            "moves": moves_played,
            "total_moves": len(moves_played),
            "final_fen": current_fen,
        }
        if use_chess_lib and board is not None and board.is_game_over():
            outcome = board.outcome()
            if outcome:
                result["winner"] = (
                    "white" if outcome.winner is True
                    else "black" if outcome.winner is False
                    else "draw"
                )
        print(json.dumps(result, indent=2))


def interactive_cmd(args):
    """Run an interactive human-vs-engine game."""
    engine = _get_engine(args)
    fen = args.fen or INITIAL_FEN

    try:
        import chess as _chess_lib
    except ImportError:
        print(c("Error: python-chess is required for interactive mode.", RED))
        print("Install with: pip install python-chess")
        sys.exit(1)

    board = _chess_lib.Board(fen)
    engine_color = "black" if args.color == "white" else "white"

    print(c("  Interactive Chess - You vs LLM Engine", BOLD))
    print(f"  You play: {c(args.color.capitalize(), CYAN)}")
    print(f"  Engine:   {c(engine_color.capitalize(), CYAN)}")
    print(c("  Type a move in UCI format (e.g., e2e4) or 'quit' to exit.", DIM))
    print()

    while not board.is_game_over():
        print_board(board.fen())

        current_side = "white" if board.turn else "black"

        if current_side == args.color:
            # Human move
            while True:
                try:
                    user_input = input(c("Your move> ", BOLD)).strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    return

                if user_input.lower() in ("quit", "exit", "q"):
                    print("Game abandoned.")
                    return

                try:
                    move = _chess_lib.Move.from_uci(user_input)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print(c(f"  Illegal move: {user_input}", RED))
                        legal = [m.uci() for m in board.legal_moves]
                        print(f"  Legal moves: {', '.join(legal[:15])}...")
                except ValueError:
                    print(c(f"  Invalid format: {user_input}. Use UCI (e.g., e2e4)", RED))
        else:
            # Engine move
            print(c("  Engine thinking...", DIM))
            analysis = asyncio.get_event_loop().run_until_complete(
                engine.analyze_position(board.fen())
            )
            move_uci = analysis.best_move
            try:
                move = _chess_lib.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                    print(f"  Engine plays: {c(move_uci, GREEN)}")
                    if analysis.routing_decision:
                        print(f"    Phase: {analysis.routing_decision.game_phase}")
                        print(f"    Strategy: {analysis.routing_decision.primary_agent}")
                else:
                    # Engine suggested illegal move — pick random legal
                    legal = list(board.legal_moves)
                    if legal:
                        fallback = legal[0]
                        board.push(fallback)
                        print(f"  Engine plays: {c(fallback.uci(), YELLOW)} (fallback)")
                    else:
                        break
            except ValueError:
                legal = list(board.legal_moves)
                if legal:
                    fallback = legal[0]
                    board.push(fallback)
                    print(f"  Engine plays: {c(fallback.uci(), YELLOW)} (fallback)")
                else:
                    break
        print()

    # Game over
    print_board(board.fen())
    outcome = board.outcome()
    if outcome:
        if outcome.winner is True:
            print(c("  White wins!", GREEN))
        elif outcome.winner is False:
            print(c("  Black wins!", GREEN))
        else:
            print(c("  Draw!", YELLOW))
        print(f"  Reason: {outcome.termination.name}")
    print()


def mcp_tools_cmd(args):
    """List available MCP chess tools."""
    from src.games.chess.mcp_chess_tools import get_chess_tool_definitions

    tools = get_chess_tool_definitions()
    if args.json:
        print(json.dumps(tools, indent=2))
    else:
        print(c("--- Available Chess MCP Tools ---", BOLD))
        print()
        for tool in tools:
            print(f"  {c(tool['name'], GREEN)}")
            print(f"    {tool['description']}")
        print()
        print(f"  Total: {len(tools)} tools")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_DEPTH = 8
DEFAULT_MAX_MOVES = 40
DEFAULT_TEMPERATURE = 0.3


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Multi-Agent MCTS - Chess Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python chess_demo.py                                  # Analyse starting position (mock)
              python chess_demo.py --provider openai --analyze      # Real LLM analysis
              python chess_demo.py --interactive --color white      # Play as white
              python chess_demo.py --self-play --depth 4            # Engine vs engine
              python chess_demo.py --mcp-tools                      # List MCP tools
        """),
    )
    parser.add_argument("--provider", default="mock", choices=["mock", "openai", "anthropic"],
                        help="LLM provider (default: mock)")
    parser.add_argument("--api-key", default=None, help="API key (or set env var)")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--fen", default=None, help="FEN string for position")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help=f"MCTS depth / iterations (default: {DEFAULT_DEPTH})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"LLM temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--color", default="white", choices=["white", "black"],
                        help="Your color in interactive mode (default: white)")
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES,
                        help=f"Max moves in self-play (default: {DEFAULT_MAX_MOVES})")

    # Mode flags
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--analyze", action="store_true", help="Analyse a position (default)")
    mode.add_argument("--interactive", action="store_true", help="Interactive human vs engine")
    mode.add_argument("--self-play", action="store_true", help="Engine vs engine game")
    mode.add_argument("--mcp-tools", action="store_true", help="List available MCP tools")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not args.json:
        print_header()
        provider_label = args.provider if args.provider != "mock" else "mock (no API key needed)"
        print(f"  Provider: {c(provider_label, CYAN)}")
        print(f"  Depth:    {args.depth}")
        print()

    if args.interactive:
        interactive_cmd(args)
    elif args.self_play:
        self_play_cmd(args)
    elif args.mcp_tools:
        mcp_tools_cmd(args)
    else:
        analyze_position_cmd(args)


if __name__ == "__main__":
    main()
