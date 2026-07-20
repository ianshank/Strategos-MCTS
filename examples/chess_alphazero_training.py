#!/usr/bin/env python3
"""
AlphaZero-Style Chess Training with Ensemble Agents.

This example demonstrates how to train a chess AI using the complete
LangGraph Multi-Agent MCTS framework with HRM, TRM, and Neural MCTS
ensemble agents.

Features demonstrated:
- Configurable AlphaZero-style training pipeline
- Ensemble agent with meta-controller routing
- Self-play game generation
- Neural network training with experience replay
- Checkpointing and evaluation
- Inference and human vs AI play

Usage:
    # Show component examples
    python examples/chess_alphazero_training.py --mode components

    # Train the model
    python examples/chess_alphazero_training.py --mode train --preset small --iterations 10

    # Run inference on a position
    python examples/chess_alphazero_training.py --mode inference --fen "starting"

    # Play against the AI
    python examples/chess_alphazero_training.py --mode play --checkpoint ./checkpoints/chess/final_model
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure stdout/stderr are UTF-8 on Windows (default cp1252 cannot encode
# box-drawing / checkmark characters used in training output).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


# ============================================================================
# Component Examples
# ============================================================================
def show_component_examples() -> None:
    """Demonstrate individual chess components."""
    print_header("Chess AlphaZero Components")

    # Import chess components
    from src.games.chess import (
        ChessActionEncoder,
        ChessBoardRepresentation,
        ChessConfig,
        ChessGameState,
        ChessMetaController,
        get_chess_small_config,
    )

    # 1. Configuration
    print("1. Chess Configuration")
    print("-" * 40)
    config = get_chess_small_config()
    print(f"   Input channels: {config.input_channels}")
    print(f"   Action space size: {config.action_size}")
    print(f"   MCTS simulations: {config.mcts.num_simulations}")
    print(f"   ResNet blocks: {config.neural_net.num_res_blocks}")
    print(f"   Device: {config.device}")

    # 2. Game State
    print("\n2. Chess Game State")
    print("-" * 40)
    state = ChessGameState.initial()
    print(f"   Current player: {'White' if state.current_player == 1 else 'Black'}")
    print(f"   Move number: {state.move_number}")
    print(f"   Legal moves: {len(state.get_legal_actions())}")
    print(f"   Game phase: {state.get_game_phase().value}")
    print(f"   FEN: {state.fen[:50]}...")

    # Play a few moves
    print("\n   Playing e4 e5 Nf3...")
    state = state.apply_action("e2e4")
    state = state.apply_action("e7e5")
    state = state.apply_action("g1f3")
    print(f"   Position after 1.e4 e5 2.Nf3:")
    print(f"   {state}")

    # 3. Action Encoder
    print("\n3. Action Space Encoder")
    print("-" * 40)
    encoder = ChessActionEncoder()
    print(f"   Total actions: {encoder.action_size}")

    # Encode some moves
    test_moves = ["e2e4", "g1f3", "e7e8q"]
    for move in test_moves:
        try:
            idx = encoder.encode_move(move)
            decoded = encoder.decode_move(idx)
            print(f"   {move} -> index {idx} -> {decoded}")
        except ValueError as e:
            print(f"   {move} -> Error: {e}")

    # 4. Board Representation
    print("\n4. Board Tensor Representation")
    print("-" * 40)
    rep = ChessBoardRepresentation()
    print(f"   Input shape: {rep.input_shape}")
    print(f"   Total planes: {rep.num_planes}")

    initial_state = ChessGameState.initial()
    tensor = initial_state.to_tensor()
    print(f"   Tensor shape: {tuple(tensor.shape)}")
    print(f"   Non-zero elements: {(tensor != 0).sum().item()}")

    # 5. Meta-Controller
    print("\n5. Meta-Controller Routing")
    print("-" * 40)
    controller = ChessMetaController(config.ensemble, device="cpu")

    # Test routing for different positions
    positions = [
        ("Initial position", ChessGameState.initial()),
        ("After 1.e4", ChessGameState.from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")),
        ("Endgame", ChessGameState.from_fen("8/8/4k3/8/4P3/8/8/4K3 w - - 0 50")),
    ]

    for name, pos in positions:
        decision = controller.route(pos)
        print(f"   {name}:")
        print(f"      Primary agent: {decision.primary_agent.value}")
        print(f"      Confidence: {decision.confidence:.2f}")
        print(f"      Reasoning: {decision.reasoning[:50]}...")

    print("\n" + "=" * 80)
    print(" Component examples completed!")
    print("=" * 80)


# ============================================================================
# Training
# ============================================================================
async def run_training(
    preset: str,
    num_iterations: int,
    checkpoint_dir: str | None,
) -> None:
    """Run the training pipeline.

    Args:
        preset: Configuration preset ("small", "medium", "large")
        num_iterations: Number of training iterations
        checkpoint_dir: Optional custom checkpoint directory
    """
    print_header(f"Chess AlphaZero Training ({preset} preset)")

    from src.games.chess import (
        ChessConfig,
        ChessTrainingOrchestrator,
    )

    # Create configuration
    config = ChessConfig.from_preset(preset)
    if checkpoint_dir:
        config.checkpoint_dir = checkpoint_dir

    print("Configuration:")
    print(f"  Preset: {preset}")
    print(f"  Device: {config.device}")
    print(f"  MCTS simulations: {config.mcts.num_simulations}")
    print(f"  Games per iteration: {config.training.games_per_iteration}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Checkpoint dir: {config.checkpoint_dir}")
    print()

    # Create orchestrator
    orchestrator = ChessTrainingOrchestrator(config)

    # Run training
    print(f"Starting training for {num_iterations} iterations...")
    print("-" * 40)

    try:
        metrics = await orchestrator.train(num_iterations)

        # Print summary
        print("\n" + "-" * 40)
        print("Training Summary:")
        print(f"  Iterations completed: {len(metrics)}")

        if metrics:
            final = metrics[-1]
            print(f"  Final policy loss: {final.policy_loss:.4f}")
            print(f"  Final value loss: {final.value_loss:.4f}")
            print(f"  Total games played: {sum(m.games_played for m in metrics)}")
            print(f"  Average game length: {sum(m.average_game_length for m in metrics) / len(metrics):.1f}")

        print(f"\n  Best model saved to: {orchestrator.best_model_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    print("\n" + "=" * 80)
    print(" Training completed!")
    print("=" * 80)


# ============================================================================
# Inference
# ============================================================================
async def run_inference(
    fen: str,
    checkpoint_path: str | None,
    num_simulations: int,
) -> None:
    """Run inference on a position.

    Args:
        fen: FEN string or "starting" for initial position
        checkpoint_path: Path to model checkpoint
        num_simulations: Number of MCTS simulations
    """
    print_header("Chess AlphaZero Inference")

    from src.games.chess import (
        ChessConfig,
        ChessEnsembleAgent,
        ChessGameState,
        get_chess_small_config,
    )

    # Create configuration
    config = get_chess_small_config()
    config.mcts.num_simulations = num_simulations

    # Create state
    if fen.lower() == "starting":
        state = ChessGameState.initial()
        print("Position: Starting position")
    else:
        state = ChessGameState.from_fen(fen)
        print(f"Position: {fen}")

    print(f"\n{state}\n")
    print(f"Side to move: {'White' if state.current_player == 1 else 'Black'}")
    print(f"Game phase: {state.get_game_phase().value}")
    print(f"Legal moves: {len(state.get_legal_actions())}")

    # Create ensemble agent
    print("\nCreating ensemble agent...")
    agent = ChessEnsembleAgent(config)

    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        agent = ChessEnsembleAgent.load(checkpoint_path)

    # Get best move
    print(f"\nRunning analysis ({num_simulations} simulations)...")
    response = await agent.get_best_move(state, temperature=0.0, use_ensemble=True)

    # Print results
    print("\n" + "-" * 40)
    print("Analysis Results:")
    print(f"  Best move: {response.best_move}")
    print(f"  Value estimate: {response.value_estimate:.3f}")
    print(f"  Confidence: {response.confidence:.3f}")
    print(f"  Thinking time: {response.thinking_time_ms:.1f}ms")
    print(f"  Ensemble method: {response.ensemble_method}")

    print("\n  Routing decision:")
    print(f"    Primary agent: {response.routing_decision.primary_agent.value}")
    print(f"    Reasoning: {response.routing_decision.reasoning}")

    print("\n  Top moves:")
    sorted_moves = sorted(response.move_probabilities.items(), key=lambda x: -x[1])[:5]
    for move, prob in sorted_moves:
        print(f"    {move}: {prob:.1%}")

    print("\n" + "=" * 80)
    print(" Inference completed!")
    print("=" * 80)


# ============================================================================
# Interactive Play
# ============================================================================
async def play_vs_ai(checkpoint_path: str | None, player_color: str) -> None:
    """Play against the AI.

    Args:
        checkpoint_path: Path to model checkpoint
        player_color: "white" or "black"
    """
    print_header("Play Chess vs AlphaZero AI")

    from src.games.chess import (
        ChessEnsembleAgent,
        ChessGameState,
        get_chess_small_config,
    )

    # Create configuration
    config = get_chess_small_config()
    config.mcts.num_simulations = 100  # Fast for interactive play

    # Create ensemble agent
    print("Creating AI agent...")
    agent = ChessEnsembleAgent(config)

    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        agent = ChessEnsembleAgent.load(checkpoint_path)

    # Setup game
    state = ChessGameState.initial()
    human_is_white = player_color.lower() == "white"

    print(f"\nYou are playing as {'White' if human_is_white else 'Black'}")
    print("Enter moves in UCI format (e.g., 'e2e4', 'g1f3')")
    print("Type 'quit' to exit, 'undo' to take back a move\n")

    move_history: list[ChessGameState] = [state]

    while not state.is_terminal():
        print(f"\n{state}\n")

        is_human_turn = (state.current_player == 1) == human_is_white

        if is_human_turn:
            # Human's turn
            while True:
                move = input("Your move: ").strip().lower()

                if move == "quit":
                    print("Thanks for playing!")
                    return

                if move == "undo" and len(move_history) > 2:
                    # Undo two moves (human + AI)
                    move_history = move_history[:-2]
                    state = move_history[-1]
                    print("Undone last two moves.")
                    break

                if move in state.get_legal_actions():
                    state = state.apply_action(move)
                    move_history.append(state)
                    break
                else:
                    print(f"Illegal move. Legal moves: {', '.join(state.get_legal_actions()[:10])}...")
        else:
            # AI's turn
            print("AI is thinking...")
            response = await agent.get_best_move(state, temperature=0.1, use_ensemble=True)
            move = response.best_move

            print(f"AI plays: {move}")
            print(f"  (Value: {response.value_estimate:.2f}, Confidence: {response.confidence:.2%})")

            state = state.apply_action(move)
            move_history.append(state)

    # Game over
    print(f"\n{state}\n")
    print("-" * 40)

    if state.is_checkmate():
        winner = "Black" if state.current_player == 1 else "White"
        print(f"Checkmate! {winner} wins!")
    elif state.is_stalemate():
        print("Stalemate! Game is a draw.")
    else:
        print("Game over (draw).")


# ============================================================================
# Main Entry Point
# ============================================================================
def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="AlphaZero-Style Chess Training with Ensemble Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode components
  %(prog)s --mode train --preset small --iterations 10
  %(prog)s --mode inference --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
  %(prog)s --mode play --player white
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["components", "train", "inference", "play"],
        default="components",
        help="Mode to run (default: components)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["small", "medium", "large"],
        default="small",
        help="Configuration preset for training (default: small)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of training iterations (default: 10)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--fen",
        type=str,
        default="starting",
        help="FEN string for inference (default: starting position)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="Number of MCTS simulations for inference (default: 100)",
    )
    parser.add_argument(
        "--player",
        type=str,
        choices=["white", "black"],
        default="white",
        help="Player color for interactive play (default: white)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Check for required dependency
    try:
        import chess  # noqa: F401
    except ImportError:
        print("Error: python-chess is required. Install with: pip install python-chess")
        sys.exit(1)

    # Run selected mode
    if args.mode == "components":
        show_component_examples()

    elif args.mode == "train":
        asyncio.run(
            run_training(
                preset=args.preset,
                num_iterations=args.iterations,
                checkpoint_dir=args.checkpoint_dir,
            )
        )

    elif args.mode == "inference":
        asyncio.run(
            run_inference(
                fen=args.fen,
                checkpoint_path=args.checkpoint,
                num_simulations=args.simulations,
            )
        )

    elif args.mode == "play":
        asyncio.run(
            play_vs_ai(
                checkpoint_path=args.checkpoint,
                player_color=args.player,
            )
        )


if __name__ == "__main__":
    main()
