"""
Example: DeepMind-Style Self-Improving AI System

This example demonstrates how to use the complete LangGraph Multi-Agent MCTS
framework with HRM, TRM, and Neural MCTS for self-improving AI systems.

Features demonstrated:
- Setting up the training pipeline
- Configuring HRM, TRM, and MCTS components
- Running self-play training iterations
- Checkpointing and evaluation
- Inference with trained models
"""

import asyncio
from pathlib import Path
import sys

# Ensure stdout/stderr are UTF-8 on Windows (default cp1252 cannot encode
# box-drawing / checkmark characters used in training output).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch

from src.framework.mcts.neural_mcts import GameState
from src.training.system_config import get_small_config
from src.training.unified_orchestrator import UnifiedTrainingOrchestrator


# ============================================================================
# Step 1: Define Your Problem Domain (Game/Task State)
# ============================================================================
class TicTacToeState(GameState):
    """
    Example: Tic-Tac-Toe game state.

    Replace this with your own domain (e.g., ARC-AGI puzzles, code generation, etc.)
    """

    def __init__(self, board=None, player=1):
        """
        Initialize game state.

        Args:
            board: 3x3 board (None for empty board)
            player: Current player (1 or -1)
        """
        if board is None:
            self.board = [[0] * 3 for _ in range(3)]
        else:
            self.board = [row[:] for row in board]  # Deep copy

        self.player = player

    def get_legal_actions(self):
        """Return list of legal moves (empty positions)."""
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    actions.append(f"{i},{j}")
        return actions

    def apply_action(self, action):
        """Apply action and return new state."""
        i, j = map(int, action.split(","))
        new_board = [row[:] for row in self.board]
        new_board[i][j] = self.player

        return TicTacToeState(new_board, -self.player)

    def is_terminal(self):
        """Check if game is over."""
        # Check for winner
        if self._check_winner() != 0:
            return True

        # Check for draw
        return all(self.board[i][j] != 0 for i in range(3) for j in range(3))

    def get_reward(self, player=1):
        """Get reward for the player."""
        winner = self._check_winner()
        if winner == player:
            return 1.0
        elif winner == -player:
            return -1.0
        return 0.0

    def _check_winner(self):
        """Check for winner (1, -1, or 0)."""
        # Check rows
        for i in range(3):
            if abs(sum(self.board[i])) == 3:
                return self.board[i][0]

        # Check columns
        for j in range(3):
            col_sum = sum(self.board[i][j] for i in range(3))
            if abs(col_sum) == 3:
                return self.board[0][j]

        # Check diagonals
        diag1 = sum(self.board[i][i] for i in range(3))
        if abs(diag1) == 3:
            return self.board[0][0]

        diag2 = sum(self.board[i][2 - i] for i in range(3))
        if abs(diag2) == 3:
            return self.board[0][2]

        return 0

    def to_tensor(self):
        """
        Convert state to tensor for neural network.

        Returns:
            [channels=3, height=3, width=3] tensor
            - Channel 0: Current player's pieces
            - Channel 1: Opponent's pieces
            - Channel 2: Empty positions
        """
        tensor = torch.zeros(3, 3, 3)

        for i in range(3):
            for j in range(3):
                if self.board[i][j] == self.player:
                    tensor[0, i, j] = 1  # Current player
                elif self.board[i][j] == -self.player:
                    tensor[1, i, j] = 1  # Opponent
                else:
                    tensor[2, i, j] = 1  # Empty

        return tensor

    def get_canonical_form(self, player):
        """Get state from perspective of given player."""
        if player == self.player:
            return self
        # Flip board perspective
        flipped_board = [[-cell for cell in row] for row in self.board]
        return TicTacToeState(flipped_board, player)

    def get_hash(self):
        """Get unique hash for this state."""
        board_str = "".join(str(self.board[i][j]) for i in range(3) for j in range(3))
        return f"{board_str}_{self.player}"

    def __str__(self):
        """String representation for debugging."""
        symbols = {1: "X", -1: "O", 0: "."}
        return "\n".join(" ".join(symbols[cell] for cell in row) for row in self.board)


# ============================================================================
# Step 2: Configure the Framework
# ============================================================================
def setup_configuration():
    """
    Setup system configuration.

    For production, use get_medium_config() or get_large_config().
    For experimentation, use get_small_config().
    """
    # Start with small config for fast iteration
    config = get_small_config()

    # Customize for Tic-Tac-Toe
    config.neural_net.input_channels = 3  # 3 channels (player, opponent, empty)
    config.neural_net.action_size = 9  # 3x3 board
    config.neural_net.num_res_blocks = 3  # Small for this simple game

    # MCTS settings
    config.mcts.num_simulations = 100  # Reasonable for Tic-Tac-Toe
    config.mcts.c_puct = 1.0
    config.mcts.temperature_threshold = 10  # Switch to greedy after 10 moves

    # Training settings
    config.training.games_per_iteration = 100  # Self-play games per iteration
    config.training.batch_size = 64
    config.training.learning_rate = 0.01

    # Enable experiment tracking
    config.use_wandb = False  # Set to True to use Weights & Biases

    # Paths
    config.checkpoint_dir = "./checkpoints"
    config.log_dir = "./logs"

    return config


# ============================================================================
# Step 3: Training Pipeline
# ============================================================================
async def train_model():
    """
    Train the complete DeepMind-style system.

    This demonstrates the full training loop:
    1. Self-play data generation
    2. Neural network training
    3. HRM and TRM training
    4. Evaluation and checkpointing
    """
    print("\n" + "=" * 80)
    print("DeepMind-Style Training Example: Tic-Tac-Toe")
    print("=" * 80 + "\n")

    # Setup configuration
    config = setup_configuration()

    # Create initial state function
    def initial_state_fn():
        return TicTacToeState()

    # Initialize orchestrator
    orchestrator = UnifiedTrainingOrchestrator(config=config, initial_state_fn=initial_state_fn, board_size=3)

    # Run training
    num_iterations = 10  # Train for 10 iterations (increase for real training)

    await orchestrator.train(num_iterations=num_iterations)

    print("\n✓ Training completed!")
    print(f"✓ Best model saved to: {orchestrator.best_model_path}")

    return orchestrator


# ============================================================================
# Step 4: Inference with Trained Model
# ============================================================================
async def inference_example(checkpoint_path: str):
    """
    Demonstrate inference with a trained model.

    Args:
        checkpoint_path: Path to model checkpoint
    """
    print("\n" + "=" * 80)
    print("Inference Example")
    print("=" * 80 + "\n")

    # Load configuration and models
    config = setup_configuration()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)

    # Create policy-value network
    from src.models.policy_value_net import create_policy_value_network

    policy_value_net = create_policy_value_network(config.neural_net, board_size=3, device=config.device)
    policy_value_net.load_state_dict(checkpoint["policy_value_net"])
    policy_value_net.eval()

    # Create MCTS
    from src.framework.mcts.neural_mcts import NeuralMCTS

    mcts = NeuralMCTS(policy_value_net, config.mcts, config.device)

    # Play a game
    state = TicTacToeState()
    print("Starting game:\n")
    print(state)
    print()

    move_count = 0
    while not state.is_terminal():
        print(f"Move {move_count + 1}:")

        # Run MCTS
        action_probs, root = await mcts.search(state, num_simulations=100, temperature=0.1, add_root_noise=False)

        # Select best action
        best_action = max(action_probs.items(), key=lambda x: x[1])[0]
        print(f"  Best action: {best_action}")
        print(f"  Action probabilities: {action_probs}")

        # Apply action
        state = state.apply_action(best_action)
        print(f"\nBoard:\n{state}\n")

        move_count += 1

        if move_count > 20:  # Safety limit
            break

    # Game result
    winner = state._check_winner()
    if winner == 1:
        print("Result: X wins!")
    elif winner == -1:
        print("Result: O wins!")
    else:
        print("Result: Draw!")


# ============================================================================
# Step 5: Using Individual Components
# ============================================================================
def component_examples():
    """Demonstrate using individual components."""
    print("\n" + "=" * 80)
    print("Component Examples")
    print("=" * 80 + "\n")

    config = setup_configuration()
    device = config.device

    # Example 1: HRM Agent
    print("1. HRM Agent (Hierarchical Reasoning)")
    from src.agents.hrm_agent import create_hrm_agent

    hrm_agent = create_hrm_agent(config.hrm, device)
    print(f"   Parameters: {hrm_agent.get_parameter_count():,}")

    # Create dummy input
    x = torch.randn(1, 4, config.hrm.h_dim).to(device)
    with torch.no_grad():
        hrm_output = hrm_agent(x, max_steps=5)
    print(f"   Halted at step: {hrm_output.halt_step}")
    print(f"   Convergence path: {hrm_output.convergence_path}")

    # Example 2: TRM Agent
    print("\n2. TRM Agent (Recursive Refinement)")
    from src.agents.trm_agent import create_trm_agent

    trm_agent = create_trm_agent(config.trm, output_dim=9, device=device)
    print(f"   Parameters: {trm_agent.get_parameter_count():,}")

    # Create dummy input
    x = torch.randn(1, config.trm.latent_dim).to(device)
    with torch.no_grad():
        trm_output = trm_agent(x, num_recursions=10)
    print(f"   Converged: {trm_output.converged}")
    print(f"   Convergence step: {trm_output.convergence_step}")
    print(f"   Recursion depth: {trm_output.recursion_depth}")

    # Example 3: Policy-Value Network
    print("\n3. Policy-Value Network (AlphaZero-style)")
    from src.models.policy_value_net import create_policy_value_network

    policy_value_net = create_policy_value_network(config.neural_net, board_size=3, device=device)
    print(f"   Parameters: {policy_value_net.get_parameter_count():,}")

    # Create dummy board state
    board_tensor = torch.randn(1, 3, 3, 3).to(device)
    with torch.no_grad():
        policy_logits, value = policy_value_net(board_tensor)
    print(f"   Policy shape: {policy_logits.shape}")
    print(f"   Value: {value.item():.3f}")

    print("\n✓ Component examples completed!")


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """Main entry point with command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="DeepMind-Style Training Example")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference", "components"],
        default="components",
        help="Mode to run: train, inference, or components",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best_model.pt",
        help="Path to checkpoint (for inference mode)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        # Run training
        asyncio.run(train_model())

    elif args.mode == "inference":
        # Run inference
        if not Path(args.checkpoint).exists():
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            print("   Run with --mode train first to create a checkpoint")
            return

        asyncio.run(inference_example(args.checkpoint))

    elif args.mode == "components":
        # Show component examples
        component_examples()


if __name__ == "__main__":
    main()
