"""
Create LangSmith RAG Evaluation Datasets.

This script creates a dataset in LangSmith specifically for evaluating RAG
(Retrieval-Augmented Generation) performance on MCTS, AlphaZero, and game
tree search topics.

Usage:
    python scripts/create_rag_eval_datasets.py

Datasets created:
- rag-eval-dataset: MCTS and game tree search Q&A with ground truth and contexts
"""

import os
from pathlib import Path
import sys
import traceback

from tests.utils.langsmith_tracing import create_test_dataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_rag_eval_dataset() -> str:
    """Create RAG evaluation dataset with MCTS-related Q&A examples."""
    examples = [
        # Basic MCTS Concepts
        {
            "inputs": {
                "question": "What is Monte Carlo Tree Search?",
                "contexts": [
                    "Monte Carlo Tree Search (MCTS) is a heuristic search algorithm for decision-making "
                    "problems. It combines the precision of tree search with the generality of random sampling.",
                    "MCTS operates by building a search tree incrementally and asymmetrically. Each iteration "
                    "consists of four phases: Selection, Expansion, Simulation, and Backpropagation.",
                    "The algorithm was first introduced in 2006 and gained prominence after being successfully "
                    "applied to computer Go programs.",
                ],
            },
            "outputs": {
                "ground_truth": "Monte Carlo Tree Search (MCTS) is a heuristic search algorithm used for "
                "decision-making that combines tree search with random sampling. It builds a search tree "
                "incrementally through iterative cycles of Selection, Expansion, Simulation, and Backpropagation, "
                "making it particularly effective for complex decision spaces like game playing.",
            },
        },
        # UCB1 and Exploration/Exploitation
        {
            "inputs": {
                "question": "How does UCB1 balance exploration and exploitation in MCTS?",
                "contexts": [
                    "UCB1 (Upper Confidence Bound 1) is a key formula in MCTS that balances exploration of "
                    "less-visited nodes with exploitation of promising paths.",
                    "The UCB1 formula is: UCB1 = Q/N + C * sqrt(ln(N_parent)/N), where Q is the total reward, "
                    "N is visit count, N_parent is parent visit count, and C is the exploration constant.",
                    "The first term (Q/N) represents exploitation - favoring nodes with high average rewards. "
                    "The second term represents exploration - favoring less-visited nodes.",
                    "As a node is visited more, the exploration term decreases, gradually shifting from exploration "
                    "to exploitation. The constant C controls this tradeoff.",
                ],
            },
            "outputs": {
                "ground_truth": "UCB1 balances exploration and exploitation through a two-term formula: "
                "UCB1 = Q/N + C * sqrt(ln(N_parent)/N). The first term (Q/N) exploits by favoring high-reward "
                "nodes, while the second term explores by preferring less-visited nodes. As visit counts increase, "
                "the exploration bonus decreases, creating an automatic shift from exploration to exploitation. "
                "The constant C controls the strength of this tradeoff.",
            },
        },
        # MCTS vs Minimax
        {
            "inputs": {
                "question": "What is the difference between MCTS and minimax?",
                "contexts": [
                    "Minimax is a traditional game tree search algorithm that exhaustively searches the tree to "
                    "a fixed depth, assuming optimal play from both players.",
                    "MCTS builds the tree selectively and asymmetrically, focusing computational resources on "
                    "more promising branches rather than uniformly expanding all nodes.",
                    "Minimax requires an evaluation function to score leaf nodes, while MCTS uses random playouts "
                    "to estimate node values.",
                    "MCTS can handle larger state spaces more effectively because it doesn't require complete "
                    "tree expansion, making it superior for games with high branching factors like Go.",
                ],
            },
            "outputs": {
                "ground_truth": "MCTS differs from minimax in several key ways: (1) MCTS builds the search tree "
                "selectively and asymmetrically, focusing on promising branches, while minimax exhaustively searches "
                "to a fixed depth. (2) MCTS uses random playouts for evaluation while minimax requires a hand-crafted "
                "evaluation function. (3) MCTS can handle larger state spaces more effectively, making it better "
                "suited for high-branching-factor games. (4) MCTS provides anytime performance - it can be stopped "
                "at any point and return the best move found so far.",
            },
        },
        # AlphaZero
        {
            "inputs": {
                "question": "What is AlphaZero and how does it differ from traditional MCTS?",
                "contexts": [
                    "AlphaZero is a deep reinforcement learning algorithm developed by DeepMind that combines "
                    "MCTS with deep neural networks for game playing.",
                    "Unlike traditional MCTS which uses random playouts, AlphaZero uses a neural network to "
                    "evaluate positions and predict move probabilities.",
                    "AlphaZero was trained purely through self-play, starting from random play and learning "
                    "without any human knowledge or game-specific heuristics.",
                    "The algorithm achieved superhuman performance in chess, shogi, and Go, defeating world "
                    "champion programs in each game after just hours of self-play training.",
                ],
            },
            "outputs": {
                "ground_truth": "AlphaZero is a reinforcement learning algorithm that enhances MCTS with deep "
                "neural networks. Instead of random playouts, it uses a neural network to evaluate positions "
                "and guide tree search. It learns entirely through self-play without human knowledge, and has "
                "achieved superhuman performance in chess, shogi, and Go. The key innovation is replacing "
                "random simulations with learned value and policy networks, making the search much more efficient.",
            },
        },
        # MCTS Phases
        {
            "inputs": {
                "question": "What are the four phases of MCTS and what happens in each?",
                "contexts": [
                    "Selection: Starting from root, traverse the tree using a selection policy (like UCB1) "
                    "until reaching a node that has unvisited children.",
                    "Expansion: Add one or more child nodes to expand the tree from the selected node.",
                    "Simulation (Rollout): From the new node, simulate a complete game to a terminal state "
                    "using a rollout policy (often random moves).",
                    "Backpropagation: Propagate the simulation result back up the tree, updating visit counts "
                    "and value estimates for all nodes in the path.",
                ],
            },
            "outputs": {
                "ground_truth": "The four phases of MCTS are: (1) Selection - traverse the tree from root using "
                "UCB1 or similar policy until finding a node with unvisited children. (2) Expansion - add one or "
                "more child nodes to the tree. (3) Simulation - play out a complete game from the new node to a "
                "terminal state using a rollout policy. (4) Backpropagation - update statistics (visits and values) "
                "for all nodes in the path from the expanded node back to the root.",
            },
        },
        # PUCT
        {
            "inputs": {
                "question": "What is PUCT and how is it used in AlphaZero?",
                "contexts": [
                    "PUCT (Predictor + Upper Confidence bounds applied to Trees) is a variant of UCB1 used in "
                    "AlphaZero's MCTS implementation.",
                    "The PUCT formula incorporates a prior probability from the policy network: "
                    "PUCT = Q + U, where U = c * P * sqrt(N_parent) / (1 + N).",
                    "P is the prior probability from the neural network's policy head, guiding exploration "
                    "toward moves the network believes are promising.",
                    "This allows AlphaZero to use learned knowledge to guide tree search more effectively than "
                    "pure UCB1, focusing on moves that the network has learned are likely to be good.",
                ],
            },
            "outputs": {
                "ground_truth": "PUCT (Predictor + Upper Confidence bounds applied to Trees) is AlphaZero's "
                "enhanced selection formula that combines Q-values with a policy prior: PUCT = Q + c * P * "
                "sqrt(N_parent) / (1 + N). The P term is the prior probability from the neural network's policy "
                "head, which guides exploration toward moves the network believes are promising. This makes tree "
                "search more efficient by incorporating learned knowledge rather than treating all unexplored "
                "moves equally.",
            },
        },
        # Virtual Loss
        {
            "inputs": {
                "question": "What is virtual loss in parallel MCTS and why is it important?",
                "contexts": [
                    "Virtual loss is a technique used in parallel MCTS implementations to coordinate multiple "
                    "simultaneous tree traversals.",
                    "When a thread selects a node for expansion, it temporarily adds a 'virtual loss' to that "
                    "node, making it appear less attractive to other threads.",
                    "This prevents multiple threads from exploring the same path simultaneously, improving "
                    "exploration diversity across parallel workers.",
                    "After the rollout completes, the virtual loss is removed and replaced with the actual "
                    "simulation result during backpropagation.",
                ],
            },
            "outputs": {
                "ground_truth": "Virtual loss is a parallelization technique in MCTS where threads temporarily "
                "add a penalty (virtual loss) to nodes they're exploring. This makes those nodes appear less "
                "attractive to other threads, preventing redundant exploration of the same path. Once the rollout "
                "completes, the virtual loss is removed and replaced with the actual result. This improves parallel "
                "efficiency by encouraging different threads to explore different parts of the tree.",
            },
        },
        # Game Tree Search
        {
            "inputs": {
                "question": "What is a game tree and how is it used in game playing AI?",
                "contexts": [
                    "A game tree is a directed graph representing all possible states and moves in a game, "
                    "where nodes represent game states and edges represent moves.",
                    "The root node represents the current game state, and leaf nodes represent terminal states "
                    "(wins, losses, or draws).",
                    "Game tree search algorithms navigate this tree to find optimal moves by evaluating "
                    "different action sequences.",
                    "For many games, the complete game tree is too large to explore exhaustively, requiring "
                    "selective search strategies like MCTS or pruning techniques like alpha-beta.",
                ],
            },
            "outputs": {
                "ground_truth": "A game tree is a graph structure where nodes represent game states and edges "
                "represent moves between states. The root is the current position, and leaf nodes are terminal "
                "states. Game playing AI uses tree search algorithms to navigate this tree and find optimal moves. "
                "Since complete game trees are typically too large to explore exhaustively, algorithms like MCTS "
                "or alpha-beta pruning are used to search selectively and efficiently.",
            },
        },
        # Exploration Constant
        {
            "inputs": {
                "question": "How do you choose the exploration constant C in UCB1?",
                "contexts": [
                    "The exploration constant C in UCB1 controls the tradeoff between exploration and exploitation. "
                    "Higher values encourage more exploration of less-visited nodes.",
                    "Theoretical analysis suggests C = sqrt(2) for optimal performance in adversarial games, "
                    "though empirical tuning often yields better results.",
                    "The optimal C value depends on the problem domain, tree depth, branching factor, and "
                    "available computation time.",
                    "In practice, C is often tuned through experimentation, with values typically ranging from "
                    "0.5 to 2.0 depending on the application.",
                ],
            },
            "outputs": {
                "ground_truth": "The exploration constant C in UCB1 is typically chosen through a combination of "
                "theory and empirical tuning. Theoretically, C = sqrt(2) is optimal for adversarial games, but in "
                "practice, values between 0.5 and 2.0 are common. The optimal value depends on the problem domain, "
                "branching factor, tree depth, and computation time. Higher C values encourage more exploration, "
                "while lower values focus more on exploitation. Domain-specific tuning through experimentation is "
                "often necessary for best results.",
            },
        },
        # RAVE
        {
            "inputs": {
                "question": "What is RAVE (Rapid Action Value Estimation) in MCTS?",
                "contexts": [
                    "RAVE is an enhancement to MCTS that speeds up learning by sharing information between moves "
                    "that visit the same state regardless of when they're played.",
                    "The key insight is that in many games, the value of playing a particular move is somewhat "
                    "independent of when it's played in the sequence.",
                    "RAVE maintains statistics for each move (action) across all positions where it was played "
                    "during simulations, not just at the current position.",
                    "As more simulations occur, RAVE gradually transitions from using these 'all moves as first' "
                    "statistics to using traditional MCTS statistics for the specific position.",
                ],
            },
            "outputs": {
                "ground_truth": "RAVE (Rapid Action Value Estimation) is an MCTS enhancement that accelerates "
                "learning by sharing value information across different positions. It tracks statistics for each "
                "move across all positions where it was played during simulations, based on the insight that a "
                "move's value is often somewhat independent of when it's played. As more data accumulates, RAVE "
                "gradually shifts from these generalized statistics to position-specific MCTS values, providing "
                "better estimates early in the search when data is sparse.",
            },
        },
        # Progressive Widening
        {
            "inputs": {
                "question": "What is progressive widening in MCTS and when is it used?",
                "contexts": [
                    "Progressive widening (or progressive unpruning) limits the number of children considered at "
                    "each node based on the node's visit count.",
                    "Instead of immediately expanding all possible moves from a position, progressive widening "
                    "adds children gradually as a node is visited more frequently.",
                    "This is particularly useful in domains with very large branching factors (many possible moves) "
                    "or continuous action spaces.",
                    "A common formula is k(n) = C * n^α, where k is the number of children allowed when the node "
                    "has been visited n times, with α typically between 0.25 and 0.5.",
                ],
            },
            "outputs": {
                "ground_truth": "Progressive widening is an MCTS technique that limits the number of child nodes "
                "expanded based on the parent's visit count, adding children gradually as the node is visited more. "
                "This is useful for domains with large or continuous action spaces where considering all moves "
                "immediately would be inefficient. A typical formula is k(n) = C * n^α (α ∈ [0.25, 0.5]), allowing "
                "more children as the node receives more visits. This focuses early exploration on a subset of moves "
                "while still allowing full expansion eventually.",
            },
        },
        # Self-Play Training
        {
            "inputs": {
                "question": "How does self-play training work in AlphaZero?",
                "contexts": [
                    "AlphaZero uses self-play to generate training data by playing games against itself using "
                    "MCTS guided by its current neural network.",
                    "Each self-play game produces training examples consisting of (state, MCTS policy, game outcome) "
                    "tuples that are used to train the neural network.",
                    "The network is trained to predict both the final game outcome (value head) and the improved "
                    "MCTS policy (policy head), creating a virtuous cycle of improvement.",
                    "This process iterates continuously: self-play generates data, the network trains on that data, "
                    "and the improved network is used for the next generation of self-play games.",
                ],
            },
            "outputs": {
                "ground_truth": "AlphaZero's self-play training works by having the agent play games against itself "
                "using MCTS guided by its neural network. Each game generates training examples of (state, MCTS search "
                "policy, game result) which train the network to predict game outcomes (value) and move probabilities "
                "(policy). The improved network then guides better MCTS search in the next generation of self-play, "
                "creating an iterative improvement cycle. This requires no human expertise or game knowledge, learning "
                "entirely from self-play experience.",
            },
        },
        # Policy and Value Networks
        {
            "inputs": {
                "question": "What are policy and value networks in AlphaZero and what do they predict?",
                "contexts": [
                    "AlphaZero uses a single neural network with two output heads: a policy head and a value head.",
                    "The policy head outputs a probability distribution over all possible moves from the current "
                    "position, representing which moves are most likely to be good.",
                    "The value head outputs a scalar estimate of the probability of winning from the current position "
                    "with perfect play, ranging from -1 (certain loss) to +1 (certain win).",
                    "During MCTS, the policy head guides exploration toward promising moves, while the value head "
                    "provides position evaluation without needing rollouts.",
                ],
            },
            "outputs": {
                "ground_truth": "AlphaZero uses a dual-head neural network: (1) The policy head outputs a probability "
                "distribution over legal moves, indicating which moves are likely to be strong. This guides MCTS "
                "exploration through the PUCT formula. (2) The value head outputs a scalar win probability estimate "
                "from -1 (certain loss) to +1 (certain win), replacing random rollouts in traditional MCTS. Both "
                "heads share convolutional layers and are trained simultaneously from self-play data to predict MCTS "
                "search policies and game outcomes.",
            },
        },
        # Tree Policy vs Rollout Policy
        {
            "inputs": {
                "question": "What is the difference between tree policy and rollout policy in MCTS?",
                "contexts": [
                    "The tree policy determines how the algorithm selects nodes during the selection phase when "
                    "traversing already-explored parts of the tree.",
                    "Common tree policies include UCB1, which balances exploration and exploitation using the "
                    "UCB1 formula to select which child node to visit next.",
                    "The rollout policy (or simulation policy) determines how moves are chosen during the "
                    "simulation phase when playing out from a newly expanded node to a terminal state.",
                    "The rollout policy is often simpler and faster than the tree policy - commonly using random "
                    "moves or lightweight heuristics - since many simulations need to be performed quickly.",
                ],
            },
            "outputs": {
                "ground_truth": "Tree policy and rollout policy serve different phases of MCTS: The tree policy "
                "(like UCB1) is used during selection to navigate the explored portion of the tree, carefully "
                "balancing exploration and exploitation. The rollout policy is used during simulation to quickly "
                "play out games from newly expanded nodes to terminal states, often using random moves or simple "
                "heuristics for speed. Tree policy is more sophisticated but only applies to explored nodes, "
                "while rollout policy must be fast since it's called many times per iteration.",
            },
        },
        # MCTS Convergence
        {
            "inputs": {
                "question": "Does MCTS converge to optimal play and under what conditions?",
                "contexts": [
                    "MCTS with UCB1 has theoretical convergence guarantees: given infinite time and memory, it "
                    "will converge to the minimax optimal strategy.",
                    "The convergence proof relies on the UCB1 selection policy ensuring that all nodes are visited "
                    "infinitely often, allowing accurate value estimates to emerge.",
                    "In practice, MCTS doesn't run infinitely - it's stopped after a time or iteration budget, "
                    "so it finds approximately optimal solutions.",
                    "The approximation quality improves with more iterations, making MCTS an 'anytime algorithm' "
                    "that can return progressively better results the longer it runs.",
                ],
            },
            "outputs": {
                "ground_truth": "MCTS with UCB1 theoretically converges to minimax-optimal play given infinite "
                "iterations and memory. The proof relies on UCB1 ensuring all nodes are visited infinitely often, "
                "allowing value estimates to converge to true values. In practice, MCTS runs with finite resources, "
                "providing approximate solutions that improve with more iterations. This 'anytime' property means "
                "MCTS can be stopped at any point and return the best move found so far, with quality improving the "
                "longer it runs.",
            },
        },
        # State Space vs Action Space
        {
            "inputs": {
                "question": "What's the difference between state space and action space in game tree search?",
                "contexts": [
                    "The state space is the set of all possible game positions or configurations that can occur "
                    "during play.",
                    "The action space (or move space) is the set of all possible moves or actions that can be taken "
                    "from any given state.",
                    "In chess, the state space includes all possible board configurations with piece positions, "
                    "castling rights, en passant, etc. The action space is all possible moves.",
                    "State space size affects memory requirements for storing positions, while action space "
                    "(branching factor) affects the width of the search tree and computational complexity.",
                ],
            },
            "outputs": {
                "ground_truth": "State space and action space are related but distinct concepts: State space is the "
                "set of all possible game positions/configurations, affecting memory and position evaluation complexity. "
                "Action space is the set of all possible moves from any position, determining the branching factor of "
                "the search tree. For example, chess has an enormous state space (~10^43 positions) and variable action "
                "space (typically 20-40 legal moves per position). Large state spaces challenge position evaluation, "
                "while large action spaces increase search tree width and computational cost.",
            },
        },
        # Dirichlet Noise
        {
            "inputs": {
                "question": "What is Dirichlet noise and why is it added to the root node in AlphaZero?",
                "contexts": [
                    "Dirichlet noise is random noise added to the policy prior probabilities at the root node "
                    "of the search tree in AlphaZero.",
                    "The noise is drawn from a Dirichlet distribution, which ensures the values remain a valid "
                    "probability distribution (sum to 1, all positive).",
                    "This noise encourages exploration during self-play training by preventing the algorithm from "
                    "becoming too deterministic and always playing the same moves.",
                    "AlphaZero uses α = 0.3 for chess and α = 0.03 for Go, with the noise fraction typically "
                    "set to 25% of the policy prior at the root.",
                ],
            },
            "outputs": {
                "ground_truth": "Dirichlet noise is random exploration noise added to policy priors at the root node "
                "during AlphaZero's self-play training. It's drawn from a Dirichlet distribution to maintain valid "
                "probabilities. The noise (typically 25% of the prior) encourages exploration and prevents the agent "
                "from playing too deterministically during training, ensuring diverse game experiences that lead to "
                "more robust learning. The α parameter controls noise concentration, with smaller values (like 0.03 "
                "for Go) creating more uniform noise for games with larger action spaces.",
            },
        },
        # Transposition Tables
        {
            "inputs": {
                "question": "What are transposition tables and how are they used in game tree search?",
                "contexts": [
                    "Transposition tables are hash tables that store previously evaluated game positions to avoid "
                    "redundant computation when the same position is reached via different move sequences.",
                    "In games like chess, the same board position can often be reached through different move orders "
                    "(transpositions), and evaluating it multiple times wastes computation.",
                    "The table typically stores the position hash, evaluation score, best move, search depth, and "
                    "node type (exact, lower bound, upper bound).",
                    "While powerful in minimax-based algorithms, transposition tables are less commonly used in "
                    "pure MCTS because MCTS builds statistics over multiple visits rather than computing a single value.",
                ],
            },
            "outputs": {
                "ground_truth": "Transposition tables are hash tables that cache evaluated positions to avoid "
                "recomputing the same position reached through different move sequences. They store position hashes, "
                "evaluations, best moves, and search depth. This is highly effective in minimax-based search where "
                "the same position is evaluated identically regardless of path. In MCTS, transpositions are handled "
                "differently since MCTS accumulates statistics over multiple visits rather than computing single "
                "values, making standard transposition tables less applicable without modification.",
            },
        },
        # Temperature Sampling
        {
            "inputs": {
                "question": "What is temperature in the context of AlphaZero and how does it affect move selection?",
                "contexts": [
                    "Temperature is a parameter that controls the randomness of move selection in AlphaZero during "
                    "self-play and actual play.",
                    "Move probabilities are computed as π_i = N_i^(1/τ) / Σ_j N_j^(1/τ), where N_i is the visit "
                    "count for move i and τ is the temperature.",
                    "High temperature (τ >> 1) makes the distribution more uniform, encouraging exploration. Low "
                    "temperature (τ → 0) makes selection more deterministic, choosing the most-visited move.",
                    "AlphaZero typically uses τ = 1 for the first 30 moves of self-play games to encourage "
                    "exploration, then τ → 0 for the remainder to play more optimally.",
                ],
            },
            "outputs": {
                "ground_truth": "Temperature (τ) controls move selection randomness in AlphaZero using the formula "
                "π_i = N_i^(1/τ) / Σ_j N_j^(1/τ). High temperature makes selection more exploratory and uniform, "
                "while low temperature makes it more deterministic, favoring the most-visited move. During self-play, "
                "AlphaZero uses τ = 1 for the opening moves to ensure diverse training positions, then decreases to "
                "τ → 0 for optimal play. This balances exploration during training with strong play for generating "
                "high-quality training data.",
            },
        },
        # Branching Factor
        {
            "inputs": {
                "question": "What is branching factor and why is it important in game tree search?",
                "contexts": [
                    "Branching factor is the average number of possible moves (children) at each node in a game tree.",
                    "High branching factors make exhaustive search exponentially more difficult - a game with "
                    "branching factor b and depth d has approximately b^d leaf nodes.",
                    "Chess has an average branching factor of about 35, while Go has around 250, making Go much "
                    "harder for traditional search methods.",
                    "MCTS handles high branching factors better than minimax because it selectively explores "
                    "promising branches rather than uniformly expanding all nodes to a fixed depth.",
                ],
            },
            "outputs": {
                "ground_truth": "Branching factor is the average number of legal moves (child nodes) at each position "
                "in a game tree. It's crucial because it determines the exponential growth rate of the tree - with "
                "branching factor b and depth d, there are approximately b^d positions to consider. High branching "
                "factors (like Go's ~250) make exhaustive search infeasible, which is why selective search methods "
                "like MCTS are important. MCTS handles high branching factors better than minimax by focusing "
                "computational effort on promising branches rather than uniform expansion.",
            },
        },
        # First Play Urgency
        {
            "inputs": {
                "question": "What is First Play Urgency (FPU) in MCTS and why is it needed?",
                "contexts": [
                    "First Play Urgency addresses how to evaluate unvisited nodes in MCTS when choosing which "
                    "child to explore first.",
                    "Without FPU, unvisited nodes would have undefined value, making it unclear how to prioritize "
                    "them against visited nodes with known statistics.",
                    "FPU assigns an initial value estimate to unvisited nodes, typically slightly lower than the "
                    "parent's value (FPU reduction) to encourage exploring known-good branches first.",
                    "AlphaZero uses FPU reduction to make the algorithm more exploitation-oriented, focusing on "
                    "branches already showing promise before exploring completely new options.",
                ],
            },
            "outputs": {
                "ground_truth": "First Play Urgency (FPU) is a mechanism for handling unvisited nodes in MCTS by "
                "assigning them an initial value estimate. Without FPU, unvisited nodes would have undefined values, "
                "making selection policy unclear. FPU typically assigns a value slightly lower than the parent's "
                "value (FPU reduction), encouraging the algorithm to explore promising known branches before completely "
                "new ones. AlphaZero uses FPU reduction to balance the neural network's policy prior (which encourages "
                "exploration of new moves) with a preference for moves already showing promise.",
            },
        },
        # Neural Network Architecture
        {
            "inputs": {
                "question": "What neural network architecture does AlphaZero use?",
                "contexts": [
                    "AlphaZero uses a deep residual convolutional neural network (ResNet) architecture adapted "
                    "for board games.",
                    "The network consists of a shared trunk of residual blocks (19-40 blocks depending on the game), "
                    "followed by separate policy and value heads.",
                    "For board games like chess and Go, the input is a stack of board position planes representing "
                    "piece positions, repetitions, castling rights, and other game-specific features.",
                    "The architecture uses batch normalization and ReLU activations, with the policy head outputting "
                    "move probabilities and the value head outputting a scalar win probability via a tanh activation.",
                ],
            },
            "outputs": {
                "ground_truth": "AlphaZero uses a deep residual convolutional neural network (ResNet) with 19-40 "
                "residual blocks in the shared trunk, followed by separate policy and value heads. Input is a stack "
                "of board position planes encoding game state. The network uses batch normalization and ReLU activations. "
                "The policy head outputs a probability distribution over moves via softmax, while the value head outputs "
                "a win probability via tanh. This architecture efficiently processes spatial board representations "
                "while sharing feature learning between policy and value predictions.",
            },
        },
        # Knowledge Distillation
        {
            "inputs": {
                "question": "How is MCTS search knowledge transferred to the neural network in AlphaZero?",
                "contexts": [
                    "AlphaZero transfers knowledge from MCTS to the neural network through supervised learning on "
                    "self-play data.",
                    "Each training example consists of a game state, the MCTS search policy (visit count distribution), "
                    "and the final game outcome.",
                    "The network is trained to minimize two losses: cross-entropy between the predicted policy and "
                    "MCTS search policy, and mean squared error between predicted value and actual game outcome.",
                    "This process is a form of knowledge distillation where the slower, more accurate MCTS search "
                    "teaches the faster neural network to approximate its behavior.",
                ],
            },
            "outputs": {
                "ground_truth": "AlphaZero transfers MCTS knowledge to the neural network through supervised learning "
                "on self-play games. Each training example includes (state, MCTS search policy from visit counts, game "
                "result). The network learns via two losses: policy loss (cross-entropy between network policy and MCTS "
                "search policy) and value loss (MSE between predicted value and game result). This is knowledge "
                "distillation - the slow but accurate MCTS search teaches the fast neural network to approximate its "
                "improved policy and value estimates.",
            },
        },
        # Replay Buffer
        {
            "inputs": {
                "question": "What is a replay buffer and how is it used in AlphaZero training?",
                "contexts": [
                    "A replay buffer is a data structure that stores historical self-play game data for training "
                    "the neural network.",
                    "AlphaZero maintains a large replay buffer containing game positions, MCTS policies, and outcomes "
                    "from recent self-play games.",
                    "Training batches are sampled uniformly from the replay buffer, which helps stabilize learning "
                    "by breaking temporal correlations in sequential game data.",
                    "The buffer typically stores the most recent N games (e.g., 500,000 positions), with old data "
                    "being discarded as new self-play games are added.",
                ],
            },
            "outputs": {
                "ground_truth": "A replay buffer stores historical self-play data for training AlphaZero's neural "
                "network. It contains game positions, MCTS search policies, and outcomes from recent games. Training "
                "batches are sampled uniformly from this buffer, which stabilizes learning by: (1) breaking temporal "
                "correlations in sequential game data, and (2) allowing the network to train on diverse positions "
                "rather than just the latest games. Old data is gradually replaced with new self-play data, keeping "
                "the buffer representative of the current playing strength.",
            },
        },
        # Multi-Armed Bandit
        {
            "inputs": {
                "question": "How does MCTS relate to the multi-armed bandit problem?",
                "contexts": [
                    "The multi-armed bandit problem involves choosing between multiple options (arms) with unknown "
                    "reward distributions to maximize total reward over time.",
                    "MCTS can be viewed as applying multi-armed bandit algorithms (like UCB1) recursively at each "
                    "node in the tree.",
                    "At each node during selection, choosing which child to visit is a bandit problem: balancing "
                    "exploration of uncertain options with exploitation of known-good choices.",
                    "UCB1 and similar bandit algorithms provide theoretical guarantees about balancing exploration "
                    "and exploitation, which transfer to MCTS through this connection.",
                ],
            },
            "outputs": {
                "ground_truth": "MCTS is fundamentally based on the multi-armed bandit problem framework. At each "
                "tree node during selection, choosing which child to visit is a bandit problem: each child is an 'arm' "
                "with an uncertain value distribution. MCTS applies bandit algorithms like UCB1 recursively at every "
                "node to balance exploring uncertain children with exploiting promising ones. This connection provides "
                "theoretical guarantees - the regret bounds from bandit theory translate into convergence guarantees "
                "for MCTS.",
            },
        },
        # Regret Minimization
        {
            "inputs": {
                "question": "What is regret minimization in the context of MCTS and UCB1?",
                "contexts": [
                    "Regret is the difference between the reward obtained and the reward that could have been "
                    "obtained by always choosing the optimal action.",
                    "UCB1 is proven to achieve logarithmic regret - regret grows as O(ln n) where n is the number "
                    "of iterations, which is near-optimal for multi-armed bandit problems.",
                    "In MCTS, minimizing regret means that as more simulations are performed, the algorithm "
                    "increasingly focuses on the truly best moves.",
                    "The logarithmic regret bound guarantees that the algorithm doesn't waste too many iterations "
                    "on suboptimal moves, leading to efficient convergence.",
                ],
            },
            "outputs": {
                "ground_truth": "Regret minimization in MCTS refers to minimizing the difference between rewards "
                "obtained and optimal rewards. UCB1 achieves logarithmic regret O(ln n), meaning wasted iterations "
                "on suboptimal moves grow very slowly. This is near-optimal for bandit problems and translates to "
                "efficient convergence in MCTS - the algorithm quickly identifies and focuses on the best moves while "
                "not wasting too many iterations on inferior options. The logarithmic bound ensures that even as the "
                "search runs longer, the proportion of iterations wasted decreases.",
            },
        },
    ]

    print("Creating rag-eval-dataset...")
    dataset_id = create_test_dataset(
        dataset_name="rag-eval-dataset",
        examples=examples,
        description="RAG evaluation dataset with MCTS, AlphaZero, and game tree search Q&A examples "
        "including ground truth answers and relevant context snippets for testing retrieval-augmented generation",
    )
    print(f"[OK] Created dataset: rag-eval-dataset (ID: {dataset_id})")
    print(f"[INFO] Dataset contains {len(examples)} Q&A examples covering:")
    print("  - Basic MCTS concepts and algorithms")
    print("  - AlphaZero architecture and training")
    print("  - UCB1, PUCT, and exploration/exploitation")
    print("  - Game tree search fundamentals")
    print("  - Advanced MCTS techniques (RAVE, progressive widening, etc.)")
    return dataset_id


def main():
    """Create RAG evaluation dataset in LangSmith."""
    # Check if LangSmith is configured
    if not os.getenv("LANGSMITH_API_KEY"):
        print("[ERROR] LANGSMITH_API_KEY environment variable not set")
        print("        Set it with: export LANGSMITH_API_KEY=your_key_here")
        sys.exit(1)

    print("=" * 70)
    print("Creating LangSmith RAG Evaluation Dataset")
    print("=" * 70)
    print()

    try:
        dataset_id = create_rag_eval_dataset()
        print()

        print("=" * 70)
        print("[SUCCESS] RAG evaluation dataset created successfully!")
        print("=" * 70)
        print()
        print(f"Dataset ID: {dataset_id}")
        print()
        print("Next steps:")
        print("  1. View dataset in LangSmith UI")
        print("  2. Use this dataset to evaluate RAG performance on MCTS topics")
        print("  3. Run experiments to compare retrieval quality and answer accuracy")
        print()

    except Exception as e:
        print(f"[ERROR] Error creating dataset: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
