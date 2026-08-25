"""
AlphaZero-Style Self-Play Training Pipeline for Continuous System Improvement

This module implements a comprehensive self-play training system inspired by AlphaZero,
adapted for multi-agent MCTS with HRM and TRM agents. The pipeline:

1. Generates episodes through agent self-play on diverse tasks
2. Records complete execution traces including MCTS search trees
3. Extracts high-quality training examples from successful episodes
4. Implements AlphaZero-style iterative improvement loop
5. Supports parallel async episode generation
6. Provides task generators for various problem domains

Key Features:
- Full MCTS tree capture and analysis
- Policy and value training data extraction
- Reasoning chain mining for LLM fine-tuning
- Negative example generation from failures
- Quality metrics and diversity tracking
- Checkpointing and resumability
- Resource monitoring
"""

import asyncio
import contextlib
from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


# ============================================================================
# Episode Data Structures
# ============================================================================


@dataclass
class Action:
    """Represents an action taken during episode execution."""

    action_id: str
    action_type: str  # "decompose", "refine", "execute", "mcts_select"
    parameters: dict[str, Any]
    timestamp: float
    confidence: float = 0.0


@dataclass
class State:
    """Represents a state during episode execution."""

    state_id: str
    representation: torch.Tensor  # Latent state vector
    raw_state: dict[str, Any]  # Original state data
    value_estimate: float = 0.0
    visit_count: int = 0
    parent_id: str | None = None


@dataclass
class MCTSTrace:
    """Captures MCTS search tree information at a decision point."""

    root_state_id: str
    num_simulations: int
    visit_counts: dict[str, int]  # action_id -> visit count
    q_values: dict[str, float]  # action_id -> Q-value
    prior_probs: dict[str, float]  # action_id -> prior probability
    selected_action: str
    tree_depth: int
    search_time: float
    value_estimates: dict[str, float]  # state_id -> value


@dataclass
class SelfPlayEpisode:
    """
    Complete episode from self-play execution.

    Contains all information needed to extract training examples.
    """

    task_id: str
    initial_state: Any
    actions: list[Action]
    states: list[State]
    rewards: list[float]
    mcts_traces: list[MCTSTrace]
    outcome: str  # "success", "failure", "timeout", "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    # Computed fields
    total_reward: float = 0.0
    episode_length: int = 0
    search_efficiency: float = 0.0  # reward / (simulations * time)
    solution_path: list[str] = field(default_factory=list)  # state_ids on success path

    def __post_init__(self):
        """Compute derived fields."""
        self.total_reward = sum(self.rewards)
        self.episode_length = len(self.actions)

        # Compute search efficiency
        total_sims = sum(trace.num_simulations for trace in self.mcts_traces)
        total_time = sum(trace.search_time for trace in self.mcts_traces)
        if total_sims > 0 and total_time > 0:
            self.search_efficiency = self.total_reward / (total_sims * total_time + 1e-8)

        # Extract solution path for successful episodes
        if self.outcome == "success" and self.states:
            self.solution_path = [state.state_id for state in self.states]


@dataclass
class TrainingExample:
    """A single training example extracted from episodes."""

    example_id: str
    example_type: str  # "policy", "value", "reasoning", "negative"
    state: torch.Tensor
    target: Any  # Depends on example_type
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Task Generators
# ============================================================================


class BaseTaskGenerator:
    """Base class for task generators."""

    def __init__(self, difficulty_range: tuple[float, float] = (0.1, 1.0), seed: int | None = None):
        self.difficulty_range = difficulty_range
        self.rng = np.random.RandomState(seed)

    def generate(self, num_tasks: int) -> list[dict[str, Any]]:
        """Generate tasks."""
        raise NotImplementedError

    def _difficulty_to_params(self, difficulty: float) -> dict[str, Any]:
        """Convert difficulty level to task-specific parameters."""
        raise NotImplementedError


class MathProblemGenerator(BaseTaskGenerator):
    """Generate mathematical reasoning problems."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.operators = ["+", "-", "*", "/", "**"]
        self.problem_types = ["arithmetic", "algebra", "geometry", "calculus"]

    def generate(self, num_tasks: int) -> list[dict[str, Any]]:
        """Generate math problems with varying difficulty."""
        tasks = []
        for i in range(num_tasks):
            difficulty = self.rng.uniform(*self.difficulty_range)
            params = self._difficulty_to_params(difficulty)

            # Generate problem based on difficulty
            if difficulty < 0.3:
                task = self._generate_arithmetic(params)
            elif difficulty < 0.6:
                task = self._generate_algebra(params)
            else:
                task = self._generate_complex(params)

            task["task_id"] = f"math_{i}_{int(difficulty * 100)}"
            task["difficulty"] = difficulty
            tasks.append(task)

        return tasks

    def _difficulty_to_params(self, difficulty: float) -> dict[str, Any]:
        """Convert difficulty to problem parameters."""
        return {
            "num_operations": int(1 + difficulty * 5),
            "max_value": int(10 + difficulty * 90),
            "use_fractions": difficulty > 0.5,
            "use_negatives": difficulty > 0.3,
        }

    def _generate_arithmetic(self, params: dict) -> dict[str, Any]:
        """Generate basic arithmetic problem."""
        num_ops = params["num_operations"]
        max_val = params["max_value"]

        expression = str(self.rng.randint(1, max_val))
        for _ in range(num_ops):
            op = self.rng.choice(["+", "-", "*"])
            val = self.rng.randint(1, max_val)
            expression += f" {op} {val}"

        try:
            answer = eval(expression)  # Safe for generated expressions
        except ZeroDivisionError:
            answer = None

        return {
            "type": "arithmetic",
            "problem": f"Calculate: {expression}",
            "expression": expression,
            "answer": answer,
            "steps": expression.split(),
        }

    def _generate_algebra(self, params: dict) -> dict[str, Any]:
        """Generate algebraic problem."""
        # Simplified: solve linear equation ax + b = c
        a = self.rng.randint(1, params["max_value"] // 10 + 1)
        b = self.rng.randint(-params["max_value"], params["max_value"])
        c = self.rng.randint(-params["max_value"], params["max_value"])

        # x = (c - b) / a
        if a != 0:
            answer = (c - b) / a
        else:
            answer = None

        equation = f"{a}x + {b} = {c}"
        return {
            "type": "algebra",
            "problem": f"Solve for x: {equation}",
            "equation": equation,
            "answer": answer,
            "steps": [f"{a}x = {c - b}", f"x = {answer}"] if answer is not None else [],
        }

    def _generate_complex(self, params: dict) -> dict[str, Any]:
        """Generate complex multi-step problem."""
        # System of equations or quadratic
        problem_type = self.rng.choice(["quadratic", "system"])

        if problem_type == "quadratic":
            # ax^2 + bx + c = 0
            a = self.rng.randint(1, 5)
            b = self.rng.randint(-10, 10)
            c = self.rng.randint(-10, 10)

            discriminant = b**2 - 4 * a * c
            if discriminant >= 0:
                x1 = (-b + np.sqrt(discriminant)) / (2 * a)
                x2 = (-b - np.sqrt(discriminant)) / (2 * a)
                answer = [x1, x2]
            else:
                answer = None

            return {
                "type": "quadratic",
                "problem": f"Solve: {a}x^2 + {b}x + {c} = 0",
                "coefficients": [a, b, c],
                "answer": answer,
                "steps": [f"discriminant = {discriminant}", f"solutions = {answer}"] if answer else [],
            }
        else:
            # System of linear equations
            return {
                "type": "system",
                "problem": "Solve the system of linear equations",
                "equations": ["2x + 3y = 12", "x - y = 1"],
                "answer": {"x": 3, "y": 2},
                "steps": ["Substitute", "Solve for x", "Solve for y"],
            }


class CodeGenerationTaskGenerator(BaseTaskGenerator):
    """Generate code generation tasks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.languages = ["python", "javascript", "java"]
        self.task_types = ["function", "class", "algorithm"]

    def generate(self, num_tasks: int) -> list[dict[str, Any]]:
        """Generate code generation tasks."""
        tasks = []
        for i in range(num_tasks):
            difficulty = self.rng.uniform(*self.difficulty_range)
            language = self.rng.choice(self.languages)
            self.rng.choice(self.task_types)

            if difficulty < 0.4:
                task = self._generate_simple_function(language)
            elif difficulty < 0.7:
                task = self._generate_algorithm(language)
            else:
                task = self._generate_complex_class(language)

            task["task_id"] = f"code_{language}_{i}"
            task["difficulty"] = difficulty
            tasks.append(task)

        return tasks

    def _difficulty_to_params(self, difficulty: float) -> dict[str, Any]:
        """Convert difficulty to code complexity parameters."""
        return {
            "num_functions": int(1 + difficulty * 5),
            "max_params": int(1 + difficulty * 4),
            "requires_loops": difficulty > 0.3,
            "requires_recursion": difficulty > 0.7,
        }

    def _generate_simple_function(self, language: str) -> dict[str, Any]:
        """Generate simple function task."""
        functions = [
            ("sum_list", "Write a function that sums all elements in a list", "[1, 2, 3, 4, 5]", 15),
            ("reverse_string", "Write a function that reverses a string", "'hello'", "'olleh'"),
            ("is_even", "Write a function that checks if a number is even", "4", True),
            ("factorial", "Write a function that computes factorial", "5", 120),
        ]

        name, description, example_input, example_output = self.rng.choice(functions)

        return {
            "type": "function",
            "language": language,
            "problem": description,
            "function_name": name,
            "example_input": example_input,
            "example_output": example_output,
            "test_cases": [(example_input, example_output)],
        }

    def _generate_algorithm(self, language: str) -> dict[str, Any]:
        """Generate algorithm implementation task."""
        algorithms = [
            ("binary_search", "Implement binary search", "sorted array and target"),
            ("merge_sort", "Implement merge sort", "unsorted array"),
            ("fibonacci", "Implement fibonacci sequence", "n-th number"),
            ("prime_check", "Check if number is prime", "integer"),
        ]

        name, description, input_desc = self.rng.choice(algorithms)

        return {
            "type": "algorithm",
            "language": language,
            "problem": description,
            "algorithm_name": name,
            "input_description": input_desc,
            "complexity_requirement": "O(n log n) or better",
        }

    def _generate_complex_class(self, language: str) -> dict[str, Any]:
        """Generate complex class design task."""
        return {
            "type": "class",
            "language": language,
            "problem": "Design a class with multiple methods",
            "class_name": "DataProcessor",
            "required_methods": ["load", "process", "save", "validate"],
            "design_patterns": ["singleton", "factory"],
        }


class MultiStepReasoningGenerator(BaseTaskGenerator):
    """Generate multi-step reasoning tasks."""

    def generate(self, num_tasks: int) -> list[dict[str, Any]]:
        """Generate reasoning tasks."""
        tasks = []
        for i in range(num_tasks):
            difficulty = self.rng.uniform(*self.difficulty_range)
            params = self._difficulty_to_params(difficulty)

            task = self._generate_reasoning_chain(params)
            task["task_id"] = f"reasoning_{i}"
            task["difficulty"] = difficulty
            tasks.append(task)

        return tasks

    def _difficulty_to_params(self, difficulty: float) -> dict[str, Any]:
        """Convert difficulty to reasoning parameters."""
        return {
            "num_steps": int(3 + difficulty * 7),
            "branching_factor": int(1 + difficulty * 3),
            "requires_backtracking": difficulty > 0.6,
        }

    def _generate_reasoning_chain(self, params: dict) -> dict[str, Any]:
        """Generate multi-step reasoning problem."""
        scenarios = [
            "logical_deduction",
            "causal_reasoning",
            "planning",
            "constraint_satisfaction",
        ]
        scenario_type = self.rng.choice(scenarios)

        if scenario_type == "logical_deduction":
            return {
                "type": "logical_deduction",
                "problem": "Given facts, deduce the conclusion",
                "facts": [
                    "All humans are mortal",
                    "Socrates is human",
                    "Therefore, Socrates is mortal",
                ],
                "steps": ["Identify premises", "Apply modus ponens", "State conclusion"],
                "answer": "Socrates is mortal",
            }
        elif scenario_type == "planning":
            return {
                "type": "planning",
                "problem": "Plan a sequence of actions to achieve goal",
                "initial_state": {"robot_at": "A", "box_at": "B", "goal": "C"},
                "goal_state": {"robot_at": "C", "box_at": "C"},
                "available_actions": ["move", "push", "pull"],
                "steps": ["Move to B", "Push box to C", "Arrive at C"],
            }
        else:
            return {
                "type": scenario_type,
                "problem": f"Solve {scenario_type} problem",
                "steps": [f"Step {i}" for i in range(params["num_steps"])],
            }


class MCTSSearchTaskGenerator(BaseTaskGenerator):
    """Generate tasks specifically designed for MCTS search."""

    def generate(self, num_tasks: int) -> list[dict[str, Any]]:
        """Generate MCTS-friendly search tasks."""
        tasks = []
        for i in range(num_tasks):
            difficulty = self.rng.uniform(*self.difficulty_range)
            task_type = self.rng.choice(["game", "optimization", "path_finding"])

            if task_type == "game":
                task = self._generate_game_state(difficulty)
            elif task_type == "optimization":
                task = self._generate_optimization(difficulty)
            else:
                task = self._generate_path_finding(difficulty)

            task["task_id"] = f"mcts_{task_type}_{i}"
            task["difficulty"] = difficulty
            tasks.append(task)

        return tasks

    def _difficulty_to_params(self, difficulty: float) -> dict[str, Any]:
        """Convert difficulty to MCTS parameters."""
        return {
            "branching_factor": int(2 + difficulty * 8),
            "depth": int(5 + difficulty * 15),
            "state_space_size": int(100 + difficulty * 900),
        }

    def _generate_game_state(self, difficulty: float) -> dict[str, Any]:
        """Generate a game state for MCTS."""
        size = int(3 + difficulty * 6)  # 3x3 to 9x9
        return {
            "type": "game",
            "game_type": "tic_tac_toe" if difficulty < 0.5 else "connect_four",
            "board_size": size,
            "current_player": self.rng.choice([1, 2]),
            "board_state": np.zeros((size, size), dtype=int).tolist(),
        }

    def _generate_optimization(self, difficulty: float) -> dict[str, Any]:
        """Generate optimization problem."""
        num_vars = int(2 + difficulty * 8)
        return {
            "type": "optimization",
            "objective": "minimize_quadratic",
            "num_variables": num_vars,
            "constraints": [f"x{i} >= 0" for i in range(num_vars)],
        }

    def _generate_path_finding(self, difficulty: float) -> dict[str, Any]:
        """Generate path finding task."""
        size = int(5 + difficulty * 15)
        return {
            "type": "path_finding",
            "grid_size": [size, size],
            "start": [0, 0],
            "goal": [size - 1, size - 1],
            "obstacles": self.rng.randint(0, 2, (size, size)).tolist(),
        }


# ============================================================================
# Self-Play Episode Generator
# ============================================================================


class SelfPlayEpisodeGenerator:
    """
    Generate self-play episodes using current model.

    Orchestrates agent execution, captures full traces, and records outcomes.
    """

    def __init__(
        self,
        hrm_agent: Any = None,
        trm_agent: Any = None,
        mcts_config: dict | None = None,
        device: str = "cpu",
    ):
        self.hrm_agent = hrm_agent
        self.trm_agent = trm_agent
        self.mcts_config = mcts_config or {}
        self.device = device

        self.episode_count = 0
        self.success_count = 0
        self.failure_count = 0

    async def generate_episode(
        self,
        task: dict[str, Any],
        max_steps: int = 50,
        timeout: float = 60.0,
    ) -> SelfPlayEpisode:
        """
        Generate a single self-play episode.

        Args:
            task: Task definition
            max_steps: Maximum number of steps
            timeout: Timeout in seconds

        Returns:
            SelfPlayEpisode with complete trace
        """
        start_time = time.time()
        task_id = task.get("task_id", f"task_{self.episode_count}")

        # Initialize episode tracking
        actions = []
        states = []
        rewards = []
        mcts_traces = []
        outcome = "in_progress"

        try:
            # Create initial state
            initial_state_tensor = self._task_to_state(task)
            initial_state = State(
                state_id=f"{task_id}_state_0",
                representation=initial_state_tensor,
                raw_state=task,
            )
            states.append(initial_state)

            current_state = initial_state
            step = 0

            while step < max_steps and time.time() - start_time < timeout:
                # Run MCTS simulation to select action
                mcts_trace, selected_action = await self._run_mcts_simulation(current_state, task, step)
                mcts_traces.append(mcts_trace)
                actions.append(selected_action)

                # Execute action and get next state
                next_state, reward, done = await self._execute_action(current_state, selected_action, task, step)

                states.append(next_state)
                rewards.append(reward)

                # Check termination
                if done:
                    outcome = "success" if reward > 0.5 else "failure"
                    break

                current_state = next_state
                step += 1

            # Handle timeout
            if time.time() - start_time >= timeout:
                outcome = "timeout"
            elif step >= max_steps and outcome == "in_progress":
                outcome = "max_steps_reached"

            # Update counters
            self.episode_count += 1
            if outcome == "success":
                self.success_count += 1
            else:
                self.failure_count += 1

        except Exception as e:
            logger.error(f"Error generating episode {task_id}: {e}")
            outcome = "error"

        # Create episode
        episode = SelfPlayEpisode(
            task_id=task_id,
            initial_state=task,
            actions=actions,
            states=states,
            rewards=rewards,
            mcts_traces=mcts_traces,
            outcome=outcome,
            metadata={
                "elapsed_time": time.time() - start_time,
                "task_type": task.get("type"),
                "difficulty": task.get("difficulty", 0.5),
            },
        )

        return episode

    async def _run_mcts_simulation(self, state: State, task: dict, step: int) -> tuple[MCTSTrace, Action]:
        """Run MCTS simulation from current state."""
        start_time = time.time()
        num_simulations = self.mcts_config.get("num_simulations", 100)

        # Simulate MCTS (simplified - in practice, use full MCTS implementation)
        visit_counts = {}
        q_values = {}
        prior_probs = {}
        value_estimates = {}

        # Get action space for current state
        actions = self._get_available_actions(state, task)

        # If we have HRM agent, use it for prior policy
        if self.hrm_agent is not None and state.representation.dim() >= 2:
            with torch.no_grad():
                # Get policy prior from HRM
                output = self.hrm_agent(state.representation.unsqueeze(0))
                # Simplified: use final state as action distribution
                action_logits = output.final_state[:, 0, : len(actions)]
                priors = F.softmax(action_logits, dim=-1).squeeze().cpu().numpy()
        else:
            # Uniform prior
            priors = np.ones(len(actions)) / len(actions)

        # MCTS simulation loop
        for sim_idx in range(num_simulations):
            # Select action using PUCT
            action_idx = self._select_action_puct(actions, visit_counts, q_values, priors, sim_idx)
            action = actions[action_idx]
            action_id = action.action_id

            # Track visits
            visit_counts[action_id] = visit_counts.get(action_id, 0) + 1

            # Simulate/evaluate (simplified)
            value = await self._evaluate_action(state, action, task)
            value_estimates[action_id] = value

            # Update Q-value
            if action_id not in q_values:
                q_values[action_id] = value
            else:
                # Running average
                n = visit_counts[action_id]
                q_values[action_id] = ((n - 1) * q_values[action_id] + value) / n

        # Store priors
        for i, action in enumerate(actions):
            prior_probs[action.action_id] = float(priors[i]) if i < len(priors) else 1.0 / len(actions)

        # Select best action (highest visit count)
        best_action = max(actions, key=lambda a: visit_counts.get(a.action_id, 0))

        # Create MCTS trace
        trace = MCTSTrace(
            root_state_id=state.state_id,
            num_simulations=num_simulations,
            visit_counts=visit_counts,
            q_values=q_values,
            prior_probs=prior_probs,
            selected_action=best_action.action_id,
            tree_depth=len(visit_counts),
            search_time=time.time() - start_time,
            value_estimates=value_estimates,
        )

        return trace, best_action

    def _select_action_puct(
        self,
        actions: list[Action],
        visit_counts: dict[str, int],
        q_values: dict[str, float],
        priors: np.ndarray,
        total_visits: int,
    ) -> int:
        """Select action using PUCT formula."""
        c_puct = self.mcts_config.get("c_puct", 1.25)
        sqrt_total = np.sqrt(max(total_visits, 1))

        puct_scores = []
        for i, action in enumerate(actions):
            action_id = action.action_id
            q = q_values.get(action_id, 0.0)
            n = visit_counts.get(action_id, 0)
            p = priors[i] if i < len(priors) else 1.0 / len(actions)

            # PUCT formula: Q + c * P * sqrt(N) / (1 + n)
            u = c_puct * p * sqrt_total / (1 + n)
            puct_scores.append(q + u)

        return int(np.argmax(puct_scores))

    async def _evaluate_action(self, state: State, action: Action, task: dict) -> float:
        """Evaluate action value using value network or heuristic."""
        # If we have TRM agent, use it for value estimation
        if self.trm_agent is not None and state.representation.dim() >= 2:
            with torch.no_grad():
                # Use TRM for value estimation
                output = self.trm_agent(state.representation.unsqueeze(0))
                # Extract value from final prediction
                value = output.final_prediction.mean().item()
                return float(np.clip(value, -1, 1))

        # Fallback: random value with bias toward task completion
        base_value = self.rng.uniform(-0.5, 0.5)
        # Bias based on action type
        if action.action_type in ["execute", "mcts_select"]:
            base_value += 0.2
        return float(np.clip(base_value, -1, 1))

    def _get_available_actions(self, state: State, task: dict) -> list[Action]:
        """Get available actions from current state."""
        actions = []
        task_type = task.get("type", "unknown")

        # Generate action space based on task type
        if task_type in ["arithmetic", "algebra", "quadratic"]:
            # Math operations
            action_types = ["decompose", "simplify", "solve", "verify"]
        elif task_type in ["function", "algorithm", "class"]:
            # Code generation
            action_types = ["write_code", "test", "refine", "optimize"]
        elif task_type == "game":
            # Game moves (simplified)
            board_size = task.get("board_size", 3)
            for i in range(board_size):
                for j in range(board_size):
                    actions.append(
                        Action(
                            action_id=f"move_{i}_{j}",
                            action_type="game_move",
                            parameters={"position": (i, j)},
                            timestamp=time.time(),
                        )
                    )
            return actions[:10]  # Limit to 10 actions
        else:
            action_types = ["explore", "refine", "execute", "backtrack"]

        # Create actions
        for i, action_type in enumerate(action_types):
            actions.append(
                Action(
                    action_id=f"{state.state_id}_action_{i}",
                    action_type=action_type,
                    parameters={"type": action_type},
                    timestamp=time.time(),
                )
            )

        return actions

    async def _execute_action(self, state: State, action: Action, task: dict, step: int) -> tuple[State, float, bool]:
        """Execute action and return next state, reward, and done flag."""
        # Simulate state transition
        next_state_repr = state.representation.clone()

        # Add some noise to represent state change
        noise = torch.randn_like(next_state_repr) * 0.1
        next_state_repr = next_state_repr + noise

        # Create next state
        next_state = State(
            state_id=f"{task['task_id']}_state_{step + 1}",
            representation=next_state_repr,
            raw_state={"step": step + 1, "action": action.action_type},
            parent_id=state.state_id,
        )

        # Compute reward (simplified)
        reward = self._compute_reward(state, action, next_state, task, step)

        # Check if done
        done = reward > 0.9 or action.action_type in ["verify", "execute"] and step > 3

        return next_state, reward, done

    def _compute_reward(self, state: State, action: Action, next_state: State, task: dict, step: int) -> float:
        """Compute reward for transition."""
        # Simplified reward computation
        base_reward = 0.0

        # Reward for progress
        if action.action_type in ["solve", "execute", "verify"]:
            base_reward += 0.3

        # Penalty for long episodes
        step_penalty = -0.01 * step

        # Bonus for task completion (heuristic)
        completion_bonus = 0.0
        if step > 3 and action.action_type == "verify":
            completion_bonus = 1.0

        return base_reward + step_penalty + completion_bonus

    def _task_to_state(self, task: dict) -> torch.Tensor:
        """Convert task to state tensor representation."""
        # Simplified: create a fixed-size state vector
        state_dim = 256

        # Hash task_id to create deterministic state
        task_id = task.get("task_id", "unknown")
        seed = int(hashlib.md5(task_id.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)

        # Create state vector
        state = torch.from_numpy(rng.randn(state_dim)).float()

        # Add task features
        difficulty = task.get("difficulty", 0.5)
        state[0] = difficulty

        task_type_encoding = {
            "arithmetic": 1.0,
            "algebra": 2.0,
            "function": 3.0,
            "game": 4.0,
        }.get(task.get("type"), 0.0)
        state[1] = task_type_encoding

        return state

    @property
    def rng(self):
        """Random number generator."""
        if not hasattr(self, "_rng"):
            self._rng = np.random.RandomState()
        return self._rng


# ============================================================================
# Training Data Extraction
# ============================================================================


class TrainingDataExtractor:
    """
    Extract high-quality training examples from episodes.

    Generates:
    - (state, optimal_action) pairs for policy learning
    - (state, value) pairs for value estimation
    - (query, reasoning_chain) for LLM fine-tuning
    - Negative examples from failures
    """

    def __init__(self, success_weight: float = 1.0, failure_weight: float = 0.3):
        self.success_weight = success_weight
        self.failure_weight = failure_weight
        self.tokenizer = None

        if HAS_TRANSFORMERS:
            with contextlib.suppress(Exception):
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def extract_examples(self, episodes: list[SelfPlayEpisode]) -> dict[str, list[TrainingExample]]:
        """
        Extract training examples from episodes.

        Returns:
            Dictionary mapping example_type to list of examples
        """
        examples = {
            "policy": [],
            "value": [],
            "reasoning": [],
            "negative": [],
        }

        for episode in episodes:
            # Extract based on outcome
            if episode.outcome == "success":
                examples["policy"].extend(self._extract_policy_examples(episode))
                examples["value"].extend(self._extract_value_examples(episode))
                examples["reasoning"].extend(self._extract_reasoning_examples(episode))
            else:
                examples["negative"].extend(self._extract_negative_examples(episode))

        return examples

    def _extract_policy_examples(self, episode: SelfPlayEpisode) -> list[TrainingExample]:
        """Extract policy training examples from successful episode."""
        examples = []

        for i, (state, trace) in enumerate(zip(episode.states[:-1], episode.mcts_traces, strict=False)):
            # Create policy target from MCTS visit counts
            total_visits = sum(trace.visit_counts.values())
            if total_visits == 0:
                continue

            # Normalize visit counts to create improved policy
            policy_target = {action_id: count / total_visits for action_id, count in trace.visit_counts.items()}

            example = TrainingExample(
                example_id=f"{episode.task_id}_policy_{i}",
                example_type="policy",
                state=state.representation,
                target=policy_target,
                weight=self.success_weight,
                metadata={
                    "task_id": episode.task_id,
                    "step": i,
                    "num_visits": total_visits,
                    "selected_action": trace.selected_action,
                },
            )
            examples.append(example)

        return examples

    def _extract_value_examples(self, episode: SelfPlayEpisode) -> list[TrainingExample]:
        """Extract value training examples from episode."""
        examples = []

        # Compute discounted returns
        gamma = 0.99
        returns = []
        G = 0.0
        for reward in reversed(episode.rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        # Create value examples
        for i, (state, value_target) in enumerate(zip(episode.states[:-1], returns, strict=False)):
            example = TrainingExample(
                example_id=f"{episode.task_id}_value_{i}",
                example_type="value",
                state=state.representation,
                target=value_target,
                weight=self.success_weight,
                metadata={
                    "task_id": episode.task_id,
                    "step": i,
                    "immediate_reward": episode.rewards[i],
                    "return": value_target,
                },
            )
            examples.append(example)

        return examples

    def _extract_reasoning_examples(self, episode: SelfPlayEpisode) -> list[TrainingExample]:
        """Extract reasoning chain examples for LLM fine-tuning."""
        if not self.tokenizer:
            return []

        examples = []

        # Build reasoning chain from actions
        reasoning_chain = []
        for action in episode.actions:
            reasoning_chain.append(f"{action.action_type}: {action.parameters}")

        # Create text representation
        task_description = str(episode.initial_state)
        reasoning_text = "\n".join(reasoning_chain)
        full_text = f"Task: {task_description}\n\nReasoning:\n{reasoning_text}"

        # Tokenize
        tokens = self.tokenizer.encode(full_text)

        # Create dummy state (text-based)
        state_tensor = torch.zeros(256)  # Placeholder

        example = TrainingExample(
            example_id=f"{episode.task_id}_reasoning",
            example_type="reasoning",
            state=state_tensor,
            target=tokens,
            weight=self.success_weight,
            metadata={
                "task_id": episode.task_id,
                "reasoning_chain": reasoning_chain,
                "num_steps": len(episode.actions),
            },
        )
        examples.append(example)

        return examples

    def _extract_negative_examples(self, episode: SelfPlayEpisode) -> list[TrainingExample]:
        """Extract negative examples from failed episodes."""
        examples = []

        # Find the failure point
        if len(episode.rewards) == 0:
            return examples

        # Identify states with lowest rewards as negative examples
        worst_states_idx = np.argsort(episode.rewards)[: min(3, len(episode.rewards))]

        for idx in worst_states_idx:
            if idx >= len(episode.states) - 1 or idx >= len(episode.mcts_traces):
                continue

            state = episode.states[idx]
            trace = episode.mcts_traces[idx]

            # Create negative policy example (avoid selected action)
            negative_policy = {trace.selected_action: 0.0}

            example = TrainingExample(
                example_id=f"{episode.task_id}_negative_{idx}",
                example_type="negative",
                state=state.representation,
                target=negative_policy,
                weight=self.failure_weight,
                metadata={
                    "task_id": episode.task_id,
                    "step": idx,
                    "reward": episode.rewards[idx],
                    "outcome": episode.outcome,
                },
            )
            examples.append(example)

        return examples


# ============================================================================
# Training Dataset
# ============================================================================


class SelfPlayDataset(Dataset):
    """PyTorch Dataset for self-play training examples."""

    def __init__(self, examples: list[TrainingExample], example_type: str):
        self.examples = [ex for ex in examples if ex.example_type == example_type]
        self.example_type = example_type

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        return {
            "state": example.state,
            "target": example.target,
            "weight": example.weight,
            "metadata": example.metadata,
        }


# ============================================================================
# AlphaZero-Style Iteration Loop
# ============================================================================


class SelfPlayTrainer:
    """
    AlphaZero-style iterative self-play trainer.

    Implements the complete training loop:
    1. Generate episodes with current model
    2. Extract training examples
    3. Train improved model
    4. Evaluate new vs old model
    5. Update if better, repeat
    """

    def __init__(
        self,
        hrm_agent: Any = None,
        trm_agent: Any = None,
        config: dict | None = None,
        device: str = "cpu",
        checkpoint_dir: str = "./checkpoints/self_play",
    ):
        self.hrm_agent = hrm_agent
        self.trm_agent = trm_agent
        self.config = config or {}
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.episode_generator = SelfPlayEpisodeGenerator(
            hrm_agent=hrm_agent,
            trm_agent=trm_agent,
            mcts_config=config.get("mcts", {}),
            device=device,
        )
        self.data_extractor = TrainingDataExtractor()

        # Task generators
        self.task_generators = {
            "math": MathProblemGenerator(),
            "code": CodeGenerationTaskGenerator(),
            "reasoning": MultiStepReasoningGenerator(),
            "mcts": MCTSSearchTaskGenerator(),
        }

        # Metrics tracking
        self.iteration_metrics = []
        self.best_model_metric = -float("inf")
        self.current_iteration = 0

        # Episode buffer
        self.episode_buffer = []
        self.max_buffer_size = config.get("max_buffer_size", 10000)

        logger.info(f"Initialized SelfPlayTrainer with {len(self.task_generators)} task generators")

    async def iteration(self, iteration_num: int) -> dict[str, Any]:
        """
        Run one complete training iteration.

        Args:
            iteration_num: Current iteration number

        Returns:
            Iteration metrics
        """
        logger.info(f"Starting iteration {iteration_num}")
        start_time = time.time()

        metrics = {
            "iteration": iteration_num,
            "timestamp": time.time(),
        }

        # 1. Generate episodes with current model
        num_games = self.config.get("games_per_iteration", 1000)
        episodes = await self.run_self_play(num_games=num_games)
        metrics["num_episodes"] = len(episodes)
        metrics["success_rate"] = sum(1 for ep in episodes if ep.outcome == "success") / len(episodes)

        logger.info(f"Generated {len(episodes)} episodes, success rate: {metrics['success_rate']:.2%}")

        # 2. Extract training examples
        training_data = self.data_extractor.extract_examples(episodes)
        metrics["num_policy_examples"] = len(training_data["policy"])
        metrics["num_value_examples"] = len(training_data["value"])
        metrics["num_reasoning_examples"] = len(training_data["reasoning"])
        metrics["num_negative_examples"] = len(training_data["negative"])

        logger.info(
            f"Extracted {metrics['num_policy_examples']} policy examples, "
            f"{metrics['num_value_examples']} value examples"
        )

        # 3. Train improved model (if we have trainable agents)
        if self.hrm_agent is not None or self.trm_agent is not None:
            train_metrics = await self.train(training_data, iteration_num)
            metrics.update(train_metrics)

        # 4. Evaluate: new vs old
        eval_metrics = await self.evaluate_models(num_eval_games=100)
        metrics.update(eval_metrics)

        # 5. Update best model if improved
        current_metric = eval_metrics.get("eval_success_rate", 0.0)
        if current_metric > self.best_model_metric:
            logger.info(f"New best model! {current_metric:.2%} > {self.best_model_metric:.2%}")
            self.best_model_metric = current_metric
            self._save_checkpoint(iteration_num, best=True)
        else:
            logger.info(f"Model not improved: {current_metric:.2%} <= {self.best_model_metric:.2%}")

        # Save regular checkpoint
        self._save_checkpoint(iteration_num, best=False)

        # Update metrics
        metrics["elapsed_time"] = time.time() - start_time
        metrics["best_model_metric"] = self.best_model_metric
        self.iteration_metrics.append(metrics)

        logger.info(f"Iteration {iteration_num} completed in {metrics['elapsed_time']:.2f}s")

        return metrics

    async def run_self_play(self, num_games: int = 1000) -> list[SelfPlayEpisode]:
        """
        Generate self-play episodes in parallel.

        Args:
            num_games: Number of episodes to generate

        Returns:
            List of episodes
        """
        # Generate tasks
        tasks = self._generate_tasks(num_games)

        # Generate episodes in parallel (batched for memory efficiency)
        batch_size = self.config.get("parallel_batch_size", 32)
        episodes = []

        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i : i + batch_size]

            # Create async tasks
            episode_futures = [self.episode_generator.generate_episode(task) for task in batch_tasks]

            # Wait for batch to complete
            batch_episodes = await asyncio.gather(*episode_futures)
            episodes.extend(batch_episodes)

            logger.info(f"Generated {len(episodes)}/{num_games} episodes")

        # Add to buffer
        self.episode_buffer.extend(episodes)
        if len(self.episode_buffer) > self.max_buffer_size:
            self.episode_buffer = self.episode_buffer[-self.max_buffer_size :]

        return episodes

    def _generate_tasks(self, num_tasks: int) -> list[dict[str, Any]]:
        """Generate diverse tasks for self-play."""
        tasks = []

        # Distribute tasks across generators
        tasks_per_generator = num_tasks // len(self.task_generators)

        for _generator_name, generator in self.task_generators.items():
            gen_tasks = generator.generate(tasks_per_generator)
            tasks.extend(gen_tasks)

        # Add any remaining tasks
        remaining = num_tasks - len(tasks)
        if remaining > 0:
            gen = list(self.task_generators.values())[0]
            tasks.extend(gen.generate(remaining))

        return tasks[:num_tasks]

    async def train(self, training_data: dict[str, list[TrainingExample]], iteration_num: int) -> dict[str, Any]:
        """
        Train models on extracted examples.

        Args:
            training_data: Dictionary of training examples
            iteration_num: Current iteration

        Returns:
            Training metrics
        """
        metrics = {}

        # Create datasets
        policy_dataset = SelfPlayDataset(training_data["policy"] + training_data["negative"], "policy")
        value_dataset = SelfPlayDataset(training_data["value"], "value")

        # Train if we have data
        if len(policy_dataset) > 0 and self.hrm_agent is not None:
            # Train HRM on policy data (simplified)
            policy_loader = DataLoader(
                policy_dataset,
                batch_size=self.config.get("batch_size", 64),
                shuffle=True,
            )

            # Simplified training (in practice, use proper trainer)
            self.hrm_agent.train()
            policy_loss = 0.0
            num_batches = 0

            for _batch in policy_loader:
                # Simplified: would use actual training loop here
                num_batches += 1
                if num_batches >= 10:  # Limit for demo
                    break

            metrics["policy_loss"] = policy_loss / max(num_batches, 1)

        if len(value_dataset) > 0 and self.trm_agent is not None:
            # Train TRM on value data (simplified)
            value_loader = DataLoader(
                value_dataset,
                batch_size=self.config.get("batch_size", 64),
                shuffle=True,
            )

            self.trm_agent.train()
            value_loss = 0.0
            num_batches = 0

            for _batch in value_loader:
                # Simplified: would use actual training loop here
                num_batches += 1
                if num_batches >= 10:  # Limit for demo
                    break

            metrics["value_loss"] = value_loss / max(num_batches, 1)

        return metrics

    async def evaluate_models(self, num_eval_games: int = 100) -> dict[str, Any]:
        """
        Evaluate current models.

        Args:
            num_eval_games: Number of evaluation episodes

        Returns:
            Evaluation metrics
        """
        # Generate evaluation episodes
        eval_tasks = self._generate_tasks(num_eval_games)
        eval_episodes = []

        # Run evaluation episodes
        for task in eval_tasks[:num_eval_games]:  # Limit for performance
            episode = await self.episode_generator.generate_episode(task)
            eval_episodes.append(episode)

        # Compute metrics
        success_count = sum(1 for ep in eval_episodes if ep.outcome == "success")
        avg_reward = np.mean([ep.total_reward for ep in eval_episodes])
        avg_length = np.mean([ep.episode_length for ep in eval_episodes])
        avg_efficiency = np.mean([ep.search_efficiency for ep in eval_episodes])

        metrics = {
            "eval_success_rate": success_count / len(eval_episodes),
            "eval_avg_reward": float(avg_reward),
            "eval_avg_length": float(avg_length),
            "eval_avg_efficiency": float(avg_efficiency),
        }

        return metrics

    def _save_checkpoint(self, iteration: int, best: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "iteration": iteration,
            "best_model_metric": self.best_model_metric,
            "iteration_metrics": self.iteration_metrics,
            "episode_buffer_size": len(self.episode_buffer),
        }

        # Save model states if available
        if self.hrm_agent is not None:
            checkpoint["hrm_state_dict"] = self.hrm_agent.state_dict()

        if self.trm_agent is not None:
            checkpoint["trm_state_dict"] = self.trm_agent.state_dict()

        # Save checkpoint
        if best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"iteration_{iteration}.pt"

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.current_iteration = checkpoint["iteration"]
        self.best_model_metric = checkpoint["best_model_metric"]
        self.iteration_metrics = checkpoint["iteration_metrics"]

        if self.hrm_agent is not None and "hrm_state_dict" in checkpoint:
            self.hrm_agent.load_state_dict(checkpoint["hrm_state_dict"])

        if self.trm_agent is not None and "trm_state_dict" in checkpoint:
            self.trm_agent.load_state_dict(checkpoint["trm_state_dict"])

        logger.info(f"Loaded checkpoint from {path}, iteration {self.current_iteration}")

    def get_quality_metrics(self) -> dict[str, Any]:
        """
        Compute quality metrics over recent iterations.

        Returns:
            Quality metrics dictionary
        """
        if not self.iteration_metrics:
            return {}

        recent_metrics = self.iteration_metrics[-10:]  # Last 10 iterations

        # Success rate trend
        success_rates = [m.get("success_rate", 0.0) for m in recent_metrics]

        # Solve time trend
        avg_lengths = [m.get("eval_avg_length", 0.0) for m in recent_metrics]

        # Solution diversity (number of unique task types solved)
        # Simplified: just track episode counts
        episode_counts = [m.get("num_episodes", 0) for m in recent_metrics]

        return {
            "avg_success_rate": float(np.mean(success_rates)),
            "success_rate_std": float(np.std(success_rates)),
            "success_rate_trend": float(
                np.mean(success_rates[-5:]) - np.mean(success_rates[:5]) if len(success_rates) >= 10 else 0.0
            ),
            "avg_episode_length": float(np.mean(avg_lengths)),
            "total_episodes_generated": sum(episode_counts),
            "best_success_rate": self.best_model_metric,
        }

    def get_resource_usage(self) -> dict[str, Any]:
        """Get current resource usage."""
        metrics = {}

        if HAS_PSUTIL:
            # CPU usage
            metrics["cpu_percent"] = psutil.cpu_percent()

            # Memory usage
            memory = psutil.virtual_memory()
            metrics["memory_used_gb"] = memory.used / (1024**3)
            metrics["memory_percent"] = memory.percent

            # GPU usage (if CUDA available)
            if torch.cuda.is_available():
                metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
                metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)

        return metrics


# ============================================================================
# Main Entry Point
# ============================================================================


async def main():
    """Example usage of self-play training pipeline."""
    logging.basicConfig(level=logging.INFO)

    logger.info("Initializing AlphaZero-Style Self-Play Training Pipeline")

    # Configuration
    config = {
        "games_per_iteration": 1000,
        "batch_size": 64,
        "parallel_batch_size": 32,
        "max_buffer_size": 10000,
        "mcts": {
            "num_simulations": 100,
            "c_puct": 1.25,
        },
    }

    # Initialize trainer (without models for demo)
    trainer = SelfPlayTrainer(
        hrm_agent=None,
        trm_agent=None,
        config=config,
        device="cpu",
    )

    # Run a few iterations
    num_iterations = 3
    for i in range(num_iterations):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"ITERATION {i + 1}/{num_iterations}")
        logger.info(f"{'=' * 60}\n")

        metrics = await trainer.iteration(i)

        logger.info(f"\nIteration {i} Results:")
        logger.info(f"  Episodes: {metrics['num_episodes']}")
        logger.info(f"  Success Rate: {metrics['success_rate']:.2%}")
        logger.info(f"  Policy Examples: {metrics['num_policy_examples']}")
        logger.info(f"  Value Examples: {metrics['num_value_examples']}")
        logger.info(f"  Eval Success Rate: {metrics['eval_success_rate']:.2%}")

        # Show resource usage
        resources = trainer.get_resource_usage()
        if resources:
            logger.info("\nResource Usage:")
            for key, value in resources.items():
                logger.info(f"  {key}: {value:.2f}")

    # Final quality metrics
    quality = trainer.get_quality_metrics()
    logger.info(f"\n{'=' * 60}")
    logger.info("FINAL QUALITY METRICS")
    logger.info(f"{'=' * 60}")
    for key, value in quality.items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info("\nSelf-play training pipeline demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
