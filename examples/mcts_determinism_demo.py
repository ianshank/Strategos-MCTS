#!/usr/bin/env python3
"""
MCTS Determinism Demonstration

Shows deterministic, testable MCTS with:
- Seeded RNG for reproducibility
- Progressive widening
- Simulation caching
- Experiment tracking

Run with: python examples/mcts_determinism_demo.py
"""

import asyncio
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.framework.mcts.config import (  # noqa: E402
    MCTSConfig,
)
from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState  # noqa: E402
from src.framework.mcts.experiments import ExperimentTracker  # noqa: E402
from src.framework.mcts.policies import (  # noqa: E402
    HybridRolloutPolicy,
    SelectionPolicy,
    ucb1,
)


async def demonstrate_deterministic_mcts():
    """
    Comprehensive demonstration of deterministic MCTS.

    Shows that identical seeds produce identical results.
    """
    print("=" * 70)
    print("DETERMINISTIC MCTS DEMONSTRATION")
    print("=" * 70)

    # Configuration
    seed = 42
    num_iterations = 100

    print("\n1. Creating MCTS Configuration")
    print(f"   Seed: {seed}")
    print(f"   Iterations: {num_iterations}")

    config = MCTSConfig(
        name="demo_config",
        seed=seed,
        num_iterations=num_iterations,
        exploration_weight=1.414,
        progressive_widening_k=1.0,
        progressive_widening_alpha=0.5,
        max_parallel_rollouts=4,
        cache_size_limit=10000,
    )

    print(f"   Progressive Widening: k={config.progressive_widening_k}, alpha={config.progressive_widening_alpha}")
    print("   Formula: expand when visits > k * n^alpha")
    print("   Interpretation: More visits needed before expanding additional children")

    # Define action generator
    def action_generator(state: MCTSState):
        """Generate available actions based on state depth."""
        depth = len(state.state_id.split("_")) - 1
        if depth == 0:
            return ["action_A", "action_B", "action_C"]
        elif depth < 3:
            return ["continue", "refine", "fallback"]
        return []  # Terminal

    # Define state transition
    def state_transition(state: MCTSState, action: str):
        """Compute next state from action."""
        new_id = f"{state.state_id}_{action}"
        new_features = state.features.copy()
        new_features["last_action"] = action
        new_features["depth"] = len(new_id.split("_")) - 1
        return MCTSState(state_id=new_id, features=new_features)

    # Define rollout policy
    rollout_policy = HybridRolloutPolicy(
        heuristic_fn=lambda _s: 0.6,  # Simple constant heuristic
        heuristic_weight=0.7,
        random_weight=0.3,
    )

    # Run multiple times with SAME seed
    print(f"\n2. Running MCTS {3} times with SAME seed to verify determinism...")
    results = []

    for run in range(3):
        # Create fresh engine with same seed
        engine = MCTSEngine(
            seed=seed,
            exploration_weight=config.exploration_weight,
            progressive_widening_k=config.progressive_widening_k,
            progressive_widening_alpha=config.progressive_widening_alpha,
            max_parallel_rollouts=config.max_parallel_rollouts,
            cache_size_limit=config.cache_size_limit,
        )

        # Create root state
        root_state = MCTSState(state_id="root", features={"run": run})
        root = MCTSNode(state=root_state, rng=engine.rng)

        # Run MCTS search
        best_action, stats = await engine.search(
            root=root,
            num_iterations=num_iterations,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=rollout_policy,
            selection_policy=SelectionPolicy.MAX_VISITS,
        )

        results.append(
            {
                "run": run + 1,
                "best_action": best_action,
                "best_visits": stats["best_action_visits"],
                "best_value": stats["best_action_value"],
                "root_visits": stats["root_visits"],
                "cache_hit_rate": stats["cache_hit_rate"],
            }
        )

        print(f"   Run {run + 1}:")
        print(f"      Best Action: {best_action}")
        print(f"      Visits: {stats['best_action_visits']}")
        print(f"      Value: {stats['best_action_value']:.4f}")
        print(f"      Cache Hit Rate: {stats['cache_hit_rate']:.2%}")

    # Verify determinism
    print("\n3. Determinism Verification:")
    is_deterministic = len({r["best_action"] for r in results}) == 1 and len({r["best_visits"] for r in results}) == 1
    print(f"   All runs identical: {is_deterministic}")

    if is_deterministic:
        print("   SUCCESS: Same seed produces identical results!")
    else:
        print("   WARNING: Results differ - check RNG isolation")

    # Demonstrate progressive widening
    print("\n4. Progressive Widening Analysis:")
    engine = MCTSEngine(seed=seed, progressive_widening_k=1.0, progressive_widening_alpha=0.5)
    root_state = MCTSState(state_id="root", features={})
    root = MCTSNode(state=root_state, rng=engine.rng)

    # Simulate expansion decisions
    for num_children in [0, 1, 2, 4, 9]:
        for visits in [1, 2, 4, 10, 20]:
            should_expand = visits > 1.0 * (num_children**0.5)
            if num_children == 0 or num_children == 4:  # Show selected cases
                print(f"   Children={num_children}, Visits={visits}: Expand={should_expand}")

    # Show cache effectiveness
    print("\n5. Cache Effectiveness:")
    engine = MCTSEngine(seed=seed)
    root_state = MCTSState(state_id="cache_test", features={})
    root = MCTSNode(state=root_state, rng=engine.rng)

    _, stats = await engine.search(
        root=root,
        num_iterations=200,
        action_generator=action_generator,
        state_transition=state_transition,
        rollout_policy=rollout_policy,
    )

    print(f"   Total Simulations: {stats['total_simulations']}")
    print(f"   Cache Hits: {stats['cache_hits']}")
    print(f"   Cache Misses: {stats['cache_misses']}")
    print(f"   Hit Rate: {stats['cache_hit_rate']:.2%}")
    print("   Cache reduces redundant evaluations for repeated states")

    # Experiment tracking
    print("\n6. Experiment Tracking:")
    tracker = ExperimentTracker(name="demo_experiments")

    for i, result in enumerate(results):
        from src.framework.mcts.experiments import ExperimentResult

        exp_result = ExperimentResult(
            experiment_id=f"demo_run_{i}",
            config=config.to_dict(),
            seed=seed,
            best_action=result["best_action"],
            best_action_value=result["best_value"],
            best_action_visits=result["best_visits"],
            root_visits=result["root_visits"],
            total_iterations=num_iterations,
            cache_hit_rate=result["cache_hit_rate"],
        )
        tracker.add_result(exp_result)

    summary = tracker.get_summary_statistics()
    print(f"   Tracked {len(tracker)} experiments")
    print(f"   Consistency Rate: {summary['action_consistency']['consistency_rate']:.2%}")
    print(f"   Most Common Action: {summary['action_consistency']['most_common_action']}")

    # UCB1 Formula Demonstration
    print("\n7. UCB1 Selection Policy:")
    print("   Formula: Q(s,a) + c * sqrt(N(parent)) / sqrt(N(child))")
    print(f"   c = {config.exploration_weight} (exploration weight)")

    # Example UCB1 calculations
    parent_visits = 100
    for visits, value_sum in [(10, 8.0), (20, 12.0), (5, 4.5)]:
        ucb_score = ucb1(value_sum, visits, parent_visits, c=1.414)
        avg_value = value_sum / visits
        exploration_term = 1.414 * (parent_visits**0.5) / (visits**0.5)
        print(f"   Visits={visits}, Value={avg_value:.2f}: UCB1={ucb_score:.3f}")
        print(f"      (Exploitation={avg_value:.2f} + Exploration={exploration_term:.2f})")

    # Final selection by max visits (not max value)
    print("\n8. Final Action Selection:")
    print("   Policy: MAX_VISITS (most robust)")
    print("   Selects action with highest visit count, not highest average value")
    print("   This is more robust as it represents the most-explored option")

    print("\n" + "=" * 70)
    print("CODE SNIPPET: Running MCTS with Seed")
    print("=" * 70)
    print("""
from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
from src.framework.mcts.config import MCTSConfig
from src.framework.mcts.policies import HybridRolloutPolicy, SelectionPolicy

# Create deterministic configuration
config = MCTSConfig(seed=42, num_iterations=100)

# Create engine with seed
engine = MCTSEngine(seed=config.seed)

# Create root node
root = MCTSNode(
    state=MCTSState(state_id="root", features={}),
    rng=engine.rng
)

# Run search
best_action, stats = await engine.search(
    root=root,
    num_iterations=config.num_iterations,
    action_generator=my_action_generator,
    state_transition=my_state_transition,
    rollout_policy=HybridRolloutPolicy(),
    selection_policy=SelectionPolicy.MAX_VISITS
)

# Results are deterministic for same seed
print(f"Best Action: {best_action}")
print(f"Statistics: {stats}")
""")

    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demonstrate_deterministic_mcts())
