"""
Create LangSmith Datasets for Experiments.

This script creates datasets in LangSmith for running experiments on HRM, TRM,
and MCTS agents with different configurations.

Usage:
    python scripts/create_langsmith_datasets.py

Datasets created:
- tactical_e2e_scenarios: Tactical military scenarios
- cybersecurity_e2e_scenarios: Cybersecurity incident response scenarios
- mcts_benchmark_scenarios: MCTS decision-making benchmarks
"""

import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.utils.langsmith_tracing import create_test_dataset  # noqa: E402


def create_tactical_dataset() -> str:
    """Create tactical E2E scenarios dataset."""
    examples = [
        {
            "inputs": {
                "query": "Enemy approaching from north. Limited visibility, night conditions. "
                "Infantry platoon, UAV support, limited ammo. Recommend defensive strategy.",
                "use_rag": True,
                "use_mcts": False,
                "scenario_type": "tactical",
            },
            "outputs": {
                "expected_elements": [
                    "defensive_position",
                    "observation_posts",
                    "uav_deployment",
                    "ammo_conservation",
                ],
                "risk_level": "medium",
                "confidence_threshold": 0.7,
            },
        },
        {
            "inputs": {
                "query": "Multi-sector threat detected. Three enemy positions identified. "
                "Limited resources, need priority targeting. Recommend engagement sequence.",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "tactical",
            },
            "outputs": {
                "expected_elements": [
                    "priority_targeting",
                    "resource_allocation",
                    "threat_assessment",
                    "engagement_sequence",
                ],
                "risk_level": "high",
                "confidence_threshold": 0.75,
            },
        },
        {
            "inputs": {
                "query": "Favorable terrain advantage identified. Enemy retreating. "
                "Recommend offensive vs defensive posture.",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "tactical",
            },
            "outputs": {
                "expected_elements": [
                    "terrain_analysis",
                    "pursuit_recommendation",
                    "risk_assessment",
                    "force_preservation",
                ],
                "risk_level": "medium",
                "confidence_threshold": 0.70,
            },
        },
    ]

    print("Creating tactical_e2e_scenarios dataset...")
    dataset_id = create_test_dataset(
        dataset_name="tactical_e2e_scenarios",
        examples=examples,
        description="E2E tactical military scenarios for testing HRM/TRM/MCTS agents",
    )
    print(f"[OK] Created dataset: tactical_e2e_scenarios (ID: {dataset_id})")
    return dataset_id


def create_cybersecurity_dataset() -> str:
    """Create cybersecurity E2E scenarios dataset."""
    examples = [
        {
            "inputs": {
                "query": "APT28 indicators detected. Credential harvesting and lateral movement observed. "
                "Recommend containment strategy.",
                "use_rag": True,
                "use_mcts": False,
                "scenario_type": "cybersecurity",
            },
            "outputs": {
                "expected_elements": [
                    "threat_actor_identification",
                    "containment_actions",
                    "credential_reset",
                    "lateral_movement_blocking",
                ],
                "severity": "critical",
                "confidence_threshold": 0.80,
            },
        },
        {
            "inputs": {
                "query": "Ransomware encryption detected across multiple systems. "
                "Backups available but compromised. Recommend recovery strategy.",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "cybersecurity",
            },
            "outputs": {
                "expected_elements": [
                    "isolation_strategy",
                    "backup_validation",
                    "recovery_sequence",
                    "forensics_preservation",
                ],
                "severity": "critical",
                "confidence_threshold": 0.85,
            },
        },
        {
            "inputs": {
                "query": "Unusual outbound traffic to known C2 servers. "
                "Multiple workstations affected. Recommend investigation and response.",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "cybersecurity",
            },
            "outputs": {
                "expected_elements": [
                    "traffic_analysis",
                    "workstation_isolation",
                    "c2_blocking",
                    "threat_hunting",
                ],
                "severity": "high",
                "confidence_threshold": 0.75,
            },
        },
    ]

    print("Creating cybersecurity_e2e_scenarios dataset...")
    dataset_id = create_test_dataset(
        dataset_name="cybersecurity_e2e_scenarios",
        examples=examples,
        description="E2E cybersecurity incident response scenarios for testing agents",
    )
    print(f"[OK] Created dataset: cybersecurity_e2e_scenarios (ID: {dataset_id})")
    return dataset_id


def create_mcts_benchmark_dataset() -> str:
    """Create MCTS benchmark scenarios dataset."""
    examples = [
        {
            "inputs": {
                "initial_state": {
                    "position": "neutral",
                    "resources": {"ammo": 100, "fuel": 80, "personnel": 25},
                    "enemy_position": "north",
                    "visibility": "low",
                },
                "possible_actions": [
                    "advance_to_alpha",
                    "hold_current_position",
                    "retreat_to_beta",
                    "flanking_maneuver",
                    "request_reinforcement",
                ],
                "objective": "secure_area_minimal_casualties",
                "iterations": 100,
            },
            "outputs": {
                "expected_win_probability_threshold": 0.60,
                "expected_best_actions": ["hold_current_position", "advance_to_alpha"],
                "convergence_iterations": 80,
            },
        },
        {
            "inputs": {
                "initial_state": {
                    "position": "defensive",
                    "resources": {"ammo": 60, "fuel": 40, "personnel": 15},
                    "enemy_position": "advancing",
                    "visibility": "high",
                },
                "possible_actions": [
                    "defensive_hold",
                    "strategic_retreat",
                    "counterattack",
                    "call_air_support",
                ],
                "objective": "minimize_casualties",
                "iterations": 200,
            },
            "outputs": {
                "expected_win_probability_threshold": 0.55,
                "expected_best_actions": ["defensive_hold", "call_air_support"],
                "convergence_iterations": 150,
            },
        },
    ]

    print("Creating mcts_benchmark_scenarios dataset...")
    dataset_id = create_test_dataset(
        dataset_name="mcts_benchmark_scenarios",
        examples=examples,
        description="MCTS decision-making benchmark scenarios for performance testing",
    )
    print(f"[OK] Created dataset: mcts_benchmark_scenarios (ID: {dataset_id})")
    return dataset_id


def create_stem_scenarios_dataset() -> str:
    """Create STEM (Math/Physics/Science/CS) scenarios dataset."""
    examples = [
        # Mathematics
        {
            "inputs": {
                "query": "Optimize resource allocation problem: 3 machines, 5 jobs, each job has "
                "different processing times on each machine. Minimize total completion time. "
                "Job times (in hours): Job1[2,3,4], Job2[3,2,3], Job3[4,3,2], Job4[2,4,3], Job5[3,2,4]",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "mathematics",
            },
            "outputs": {
                "expected_elements": [
                    "optimization_algorithm",
                    "scheduling_strategy",
                    "total_time_calculation",
                    "machine_assignment",
                ],
                "confidence_threshold": 0.75,
            },
        },
        {
            "inputs": {
                "query": "Graph theory problem: Find shortest path in weighted graph with 10 nodes, "
                "15 edges. Negative weights possible. Recommend algorithm and explain complexity.",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "mathematics",
            },
            "outputs": {
                "expected_elements": [
                    "algorithm_selection",
                    "complexity_analysis",
                    "negative_weight_handling",
                    "implementation_approach",
                ],
                "confidence_threshold": 0.80,
            },
        },
        # Physics
        {
            "inputs": {
                "query": "Projectile motion analysis: Object launched at 45° angle, initial velocity 50 m/s, "
                "from 100m height. Calculate maximum height, range, and time of flight. Air resistance negligible.",
                "use_rag": False,
                "use_mcts": False,
                "scenario_type": "physics",
            },
            "outputs": {
                "expected_elements": [
                    "kinematic_equations",
                    "maximum_height_calculation",
                    "range_calculation",
                    "time_of_flight",
                ],
                "confidence_threshold": 0.85,
            },
        },
        {
            "inputs": {
                "query": "Thermodynamics problem: Design heat engine operating between 600K and 300K reservoirs. "
                "Calculate maximum theoretical efficiency (Carnot) and suggest practical improvements.",
                "use_rag": True,
                "use_mcts": False,
                "scenario_type": "physics",
            },
            "outputs": {
                "expected_elements": [
                    "carnot_efficiency",
                    "theoretical_maximum",
                    "practical_considerations",
                    "improvement_suggestions",
                ],
                "confidence_threshold": 0.75,
            },
        },
        # Computer Science - Algorithms
        {
            "inputs": {
                "query": "Design algorithm for real-time anomaly detection in streaming data. "
                "Requirements: 100k events/sec, < 10ms latency, 99.9% accuracy. Recommend approach.",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "computer_science",
            },
            "outputs": {
                "expected_elements": [
                    "streaming_algorithm",
                    "latency_optimization",
                    "accuracy_tradeoffs",
                    "scalability_considerations",
                ],
                "confidence_threshold": 0.70,
            },
        },
        {
            "inputs": {
                "query": "Database query optimization: Join 4 tables (10M, 5M, 1M, 500K rows). "
                "Current query takes 45s. Analyze execution plan and recommend optimizations.",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "computer_science",
            },
            "outputs": {
                "expected_elements": [
                    "execution_plan_analysis",
                    "index_recommendations",
                    "join_order_optimization",
                    "performance_estimation",
                ],
                "confidence_threshold": 0.75,
            },
        },
        # Computer Science - Architecture
        {
            "inputs": {
                "query": "Microservices architecture design: E-commerce platform expecting 1M users, "
                "10K concurrent requests. Design service boundaries, data consistency strategy, and scaling approach.",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "computer_science",
            },
            "outputs": {
                "expected_elements": [
                    "service_decomposition",
                    "data_consistency_strategy",
                    "scaling_plan",
                    "fault_tolerance",
                ],
                "confidence_threshold": 0.70,
            },
        },
        # Chemistry
        {
            "inputs": {
                "query": "Chemical equilibrium problem: Reaction A + B ⇌ C + D at 500K. "
                "Given Kp = 2.5, initial pressures PA=2atm, PB=3atm, PC=0, PD=0. Calculate equilibrium pressures.",
                "use_rag": False,
                "use_mcts": False,
                "scenario_type": "chemistry",
            },
            "outputs": {
                "expected_elements": [
                    "equilibrium_expression",
                    "ice_table",
                    "equilibrium_pressures",
                    "verification",
                ],
                "confidence_threshold": 0.80,
            },
        },
        # Data Science
        {
            "inputs": {
                "query": "ML model selection: Binary classification, 100K samples, 50 features, 30% class imbalance. "
                "Production requirements: <100ms inference, 95% recall. Recommend model and training strategy.",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "data_science",
            },
            "outputs": {
                "expected_elements": [
                    "model_recommendation",
                    "class_imbalance_handling",
                    "performance_optimization",
                    "evaluation_strategy",
                ],
                "confidence_threshold": 0.75,
            },
        },
        # Computer Science - Distributed Systems
        {
            "inputs": {
                "query": "Distributed consensus problem: 5-node cluster, Byzantine fault tolerance required. "
                "Network partition possible. Recommend consensus algorithm and analyze failure scenarios.",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "computer_science",
            },
            "outputs": {
                "expected_elements": [
                    "consensus_algorithm",
                    "byzantine_tolerance",
                    "partition_handling",
                    "failure_analysis",
                ],
                "confidence_threshold": 0.70,
            },
        },
        # Applied Mathematics - Cryptography
        {
            "inputs": {
                "query": "Cryptographic protocol design: Secure key exchange for IoT devices with limited CPU (50MHz). "
                "Requirements: Forward secrecy, resistance to quantum attacks. Recommend protocol.",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "mathematics",
            },
            "outputs": {
                "expected_elements": [
                    "protocol_selection",
                    "computational_constraints",
                    "post_quantum_security",
                    "implementation_guidance",
                ],
                "confidence_threshold": 0.75,
            },
        },
        # Computational Biology
        {
            "inputs": {
                "query": "Protein folding prediction: Sequence of 200 amino acids. "
                "Recommend computational approach balancing accuracy and runtime (< 24 hours).",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "computational_biology",
            },
            "outputs": {
                "expected_elements": [
                    "algorithm_selection",
                    "accuracy_runtime_tradeoff",
                    "resource_requirements",
                    "validation_approach",
                ],
                "confidence_threshold": 0.65,
            },
        },
    ]

    print("Creating stem_scenarios dataset...")
    dataset_id = create_test_dataset(
        dataset_name="stem_scenarios",
        examples=examples,
        description="STEM (Math, Physics, Science, Computer Science) problem-solving scenarios",
    )
    print(f"[OK] Created dataset: stem_scenarios (ID: {dataset_id})")
    return dataset_id


def create_generic_scenarios_dataset() -> str:
    """Create generic test scenarios dataset for general-purpose testing."""
    examples = [
        {
            "inputs": {
                "query": "Analyze the current situation and provide a strategic recommendation.",
                "use_rag": True,
                "use_mcts": False,
                "scenario_type": "general",
            },
            "outputs": {
                "expected_elements": [
                    "situation_analysis",
                    "strategic_recommendation",
                    "risk_assessment",
                ],
                "confidence_threshold": 0.70,
            },
        },
        {
            "inputs": {
                "query": "Multiple competing priorities identified. Recommend prioritization strategy "
                "and resource allocation.",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "general",
            },
            "outputs": {
                "expected_elements": [
                    "priority_ranking",
                    "resource_allocation",
                    "trade_off_analysis",
                    "timeline",
                ],
                "confidence_threshold": 0.75,
            },
        },
        {
            "inputs": {
                "query": "Complex decision with uncertain outcomes. Available information is incomplete. "
                "Recommend best course of action with risk mitigation.",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "general",
            },
            "outputs": {
                "expected_elements": [
                    "decision_recommendation",
                    "uncertainty_quantification",
                    "risk_mitigation",
                    "contingency_planning",
                ],
                "confidence_threshold": 0.65,
            },
        },
        {
            "inputs": {
                "query": "Time-sensitive decision required. Limited information available. "
                "Provide rapid assessment and actionable recommendation.",
                "use_rag": False,
                "use_mcts": False,
                "scenario_type": "general",
            },
            "outputs": {
                "expected_elements": [
                    "rapid_assessment",
                    "immediate_action",
                    "follow_up_tasks",
                ],
                "confidence_threshold": 0.60,
            },
        },
        {
            "inputs": {
                "query": "Long-term planning scenario. Multiple phases required. "
                "Develop comprehensive strategy with milestones and success criteria.",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "general",
            },
            "outputs": {
                "expected_elements": [
                    "phase_breakdown",
                    "milestones",
                    "success_criteria",
                    "resource_planning",
                    "risk_management",
                ],
                "confidence_threshold": 0.80,
            },
        },
    ]

    print("Creating generic_scenarios dataset...")
    dataset_id = create_test_dataset(
        dataset_name="generic_scenarios",
        examples=examples,
        description="Generic test scenarios for general-purpose agent testing and development",
    )
    print(f"[OK] Created dataset: generic_scenarios (ID: {dataset_id})")
    return dataset_id


def main():
    """Create all LangSmith datasets."""
    # Check if LangSmith is configured
    if not os.getenv("LANGSMITH_API_KEY"):
        print("[ERROR] LANGSMITH_API_KEY environment variable not set")
        print("        Set it with: export LANGSMITH_API_KEY=your_key_here")
        sys.exit(1)

    print("=" * 70)
    print("Creating LangSmith Datasets for Experiments")
    print("=" * 70)
    print()

    created_datasets = {}
    dataset_functions = [
        ("tactical_e2e_scenarios", create_tactical_dataset),
        ("cybersecurity_e2e_scenarios", create_cybersecurity_dataset),
        ("mcts_benchmark_scenarios", create_mcts_benchmark_dataset),
        ("stem_scenarios", create_stem_scenarios_dataset),
        ("generic_scenarios", create_generic_scenarios_dataset),
    ]

    for dataset_name, create_func in dataset_functions:
        try:
            dataset_id = create_func()
            created_datasets[dataset_name] = dataset_id
            print()
        except Exception as e:
            if "409" in str(e) or "already exists" in str(e).lower():
                print(f"[SKIP] Dataset '{dataset_name}' already exists, skipping...")
                created_datasets[dataset_name] = "existing"
                print()
            else:
                print(f"[ERROR] Failed to create '{dataset_name}': {e}")
                print()

    print("=" * 70)
    print("[COMPLETE] Dataset creation finished!")
    print("=" * 70)
    print()
    print("Dataset Status:")
    for name, dataset_id in created_datasets.items():
        if dataset_id == "existing":
            print(f"  - {name}: [EXISTING]")
        else:
            print(f"  - {name}: [CREATED] (ID: {dataset_id})")
    print()
    print("Next steps:")
    print("  1. View datasets in LangSmith UI")
    print("  2. Run experiments with: python scripts/run_langsmith_experiments.py")
    print()


if __name__ == "__main__":
    main()
