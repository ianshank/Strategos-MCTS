---
id: ddp_orchestrator
goal: Implement Distributed Data Parallel (DDP) for multi-GPU scaling in the MCTS orchestrator
module: src/training/
milestone: Phase 1
status: approved
---

# Goal

Scale the `UnifiedTrainingOrchestrator` and `SelfPlayTrainer` to utilize multiple GPUs via PyTorch Distributed Data Parallel (DDP). This will allow parallel self-play game generation across ranks and synchronization of gradients during policy-value network updates, fulfilling the first step of Phase 1 of the 2026 H2 Roadmap.

# Acceptance Criteria

- AC-1: The configuration system (`SystemConfig` and `Settings`) securely parses `LOCAL_RANK`, `RANK`, and `WORLD_SIZE` from the environment if running under `torchrun`, defaulting to single-node behavior if absent. Intended test: `tests/unit/training/test_system_config.py`.
- AC-2: `UnifiedTrainingOrchestrator` wraps its `policy_value_net` in a `DistributedDataParallel` module if `config.distributed` is `True`. 
- AC-3: `SelfPlayTrainer` accurately distributes self-play load: each local GPU rank maintains a local replay buffer and runs its own self-play games. During network updates, DDP averages the gradients across all participating nodes. 
- AC-4: Race condition fencing: Only `RANK == 0` persists model checkpoints to disk or emits observability metrics to WandB or LangSmith.

# Constraints

- Backwards Compatibility: The system must gracefully fall back to `distributed=False` standard single-GPU operation when `torchrun` is not used.
- Gradient Averaging only: Do not implement complex shared-memory RPC replay buffers. The strategy is "Local Buffer + Gradient Sync".

# Invariants

- DDP initialization (`init_process_group`) occurs only at the outermost execution boundary (e.g., `self_play_convergence.py` or entry points) and gracefully shuts down via `destroy_process_group()`.

# Out of Scope

- Distributed self-play actor frameworks (MPI/RPC remote actors) decoupled from the learner node. That is Phase 1.2.
- TensorRT inference optimizations. That is Phase 1.3.
