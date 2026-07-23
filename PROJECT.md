# Project: Strategos-MCTS PR #85 GPU/Gameplay Hardening

## Architecture
Strategos-MCTS is a high-performance Python framework for Monte Carlo Tree Search and AlphaZero-style self-play training, featuring multi-agent reinforcement learning, neural network policy/value evaluation, and GPU introspection.

## Code Layout
- `src/config/`: Configuration settings and constants (`settings.py`, `constants.py`).
- `src/models/`: Neural network architecture and loss functions (`policy_value_net.py`).
- `src/training/`: Self-play training loops, system config, convergence routines (`self_play_trainer.py`, `system_config.py`, `training_config.py`, `self_play_convergence.py`).
- `src/utils/`: Hardware introspection and GPU utilities (`gpu_utils.py`).
- `docs/`: Framework documentation (`GAME_DOMAINS.md`, `CHANGELOG.md`).
- `tests/`: Integration and unit test suites (`test_deployed_models.py`, etc.).

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: CI & Config Hygiene | R1, R4.1, R4.4, R4.7, R4.8 | none | DONE |
| 2 | M2: Core Training & Model Bugs | R2.1, R2.2, R2.3, R2.4, R4.2, R4.3, R4.6 | M1 | DONE |
| 3 | M3: System Config & GPU Memory | R3.2, R3.3, R3.6 | M2 | DONE |
| 4 | M4: Tests & Docs Hardening | R3.1, R3.4, R3.5, R4.5 | M3 | DONE |
| 5 | M5: Full Regression & Push | R5 | M4 | DONE |

## Interface Contracts
- `SystemConfig.from_settings()` re-validates device and precision invariants after applying `Settings`.
- `AlphaZeroLoss.forward()` gracefully handles `-inf` logits without producing NaN gradients or outputs.
- `check_gpu_ready()` returns hardware-accurate free VRAM using `torch.cuda.mem_get_info()`.
- `SelfPlayTrainer.train_step()` pins CPU memory prior to CUDA transfer with `non_blocking=True`.
