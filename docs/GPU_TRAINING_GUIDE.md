# GPU Training Guide for Strategos-MCTS

This guide covers configuring, executing, and optimizing GPU-accelerated self-play training and neural MCTS.

## Hardware Requirements

- **Minimum**: NVIDIA GPU with >= 4GB VRAM (Compute Capability 7.0+)
- **Recommended**: NVIDIA RTX 3080/4080 (10-16GB VRAM) or A100/H100 for large-scale chess convergence.
- **Software**: CUDA Toolkit 12.1+, PyTorch 2.0+ with CUDA support.

## Configuration via Settings (`.env`)

Configure GPU training parameters directly via environment variables:

```bash
# Torch device override ('cuda', 'cpu', or 'mps')
TORCH_DEVICE_OVERRIDE=cuda

# Mixed precision (FP16) training
TRAINING_USE_MIXED_PRECISION=true

# PyTorch 2.0 graph compilation
TRAINING_COMPILE_MODEL=false

# CUDA memory fraction cap (0.1 to 1.0)
TRAINING_CUDA_MEMORY_FRACTION=0.9

# DataLoader memory pinning
TRAINING_PIN_MEMORY=true

# Distributed training backend ('nccl' or 'gloo')
TRAINING_BACKEND=nccl
```

## Operational Profiles

Use pre-configured training profiles for consistent setups across environments:

- **`smoke`**: 2 iterations, 4 games, 8 MCTS sims (CPU, quick plumbing check)
- **`dev`**: 20 iterations, 50 games, 200 MCTS sims (GPU, local development)
- **`full`**: 200 iterations, 500 games, 800 MCTS sims (Production GPU training)

### CLI Usage

```bash
# Connect Four on GPU with dev profile
python -m src.training.self_play_convergence \
    --domain connect_four \
    --profile dev \
    --checkpoint-dir checkpoints/connect_four

# Chess with custom GPU options
python -m src.training.self_play_convergence \
    --domain chess \
    --iterations 50 \
    --num-simulations 800 \
    --device cuda \
    --mixed-precision \
    --checkpoint-dir checkpoints/chess
```

## Docker Compose GPU Deployment

Run multi-gpu or isolated container training with Docker Compose:

```bash
docker-compose -f docker-compose.train.yml up training-connect-four
```

## GPU Pre-flight & Memory Introspection

Use the Python utilities in `src.utils.gpu_utils`:

```python
from src.utils.gpu_utils import get_gpu_info, check_gpu_ready, GPUMemoryTracker

# Check GPU readiness before starting training
if check_gpu_ready(min_memory_gb=4.0):
    print("GPU ready:", get_gpu_info())

# Track peak memory usage during an operation
with GPUMemoryTracker("self_play_iteration"):
    # training code
    pass
```
