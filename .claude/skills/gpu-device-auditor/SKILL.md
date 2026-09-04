---
name: gpu-device-auditor
description: Accelerator verification skill. Inspects CUDA topology, checks AMP/TF32 consistency, enforces memory limits via GPUMemoryTracker, and validates multi-GPU DDP process group rank fencing.
---
# gpu-device-auditor

Use this skill to audit GPU accelerator readiness before training or heavy inference.

## Instructions

1. **Inspect Topology**:
   Check the available CUDA devices and drivers using `python -c "import torch; print(torch.cuda.device_count()); print(torch.version.cuda)"`

2. **Validate DDP Rank Fencing**:
   Check if the system handles `GLOO` loopback or `NCCL` properly by executing `tests/e2e/test_ddp_two_rank_cpu_e2e.py` if present.

3. **Check GPUMemoryTracker**:
   Ensure `src.utils.gpu_utils` is functioning and tracking allocations limits.
   
4. **TF32 / AMP**:
   Ensure TensorFloat32 and Automatic Mixed Precision configurations match the target architecture specifications.
