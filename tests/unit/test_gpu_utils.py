"""Alias / redirect for unit tests in tests/unit/utils/test_gpu_utils.py."""

from tests.unit.utils.test_gpu_utils import (
    test_check_gpu_ready_false_on_mem_get_info_error,
    test_check_gpu_ready_false_when_insufficient_memory,
    test_check_gpu_ready_false_when_no_cuda,
    test_check_gpu_ready_true_when_enough_memory,
    test_get_gpu_info_cpu_fallback,
    test_get_gpu_info_cuda_available,
    test_gpu_memory_tracker_context_manager,
    test_set_cuda_memory_fraction,
)

__all__ = [
    "test_check_gpu_ready_false_on_mem_get_info_error",
    "test_check_gpu_ready_false_when_insufficient_memory",
    "test_check_gpu_ready_false_when_no_cuda",
    "test_check_gpu_ready_true_when_enough_memory",
    "test_get_gpu_info_cpu_fallback",
    "test_get_gpu_info_cuda_available",
    "test_gpu_memory_tracker_context_manager",
    "test_set_cuda_memory_fraction",
]
