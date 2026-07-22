"""Unit tests for GPU utilities in src/utils/gpu_utils.py."""

from __future__ import annotations

import unittest.mock as mock

import pytest

from src.utils.gpu_utils import (
    GPUMemoryTracker,
    check_gpu_ready,
    get_gpu_info,
    set_cuda_memory_fraction,
)


@pytest.mark.unit
def test_get_gpu_info_cpu_fallback() -> None:
    with mock.patch("torch.cuda.is_available", return_value=False):
        info = get_gpu_info()
        assert info["cuda_available"] is False
        assert info["device_count"] == 0
        assert info["memory_total_gb"] == 0.0


@pytest.mark.unit
def test_get_gpu_info_cuda_available() -> None:
    mock_props = mock.MagicMock()
    mock_props.name = "NVIDIA GeForce RTX 4090"
    mock_props.total_memory = 24 * (1024**3)
    mock_props.major = 8
    mock_props.minor = 9

    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.current_device", return_value=0),
        mock.patch("torch.cuda.device_count", return_value=1),
        mock.patch("torch.cuda.get_device_properties", return_value=mock_props),
        mock.patch("torch.cuda.memory_allocated", return_value=2 * (1024**3)),
        mock.patch("torch.cuda.memory_reserved", return_value=4 * (1024**3)),
    ):
        info = get_gpu_info()
        assert info["cuda_available"] is True
        assert info["device_name"] == "NVIDIA GeForce RTX 4090"
        assert info["memory_total_gb"] == 24.0
        assert info["memory_allocated_gb"] == 2.0
        assert info["capability"] == (8, 9)


@pytest.mark.unit
def test_check_gpu_ready_false_when_no_cuda() -> None:
    with mock.patch("src.utils.gpu_utils.get_gpu_info", return_value={"cuda_available": False}):
        assert check_gpu_ready(min_memory_gb=2.0) is False


@pytest.mark.unit
def test_check_gpu_ready_true_when_enough_memory() -> None:
    info = {
        "cuda_available": True,
        "device_name": "Test GPU",
        "memory_total_gb": 16.0,
        "memory_allocated_gb": 2.0,
    }
    with mock.patch("src.utils.gpu_utils.get_gpu_info", return_value=info):
        assert check_gpu_ready(min_memory_gb=4.0) is True


@pytest.mark.unit
def test_check_gpu_ready_false_when_insufficient_memory() -> None:
    info = {
        "cuda_available": True,
        "device_name": "Test GPU",
        "memory_total_gb": 8.0,
        "memory_allocated_gb": 7.0,
    }
    with mock.patch("src.utils.gpu_utils.get_gpu_info", return_value=info):
        assert check_gpu_ready(min_memory_gb=4.0) is False


@pytest.mark.unit
def test_set_cuda_memory_fraction() -> None:
    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.set_per_process_memory_fraction") as mock_set,
    ):
        set_cuda_memory_fraction(0.8)
        mock_set.assert_called_once_with(0.8)


@pytest.mark.unit
def test_gpu_memory_tracker_context_manager() -> None:
    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.reset_peak_memory_stats") as mock_reset,
        mock.patch("torch.cuda.memory_allocated", return_value=100 * (1024**2)),
        mock.patch("torch.cuda.max_memory_allocated", return_value=500 * (1024**2)),
        mock.patch("torch.cuda.max_memory_reserved", return_value=600 * (1024**2)),
    ):
        with GPUMemoryTracker("test_op") as tracker:
            pass
        mock_reset.assert_called_once()
        assert tracker.peak_allocated_mb == 500.0
        assert tracker.peak_reserved_mb == 600.0
