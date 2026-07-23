"""Unit tests for distributed utilities."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from src.utils import distributed


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure a clean environment for each test."""
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)


def test_is_distributed_true():
    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
    ):
        assert distributed.is_distributed() is True


def test_is_distributed_false():
    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=False),
    ):
        assert distributed.is_distributed() is False


def test_init_distributed_already_initialized():
    with patch("src.utils.distributed.is_distributed", return_value=True):
        assert distributed.init_distributed() is True


def test_init_distributed_success():
    with (
        patch("src.utils.distributed.is_distributed", return_value=False),
        patch("torch.distributed.init_process_group") as mock_init,
    ):
        assert distributed.init_distributed("gloo") is True
        mock_init.assert_called_once_with(backend="gloo")


def test_init_distributed_failure():
    with (
        patch("src.utils.distributed.is_distributed", return_value=False),
        patch("torch.distributed.init_process_group", side_effect=RuntimeError("test error")),
    ):
        assert distributed.init_distributed() is False


def test_cleanup_distributed():
    with (
        patch("src.utils.distributed.is_distributed", return_value=True),
        patch("torch.distributed.destroy_process_group") as mock_destroy,
    ):
        distributed.cleanup_distributed()
        mock_destroy.assert_called_once()


def test_cleanup_distributed_not_initialized():
    with (
        patch("src.utils.distributed.is_distributed", return_value=False),
        patch("torch.distributed.destroy_process_group") as mock_destroy,
    ):
        distributed.cleanup_distributed()
        mock_destroy.assert_not_called()


def test_get_rank_distributed():
    with (
        patch("src.utils.distributed.is_distributed", return_value=True),
        patch("torch.distributed.get_rank", return_value=3),
    ):
        assert distributed.get_rank() == 3


def test_get_rank_env():
    with patch("src.utils.distributed.is_distributed", return_value=False):
        os.environ["RANK"] = "2"
        assert distributed.get_rank() == 2


def test_get_local_rank():
    os.environ["LOCAL_RANK"] = "1"
    assert distributed.get_local_rank() == 1


def test_get_world_size_distributed():
    with (
        patch("src.utils.distributed.is_distributed", return_value=True),
        patch("torch.distributed.get_world_size", return_value=4),
    ):
        assert distributed.get_world_size() == 4


def test_get_world_size_env():
    with patch("src.utils.distributed.is_distributed", return_value=False):
        os.environ["WORLD_SIZE"] = "8"
        assert distributed.get_world_size() == 8


def test_is_main_process():
    with patch("src.utils.distributed.get_rank", return_value=0):
        assert distributed.is_main_process() is True

    with patch("src.utils.distributed.get_rank", return_value=1):
        assert distributed.is_main_process() is False


def test_wrap_ddp_not_distributed():
    with patch("src.utils.distributed.is_distributed", return_value=False):
        module = nn.Linear(10, 10)
        assert distributed.wrap_ddp(module, "cpu") is module


def test_wrap_ddp_cpu():
    with (
        patch("src.utils.distributed.is_distributed", return_value=True),
        patch("torch.nn.parallel.DistributedDataParallel") as mock_ddp,
    ):
        module = nn.Linear(10, 10)
        mock_ddp.return_value = "wrapped"
        assert distributed.wrap_ddp(module, "cpu") == "wrapped"
        mock_ddp.assert_called_once_with(module, device_ids=None)


def test_wrap_ddp_cuda():
    with (
        patch("src.utils.distributed.is_distributed", return_value=True),
        patch("src.utils.distributed.get_local_rank", return_value=2),
        patch("torch.nn.parallel.DistributedDataParallel") as mock_ddp,
    ):
        module = nn.Linear(10, 10)
        mock_ddp.return_value = "wrapped_cuda"
        assert distributed.wrap_ddp(module, "cuda:2") == "wrapped_cuda"
        mock_ddp.assert_called_once_with(module, device_ids=[2])


def test_wrap_ddp_failure():
    with (
        patch("src.utils.distributed.is_distributed", return_value=True),
        patch("torch.nn.parallel.DistributedDataParallel", side_effect=Exception("mocked error")),
    ):
        module = nn.Linear(10, 10)
        assert distributed.wrap_ddp(module, "cpu") is module


def test_unwrap_model():
    base_module = nn.Linear(10, 10)

    # Not wrapped
    assert distributed.unwrap_model(base_module) is base_module

    # Wrapped
    wrapped = MagicMock()
    wrapped.module = base_module
    assert distributed.unwrap_model(wrapped) is base_module
