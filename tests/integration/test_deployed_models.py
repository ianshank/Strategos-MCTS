"""
Integration Tests for Deployed Models
=====================================

Tests the production readiness of deployed models exported to `models/production/`.
Verifies model loading, inference, and basic performance criteria.
"""

from pathlib import Path
import time

import pytest

# Skip all tests in this module if torch is not available
torch = pytest.importorskip("torch", reason="PyTorch required for deployed model tests")
yaml = pytest.importorskip("yaml", reason="PyYAML required for config loading")

PROJECT_ROOT = Path(__file__).parent.parent.parent

try:
    from training.agent_trainer import HRMTrainer, TRMTrainer
except ImportError:
    HRMTrainer = None
    TRMTrainer = None


@pytest.fixture(autouse=True)
def safe_torch_load():
    """Fixture to set up safe torch loading for all tests in the module.

    Checkpoints may have been saved with a numpy version where the canonical
    path is ``numpy._core.multiarray.scalar``. On numpy 1.26.x the ``_core``
    subpackage isn't directly importable, so we alias it in ``sys.modules``
    AND register the ``scalar`` type with ``add_safe_globals``.
    """
    if hasattr(torch.serialization, "add_safe_globals"):
        import sys

        import numpy as np

        # Ensure numpy._core.multiarray is resolvable by the unpickler.
        # On numpy 1.x, np._core may not exist or may lack multiarray.
        if "numpy._core" not in sys.modules:
            sys.modules["numpy._core"] = np.core  # type: ignore[attr-defined]
        if "numpy._core.multiarray" not in sys.modules:
            sys.modules["numpy._core.multiarray"] = np.core.multiarray  # type: ignore[attr-defined]

        from numpy.core.multiarray import scalar as np_scalar  # type: ignore[attr-defined]

        safe_globals = [np_scalar, np.dtype]
        if hasattr(np, "dtypes") and hasattr(np.dtypes, "Float64DType"):
            safe_globals.append(np.dtypes.Float64DType)
        torch.serialization.add_safe_globals(safe_globals)


@pytest.fixture
def production_models_dir():
    """Path to production models."""
    return PROJECT_ROOT / "models" / "production"


@pytest.fixture
def production_config():
    """Load production configuration."""
    config_path = PROJECT_ROOT / "training" / "configs" / "production_config.yaml"
    if not config_path.exists():
        # Fallback to main config if production config not generated yet
        config_path = PROJECT_ROOT / "training" / "config.yaml"

    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.mark.integration
def test_deployed_models_exist(production_models_dir):
    """Verify all required models are deployed."""
    required_models = ["hrm_production.pt", "trm_production.pt", "mcts_production.pt", "meta_production.pt"]

    missing = []
    for model in required_models:
        if not (production_models_dir / model).exists():
            missing.append(model)

    assert not missing, f"Missing deployed models: {', '.join(missing)}"


@pytest.mark.integration
def test_hrm_model_loading_and_inference(production_models_dir, production_config):
    """Test that deployed HRM model loads and runs inference."""
    model_path = production_models_dir / "hrm_production.pt"

    if not model_path.exists():
        pytest.skip("HRM model not deployed")

    try:
        trainer = HRMTrainer(production_config)
    except (ImportError, ValueError, OSError) as err:
        err_msg = str(err).lower()
        if (
            "token" in err_msg
            or "pretrained" in err_msg
            or "transformers" in err_msg
            or isinstance(err, (ImportError, OSError))
        ):
            pytest.skip(f"Tokenizer or model dependency missing: {err}")
        raise

    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()
        # Ensure model is on the correct device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer.model.to(device)
        if device == "cpu":
            # CPU has no Half matmul support; upcast in case any weights
            # (fp16 hub snapshot or fp16-exported checkpoint) came in as Half
            trainer.model.float()
    except (OSError, ValueError) as e:
        pytest.fail(f"Failed to load HRM model: {e}")

    # Run dummy inference
    # Ensure input is on correct device (cuda)
    input_ids = torch.randint(0, 1000, (1, 128)).to(device)

    with torch.no_grad():
        start_time = time.time()
        outputs = trainer.model(input_ids)
        duration = (time.time() - start_time) * 1000  # ms

    assert "logits" in outputs
    assert outputs["logits"].shape[-1] == production_config["agents"]["hrm"]["num_labels"]

    # Performance check (loose threshold for CI/CD environments)
    # Initial run might be slow due to CUDA initialization overhead
    assert duration < 5000, f"Inference too slow: {duration:.2f}ms"


@pytest.mark.integration
def test_trm_model_loading_and_inference(production_models_dir, production_config):
    """Test that deployed TRM model loads and runs inference."""
    model_path = production_models_dir / "trm_production.pt"

    if not model_path.exists():
        pytest.skip("TRM model not deployed")

    try:
        trainer = TRMTrainer(production_config)
    except (ImportError, ValueError, OSError) as err:
        err_msg = str(err).lower()
        if (
            "token" in err_msg
            or "pretrained" in err_msg
            or "transformers" in err_msg
            or isinstance(err, (ImportError, OSError))
        ):
            pytest.skip(f"Tokenizer or model dependency missing: {err}")
        raise

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()
        # Ensure model is on the correct device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer.model.to(device)
        if device == "cpu":
            # CPU has no Half matmul support; upcast in case any weights
            # (fp16 hub snapshot or fp16-exported checkpoint) came in as Half
            trainer.model.float()
    except (OSError, ValueError) as e:
        pytest.fail(f"Failed to load TRM model: {e}")

    input_ids = torch.randint(0, 1000, (1, 128)).to(device)

    with torch.no_grad():
        outputs = trainer.model(input_ids)

    assert "improvement_predictions" in outputs
    # Check max iterations match config
    expected_iters = production_config["agents"]["trm"]["max_refinement_iterations"]
    assert outputs["improvement_predictions"].shape[-1] == expected_iters


@pytest.mark.integration
def test_meta_controller_loading(production_models_dir):
    """Test that deployed meta-controller loads."""
    model_path = production_models_dir / "meta_production.pt"

    if not model_path.exists():
        pytest.skip("Meta-controller not deployed")

    try:
        # weights_only=False because this trusted checkpoint was saved with a numpy
        # version whose pickle globals reference numpy._core.multiarray.scalar, which
        # can't be add_safe_globals'd on numpy 1.26.x (different module path).
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint
    except (OSError, ValueError) as e:
        pytest.fail(f"Failed to load meta-controller: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
