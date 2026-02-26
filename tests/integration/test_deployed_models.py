"""
Integration Tests for Deployed Models
=====================================

Tests the production readiness of deployed models exported to `models/production/`.
Verifies model loading, inference, and basic performance criteria.
"""

import sys
import time
from pathlib import Path

import pytest

# Skip all tests in this module if torch is not available
torch = pytest.importorskip("torch", reason="PyTorch required for deployed model tests")
yaml = pytest.importorskip("yaml", reason="PyYAML required for config loading")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from training.agent_trainer import HRMTrainer, TRMTrainer
except ImportError:
    HRMTrainer = None
    TRMTrainer = None


@pytest.fixture(autouse=True)
def safe_torch_load():
    """Fixture to set up safe torch loading for all tests in the module."""
    if hasattr(torch.serialization, "add_safe_globals"):
        import numpy as np

        # Add all needed numpy types
        torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])


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

    # Initialize trainer wrapper
    trainer = HRMTrainer(production_config)

    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()
        # Ensure model is on the correct device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer.model.to(device)
    except Exception as e:
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

    trainer = TRMTrainer(production_config)

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()
        # Ensure model is on the correct device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer.model.to(device)
    except Exception as e:
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
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint
    except Exception as e:
        pytest.fail(f"Failed to load meta-controller: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
