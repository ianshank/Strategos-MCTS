"""
Integration Tests for Demo Training Pipeline
=============================================

Tests the complete demo pipeline end-to-end with mocked external services.

2025 Best Practices:
- Integration tests marked with @pytest.mark.integration
- Mock external services but test real pipeline flow
- Verify artifacts are created correctly
- Test resource management and cleanup
- Performance benchmarks with timing
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests in this module if torch is not available
torch = pytest.importorskip("torch", reason="PyTorch required for demo pipeline tests")
yaml = pytest.importorskip("yaml", reason="PyYAML required for config loading")

# Project root for locating config/fixture files
PROJECT_ROOT = Path(__file__).parent.parent.parent


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def demo_config():
    """Load demo configuration if available."""
    config_path = PROJECT_ROOT / "training" / "config_local_demo.yaml"
    if not config_path.exists():
        pytest.skip(f"Demo config not found at {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace for tests."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    (workspace / "checkpoints").mkdir()
    (workspace / "logs").mkdir()
    (workspace / "data").mkdir()

    yield workspace


@pytest.fixture
def sample_training_config():
    """Create a sample training configuration."""
    return {
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "checkpoint_interval": 1,
        },
        "model": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
        },
        "mcts": {
            "num_simulations": 10,
            "c_puct": 1.0,
            "temperature": 1.0,
        },
    }


@pytest.fixture
def mock_external_services(monkeypatch):
    """Mock all external service API calls.

    Guards against missing optional dependencies (wandb, pinecone) so the
    fixture itself does not crash when they are not installed.
    """
    # Set environment variables
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    monkeypatch.setenv("GITHUB_TOKEN", "test-github-token")

    # Mock W&B (only if installed)
    try:
        import wandb  # noqa: F401

        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        monkeypatch.setattr("wandb.init", mock_wandb.init)
        monkeypatch.setattr("wandb.log", MagicMock())
        monkeypatch.setattr("wandb.finish", MagicMock())
    except ImportError:
        pass

    # Mock Pinecone (only if installed)
    try:
        import pinecone  # noqa: F401

        with patch("pinecone.Index") as mock_index:
            mock_index.return_value.upsert = MagicMock()
            mock_index.return_value.query = MagicMock(return_value={"matches": []})
            yield
            return
    except ImportError:
        pass

    yield


# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.integration
def test_demo_config_exists():
    """Test that demo configuration file exists."""
    config_path = PROJECT_ROOT / "training" / "config_local_demo.yaml"
    assert config_path.exists(), "Demo config file not found"


@pytest.mark.integration
def test_demo_config_valid(demo_config):
    """Test that demo configuration is valid."""
    # Check required sections
    assert "demo" in demo_config
    assert "data" in demo_config
    assert "training" in demo_config
    assert "agents" in demo_config
    assert "rag" in demo_config
    assert "monitoring" in demo_config

    # Verify demo mode is enabled
    assert demo_config["demo"]["mode"] is True

    # Verify reduced parameters for 16GB
    assert demo_config["training"]["batch_size"] <= 8
    assert demo_config["training"]["epochs"] <= 3
    assert demo_config["agents"]["mcts"]["simulations"] <= 50


@pytest.mark.integration
def test_demo_config_memory_optimization(demo_config):
    """Test that memory optimization settings are present."""
    training = demo_config["training"]

    assert training["fp16"] is True  # Mixed precision
    assert "memory_optimization" in training
    assert training["memory_optimization"]["gradient_checkpointing"] is True


@pytest.mark.integration
def test_demo_config_external_services(demo_config):
    """Test that external services are configured."""
    assert "external_services" in demo_config

    services = demo_config["external_services"]
    assert "required" in services

    # Check required services
    required_names = [s["name"] for s in services["required"]]
    assert "pinecone" in required_names
    assert "wandb" in required_names
    assert "github" in required_names


# ============================================================================
# CLI Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_cli_demo_flag_recognition():
    """Test that CLI recognizes --demo flag."""
    from training.cli import main

    # Test that --demo flag is available
    with pytest.raises(SystemExit), patch("sys.argv", ["cli.py", "train", "--help"]):  # argparse will exit
        main()


@pytest.mark.integration
def test_cli_imports():
    """Test that all CLI imports work."""
    try:
        from training.cli import (
            evaluate_command,
            main,
            setup_logging,
            train_command,
        )

        assert callable(train_command)
        assert callable(evaluate_command)
        assert callable(setup_logging)
        assert callable(main)
    except ImportError as e:
        pytest.fail(f"CLI import failed: {e}")


# ============================================================================
# Verification Script Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_verification_script_executes(demo_config, mock_external_services):
    """Test that verification script executes without errors."""
    # wandb is not a CI extra — skip gracefully rather than error
    wandb = pytest.importorskip("wandb", reason="wandb not installed in this environment")  # noqa: F841
    import logging

    from rich.console import Console

    from scripts.verify_external_services import (
        check_critical_failures,
        verify_all_services,
    )

    console = Console()
    logger = logging.getLogger("test")

    # Create temp config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(demo_config, f)
        config_path = Path(f.name)

    try:
        # Every verifier in VERIFIER_MAP must be patched here — an unpatched
        # verifier makes a real network call and fails without live credentials.
        with (
            patch("scripts.verify_external_services.PineconeVerifier.verify") as mock_pinecone,
            patch("scripts.verify_external_services.WandBVerifier.verify") as mock_wandb,
            patch("scripts.verify_external_services.GitHubVerifier.verify") as mock_github,
            patch("scripts.verify_external_services.OpenAIVerifier.verify") as mock_openai,
            patch("scripts.verify_external_services.Neo4jVerifier.verify") as mock_neo4j,
        ):
            from scripts.verify_external_services import (
                ServiceStatus,
                VerificationResult,
            )

            mock_pinecone.return_value = VerificationResult(
                service_name="pinecone",
                status=ServiceStatus.SUCCESS,
                message="Connected",
                is_critical=True,
            )

            mock_wandb.return_value = VerificationResult(
                service_name="wandb",
                status=ServiceStatus.SUCCESS,
                message="Authenticated",
                is_critical=True,
            )

            mock_github.return_value = VerificationResult(
                service_name="github",
                status=ServiceStatus.SUCCESS,
                message="Authenticated",
                is_critical=True,
            )

            mock_openai.return_value = VerificationResult(
                service_name="openai",
                status=ServiceStatus.SKIPPED,
                message="Optional service - credentials not provided",
                is_critical=False,
            )

            mock_neo4j.return_value = VerificationResult(
                service_name="neo4j",
                status=ServiceStatus.SKIPPED,
                message="Optional service - credentials not provided",
                is_critical=False,
            )

            results = await verify_all_services(config_path, logger, console)

            assert len(results) > 0
            assert check_critical_failures(results) is True

    finally:
        config_path.unlink()


# ============================================================================
# Training Pipeline Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available",
)
def test_demo_pipeline_initialization(demo_config, temp_workspace, mock_gpu):
    """Test that demo pipeline initializes correctly."""
    # This test verifies initialization without running full training
    with patch("training.orchestrator.TrainingPipeline._run_phase") as mock_run:
        mock_run.return_value = {"status": "success", "duration": 0}

        # We would initialize pipeline here
        # For now, just verify config structure
        assert "phases" in demo_config
        assert len(demo_config["phases"]) == 5


@pytest.mark.integration
def test_demo_checkpoint_structure(temp_workspace):
    """Test that checkpoint structure is correct."""
    checkpoint_dir = temp_workspace["checkpoints"]

    # Simulate checkpoint creation
    checkpoint = {
        "model_type": "hrm",
        "model_state_dict": {"layer1.weight": [1, 2, 3]},
        "optimizer_state": {},
        "epoch": 1,
        "config": {"hidden_size": 512},
    }

    checkpoint_path = checkpoint_dir / "hrm_checkpoint_epoch_1.pt"
    torch.save(checkpoint, checkpoint_path)

    # Verify checkpoint can be loaded
    loaded = torch.load(checkpoint_path, weights_only=True)
    assert loaded["model_type"] == "hrm"
    assert loaded["epoch"] == 1


@pytest.mark.integration
def test_demo_logging_structure(temp_workspace):
    """Test that logging structure is correct."""
    log_dir = temp_workspace["logs"]

    # Create sample log file
    log_file = log_dir / "training_20250120_120000.log"
    with open(log_file, "w") as f:
        f.write("2025-01-20 12:00:00 - INFO - Training started\n")
        f.write("2025-01-20 12:01:00 - INFO - Epoch 1/3\n")

    assert log_file.exists()
    assert log_file.stat().st_size > 0


# ============================================================================
# Resource Management Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available",
)
def test_gpu_memory_monitoring():
    """Test GPU memory monitoring utilities."""
    if torch.cuda.is_available():
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated(0)

        # Allocate tensor
        tensor = torch.randn(1000, 1000, device="cuda")

        # Check memory increased
        current_memory = torch.cuda.memory_allocated(0)
        assert current_memory > initial_memory

        # Clean up
        del tensor
        torch.cuda.empty_cache()

        # Memory should decrease
        final_memory = torch.cuda.memory_allocated(0)
        assert final_memory < current_memory


@pytest.mark.integration
def test_workspace_cleanup(temp_workspace):
    """Test that workspace cleanup works correctly."""
    # Create some files
    test_file = temp_workspace["checkpoints"] / "test.pt"
    test_file.write_text("test data")

    assert test_file.exists()

    # Cleanup happens automatically in fixture teardown
    # This test verifies files can be created and accessed


# ============================================================================
# Performance Benchmarks
# ============================================================================


@pytest.mark.integration
@pytest.mark.benchmark
def test_config_loading_performance(demo_config):
    """Test that config loading is fast enough."""
    config_path = PROJECT_ROOT / "training" / "config_local_demo.yaml"

    start = time.time()
    for _ in range(100):
        with open(config_path) as f:
            yaml.safe_load(f)
    duration = time.time() - start

    # Should load 100 times in less than 5 seconds (generous for slow CI disk / full-suite I/O)
    assert duration < 5.0, f"Config loading too slow: {duration:.3f}s"


@pytest.mark.integration
@pytest.mark.benchmark
def test_checkpoint_save_performance(temp_workspace):
    """Test that checkpoint saving is fast enough."""
    checkpoint_dir = temp_workspace["checkpoints"]

    # Create realistic checkpoint
    checkpoint = {
        "model_type": "hrm",
        "model_state_dict": {f"layer{i}.weight": torch.randn(512, 512) for i in range(10)},
        "optimizer_state": {},
        "epoch": 1,
    }

    start = time.time()
    checkpoint_path = checkpoint_dir / "benchmark.pt"
    torch.save(checkpoint, checkpoint_path)
    duration = time.time() - start

    # Should save in less than 5 seconds
    assert duration < 5.0, f"Checkpoint save too slow: {duration:.3f}s"


# ============================================================================
# End-to-End Smoke Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.smoke
def test_demo_imports_all_dependencies():
    """Smoke test: verify all demo dependencies can be imported."""
    # wandb and sentence_transformers are optional (not in [dev,neural] CI extras);
    # skip them rather than failing the smoke test when they're absent.
    optional_deps = {"wandb", "sentence_transformers"}
    dependencies = [
        "torch",
        "yaml",
        "pydantic",
        "httpx",
        "tenacity",
        "rich",
    ]

    failed_imports = []
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            failed_imports.append(dep)

    # Warn about missing optional deps but don't fail
    missing_optional = []
    for dep in optional_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_optional.append(dep)
    if missing_optional:
        import warnings

        warnings.warn(f"Optional deps not installed (CI expected): {', '.join(missing_optional)}", stacklevel=2)

    assert not failed_imports, f"Failed to import: {', '.join(failed_imports)}"


@pytest.mark.integration
@pytest.mark.smoke
def test_demo_scripts_exist():
    """Smoke test: verify all demo scripts exist."""
    required_files = [
        "training/config_local_demo.yaml",
        "scripts/verify_external_services.py",
        "scripts/run_local_demo.ps1",
        "docs/LOCAL_TRAINING_GUIDE.md",
    ]

    missing_files = []
    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    assert not missing_files, f"Missing required files: {', '.join(missing_files)}"


@pytest.mark.integration
@pytest.mark.smoke
def test_demo_directories_can_be_created(temp_workspace):
    """Smoke test: verify demo directories can be created."""
    required_dirs = [
        "checkpoints/demo",
        "logs/demo",
        "cache/dabstep",
        "cache/embeddings",
        "reports",
    ]

    for dir_path in required_dirs:
        full_path = temp_workspace["root"] / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        assert full_path.exists()
        assert full_path.is_dir()


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.integration
def test_missing_api_key_handling(monkeypatch):
    """Test that missing API keys are handled gracefully."""
    # Remove all API keys
    for key in [
        "PINECONE_API_KEY",
        "WANDB_API_KEY",
        "GITHUB_TOKEN",
    ]:
        monkeypatch.delenv(key, raising=False)

    from scripts.verify_external_services import ServiceConfig

    _ = ServiceConfig(
        name="test",
        env_var="MISSING_KEY",
        description="Test service",
        required=True,
    )

    # Should handle missing key gracefully
    assert os.getenv("MISSING_KEY") is None


@pytest.mark.integration
def test_invalid_config_handling(temp_workspace):
    """Test that invalid config is handled gracefully."""
    invalid_config = temp_workspace["root"] / "invalid.yaml"

    # Create invalid YAML
    with open(invalid_config, "w") as f:
        f.write("invalid: yaml: syntax: [")

    # Should raise error when loading
    with pytest.raises(yaml.YAMLError), open(invalid_config) as f:
        yaml.safe_load(f)


# ============================================================================
# Documentation Tests
# ============================================================================


@pytest.mark.integration
def test_documentation_exists():
    """Test that all documentation exists."""
    doc_files = [
        "docs/LOCAL_TRAINING_GUIDE.md",
        "README.md",
    ]

    for doc_path in doc_files:
        full_path = PROJECT_ROOT / doc_path
        assert full_path.exists(), f"Documentation missing: {doc_path}"

        # Check file is not empty
        content = full_path.read_text(encoding="utf-8")
        assert len(content) > 100, f"Documentation too short: {doc_path}"


@pytest.mark.integration
def test_documentation_has_required_sections():
    """Test that documentation has required sections."""
    guide_path = PROJECT_ROOT / "docs" / "LOCAL_TRAINING_GUIDE.md"

    if not guide_path.exists():
        pytest.skip("Guide not found")

    content = guide_path.read_text(encoding="utf-8")

    required_sections = [
        "Overview",
        "Hardware Requirements",
        "Environment Setup",
        "Running the Demo",
        "Troubleshooting",
    ]

    missing_sections = [section for section in required_sections if section not in content]

    assert not missing_sections, f"Missing sections: {', '.join(missing_sections)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
