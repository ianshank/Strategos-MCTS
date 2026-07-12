"""
Dtype-safety regression tests for the deployed-model loading path.

The HuggingFace Hub snapshot of ``microsoft/deberta-v3-base`` stores its
safetensors weights in fp16. transformers >= 5 loads checkpoints with
``dtype="auto"`` by default, which honors that stored dtype, so the HRM/TRM
backbone comes out in Half while the freshly-created heads are Float.
``load_state_dict`` preserves per-parameter dtypes, so the mix survives
checkpoint loading and CPU inference fails with::

    RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float

(observed in tests/integration/test_deployed_models.py). These tests pin the
fix: ``build_model`` must normalize the assembled model to fp32 regardless of
the dtype ``from_pretrained`` hands back.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for agent trainer tests")
pytest.importorskip("transformers", reason="transformers required for the DeBERTa-backed build path")

import torch.nn as nn  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import training.agent_trainer as agent_trainer  # noqa: E402

HIDDEN_SIZE = 64
VOCAB_SIZE = 1000


class _HalfBackbone(nn.Module):
    """Stand-in for an fp16 Hub snapshot loaded under dtype="auto" defaults."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.encoder = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor]:
        hidden = self.encoder(self.embeddings(input_ids))
        return (hidden,)


class _StubAutoModel:
    @staticmethod
    def from_pretrained(_model_name: str, **_kwargs) -> nn.Module:
        return _HalfBackbone().half()


class _StubTokenizer:
    @staticmethod
    def from_pretrained(_model_name: str, **_kwargs) -> None:
        return None


@pytest.fixture
def dtype_config() -> dict:
    """Minimal config for building HRM/TRM trainers against the stub backbone."""
    return {
        "training": {
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "epochs": 1,
            "warmup_ratio": 0.1,
            "gradient_accumulation_steps": 1,
            "gradient_clip_norm": 1.0,
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.1, "target_modules": ["query", "key"]},
        },
        "agents": {
            "hrm": {
                "model_name": "stub/fp16-backbone",
                "max_decomposition_depth": 5,
                "lora_rank": 8,
                "hidden_size": HIDDEN_SIZE,
                "num_labels": 3,
            },
            "trm": {
                "model_name": "stub/fp16-backbone",
                "max_refinement_iterations": 3,
                "convergence_threshold": 0.95,
                "lora_rank": 8,
                "hidden_size": HIDDEN_SIZE,
            },
        },
    }


@pytest.fixture
def fp16_backbone_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the trainers build against an fp16 backbone without network or PEFT."""
    monkeypatch.setattr(agent_trainer, "AutoModel", _StubAutoModel)
    monkeypatch.setattr(agent_trainer, "DebertaV2Tokenizer", _StubTokenizer)
    monkeypatch.setattr(agent_trainer, "HAS_PEFT", False)


def _assert_all_fp32(model: nn.Module) -> None:
    non_fp32 = {name: str(p.dtype) for name, p in model.named_parameters() if p.dtype != torch.float32}
    assert not non_fp32, f"Expected all parameters in float32, found: {non_fp32}"


@pytest.mark.unit
class TestHRMTrainerDtype:
    def test_build_model_upcasts_fp16_backbone_to_fp32(self, dtype_config, fp16_backbone_env):
        trainer = agent_trainer.HRMTrainer(dtype_config, device="cpu")
        _assert_all_fp32(trainer.model)

    def test_cpu_inference_after_fp16_backbone_build(self, dtype_config, fp16_backbone_env):
        trainer = agent_trainer.HRMTrainer(dtype_config, device="cpu")
        trainer.model.eval()

        input_ids = torch.randint(0, VOCAB_SIZE, (1, 16))
        with torch.no_grad():
            outputs = trainer.model(input_ids)

        assert outputs["logits"].shape == (1, dtype_config["agents"]["hrm"]["num_labels"])
        assert torch.isfinite(outputs["logits"]).all()

    def test_fp16_checkpoint_state_dict_loads_as_fp32(self, dtype_config, fp16_backbone_env):
        """An fp16-saved checkpoint must not reintroduce Half params on load."""
        trainer = agent_trainer.HRMTrainer(dtype_config, device="cpu")
        # Mirror Module.half() semantics: real fp16 exports cast only floating-point
        # tensors, leaving integer/bool buffers (e.g. position ids) untouched
        fp16_state = {
            key: value.half() if value.is_floating_point() else value
            for key, value in trainer.model.state_dict().items()
        }

        trainer.model.load_state_dict(fp16_state)

        _assert_all_fp32(trainer.model)
        with torch.no_grad():
            outputs = trainer.model(torch.randint(0, VOCAB_SIZE, (1, 16)))
        assert torch.isfinite(outputs["logits"]).all()


@pytest.mark.unit
class TestTRMTrainerDtype:
    def test_build_model_upcasts_fp16_backbone_to_fp32(self, dtype_config, fp16_backbone_env):
        trainer = agent_trainer.TRMTrainer(dtype_config, device="cpu")
        _assert_all_fp32(trainer.model)

    def test_cpu_inference_after_fp16_backbone_build(self, dtype_config, fp16_backbone_env):
        trainer = agent_trainer.TRMTrainer(dtype_config, device="cpu")
        trainer.model.eval()

        input_ids = torch.randint(0, VOCAB_SIZE, (1, 16))
        with torch.no_grad():
            outputs = trainer.model(input_ids)

        expected_iters = dtype_config["agents"]["trm"]["max_refinement_iterations"]
        assert outputs["improvement_predictions"].shape == (1, expected_iters)
        assert torch.isfinite(outputs["improvement_predictions"]).all()
