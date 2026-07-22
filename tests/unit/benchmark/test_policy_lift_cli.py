"""Tests for the M5 policy-lift CLI (src.benchmark.policy_lift).

Covers argument parsing, architecture resolution (explicit config, sidecar, MLP shape
inference), the end-to-end run() path on the reasoning smoke-test domain, JSON artifact
structure, and error exit codes.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import torch

from src.benchmark import policy_lift as pl
from src.framework.domain_registry import DomainRegistry
from src.framework.mcts.single_agent_domains import make_reasoning_state
from src.models.policy_value_net import MLPPolicyValueNetwork

pytestmark = [pytest.mark.unit]


def _reasoning_mlp(hidden_dims: list[int] | None = None) -> MLPPolicyValueNetwork:
    state_dim = make_reasoning_state().to_tensor().shape[0]
    action_size = DomainRegistry.action_space_size("reasoning")
    return MLPPolicyValueNetwork(
        state_dim=state_dim,
        action_size=action_size,
        hidden_dims=hidden_dims or [16],
    )


class TestParser:
    def test_defaults(self):
        args = pl.build_parser().parse_args(["--domain", "reasoning", "--checkpoint", "x.pt"])
        assert args.domain == "reasoning"
        assert args.num_games is None  # resolved per metric downstream
        assert args.seed == 0
        assert args.device == "cpu"
        assert args.confidence == pytest.approx(0.95)
        assert args.target_lift == pytest.approx(20.0)
        assert args.baseline_checkpoint is None
        assert args.output is None

    def test_required_args(self, capsys):
        with pytest.raises(SystemExit):
            pl.build_parser().parse_args(["--domain", "reasoning"])

    def test_confidence_choices(self):
        with pytest.raises(SystemExit):
            pl.build_parser().parse_args(["--domain", "reasoning", "--checkpoint", "x.pt", "--confidence", "0.5"])


class TestArchitectureInference:
    def test_infer_mlp_from_state_dict(self):
        network = _reasoning_mlp(hidden_dims=[32, 16])
        arch = pl.infer_mlp_architecture(network.state_dict())
        assert arch["type"] == "mlp"
        assert arch["state_dim"] == network.state_dim
        assert arch["hidden_dims"] == [32, 16]
        assert arch["action_size"] == network.action_size
        assert arch["use_batch_norm"] is True

    def test_infer_rejects_non_mlp_layout(self):
        with pytest.raises(pl.ArchitectureError):
            pl.infer_mlp_architecture({"policy.weight": torch.zeros(4, 8), "value.weight": torch.zeros(1, 8)})

    def test_round_trip_build_and_load(self):
        """An inferred architecture rebuilds a network the checkpoint loads into."""
        network = _reasoning_mlp()
        state_dict = network.state_dict()
        arch = pl.infer_mlp_architecture(state_dict)
        spec = DomainRegistry.get("reasoning")
        rebuilt = pl.build_network(arch, spec, "cpu", state_dict=state_dict)
        rebuilt.load_state_dict(state_dict)  # raises on any mismatch

    def test_action_size_mismatch_is_an_error(self, tmp_path):
        state_dim = make_reasoning_state().to_tensor().shape[0]
        wrong = MLPPolicyValueNetwork(state_dim=state_dim, action_size=3, hidden_dims=[8])
        checkpoint = tmp_path / "wrong.pt"
        torch.save(wrong.state_dict(), checkpoint)
        args = pl.build_parser().parse_args(["--domain", "reasoning", "--checkpoint", str(checkpoint)])
        spec = DomainRegistry.get("reasoning")
        with pytest.raises(pl.ArchitectureError, match="expects"):
            pl.load_architecture(checkpoint, args, spec, wrong.state_dict())

    def test_sidecar_takes_precedence_over_inference(self, tmp_path):
        network = _reasoning_mlp(hidden_dims=[16])
        checkpoint = tmp_path / "net.pt"
        torch.save(network.state_dict(), checkpoint)
        sidecar_arch = {
            "type": "mlp",
            "state_dim": network.state_dim,
            "hidden_dims": [16],
            "action_size": network.action_size,
            "use_batch_norm": True,
            "dropout": 0.1,
        }
        (tmp_path / "net.pt.meta.json").write_text(json.dumps({"schema_version": 1, "network": sidecar_arch}))
        args = pl.build_parser().parse_args(["--domain", "reasoning", "--checkpoint", str(checkpoint)])
        spec = DomainRegistry.get("reasoning")
        arch = pl.load_architecture(checkpoint, args, spec, network.state_dict())
        assert arch == sidecar_arch

    def test_explicit_network_config_wins(self, tmp_path):
        config_path = tmp_path / "arch.json"
        config_path.write_text(json.dumps({"type": "mlp", "state_dim": 7, "hidden_dims": [4]}))
        args = pl.build_parser().parse_args(
            ["--domain", "reasoning", "--checkpoint", "x.pt", "--network-config", str(config_path)]
        )
        spec = DomainRegistry.get("reasoning")
        arch = pl.load_architecture(tmp_path / "x.pt", args, spec, {})
        assert arch == {"type": "mlp", "state_dim": 7, "hidden_dims": [4]}

    def test_unknown_arch_type_is_an_error(self):
        spec = DomainRegistry.get("reasoning")
        with pytest.raises(pl.ArchitectureError, match="Unknown network architecture"):
            pl.build_network({"type": "transformer"}, spec, "cpu")


class TestRunEndToEnd:
    def _write_checkpoint(self, tmp_path, name="trained.pt"):
        network = _reasoning_mlp()
        checkpoint = tmp_path / name
        torch.save(network.state_dict(), checkpoint)
        return checkpoint

    def test_run_produces_artifact_and_exit_code(self, tmp_path):
        checkpoint = self._write_checkpoint(tmp_path)
        output = tmp_path / "out" / "lift.json"
        args = pl.build_parser().parse_args(
            [
                "--domain",
                "reasoning",
                "--checkpoint",
                str(checkpoint),
                "--num-games",
                "2",
                "--num-simulations",
                "4",
                "--max-moves",
                "5",
                "--seed",
                "7",
                "--output",
                str(output),
            ]
        )
        exit_code = asyncio.run(pl.run(args))
        assert exit_code in (pl.EXIT_GATE_MET, pl.EXIT_GATE_NOT_MET)

        artifact = json.loads(output.read_text())
        for key in (
            "domain",
            "metric",
            "baseline_score",
            "trained_score",
            "lift_pct",
            "lift_ci_lower_pct",
            "lift_ci_upper_pct",
            "confidence",
            "num_games",
            "meets_target",
            "point_meets_target",
            "run",
        ):
            assert key in artifact, f"missing artifact key: {key}"
        assert artifact["domain"] == "reasoning"
        assert artifact["num_games"] == 2
        assert artifact["run"]["seed"] == 7
        assert artifact["run"]["num_simulations"] == 4
        assert artifact["run"]["checkpoint"] == str(checkpoint)
        assert artifact["run"]["network"]["type"] == "mlp"
        # Identical untrained-vs-trained-ish tiny nets cannot credibly clear +20%.
        assert artifact["meets_target"] == (exit_code == pl.EXIT_GATE_MET)

    def test_run_with_explicit_baseline_checkpoint(self, tmp_path):
        trained = self._write_checkpoint(tmp_path, "trained.pt")
        baseline = self._write_checkpoint(tmp_path, "baseline.pt")
        args = pl.build_parser().parse_args(
            [
                "--domain",
                "reasoning",
                "--checkpoint",
                str(trained),
                "--baseline-checkpoint",
                str(baseline),
                "--num-games",
                "1",
                "--num-simulations",
                "2",
                "--max-moves",
                "3",
            ]
        )
        assert asyncio.run(pl.run(args)) in (pl.EXIT_GATE_MET, pl.EXIT_GATE_NOT_MET)

    def test_unknown_domain_exits_2(self, tmp_path, capsys):
        checkpoint = self._write_checkpoint(tmp_path)
        args = pl.build_parser().parse_args(["--domain", "nonexistent", "--checkpoint", str(checkpoint)])
        assert asyncio.run(pl.run(args)) == pl.EXIT_ERROR
        assert "Unknown domain" in capsys.readouterr().err

    def test_missing_checkpoint_exits_2(self, tmp_path, capsys):
        args = pl.build_parser().parse_args(["--domain", "reasoning", "--checkpoint", str(tmp_path / "missing.pt")])
        assert asyncio.run(pl.run(args)) == pl.EXIT_ERROR
        assert "not found" in capsys.readouterr().err

    def test_non_state_dict_checkpoint_exits_2(self, tmp_path, capsys):
        checkpoint = tmp_path / "tensor.pt"
        torch.save(torch.zeros(3), checkpoint)
        args = pl.build_parser().parse_args(["--domain", "reasoning", "--checkpoint", str(checkpoint)])
        assert asyncio.run(pl.run(args)) == pl.EXIT_ERROR
        assert "state_dict" in capsys.readouterr().err


class TestCheckpointSidecarWriter:
    def test_save_checkpoint_writes_sidecar_when_metadata_passed(self, tmp_path):
        from src.training.self_play_trainer import SelfPlayTrainer

        network = _reasoning_mlp()
        trainer = SelfPlayTrainer(
            network,
            make_reasoning_state,
            DomainRegistry.action_space_size("reasoning"),
            single_agent=True,
            seed=0,
        )
        checkpoint = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(checkpoint, metadata={"network": {"type": "mlp"}, "domain": "reasoning"})
        sidecar = tmp_path / "ckpt.pt.meta.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["schema_version"] == 1
        assert meta["network"] == {"type": "mlp"}
        assert meta["domain"] == "reasoning"

    def test_save_checkpoint_without_metadata_writes_no_sidecar(self, tmp_path):
        from src.training.self_play_trainer import SelfPlayTrainer

        network = _reasoning_mlp()
        trainer = SelfPlayTrainer(
            network,
            make_reasoning_state,
            DomainRegistry.action_space_size("reasoning"),
            single_agent=True,
            seed=0,
        )
        checkpoint = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(checkpoint)
        assert not (tmp_path / "ckpt.pt.meta.json").exists()


class TestValidationAndErrorPaths:
    def test_hidden_dims_without_input_dim_exits_2(self, capsys):
        args = pl.build_parser().parse_args(
            ["--domain", "reasoning", "--checkpoint", "x.pt", "--hidden-dims", "32", "16"]
        )
        assert asyncio.run(pl.run(args)) == pl.EXIT_ERROR
        assert "--hidden-dims requires --input-dim" in capsys.readouterr().err

    def test_num_games_zero_exits_2(self, tmp_path, capsys):
        network = _reasoning_mlp()
        checkpoint = tmp_path / "net.pt"
        torch.save(network.state_dict(), checkpoint)
        args = pl.build_parser().parse_args(
            ["--domain", "reasoning", "--checkpoint", str(checkpoint), "--num-games", "0"]
        )
        assert asyncio.run(pl.run(args)) == pl.EXIT_ERROR
        assert "num_games must be greater than 0" in capsys.readouterr().err

    def test_unwritable_output_exits_2(self, tmp_path, capsys):
        network = _reasoning_mlp()
        checkpoint = tmp_path / "net.pt"
        torch.save(network.state_dict(), checkpoint)
        blocked = tmp_path / "blocked"
        blocked.mkdir()  # output path is a directory -> OSError on write_text
        args = pl.build_parser().parse_args(
            [
                "--domain",
                "reasoning",
                "--checkpoint",
                str(checkpoint),
                "--num-games",
                "1",
                "--num-simulations",
                "2",
                "--max-moves",
                "3",
                "--output",
                str(blocked),
            ]
        )
        assert asyncio.run(pl.run(args)) == pl.EXIT_ERROR
        captured = capsys.readouterr()
        assert "could not write artifact" in captured.err
        assert '"lift_pct"' in captured.out  # the result is still printed to stdout

    def test_network_config_must_be_a_json_object(self, tmp_path):
        config_path = tmp_path / "arch.json"
        config_path.write_text(json.dumps(["mlp", 7]))
        args = pl.build_parser().parse_args(
            ["--domain", "reasoning", "--checkpoint", "x.pt", "--network-config", str(config_path)]
        )
        spec = DomainRegistry.get("reasoning")
        with pytest.raises(pl.ArchitectureError, match="JSON object"):
            pl.load_architecture(tmp_path / "x.pt", args, spec, {})

    def test_main_exits_with_error_code(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["policy-lift", "--domain", "nonexistent_domain", "--checkpoint", "x.pt"])
        with pytest.raises(SystemExit) as excinfo:
            pl.main()
        assert excinfo.value.code == pl.EXIT_ERROR


class TestArchitectureResolutionFallbacks:
    def test_input_dim_cli_args_build_mlp_arch(self, tmp_path):
        args = pl.build_parser().parse_args(
            ["--domain", "reasoning", "--checkpoint", "x.pt", "--input-dim", "12", "--hidden-dims", "8", "4"]
        )
        spec = DomainRegistry.get("reasoning")
        arch = pl.load_architecture(tmp_path / "x.pt", args, spec, {})
        assert arch == {
            "type": "mlp",
            "state_dim": 12,
            "hidden_dims": [8, 4],
            "action_size": spec.action_space_size,
        }

    def test_sidecar_without_network_key_falls_back_to_inference(self, tmp_path):
        network = _reasoning_mlp()
        checkpoint = tmp_path / "net.pt"
        torch.save(network.state_dict(), checkpoint)
        (tmp_path / "net.pt.meta.json").write_text(json.dumps({"schema_version": 1, "domain": "reasoning"}))
        args = pl.build_parser().parse_args(["--domain", "reasoning", "--checkpoint", str(checkpoint)])
        spec = DomainRegistry.get("reasoning")
        arch = pl.load_architecture(checkpoint, args, spec, network.state_dict())
        assert arch["type"] == "mlp"  # inferred from the state_dict, not the sidecar
        assert arch["state_dim"] == network.state_dim

    def test_win_rate_domain_falls_back_to_conv_defaults(self):
        from src.framework.domain_registry import DomainSpec

        spec = DomainSpec(
            name="fake_board_domain",
            initial_state_fn=make_reasoning_state,
            action_space_size=4672,
            single_agent=False,
            metric="win_rate",
        )
        arch = pl.chess_default_architecture(spec)
        assert arch["type"] == "resnet"
        assert arch["action_size"] == 4672
        assert arch["input_channels"] > 0
        assert arch["board_size"] == 8

    def test_load_architecture_uses_conv_defaults_for_win_rate_non_mlp_checkpoint(self, tmp_path):
        from src.framework.domain_registry import DomainSpec

        spec = DomainSpec(
            name="fake_board_domain",
            initial_state_fn=make_reasoning_state,
            action_space_size=4672,
            single_agent=False,
            metric="win_rate",
        )
        args = pl.build_parser().parse_args(["--domain", "fake_board_domain", "--checkpoint", "x.pt"])
        non_mlp_state = {"conv_input.0.weight": torch.zeros(8, 22, 3, 3)}
        arch = pl.load_architecture(tmp_path / "x.pt", args, spec, non_mlp_state)
        assert arch["type"] == "resnet"

    def test_build_network_resnet_round_trip(self):
        from src.framework.domain_registry import DomainSpec

        spec = DomainSpec(
            name="fake_board_domain",
            initial_state_fn=make_reasoning_state,
            action_space_size=16,
            single_agent=False,
            metric="win_rate",
        )
        arch = {"type": "resnet", "num_res_blocks": 1, "num_channels": 4, "input_channels": 3, "board_size": 4}
        network = pl.build_network(arch, spec, "cpu")
        policy, value = network(torch.zeros(1, 3, 4, 4))
        assert policy.shape == (1, 16)  # action_size defaulted from the spec
        assert value.shape == (1, 1)


class TestDropoutLayoutResolution:
    def test_retry_resolves_dropout_free_checkpoints(self):
        """Candidates are tried in order until the state_dict key set matches."""
        state_dim = make_reasoning_state().to_tensor().shape[0]
        network = MLPPolicyValueNetwork(state_dim=state_dim, action_size=9, hidden_dims=[8, 8], dropout=0.0)
        state_dict = network.state_dict()
        arch = {"type": "mlp", "state_dim": state_dim, "hidden_dims": [8, 8], "action_size": 9}
        spec = DomainRegistry.get("reasoning")
        rebuilt = pl.build_network(arch, spec, "cpu", state_dict=state_dict)
        rebuilt.load_state_dict(state_dict)  # raises on any layout mismatch

    def test_explicit_wrong_dropout_is_an_error(self):
        state_dim = make_reasoning_state().to_tensor().shape[0]
        network = MLPPolicyValueNetwork(state_dim=state_dim, action_size=9, hidden_dims=[8, 8], dropout=0.0)
        arch = {
            "type": "mlp",
            "state_dim": state_dim,
            "hidden_dims": [8, 8],
            "action_size": 9,
            "dropout": 0.1,  # wrong: shifts the Sequential indices vs the checkpoint
        }
        spec = DomainRegistry.get("reasoning")
        with pytest.raises(pl.ArchitectureError, match="layout mismatch"):
            pl.build_network(arch, spec, "cpu", state_dict=network.state_dict())


class TestReviewHardening:
    """Regression tests for review findings: sidecar shape, state_dict validation,
    schema_version protection, and baseline architecture mismatch."""

    def test_non_object_sidecar_json_falls_back_cleanly(self, tmp_path):
        """A sidecar whose JSON is a list must fall back to inference, not crash."""
        network = _reasoning_mlp()
        checkpoint = tmp_path / "net.pt"
        torch.save(network.state_dict(), checkpoint)
        (tmp_path / "net.pt.meta.json").write_text(json.dumps(["not", "an", "object"]))
        args = pl.build_parser().parse_args(["--domain", "reasoning", "--checkpoint", str(checkpoint)])
        spec = DomainRegistry.get("reasoning")
        arch = pl.load_architecture(checkpoint, args, spec, network.state_dict())
        assert arch["type"] == "mlp"  # inferred, no AttributeError

    def test_non_tensor_state_dict_values_exit_2(self, tmp_path, capsys):
        checkpoint = tmp_path / "bogus.pt"
        torch.save({"weights": [1, 2, 3], "config": "not a tensor"}, checkpoint)
        args = pl.build_parser().parse_args(["--domain", "reasoning", "--checkpoint", str(checkpoint)])
        assert asyncio.run(pl.run(args)) == pl.EXIT_ERROR
        assert "state_dict of tensors" in capsys.readouterr().err

    def test_baseline_architecture_mismatch_exits_2(self, tmp_path, capsys):
        """A baseline checkpoint with a different layout must fail fast, not silently
        compare mismatched networks."""
        trained = _reasoning_mlp(hidden_dims=[16])
        mismatched = _reasoning_mlp(hidden_dims=[32, 16])
        trained_path = tmp_path / "trained.pt"
        baseline_path = tmp_path / "baseline.pt"
        torch.save(trained.state_dict(), trained_path)
        torch.save(mismatched.state_dict(), baseline_path)
        args = pl.build_parser().parse_args(
            [
                "--domain",
                "reasoning",
                "--checkpoint",
                str(trained_path),
                "--baseline-checkpoint",
                str(baseline_path),
                "--num-games",
                "1",
                "--num-simulations",
                "2",
            ]
        )
        assert asyncio.run(pl.run(args)) == pl.EXIT_ERROR
        assert "does not match the trained" in capsys.readouterr().err

    def test_sidecar_schema_version_cannot_be_overridden(self, tmp_path):
        from src.training.self_play_trainer import SelfPlayTrainer

        trainer = SelfPlayTrainer(
            _reasoning_mlp(),
            make_reasoning_state,
            DomainRegistry.action_space_size("reasoning"),
            single_agent=True,
            seed=0,
        )
        checkpoint = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(checkpoint, metadata={"schema_version": 999, "domain": "reasoning"})
        meta = json.loads((tmp_path / "ckpt.pt.meta.json").read_text())
        assert meta["schema_version"] == 1  # fixed key wins over caller metadata
        assert meta["domain"] == "reasoning"
