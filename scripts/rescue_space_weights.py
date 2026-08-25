#!/usr/bin/env python3
"""
Rescue the trained meta-controller weights stranded in the demo Space.

The Space at ``ianshank/langgraph-mcts-demo`` served a vendored copy of ``src/``
that this repository deleted for silently diverging (see ``CHANGELOG.md``). That
Space repo is the only remaining home of the real RNN and BERT-LoRA weights —
this tree carries them as Git-LFS pointer stubs. Redeploying the Space
force-pushes over that history, so the weights must be rescued to a model
repository *first*.

The point of this script is the verification, not the copy: it loads each
checkpoint through the *current* controller code paths and runs a forward pass,
and refuses to publish anything it cannot load. Publishing weights that load
"successfully" into a mismatched architecture is exactly the unreproducible
claim CHARTER NG-3 forbids, and for the RNN it is worse than cosmetic —
``app.py`` calls ``load_state_dict`` unwrapped, so shape-mismatched weights
crash every query instead of degrading to the reduced-mode banner.

Usage::

    # 1. download + verify (no token needed; the source Space is public)
    python scripts/rescue_space_weights.py --verify

    # 2. publish only after step 1 is green (needs a write-scoped HF token)
    HF_TOKEN=hf_... python scripts/rescue_space_weights.py --upload

    # 3. prove the published copy round-trips
    python scripts/rescue_space_weights.py --verify --from-repo

Verification constructs ``BERTMetaController``, which resolves settings, so the
process needs a provider selection that constructs without secrets. The script
applies the same keyless recipe the Space uses (``LLM_PROVIDER=lmstudio`` with a
dead local URL) unless the caller has already chosen one.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

# Keyless-safe provider selection must be in place before any src.config.settings
# import runs, so it precedes the first-party imports below (and is why this
# module carries a module-level statement ahead of them).
os.environ.setdefault("LLM_PROVIDER", "lmstudio")
os.environ.setdefault("LMSTUDIO_BASE_URL", "http://127.0.0.1:9/v1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_SPACE = "ianshank/langgraph-mcts-demo"
TARGET_MODEL_REPO = os.environ.get("STRATEGOS_CHECKPOINT_REPO", "ianshank/strategos-mcts-checkpoints")

# Source path in the Space -> destination path in the model repo. The destination
# layout mirrors src.ui.status.DEFAULT_CHECKPOINTS (minus the leading ``models/``)
# so a snapshot download into ``models/`` lands where the app already looks.
RESCUE_FILES: dict[str, str] = {
    "models/rnn_meta_controller.pt": "rnn_meta_controller.pt",
    "models/bert_lora/final_model/adapter_config.json": "bert_lora/final_model/adapter_config.json",
    "models/bert_lora/final_model/adapter_model.safetensors": "bert_lora/final_model/adapter_model.safetensors",
}

# Deliberately NOT rescued: models/bert_lora/training_results.json and the
# fork-era model README. They carry training metrics produced by the deleted
# fork that no command in this tree reproduces (CHARTER NG-3).

MODEL_CARD = """---
license: mit
library_name: pytorch
tags:
  - multi-agent
  - meta-controller
  - lora
---

# Strategos-MCTS meta-controller checkpoints

Checkpoints for the meta-controllers in
[Strategos-MCTS](https://github.com/ianshank/Strategos-MCTS): a GRU routing model
and a LoRA adapter over `prajjwal1/bert-mini`, both selecting among three agents
(HRM / TRM / MCTS).

## Provenance

These files were rescued from the repository's earlier demo Space, which served a
vendored copy of `src/` that the project deleted for silently diverging from the
canonical tree. That Space was the only remaining copy; the checkpoints in the Git
repository are Git-LFS pointer stubs.

**Nothing about training is claimed here.** The fork's training-results artifacts
were deliberately not republished, because no command in the current tree
reproduces their numbers. The single claim made by this repository is
load-compatibility, and it is reproducible:

```bash
git clone https://github.com/ianshank/Strategos-MCTS && cd Strategos-MCTS
pip install -e ".[neural]"
python scripts/rescue_space_weights.py --verify --from-repo
```

That command loads both checkpoints through the current controller code paths,
asserts the RNN state-dict contract, and runs a forward pass through each.

## Layout

| File | Loaded by |
|---|---|
| `rnn_meta_controller.pt` | `RNNMetaController` (bare `state_dict`) |
| `bert_lora/final_model/` | `BERTMetaController.load_model()` (PEFT adapter) |

## License

MIT, matching the source repository.
"""


def _sample_features():
    """One representative feature vector, shared by both forward-pass checks."""
    from src.agents.meta_controller.base import MetaControllerFeatures

    return MetaControllerFeatures(
        hrm_confidence=0.72,
        trm_confidence=0.58,
        mcts_value=0.65,
        consensus_score=0.61,
        last_agent="hrm",
        iteration=2,
        query_length=64,
        has_rag_context=False,
        rag_relevance_score=0.0,
        is_technical_query=True,
    )


def _expected_rnn_state(device: str) -> dict[str, tuple[int, ...]]:
    """Shapes the current RNNMetaControllerModel expects, read from the model itself."""
    from src.agents.meta_controller.rnn_controller import RNNMetaController

    controller = RNNMetaController(name="contract", seed=42, device=device)
    return {key: tuple(value.shape) for key, value in controller.model.state_dict().items()}


def _download(logger: logging.Logger, dest: Path, from_repo: bool) -> dict[str, Path]:
    """Fetch the checkpoint set from the source Space, or from the published model repo."""
    from huggingface_hub import hf_hub_download

    repo_id = TARGET_MODEL_REPO if from_repo else SOURCE_SPACE
    repo_type = "model" if from_repo else "space"
    local: dict[str, Path] = {}
    for source_path, target_path in RESCUE_FILES.items():
        remote = target_path if from_repo else source_path
        logger.info("downloading %s from %s (%s)", remote, repo_id, repo_type)
        got = hf_hub_download(repo_id=repo_id, filename=remote, repo_type=repo_type)
        staged = dest / target_path
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(Path(got).read_bytes())
        local[target_path] = staged
    return local


def _verify_rnn(logger: logging.Logger, path: Path, device: str) -> None:
    """Load the RNN checkpoint exactly as app.py does; raise if the contract is broken."""
    import torch

    from src.agents.meta_controller.rnn_controller import RNNMetaController

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    expected = _expected_rnn_state(device)
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"RNN checkpoint is {type(checkpoint).__name__}, not a state_dict — refusing to publish")

    # A wrapped checkpoint ({"model_state_dict": ...}) is recoverable, but app.py
    # passes whatever it loads straight to load_state_dict, so normalise here and
    # publish the bare form rather than shipping something the app cannot use.
    if set(checkpoint) != set(expected):
        for wrapper in ("model_state_dict", "state_dict"):
            inner = checkpoint.get(wrapper)
            if isinstance(inner, dict) and set(inner) == set(expected):
                logger.warning("unwrapping RNN checkpoint from %r and re-saving as a bare state_dict", wrapper)
                checkpoint = inner
                torch.save(checkpoint, path)
                break

    actual = {key: tuple(value.shape) for key, value in checkpoint.items()}
    if actual != expected:
        raise SystemExit(
            "RNN state_dict does not match the current architecture — refusing to publish.\n"
            f"  expected: {expected}\n  actual:   {actual}"
        )

    controller = RNNMetaController(name="verify", seed=42, device=device)
    controller.model.load_state_dict(checkpoint)
    controller.model.eval()

    prediction = controller.predict(_sample_features())
    logger.info(
        "RNN forward pass OK: agent=%s confidence=%.4f probabilities=%s",
        prediction.agent,
        prediction.confidence,
        prediction.probabilities,
    )


def _verify_bert(logger: logging.Logger, adapter_dir: Path, device: str) -> None:
    """Load the LoRA adapter through BERTMetaController and run one forward pass."""
    from src.agents.meta_controller.bert_controller import BERTMetaController

    controller = BERTMetaController(name="verify", seed=42, device=device, use_lora=True)
    controller.load_model(str(adapter_dir))

    prediction = controller.predict(_sample_features())
    logger.info(
        "BERT forward pass OK: agent=%s confidence=%.4f probabilities=%s",
        prediction.agent,
        prediction.confidence,
        prediction.probabilities,
    )


def _upload(logger: logging.Logger, staged: Path) -> None:
    """Publish the verified checkpoint set plus a provenance-bounded model card."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required to upload (write scope on the target model repo)")

    (staged / "README.md").write_text(MODEL_CARD, encoding="utf-8")

    api = HfApi(token=token)
    api.create_repo(repo_id=TARGET_MODEL_REPO, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=TARGET_MODEL_REPO,
        repo_type="model",
        folder_path=str(staged),
        commit_message="Rescue meta-controller checkpoints from the demo Space",
    )
    logger.info("published to https://huggingface.co/%s", TARGET_MODEL_REPO)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="download and load-verify the checkpoints")
    parser.add_argument("--upload", action="store_true", help="publish verified checkpoints (implies --verify)")
    parser.add_argument(
        "--from-repo",
        action="store_true",
        help="verify against the published model repo instead of the source Space",
    )
    parser.add_argument("--device", default="cpu", help="torch device for verification (default: cpu)")
    parser.add_argument(
        "--staging-dir",
        default="",
        help="directory for downloaded checkpoints (default: a temporary directory)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("rescue")

    if not (args.verify or args.upload):
        parser.error("nothing to do: pass --verify and/or --upload")

    if args.staging_dir:
        staged = Path(args.staging_dir)
        staged.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile

        staged = Path(tempfile.mkdtemp(prefix="strategos-rescue-"))
    logger.info("staging directory: %s", staged)

    local = _download(logger, staged, from_repo=args.from_repo)

    _verify_rnn(logger, local["rnn_meta_controller.pt"], args.device)
    _verify_bert(logger, staged / "bert_lora" / "final_model", args.device)
    logger.info("verification passed: both checkpoints load through the current code paths")

    if args.upload:
        _upload(logger, staged)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
