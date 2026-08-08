"""
Hugging Face Space entrypoint: fetch checkpoints, pick a provider, start the demo.

This sits at the repository root, next to ``app.py``, for the reason
``specs/ui_runtime_integrity.SPEC.md`` gives for ``app.py`` itself: deployment
glue belongs outside ``src/``, where the coverage gate measures behaviour that
matters. It contains no logic of its own beyond wiring — every decision it makes
is either an environment variable or a call into already-measured code.

Two jobs, neither of which the app can do for itself:

1. **Checkpoints.** The Git repository carries the meta-controller weights as
   Git-LFS pointer stubs, so a Space cloned from it has no usable models. The
   real weights live in a Hub model repository; they are fetched here, into the
   layout ``src.ui.status.DEFAULT_CHECKPOINTS`` already expects. Failure is not
   fatal — the app's runtime banner reports honestly on what actually loaded.

2. **Provider selection.** ``app.py`` resolves settings eagerly when it builds
   the UI, and settings refuse to construct when the selected provider has no
   credentials. With no key configured the container therefore needs a provider
   that constructs without secrets; the repository already uses ``lmstudio`` for
   exactly this in ``Dockerfile``. Pointed at a dead local address it makes no
   external calls and every answer degrades through the app's labelled path.

Setting ``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY`` as a Space secret is all it
takes to switch to real inference on the next restart.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
CHECKPOINT_REPO = os.environ.get("STRATEGOS_CHECKPOINT_REPO", "ianshank/strategos-mcts-checkpoints")

# Dead by design: a closed port on loopback. It keeps Settings constructible with
# zero secrets while guaranteeing no request ever leaves the container.
UNREACHABLE_LMSTUDIO_URL = "http://127.0.0.1:9/v1"

logger = logging.getLogger("space_bootstrap")


def fetch_checkpoints() -> None:
    """Download the meta-controller weights, tolerating any failure."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=CHECKPOINT_REPO, local_dir=str(MODELS_DIR))
        logger.info("checkpoints downloaded from %s into %s", CHECKPOINT_REPO, MODELS_DIR)
    except Exception as exc:  # noqa: BLE001 - hub errors are many; none should stop the demo
        logger.warning(
            "checkpoint download from %s failed (%s); the UI will report reduced mode",
            CHECKPOINT_REPO,
            exc,
        )


def select_provider() -> None:
    """Choose an LLM provider that settings can construct, and say why."""
    if os.environ.get("LLM_PROVIDER"):
        logger.info("LLM_PROVIDER=%s set explicitly; leaving it alone", os.environ["LLM_PROVIDER"])
        return

    if os.environ.get("OPENAI_API_KEY"):
        os.environ["LLM_PROVIDER"] = "openai"
        logger.info("OPENAI_API_KEY present; using the openai provider")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["LLM_PROVIDER"] = "anthropic"
        logger.info("ANTHROPIC_API_KEY present; using the anthropic provider")
    else:
        os.environ["LLM_PROVIDER"] = "lmstudio"
        os.environ.setdefault("LMSTUDIO_BASE_URL", UNREACHABLE_LMSTUDIO_URL)
        logger.info(
            "no provider key configured; selecting lmstudio at %s so settings construct without "
            "secrets. LLM-synthesised answers will carry the degraded-mode label. What the "
            "meta-controllers are running on is reported by the app's own checkpoint banner, "
            "not asserted here.",
            os.environ["LMSTUDIO_BASE_URL"],
        )


def main() -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s %(message)s")
    fetch_checkpoints()
    select_provider()

    # Hand off to app.py's own __main__, which passes the server_name the Space
    # needs (Hugging Face does not set Gradio's env vars, and Gradio's default
    # bind address would fail the port check).
    os.execv(sys.executable, [sys.executable, str(APP_DIR / "app.py")])


if __name__ == "__main__":
    raise SystemExit(main())
