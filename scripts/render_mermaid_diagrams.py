#!/usr/bin/env python3
"""Render the Mermaid diagrams embedded in the architecture docs to image files.

Extracts every ```mermaid fenced block from the given Markdown docs and renders each to
an SVG under ``docs/diagrams/`` using ``@mermaid-js/mermaid-cli`` (``mmdc``).

Reproducible / no hardcoded paths:
- ``mmdc`` is located on PATH, else via ``MMDC_BIN``, else ``npx mmdc``.
- The Chromium executable is discovered under ``PLAYWRIGHT_BROWSERS_PATH`` (falls back to
  whatever Puppeteer bundles) and written to a temporary puppeteer config.

Usage:
    npm install -g @mermaid-js/mermaid-cli   # or rely on npx
    python scripts/render_mermaid_diagrams.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "diagrams"
SOURCE_DOCS = [
    REPO_ROOT / "docs" / "C4_ARCHITECTURE.md",
    REPO_ROOT / "docs" / "C4_MERMAID_ARCHITECTURE.md",
]
_MERMAID_BLOCK = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)


def _find_chromium() -> str | None:
    """Locate a Chromium/Chrome executable under PLAYWRIGHT_BROWSERS_PATH, if any."""
    base = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if not base or not Path(base).is_dir():
        return None
    for name in ("chrome", "headless_shell"):
        for path in Path(base).rglob(name):
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
    return None


def _mmdc_command() -> list[str]:
    """Resolve how to invoke mermaid-cli (PATH > MMDC_BIN > npx)."""
    if shutil.which("mmdc"):
        return ["mmdc"]
    env_bin = os.environ.get("MMDC_BIN")
    if env_bin and Path(env_bin).exists():
        return [env_bin]
    return ["npx", "--no-install", "mmdc"]


def _puppeteer_config(tmp: Path) -> Path | None:
    """Write a puppeteer config pointing at the discovered Chromium, if found."""
    chromium = _find_chromium()
    if not chromium:
        return None
    cfg = tmp / "puppeteer.json"
    cfg.write_text(
        json.dumps({"executablePath": chromium, "args": ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]}),
        encoding="utf-8",
    )
    return cfg


def render() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mmdc = _mmdc_command()
    rendered: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        puppeteer = _puppeteer_config(tmp)

        for doc in SOURCE_DOCS:
            if not doc.exists():
                print(f"skip (missing): {doc}")
                continue
            blocks = _MERMAID_BLOCK.findall(doc.read_text(encoding="utf-8"))
            for idx, block in enumerate(blocks, start=1):
                stem = f"{doc.stem.lower()}-{idx:02d}"
                src = tmp / f"{stem}.mmd"
                src.write_text(block.strip() + "\n", encoding="utf-8")
                out = OUTPUT_DIR / f"{stem}.svg"
                cmd = [*mmdc, "-i", str(src), "-o", str(out), "-b", "transparent"]
                if puppeteer:
                    cmd += ["-p", str(puppeteer)]
                subprocess.run(cmd, check=True, cwd=REPO_ROOT)
                rendered.append(out.name)
                print(f"rendered {out.relative_to(REPO_ROOT)}")

    if not rendered:
        print("No mermaid blocks found.", file=sys.stderr)
        return 1
    print(f"\nDone: {len(rendered)} diagram(s) -> {OUTPUT_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(render())
