#!/usr/bin/env python3
"""Thin shim for the context-doc validator.

The engine lives in :mod:`src.tools.context_docs` (importable, type-checked, coverage-gated). This
wrapper is kept so ``python scripts/validate_context_docs.py`` — referenced in ``AGENTS.md`` and the
``validate-context`` skill — keeps working. Prefer the ``validate-context-docs`` console script or
``python -m src.tools.context_docs``.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.context_docs import main  # noqa: E402  (path set up above)

if __name__ == "__main__":
    raise SystemExit(main())
