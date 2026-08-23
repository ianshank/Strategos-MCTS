"""Shared scaffolding for tests that build a synthetic repo tree around the claim ledger.

Spec: ``specs/evidence_claim_ledger.SPEC.md``.

The claim-surface ratchet fails closed: a synthetic tree with no baseline is *invalid*, not
"trivially clean". That is deliberate, but it means every test tree needs a baseline, and more than
one test module needs one — hence a single helper here rather than a copy per module.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config.constants import (
    CLAIM_SURFACE_BASELINE_RELATIVE_PATH,
    CLAIM_SURFACE_BASELINE_SCHEMA_VERSION,
    CLAIM_SURFACE_KEY_BULLETS,
    CLAIM_SURFACE_KEY_PATH,
    CLAIM_SURFACE_KEY_ROWS,
    CLAIM_SURFACE_KEY_SECTION,
    CLAIM_SURFACE_KEY_SURFACES,
    CLAIM_SURFACE_KEY_SURPLUS,
)

__all__ = ["write_claim_surface_baseline"]


def write_claim_surface_baseline(
    tree: Path,
    *,
    path: str = "README.md",
    section: str = "",
    surplus: int = 0,
    bullets: int = 0,
    rows: int = 0,
    schema_version: int = CLAIM_SURFACE_BASELINE_SCHEMA_VERSION,
) -> Path:
    """Write a one-surface claim-surface baseline into ``tree`` and return its path.

    Every field is a parameter so a test can falsify exactly one property (an unsupported
    ``schema_version``, a stale ``surplus``, a surface that does not exist) without hand-rolling
    JSON. Defaults describe the smallest passing case: a surface with no claim-shaped bullets, whose
    surplus is therefore zero whatever the ledger says.
    """
    target = tree / CLAIM_SURFACE_BASELINE_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                CLAIM_SURFACE_KEY_SURFACES: [
                    {
                        CLAIM_SURFACE_KEY_PATH: path,
                        CLAIM_SURFACE_KEY_SECTION: section,
                        CLAIM_SURFACE_KEY_BULLETS: bullets,
                        CLAIM_SURFACE_KEY_ROWS: rows,
                        CLAIM_SURFACE_KEY_SURPLUS: surplus,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return target
