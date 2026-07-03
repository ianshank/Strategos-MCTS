---
id: phase_8_repo_organization
goal: Reorganize the repository into an enterprise-standard file/directory layout
module: docs/
phase: "8"
milestone: M5
status: implemented
---

# Goal

Reduce root clutter and remove ambiguity in the repository layout: consolidate ~40 loose root markdown
files into `docs/`, stop tracking model binaries, collapse overlapping demo trees, and clarify the
`config/` (runtime) vs `src/config/` (code) and `training/` vs `src/training/` split. All moves use
`git mv`, sequenced by risk, each followed by a reference grep and the full local gate.

# Acceptance Criteria

- AC-1: Non-canonical root `.md` files are moved into `docs/` subfolders (`reports/`, `summaries/`,
  `plans/`, `quickstart/`); only `README.md`, `CLAUDE.md`, `AGENTS.md`, `CHANGELOG.md`, `ATTRIBUTION.md`,
  `PROJECT_STRUCTURE.md`, and the three template files remain at root, with internal references fixed. Root
  markdown went from 45 to 9 files.
- AC-2: **Layout ambiguity is removed by authoritative documentation** in `PROJECT_STRUCTURE.md` — a
  "Layout & naming disambiguation" section explains `config/` vs `src/config/`, `training/` vs
  `src/training/`, the demo/example entry points, and the tracked `models/` artifacts, including the import
  and build-context contracts that make them distinct.
- AC-3: The physical moves originally proposed (untrack `models/`, merge the demo trees, rename `config/`,
  relocate `training/tests/`) are **deliberately not performed** because inspection showed each would break
  a real contract (documented in the Constraints below). The goal — an unambiguous, discoverable layout —
  is met without the breakage.
- AC-4: The full local gate stays green; `docs/STATUS.md` / `PROJECT_STRUCTURE.md` reflect the current layout.

# Constraints (why the physical moves were rejected)

- `models/*.pt` + LoRA adapter (~0.4 MB) are consumed by `tests/integration/test_deployed_models.py`;
  untracking them breaks fresh-clone/CI tests and there is no retrieval mechanism — keep tracked.
- `demo_src/` is imported as a bare top-level package by `huggingface_space/app_mock.py` and
  `scripts/run_e2e_workflow.py`; `demo.py` is import-tested by `tests/unit/test_llm_mcts.py`; `examples/`
  is intentionally kept without a root `__init__.py`. Merging these breaks those import contracts.
- `config/` is `COPY`-ed by `Dockerfile`; renaming needs Docker + loader changes for no functional gain.
- Backward compatible; no hardcoded values; full local gate green before push.
