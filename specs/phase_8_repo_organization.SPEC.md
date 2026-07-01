---
goal: Reorganize the repository into an enterprise-standard file/directory layout
phase: "8"
milestone: M5
status: active
---

# Goal

Reduce root clutter and remove ambiguity in the repository layout: consolidate ~40 loose root markdown
files into `docs/`, stop tracking model binaries, collapse overlapping demo trees, and clarify the
`config/` (runtime) vs `src/config/` (code) and `training/` vs `src/training/` split. All moves use
`git mv`, sequenced by risk, each followed by a reference grep and the full local gate.

# Acceptance Criteria

- Non-canonical root `.md` files are moved into `docs/` subfolders (`reports/`, `summaries/`, `plans/`,
  `templates/`, `quickstart/`); only `README.md`, `CLAUDE.md`, `AGENTS.md`, `CHANGELOG.md`,
  `ATTRIBUTION.md`, `PROJECT_STRUCTURE.md` remain at root, and internal links to moved files are fixed.
- Committed model binaries under `models/` are untracked (`git rm --cached`) and covered by `.gitignore`,
  with a documented retrieval path; git history is not rewritten in this pass.
- The demo entry points (`demo.py`, `chess_demo.py`, `demo_src/`, `demos/`) are consolidated into a single
  `examples/` tree, with `Dockerfile` COPY paths, `app.py`, and docs updated accordingly.
- The `config/` vs `src/config/` and `training/` vs `src/training/` distinction is either documented
  prominently in `PROJECT_STRUCTURE.md` or resolved by renaming, with `Dockerfile`, `pyproject.toml`
  per-file-ignores, and `.gitignore` updated to match; root `training/tests/` is merged into `tests/`.
- After all moves, `docs/STATUS.md` and `PROJECT_STRUCTURE.md` are refreshed and the full gate is green.

# Constraints

- Every move uses `git mv`; each of the sub-steps is its own commit so a bad move is trivially revertible.
- No import breakage: `git grep` for references before each code/dir move and update them.
- Backward compatible; no hardcoded values; full local gate green before push.
