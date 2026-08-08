---
id: ui_runtime_integrity
goal: Make every UI surface actually start, answer with real agent output, and report honestly when it cannot
module: src/ui/
milestone: M4
status: draft
---

# Goal

An audit of every web surface found no working UI. The root Gradio app was unimportable without a
provider key; the chess UI raised `TypeError` before opening a socket; every reasoning, streaming and
graph route in the REST API returned 503; and the demo advertised "REAL trained models" in a
repository where all six checkpoints are Git-LFS pointer stubs.

Three root causes explain nearly all of it, and each is worth fixing once at the contract rather than
patched per call site: an injected-logger signature mismatch, settings resolved at import time, and a
Gradio major-version API removal inside a declared-supported range. Fix those, route the UI's answers
through the service layer the REST server already uses, and make the UI's claims track what actually
loaded.

# Acceptance Criteria

- AC-1: `StructuredLogger` accepts both the structured (`message, **extra`) and stdlib printf
  (`message, *args`) calling conventions, forwards `exc_info`/`stack_info`/`stacklevel` to stdlib rather
  than folding them into `extra`, and renames structured fields that would collide with reserved
  `LogRecord` slots instead of raising at the call site. A test asserts the four-positional-argument shape
  used at `src/framework/graph/builder.py` succeeds against an injected `StructuredLogger`.
- AC-2: `FrameworkService` reaches `framework_mode="integrated"` rather than leaving `framework` as
  `None`; its fallback catches broadly but logs unexpected exception types at `exception` level with the
  traceback, so a defect cannot degrade silently. `/ready` reports not-ready while the framework is
  uninitialized, gated by a settings flag rather than a hardcoded stance.
- AC-3: A reusable checkpoint module classifies a checkpoint before any deserializer sees it —
  distinguishing missing, Git-LFS pointer stub, unreadable and OK, for both files and adapter directories
  — and a tolerant loader returns `None` with an actionable structured warning instead of raising. No
  literal checkpoint path or magic signature appears outside `src/config/constants.py`.
- AC-4: `import app` succeeds with no provider key configured and with the `[neural]` extra absent;
  `app.APP_VERSION` and `app.demo` continue to resolve for existing importers. A test asserts the
  keyless import, and CI runs it.
- AC-5: `create_chess_ui()` returns a Blocks graph on every Gradio major the project declares support
  for, via runtime capability detection rather than a narrowed version pin; both the `gr.Timer` and the
  `Blocks.load(every=)` branches are tested against injected stand-ins.
- AC-6: The UI's query handlers route through the same `FrameworkService` the REST server uses, taking
  confidence, agent attribution and reasoning trace from framework output; no hardcoded confidence value
  remains. A framework failure yields a visibly-labelled degraded result with zero confidence rather than
  a confident-looking answer.
- AC-7: The UI header reports checkpoint state measured at runtime — claiming trained-model operation
  only when every configured checkpoint is genuinely readable, and otherwise naming each unusable
  checkpoint with its remedy. No claim remains in the UI copy that a command cannot reproduce.

# Constraints

- Backward compatible: existing importers of `app.APP_VERSION`, `app.demo` and every current
  `StructuredLogger` call site keep working unchanged.
- No hardcoded values: version defaults, checkpoint paths, LFS signatures, win markers, refresh intervals
  and degraded-mode markers all live in `src/config/constants.py` or settings.
- Logic that must be measured lives under `src/`, not the root `app.py`, which sits outside
  `[tool.coverage.run] source = ["src"]` and is invisible to the coverage gate by construction.
- Structured logging with correlation IDs on every degradation path; degradation is never silent.
- No real network or provider calls in unit tests; tests requiring a downloadable model skip with an
  explicit reason naming the cause rather than failing opaquely.
- Full local gate (black / ruff / mypy `src/` / pytest) green before push.
