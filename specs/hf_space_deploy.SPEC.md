---
id: hf_space_deploy
goal: Publish the existing Gradio demo as a Hugging Face Space built from this tree, with every claim it makes reproducible and every degraded path labelled
module: Dockerfile.space
milestone: M4
status: draft
---

# Goal

The demo Space at `ianshank/langgraph-mcts-demo` still serves the vendored `src/` fork this repository
deleted for silently diverging in 51 files (`CHANGELOG.md`). It is the last live copy of that fork, and
it advertises trained-model operation the way `ui_runtime_integrity` records the UI once did.

Replace it with a deployment built from this tree on every green push to `main`, so the Space cannot
diverge again: one dependency source (`pyproject.toml`), one writer (the workflow), and a Space card
whose claims are reproduced by named commands. The Space is a demo showcase, not a service.

Two facts about a keyless deployment shape the contract. `_build_demo()` resolves settings eagerly, so
the container needs a provider selection that constructs without secrets — the repository already
solves this for containers with `LLM_PROVIDER=lmstudio`. And `ALLOW_MOCK_LLM_FALLBACK` cannot fire in
that configuration, because the fallback it guards engages only when LLM client *creation* raises,
which the LMStudio client never does. Setting it would advertise behaviour that cannot occur.

# Acceptance Criteria

- AC-1: A Docker-SDK Space definition builds from this tree with no second dependency manifest: the
  image derives its install set from `pyproject.toml` (`[project.dependencies]` plus the `ui` and
  `neural` extras) rather than restating it. The root `requirements.txt` — a demo/E2E CI manifest — is
  not shipped to the Space and not modified. No `tests/unit/` coverage exists for root-level deployment
  glue (Constraints); intended verification: `grep -c requirements.txt Dockerfile.space
  .github/workflows/deploy-space.yml` returns `0` for both, and `docs/HUGGINGFACE_SPACE.md`'s "What is
  not left to chance" section documents why.
- AC-2: The container starts with no provider key and no API secrets configured, and serves the Gradio
  UI on the port declared by the Space card. Intended verification: the command sequence in
  `docs/HUGGINGFACE_SPACE.md`'s "Verifying locally" section (`pip install -e ".[ui,neural]"` then
  `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY GRADIO_SERVER_PORT=7861 python space_bootstrap.py`).
- AC-3: `ALLOW_MOCK_LLM_FALLBACK` is not set anywhere in the deployment, and the runbook records why
  with the code citations that make it unreachable. Every environment variable the deployment sets that
  differs from its in-repo default is named in the image definition and explained in the runbook.
  Intended verification: `grep -c ALLOW_MOCK_LLM_FALLBACK Dockerfile.space` returns `0`; the `ENV` block
  in `Dockerfile.space` cross-checked line-for-line against the table in `docs/HUGGINGFACE_SPACE.md`'s
  "Environment" section.
- AC-4: With no key configured, LLM-synthesised answers carry the degraded-mode marker, and the
  comparison surface runs its explicit `mock` provider. Neither path emits an unlabelled answer that
  could be mistaken for a real provider call. Intended verification: the `gradio_client` snippet in
  `docs/HUGGINGFACE_SPACE.md`'s "Verifying locally" section, which asserts the degraded-mode label
  appears in `result[0]` against a running keyless instance — not merely assumed.
- AC-5: Model checkpoints reach the Space from a Hugging Face model repository at startup, and a
  download failure is non-fatal — the existing runtime banner reports what actually loaded. The
  checkpoint layout matches `DEFAULT_CHECKPOINTS` so the configured paths resolve unchanged. Intended
  test: `tests/unit/test_ui_status.py::TestPathResolution::test_defaults_are_relative_to_the_repo_root`
  covers the `DEFAULT_CHECKPOINTS` layout `space_bootstrap.py` targets. The download-failure-is-non-fatal
  behavior in `space_bootstrap.py::fetch_checkpoints` (a bare `except Exception` that logs and continues)
  has no automated test as of this spec — reviewed by inspection only; falsified by any change that lets
  a Hub failure raise instead of degrade.
- AC-6: Checkpoints are published only after a committed script loads them through the *current*
  controller code paths and runs a forward pass. The script refuses to publish on any state-dict
  mismatch, and publishes no artefact whose numbers this tree cannot reproduce. The published model
  card claims load-compatibility and provenance, nothing more. Intended verification: `python
  scripts/rescue_space_weights.py --verify` (no pytest wrapper exists; the script is its own committed
  verification tool per the Constraints below).
- AC-7: Deployment is triggered only by a successful CI run on the default branch, assembles the Space
  tree from an explicit allowlist, fails loudly on any file exceeding the Hub's non-LFS size limit, and
  fails the job when the Space does not reach a running state. A green deployment run means a live
  Space. Intended test: `tests/unit/test_ci_workflow_invariants.py`, which glob-discovers every file
  under `.github/workflows/` (including `deploy-space.yml`) and checks timeout, concurrency, and
  disarmed-step invariants generically. **A green run additionally requires the `HF_TOKEN` repository
  secret to be provisioned out-of-band** (`docs/HUGGINGFACE_SPACE.md`'s "One-time setup"); this spec
  does not provision it, and the workflow fails loudly and correctly when it is absent
  (`deploy-space.yml`'s explicit `HF_TOKEN is not set` check) rather than silently skipping.
- AC-8: The Space card states the demo's status without restating any measured value that another
  artefact generates, links the status document rather than copying its numbers, and maps the project
  surfaces the UI does not demonstrate to the commands that run them. No automated check covers
  `README.space.md` or `docs/HUGGINGFACE_SPACE.md` — `src/tools/context_docs.py`'s `GOVERNANCE_DOCS`
  is `("CHARTER.md",)` only, and its skill/agent doc discovery does not include either file. Verdict,
  stated honestly per this project's own convention (`CHARTER.md` §4): **ASPIRATIONAL** — review only,
  no gate fires if this claim drifts.

# Constraints

- No change under `src/`: the deployment is assembled from `Dockerfile.space`, `space_bootstrap.py`,
  `README.space.md`, `.github/workflows/deploy-space.yml`, `scripts/rescue_space_weights.py` and
  `docs/HUGGINGFACE_SPACE.md`. Deployment glue stays at the root and outside the coverage gate, for the
  same reason `app.py` does; anything that must be measured belongs under `src/`.
- The Space repository is a build artefact with exactly one writer. Nothing is hand-edited there, and
  no copy of `src/` is vendored into this repository to serve it.
- Checkpoint rescue strictly precedes the first deployment push: the force-push destroys the only
  remaining copy of those weights.
- Publishing a public deployment surface touches a zero-budget non-goal. The scope ruling is a human
  decision recorded before merge, not an assumption this spec resolves.
- The honesty behaviour this deployment depends on — the runtime checkpoint banner and the labelled
  degraded path — is specified by `ui_runtime_integrity`, which is still `draft`. The verification
  commands here are the compensating check.
- Full local gate (black / ruff / mypy `src/` / pytest) green before push, including the workflow
  invariants test that governs every file under `.github/workflows/`.
