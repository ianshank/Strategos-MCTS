# Hugging Face Space runbook

The demo Space at [ianshank/langgraph-mcts-demo](https://huggingface.co/spaces/ianshank/langgraph-mcts-demo)
serves the root Gradio app (`app.py`) from this tree.

**It is a demo showcase, not a service.** It makes no uptime, latency, or
support-response commitment — `CHARTER.md` NG-1 stands, and the Space card says so
in the same words. It sleeps after 48 hours of inactivity and restarts when
someone visits.

---

## How a deploy happens

`.github/workflows/deploy-space.yml` runs after a successful **CI Pipeline** run
on `main` (or on manual `workflow_dispatch`). It:

1. checks out the exact commit CI validated — not the branch tip, which may have moved;
2. assembles a tree from an **allowlist**: `src/`, `app.py`, `space_bootstrap.py`,
   `pyproject.toml`, `LICENSE`, plus `Dockerfile.space` → `Dockerfile` and
   `README.space.md` → `README.md`;
3. fails if any file exceeds 10MB, which the Hub rejects for non-LFS files;
4. force-pushes a single-commit history to the Space;
5. polls the Hub runtime API and **fails the job** unless the Space reaches `RUNNING`.

That last step is the point: a green deploy run means a live Space, not merely an
accepted push.

### The Space has exactly one writer

Everything in the Space repository is overwritten on the next deploy. Do not edit
files there, and do not vendor a copy of `src/` into this repository to serve it —
that is precisely how the previous Space became a fork that diverged in 51 files
(`CHANGELOG.md`).

### Rollback

Re-run the workflow from a known-good commit:

```bash
gh workflow run "Deploy Hugging Face Space" --ref <good-sha>   # or use the Actions UI
```

There is no separate rollback path — redeploying an earlier commit *is* the
rollback, because the Space holds no state of its own.

---

## What is not left to chance: `models/` and `requirements.txt`

Two exclusions are deliberate and easy to "fix" wrongly:

- **`models/` is not deployed.** Every checkpoint in Git is a Git-LFS pointer stub.
  Pushing those to the Space would ship 130-byte files that look like weights.
  `space_bootstrap.py` downloads the real ones from the Hub at startup instead.
- **`requirements.txt` is not deployed.** It is the demo/E2E CI manifest (selenium,
  wandb, pytest), as its own header says. The Space image derives its install set
  from `pyproject.toml` — `[project.dependencies]` plus the `ui` and `neural`
  extras — so there is no second dependency list to keep in sync.

---

## Environment

Set in `Dockerfile.space`, and each differs from its in-repo default:

| Variable | Value | Why |
|---|---|---|
| `ENABLE_DEMO_COMPARISON` | `true` | Default `False` in `src/config/settings.py`. Turns on the MCTS-vs-single-shot surface, whose explicit `mock` provider is the one thing that runs end-to-end with no API key. A feature flag, not a degraded path. |
| `GRADIO_SERVER_PORT` | `7860` | Must match `app_port` in the Space card. Hugging Face does not set Gradio's env vars. |
| `HF_HOME` | `/home/user/.cache/huggingface` | Hub cache under the uid-1000 home, so the prewarmed models are readable at runtime. |
| `STRATEGOS_CHECKPOINT_REPO` | `ianshank/strategos-mcts-checkpoints` | Read only by `space_bootstrap.py`. |
| `LOG_LEVEL` | `INFO` | |

`ENABLE_GRAPH_VISUALIZATION` and `ENABLE_STREAMING` already default to `true`; they
are deliberately **not** restated, so the image shows only real deviations.

### Why `ALLOW_MOCK_LLM_FALLBACK` is not set

It would do nothing, and saying otherwise would be a claim no code supports.

The flag guards a fallback in `src/api/framework_service.py` that engages only when
LLM client *creation* raises. In this container it cannot:

- with no key, `space_bootstrap.py` selects `lmstudio`, and `LMStudioClient`
  construction is lazy — it opens no connection, so it never raises;
- the alternative (`openai` with no key) fails earlier still, inside
  `validate_provider_credentials()` in `src/config/settings.py`, which stops
  settings from constructing at all — and `app.py` resolves settings eagerly when
  it builds the UI.

So the honest keyless configuration is a reachable provider that has nothing behind
it. Requests fail per-call and the app's labelled degraded path renders the result.

---

## Provider selection

`space_bootstrap.py` picks, in order: an explicit `LLM_PROVIDER` if set; else
`openai` if `OPENAI_API_KEY` is present; else `anthropic` if `ANTHROPIC_API_KEY` is
present; else `lmstudio` pointed at `http://127.0.0.1:9/v1` — a closed loopback
port, so no request ever leaves the container.

### Enabling real inference

In the Space UI: **Settings → Variables and secrets → New secret**, name
`OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`), then restart the Space. Runtime secrets
arrive as ordinary environment variables in Docker Spaces, so nothing else changes.

> **Every visitor then spends your credits.** The Space is public and has no rate
> limiting of its own. Prefer leaving it keyless unless you are actively
> demonstrating something, and remove the secret afterwards.

If both keys are set, `openai` wins; override with a Space **variable**
`LLM_PROVIDER=anthropic`.

---

## Checkpoints

The weights live in
[ianshank/strategos-mcts-checkpoints](https://huggingface.co/ianshank/strategos-mcts-checkpoints)
and are downloaded into `models/` at startup, matching the layout
`src/ui/status.py` (`DEFAULT_CHECKPOINTS`) expects. A failed download is
non-fatal — the app's runtime banner reports reduced mode.

They were rescued from the old Space, which was their last remaining copy, by
`scripts/rescue_space_weights.py`.

### Re-uploading checkpoints

**Run the verifier first, every time:**

```bash
python scripts/rescue_space_weights.py --verify --from-repo
```

The runtime banner classifies a checkpoint's *container format* (missing, LFS stub,
unreadable, OK) — it cannot detect an architecture mismatch. A real-but-mismatched
RNN state dict passes the banner and then crashes every query, because `app.py`
calls `load_state_dict` on it unwrapped. The rescue script asserts the state-dict
contract and refuses to publish on a mismatch; skipping it removes the only check
that catches this.

---

## One-time setup

1. **HF token.** huggingface.co → Settings → Access Tokens → fine-grained, **write**
   access to `spaces/ianshank/langgraph-mcts-demo` and
   `models/ianshank/strategos-mcts-checkpoints`.
2. **GitHub secret.** Repository → Settings → Secrets and variables → Actions → New
   repository secret, named `HF_TOKEN`.

### Account requirement

Gradio and Docker Spaces require a paid Hugging Face plan to create; the owning
account is PRO. `cpu-basic` hardware carries no hourly cost, but "free tier" is the
wrong phrase — if the subscription lapses, this Space stops being schedulable. That
is a platform constraint, not a repository defect.

---

## Verifying locally

```bash
pip install -e ".[ui,neural]"
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY GRADIO_SERVER_PORT=7861 python space_bootstrap.py
```

Then, against the running app:

```bash
python - <<'PY'
from gradio_client import Client
client = Client("http://127.0.0.1:7861")
result = client.predict("Design a rate limiter", "RNN", api_name="/process_query")
print(result[2])            # routing probabilities — must be real numbers
print(result[0][:200])      # answer — must carry the degraded-mode label when keyless
PY
```

### What a restricted network can and cannot verify

Some environments (including Claude Code sessions behind an egress policy) cannot
reach `huggingface.co` or `download.pytorch.org`. There, the checks split:

| Check | Behind a blocked-Hub proxy |
|---|---|
| Keyless boot serves the UI | **Works.** The checkpoint download fails, the bootstrap logs the failure, and the banner reports reduced mode — which is the designed behaviour. |
| Checkpoint contract logic | **Works.** `_verify_rnn` can be exercised against synthetic state dicts without any network. |
| A query end-to-end | **Blocked.** The first query constructs `BERTMetaController`, which pulls `prajjwal1/bert-mini` from the Hub, and the call fails hard rather than degrading. |
| Docker image build | **Blocked.** The CPU wheel index and the prewarm layer both need network. |

That last row is why the image prewarms `prajjwal1/bert-mini` and
`all-MiniLM-L6-v2` at build time: it is not only a cold-start optimisation, it is
what keeps a running Space off the Hub's availability on the query path.
Hugging Face's builder has full network access, so the image builds there.

---

## Related

- `specs/hf_space_deploy.SPEC.md` — the contract this deployment implements
- `specs/ui_runtime_integrity.SPEC.md` — the checkpoint banner and labelled
  degraded path the Space's honesty depends on (still `draft`)
- `docs/DOCKER_DEPLOYMENT.md` — the production API image, a different artifact
