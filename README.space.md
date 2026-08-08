---
title: Strategos MCTS — Multi-Agent Reasoning Demo
emoji: 🌳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
models:
  - ianshank/strategos-mcts-checkpoints
tags:
  - multi-agent
  - mcts
  - langgraph
  - meta-controller
short_description: Neural meta-controllers routing queries across HRM/TRM/MCTS agents
---

# Strategos-MCTS

A demo of [Strategos-MCTS](https://github.com/ianshank/Strategos-MCTS), a LangGraph
multi-agent framework where trained neural meta-controllers decide which reasoning
agent should answer a query.

> **This is a demo showcase, not a service.** It makes no uptime, latency, or
> support commitment, and it sleeps when idle. The Space is a build artifact:
> every file here is force-pushed from the GitHub repository's default branch by
> CI. Editing it directly accomplishes nothing — the next deploy overwrites it.

## What you can try

Enter a query and pick a controller. The **RNN** (GRU) and **BERT + LoRA**
meta-controllers each route it to one of three agents — **HRM** (hierarchical
decomposition), **TRM** (iterative refinement), or **MCTS** (strategic
exploration) — and show you the routing probabilities and extracted features
behind the decision.

The **Single-Shot vs MCTS Comparison** section runs a direct answer against MCTS
multi-strategy exploration and prints the search tree.

## What is real here, and what is not

This matters more than the usual demo disclaimer, so it is spelled out:

| Surface | Status without an API key |
|---|---|
| Meta-controller routing (RNN / BERT) | **Real.** Runs the published checkpoints. The banner at the top of the app reports what actually loaded at runtime — trust it over this table. |
| Single-shot vs MCTS comparison | **Real search, mock LLM.** The `mock` provider is an explicit choice in the dropdown, not a silent substitution. |
| Graph structure / streaming views | **Real.** Rendered from the live LangGraph structure. |
| LLM-synthesised answers | **Degraded and labelled.** With no provider configured there is nothing to call, so answers carry an explicit degraded-mode marker. They are never quietly replaced with canned text. |

Model weights come from
[ianshank/strategos-mcts-checkpoints](https://huggingface.co/ianshank/strategos-mcts-checkpoints),
which documents their provenance and the command that verifies they load. If that
download fails, the app says so rather than pretending.

The Space owner can set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` as a Space secret
to enable real LLM inference on the next restart.

## The rest of the project

The UI demonstrates one slice. These run from a clone of the repository:

| Capability | Command |
|---|---|
| MCTS vs single-shot, in the terminal (no API key) | `python demo.py --compare` |
| MCTS reasoning with a visible search tree | `python demo.py --tree --iterations 15` |
| Chess adversarial domain, with its own UI | `python chess_demo.py` |
| Benchmark harness across systems and tasks | `python -m src.benchmark --dry-run` |
| Deterministic agent harness, spec-driven | `harness dry-run --spec specs/phase_1_correctness.SPEC.md` |

A chess tab is deliberately not mounted here: it needs an extra dependency and
spawns background self-play and training threads, which is not a good neighbour on
shared demo hardware.

## Status and limitations

Test and coverage status lives in
[`docs/STATUS.md`](https://github.com/ianshank/Strategos-MCTS/blob/main/docs/STATUS.md),
and the project's scope and non-goals in
[`CHARTER.md`](https://github.com/ianshank/Strategos-MCTS/blob/main/CHARTER.md).
Those documents are generated and reviewed against the tree; no numbers are
restated here, so that this card cannot drift away from them.

Licensed MIT, matching the source repository.
