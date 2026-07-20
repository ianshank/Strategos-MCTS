# Running Strategos-MCTS Locally

This guide provides instructions for executing the full suite of Strategos-MCTS training pipelines, components, and demos locally, even without external API dependencies or matching CUDA environments.

## Prerequisites

- **Python 3.11** or higher.
- A virtual environment initialized and activated (`python -m venv .venv`).
- Ensure `.env` is configured (copy `.env.example` to `.env`).

## Installing Dependencies

If you run into issues installing the `chess` library due to legacy `distutils` limitations on Python 3.11, use the following workaround on Windows:

```powershell
$env:SETUPTOOLS_USE_DISTUTILS="stdlib"
pip install chess
```

## Running the Complete Suite

A PowerShell script is provided to sequentially execute all demos and training loops. 

```powershell
.\scripts\run_all_local.ps1
```

### What the script handles automatically:
- Sets `PYTHONPATH="."` to resolve import errors for `src` modules.
- Sets `PYTHONIOENCODING="utf-8"` to prevent Windows console Unicode crashes when components print success checks (`✓`).
- Temporarily sets `CUDA_VISIBLE_DEVICES="-1"` to force PyTorch to use the CPU. This acts as a reliable fallback if there are CUDA binary compatibility mismatches on your local GPU (e.g., RTX 50-series and older `torch` versions).

### Modules Run in the Suite:
- **Tier 1 (Core)**: `demo.py`, `chess_demo.py`, `healthcheck.py`, `huggingface_space/app_minimal_fallback.py`
- **Tier 2 (Neural & MCTS)**: `train_rnn.py`, `train_bert_lora.py`, `mcts_determinism_demo.py`, `neural_training_demo.py`, `deepmind_style_training.py`, `advanced_mcts_demo.py`, `hybrid_agent_demo.py`, `chess_alphazero_training.py`
- **Tier 3 (Benchmarks & Evaluation)**: `src.benchmark --dry-run`, `src.framework.harness.cli validate-spec`

## Optional: GUI & API Servers

If you want to run the REST API Server or Gradio UI manually, ensure you have an LLM Provider running locally (like LMStudio on port 1234) or actual API keys in your `.env`.

**REST API Server:**
```powershell
$env:PYTHONPATH="."
python src/api/rest_server.py
```
**Gradio UI:**
```powershell
$env:PYTHONPATH="."
python huggingface_space/app.py
```
