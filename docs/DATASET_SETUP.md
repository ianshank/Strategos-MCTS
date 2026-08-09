# Dataset Setup Guide

This guide explains how to set up and access the datasets used for training and evaluation in the Multi-Agent MCTS system.

## Quick Start

### DABStep (No Authentication Required)

DABStep is an open-access dataset that can be loaded immediately:

```python
from src.data.dataset_loader import DABStepLoader

loader = DABStepLoader()
samples = loader.load(split="train")  # Uses 'default' split
print(f"Loaded {len(samples)} samples")
```

**License**: CC-BY-4.0 (Creative Commons Attribution 4.0)
**Size**: 450+ multi-step reasoning tasks
**Use Case**: HRM/TRM agent training, tactical reasoning

### PRIMUS (Authentication Required)

PRIMUS is a **gated dataset** that requires HuggingFace authentication.

#### Step 1: Create HuggingFace Account
1. Go to https://huggingface.co/join
2. Create a free account

#### Step 2: Accept Dataset Terms
1. Visit https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed
2. Click "Agree and access repository"
3. Also accept terms for https://huggingface.co/datasets/trendmicro-ailab/Primus-Instruct

#### Step 3: Generate Access Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "mcts-training")
4. Select "Read" access
5. Click "Generate"
6. Copy the token (starts with `hf_`)

#### Step 4: Authenticate CLI

If you've installed the `neural` extra (`pip install -e ".[neural]"`), the `hf` CLI is already
available — it ships with `huggingface_hub` as of the `>=0.34.0` this project pins. Otherwise,
install the standalone `hf` CLI, which runs independent of any project's Python environment. If
you'd rather not pipe a remote script straight into your shell, download and inspect it first:

```bash
curl -LsSf https://hf.co/cli/install.sh -o install-hf-cli.sh   # download
less install-hf-cli.sh                                         # inspect, then:
bash install-hf-cli.sh                                          # macOS/Linux
```

or accept the same trust as any other one-line install:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash        # macOS/Linux
# powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"   # Windows
```

Then authenticate:

```bash
# Option 1: Interactive login (opens a browser)
hf auth login

# Option 2: Non-interactive, passing a token directly
hf auth login --token hf_your_token_here

# Option 3: Set the environment variable directly (no CLI login needed)
export HF_TOKEN=hf_your_token_here     # Linux/Mac, PowerShell: $env:HF_TOKEN = "hf_your_token_here"
set HF_TOKEN=hf_your_token_here        # Windows cmd.exe
```

> The older `huggingface-cli login` also still works (deprecated with a warning; not removed
> upstream until `huggingface_hub` v1.0.0), but `hf` is the current command going forward. Some
> in-repo runtime error messages (e.g. `src/data/dataset_loader.py`'s gated-dataset error) still
> print `huggingface-cli login` as the suggested fix — that wording predates this doc update;
> `hf auth login` is the direct replacement.

#### Step 5: Load PRIMUS
```python
from src.data.dataset_loader import PRIMUSLoader

loader = PRIMUSLoader()
seed_samples = loader.load_seed(max_samples=1000)
instruct_samples = loader.load_instruct()
print(f"Loaded {len(seed_samples)} seed, {len(instruct_samples)} instruct")
```

**License**: ODC-BY (Open Data Commons Attribution)
**Size**: 674,848 cybersecurity documents + 835 instruction samples
**Use Case**: Cybersecurity domain knowledge, RAG knowledge base

## Dataset Details

### DABStep (Data Analysis Benchmark Step-by-Step)

- **Source**: https://huggingface.co/datasets/adyen/DABstep
- **License**: CC-BY-4.0
- **Splits**: `default` (450 samples), `dev` (10 samples)
- **Schema**:
  ```python
  {
      "question": str,  # The multi-step reasoning question
      "answer": str,    # Expected answer
      "difficulty": str # Task difficulty level
  }
  ```
- **Use Cases**:
  - Multi-step reasoning training
  - Tactical analysis validation
  - Agent decision quality evaluation

### PRIMUS-Seed

- **Source**: https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed
- **License**: ODC-BY
- **Size**: 674,848 documents (~190M tokens)
- **Domains**:
  - MITRE ATT&CK techniques
  - Wikipedia cybersecurity articles
  - Company security sites
  - Threat intelligence reports
  - Vulnerability databases
- **Use Cases**:
  - RAG knowledge base
  - Domain-specific grounding
  - Cybersecurity context enhancement

### PRIMUS-Instruct

- **Source**: https://huggingface.co/datasets/trendmicro-ailab/Primus-Instruct
- **License**: ODC-BY
- **Size**: 835 instruction-tuning samples
- **Format**: Instruction-Response pairs
- **Use Cases**:
  - Instruction fine-tuning
  - Cybersecurity Q&A training
  - Agent response quality improvement

## Combined Loading

Load all datasets with a single interface:

```python
from src.data.dataset_loader import CombinedDatasetLoader

loader = CombinedDatasetLoader()
all_samples = loader.load_all(
    dabstep_split="train",
    primus_max_samples=10000,  # Load subset of PRIMUS-Seed
    include_instruct=True
)

# Get domain distribution
distribution = loader.get_domain_distribution()
print(distribution)

# Filter by domain
data_analysis = loader.filter_by_domain("data_analysis")
cybersec = loader.filter_by_domain("mitre_attack")

# Export for training
loader.export_for_training("output/training_data.jsonl", format="jsonl")
```

## Caching

Datasets are cached locally to avoid repeated downloads:

- **Default cache**: `~/.cache/huggingface/hub/`
- **Custom cache**: Pass `cache_dir` parameter to loaders
- **First load**: May take 5-30 minutes depending on dataset size and network speed
- **Subsequent loads**: Fast (uses local cache)

### Clear Cache

```bash
# View cache size
du -sh ~/.cache/huggingface/hub/

# Clear specific dataset
rm -rf ~/.cache/huggingface/hub/datasets--adyen--DABstep

# Clear all cache (caution: removes all cached models/datasets)
rm -rf ~/.cache/huggingface/hub/
```

## License Attribution Requirements

### CC-BY-4.0 (DABStep)
When using DABStep, you must:
1. Credit the original creators (Adyen)
2. Include link to the license
3. Indicate if changes were made

Example attribution:
```
This work uses the DABStep dataset by Adyen, licensed under CC-BY-4.0.
https://huggingface.co/datasets/adyen/DABstep
```

### ODC-BY (PRIMUS)
When using PRIMUS, you must:
1. Attribute Trend Micro AI Lab
2. Share attribution with any derivatives

Example attribution:
```
This work uses the PRIMUS dataset by Trend Micro AI Lab, licensed under ODC-BY.
https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed
```

## Troubleshooting

### "Gated dataset" Error
```
DatasetNotFoundError: Dataset 'trendmicro-ailab/Primus-Seed' is a gated dataset
```
**Solution**: Follow authentication steps above

### "No module 'datasets'" Error
```bash
pip install datasets
```

### Slow Downloads
- Use wired connection instead of WiFi
- Set custom cache to SSD
- Download during off-peak hours
- Consider `max_samples` parameter for initial testing

### Memory Issues with Large Datasets
```python
# Load in batches instead of all at once
loader = PRIMUSLoader()
for batch in loader.iterate_samples(batch_size=100):
    process_batch(batch)
```

## Validation Script

Run the validation script to ensure datasets are properly configured:

```bash
python scripts/validate_datasets.py
```

Expected output:
```
DABSTEP: [PASS]
PRIMUS: [PASS]  # After authentication
COMBINED: [PASS]
HUGGINGFACE: [PASS]
```

## Cost Summary

| Dataset | License | Cost | Size |
|---------|---------|------|------|
| DABStep | CC-BY-4.0 | $0 | 450+ samples |
| PRIMUS-Seed | ODC-BY | $0 | 674K documents |
| PRIMUS-Instruct | ODC-BY | $0 | 835 samples |
| **Total** | - | **$0** | ~675K samples |

All datasets are free to use with proper attribution.
