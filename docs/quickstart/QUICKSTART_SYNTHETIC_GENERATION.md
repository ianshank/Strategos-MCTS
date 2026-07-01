# Quick Start: Synthetic Knowledge Generation

Generate 1,000 high-quality Q&A pairs in under 1 hour.

## Prerequisites (2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key (choose one)
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
# or use local LM Studio (no key needed)

# 3. Verify installation
python scripts/verify_synthetic_generator.py
```

## Generate Your First 10 Samples (5 minutes)

```bash
python scripts/generate_synthetic_training_data.py \
    --num-samples 10 \
    --model gpt-3.5-turbo

# Output:
# ✓ Generated 10 Q&A pairs
# ✓ Saved to training/synthetic_data/
# ✓ Cost: ~$0.10
```

## Review the Results

```bash
# View first sample
cat training/synthetic_data/synthetic_qa_langsmith_*.json | jq '.[0]'

# Check statistics
cat training/synthetic_data/generation_stats.json | jq
```

## Scale to 1,000 Samples (30-50 minutes)

```bash
python scripts/generate_synthetic_training_data.py \
    --num-samples 1000 \
    --model gpt-4-turbo-preview \
    --batch-size 20 \
    --min-quality 0.6 \
    --upload-langsmith

# Expected:
# - Time: 30-50 minutes
# - Cost: $50-100
# - High-quality pairs: 800-900
```

## What You Get

### Output Files

```
training/synthetic_data/
├── synthetic_qa_langsmith_*.json  # LangSmith format (ready to upload)
├── synthetic_qa_raw_*.json        # Raw format (for custom processing)
├── generation_stats.json          # Detailed statistics
└── checkpoint.json                # Resume point
```

### Sample Output

```json
{
  "inputs": {
    "question": "How does UCB1 balance exploration and exploitation in MCTS?",
    "contexts": [
      "UCB1 formula: Q/N + C*sqrt(ln(N_parent)/N)",
      "The first term exploits, second explores",
      "Visit count N determines the balance"
    ]
  },
  "outputs": {
    "ground_truth": "UCB1 balances exploration and exploitation through..."
  },
  "metadata": {
    "category": "exploration_exploitation",
    "difficulty": "medium",
    "quality_score": 0.75
  }
}
```

## Common Use Cases

### Use Case 1: Quick Test (10 samples, $0.10)

```bash
python scripts/generate_synthetic_training_data.py \
    --num-samples 10 \
    --model gpt-3.5-turbo
```

### Use Case 2: Standard Generation (1,000 samples, $50)

```bash
python scripts/generate_synthetic_training_data.py \
    --num-samples 1000 \
    --model gpt-4-turbo-preview \
    --batch-size 20 \
    --min-quality 0.6
```

### Use Case 3: Production Scale (10,000 samples, $500)

```bash
python scripts/generate_synthetic_training_data.py \
    --num-samples 10000 \
    --model gpt-4-turbo-preview \
    --batch-size 50 \
    --min-quality 0.6 \
    --max-cost 100.0 \
    --resume \
    --upload-langsmith \
    --log-file training/logs/generation.log
```

### Use Case 4: Specific Categories

```bash
python scripts/generate_synthetic_training_data.py \
    --num-samples 500 \
    --categories mcts_algorithms alphazero_neural advanced_mcts \
    --min-quality 0.7
```

### Use Case 5: Local LLM (Free)

```bash
# Start LM Studio first, then:
python scripts/generate_synthetic_training_data.py \
    --provider lmstudio \
    --base-url http://localhost:1234/v1 \
    --num-samples 100
```

## Monitor Progress

```bash
# Real-time stats
watch -n 5 "cat training/synthetic_data/generation_stats.json | jq '.valid_pairs, .total_cost'"

# Tail logs
tail -f training/logs/generation.log
```

## Upload to LangSmith

```bash
# Set LangSmith API key
export LANGSMITH_API_KEY="lsv2_..."

# Generate and upload
python scripts/generate_synthetic_training_data.py \
    --num-samples 1000 \
    --upload-langsmith \
    --langsmith-dataset "my-training-data"
```

## Cost Management

### Budget Control

```bash
# Stop at $50
--max-cost 50.0

# Stop at $100
--max-cost 100.0
```

### Cost Estimates

| Model | 100 Pairs | 1,000 Pairs | 10,000 Pairs |
|-------|-----------|-------------|--------------|
| GPT-3.5 | $0.50 | $5 | $50 |
| GPT-4 | $5 | $50 | $500 |
| Claude-3-Sonnet | $2 | $20 | $200 |
| Claude-3-Haiku | $0.20 | $2 | $20 |

### Optimization Tips

1. **Use GPT-3.5 for bulk**: 10x cheaper
2. **Use GPT-4 for quality subset**: Higher scores
3. **Set quality threshold**: `--min-quality 0.6`
4. **Monitor costs**: Check `generation_stats.json`

## Troubleshooting

### Rate Limit Error

```bash
# Reduce batch size and rate
--batch-size 5 --rate-limit 30
```

### Low Quality Scores

```bash
# Use better model
--model gpt-4-turbo-preview

# Lower threshold
--min-quality 0.5
```

### Resume After Interruption

```bash
# Automatically resumes from checkpoint
--resume
```

## Next Steps

1. ✅ Generate test batch (10 samples)
2. ✅ Review quality and cost
3. ✅ Scale to production (1,000+ samples)
4. ✅ Upload to LangSmith
5. ✅ Use in training pipeline

## Full Documentation

- **This Guide**: Quick start
- **Full Guide**: `training/SYNTHETIC_DATA_GENERATION_GUIDE.md` (800+ lines)
- **Quick Ref**: `training/README_SYNTHETIC_GENERATOR.md` (300+ lines)
- **Examples**: `examples/synthetic_data_generation_example.py` (7 examples)
- **Implementation**: `SYNTHETIC_KNOWLEDGE_GENERATOR_SUMMARY.md`

## Support

```bash
# Run verification
python scripts/verify_synthetic_generator.py

# Run examples
python examples/synthetic_data_generation_example.py

# Run tests
pytest tests/integration/test_synthetic_knowledge_generator.py -v
```

---

**Start now**:

```bash
python scripts/generate_synthetic_training_data.py --num-samples 10
```
