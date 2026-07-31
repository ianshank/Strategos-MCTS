# API Quick Reference

## Provider Configuration (example shape — never commit real key material)

### OpenAI (Default)
```bash
Provider: openai
API Key: <set in .env, never committed>
Model: gpt-4-turbo-preview
Status: ✅ Working
```

### Anthropic
```bash
Provider: anthropic  
API Key: <set in .env, never committed>
Model: claude-3-haiku-20240307
Status: ⚠️ Check model availability
```

### Weights & Biases
```bash
API Key: <set in .env, never committed>
Project: langgraph-mcts
Status: ✅ Configured
```

### LM Studio (Local)
```bash
Provider: lmstudio
URL: http://127.0.0.1:1234/v1
Model: llama-3.2-8x3b-moe...
Status: ✅ Configured
```

## Quick Commands

### Switch Providers
```bash
# Windows PowerShell
$env:LLM_PROVIDER = "openai"     # Use OpenAI
$env:LLM_PROVIDER = "anthropic"  # Use Anthropic
$env:LLM_PROVIDER = "lmstudio"   # Use LM Studio

# Linux/Mac
export LLM_PROVIDER=openai
```

### Test Connections
```bash
python scripts/verification/test_api_integrations.py
```

### Create/Update .env
```bash
python setup_api_keys.py
```

## Python Usage

### Basic Example
```python
from src.adapters.llm import create_client

# Quick test
async def test():
    client = create_client(provider="openai")
    async with client:
        response = await client.generate(
            prompt="Hello, world!",
            max_tokens=50
        )
        print(response.text)

# Run: asyncio.run(test())
```

### With MCTS Framework
```python
# The framework will use the provider from .env
from src.agents.graph_builder import GraphBuilder

builder = GraphBuilder(...)
graph = builder.build()

# Or override temporarily
import os
os.environ["LLM_PROVIDER"] = "anthropic"
```

## Cost Estimates

| Task | OpenAI GPT-4 | Anthropic Claude | LM Studio |
|------|--------------|------------------|-----------|
| Simple query | ~$0.001 | ~$0.0005 | Free |
| Complex reasoning | ~$0.05 | ~$0.02 | Free |
| Full MCTS run | ~$0.50 | ~$0.20 | Free |

## Status Check

Run this to verify your setup:
```python
python -c "from src.config.settings import get_settings; s=get_settings(); print(f'Provider: {s.LLM_PROVIDER}\\nModel: {getattr(s, f\"{s.LLM_PROVIDER.value.upper()}_MODEL\", \"default\")}')"
```
