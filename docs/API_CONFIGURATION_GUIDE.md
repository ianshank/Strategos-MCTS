# API Configuration Guide

## Overview

This guide covers the configuration of external API providers for the Multi-Agent MCTS Framework:
- **OpenAI** - GPT-4 and other models
- **Anthropic** - Claude models
- **Weights & Biases** - Experiment tracking
- **LM Studio** - Local model inference

## Quick Start

1. **Environment Variables**: API keys are stored in the `.env` file (created by `setup_api_keys.py`)

2. **Default Provider**: Set in `.env` as `LLM_PROVIDER=openai`

3. **Test Connections**: Run `python scripts/verification/test_api_integrations.py`

## Supported Providers

### OpenAI

**Models Available**:
- `gpt-4-turbo-preview` (recommended)
- `gpt-4-0125-preview`
- `gpt-4-1106-preview`
- `gpt-3.5-turbo`

**Configuration**:
```python
# In .env
OPENAI_API_KEY=sk-proj-...
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4-turbo-preview

# In code
from src.adapters.llm import create_client

client = create_client(
    provider="openai",
    model="gpt-4-turbo-preview"
)
```

### Anthropic

**Models Available**:
- `claude-3-opus-20240229` (most capable)
- `claude-3-sonnet-20240229` (balanced)
- `claude-3-haiku-20240307` (fastest)

**Configuration**:
```python
# In .env
ANTHROPIC_API_KEY=sk-ant-api03-...
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-3-haiku-20240307

# In code
client = create_client(
    provider="anthropic",
    model="claude-3-haiku-20240307"
)
```

### LM Studio (Local)

**Configuration**:
```python
# In .env
LLM_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_MODEL=llama-3.2-8x3b-moe...

# In code
client = create_client(
    provider="lmstudio",
    base_url="http://127.0.0.1:1234/v1"
)
```

### Weights & Biases

**Configuration**:
```python
# In .env
WANDB_API_KEY=26a08535e80a6d5d6f4a941f6a742264b25f4819

# In code
import wandb

wandb.init(
    project="langgraph-mcts",
    config={
        "model": "gpt-4",
        "task": "multi-agent-reasoning"
    }
)
```

## Usage Examples

### Basic Usage

```python
from src.adapters.llm import create_client
from src.config.settings import get_settings

# Use default provider from .env
settings = get_settings()
client = create_client(provider=settings.LLM_PROVIDER.value)

# Generate response
async with client:
    response = await client.generate(
        prompt="Explain MCTS in one sentence.",
        temperature=0.7,
        max_tokens=100
    )
    print(response.text)
```

### Switching Providers

```python
# Method 1: Environment variable
import os
os.environ["LLM_PROVIDER"] = "anthropic"

# Method 2: Explicit provider
openai_client = create_client(provider="openai")
anthropic_client = create_client(provider="anthropic")
lmstudio_client = create_client(provider="lmstudio")
```

### Multi-Provider Example

```python
async def compare_providers(prompt: str):
    """Compare responses from different providers."""
    
    providers = ["openai", "anthropic", "lmstudio"]
    responses = {}
    
    for provider in providers:
        try:
            client = create_client(provider=provider)
            async with client:
                response = await client.generate(
                    prompt=prompt,
                    temperature=0.7
                )
                responses[provider] = response.text
        except Exception as e:
            responses[provider] = f"Error: {e}"
    
    return responses
```

### With MCTS Framework

```python
from src.agents.graph_builder import GraphBuilder

# Configure provider for agents
os.environ["LLM_PROVIDER"] = "openai"  # or "anthropic"

# Build graph
builder = GraphBuilder(
    hrm_agent=hrm_agent,
    trm_agent=trm_agent,
    model_adapter=adapter,
    meta_controller_config=config
)

graph = builder.build()
```

## Cost Optimization

### Provider Costs (approximate)
- **OpenAI GPT-4**: $0.03/1K input tokens, $0.06/1K output tokens
- **Anthropic Claude 3**: $0.015-0.075/1K tokens (varies by model)
- **LM Studio**: Free (local inference)

### Tips:
1. Use `claude-3-haiku` for rapid prototyping (cheapest)
2. Use `gpt-3.5-turbo` for simple tasks
3. Use LM Studio for development/testing
4. Reserve `gpt-4` or `claude-3-opus` for complex reasoning

## Environment Variables Reference

```bash
# LLM Provider Selection
LLM_PROVIDER=openai            # Options: openai, anthropic, lmstudio

# API Keys
OPENAI_API_KEY=sk-proj-...     # OpenAI API key
ANTHROPIC_API_KEY=sk-ant-...   # Anthropic API key
WANDB_API_KEY=...              # Weights & Biases key

# Model Configuration
OPENAI_MODEL=gpt-4-turbo-preview
ANTHROPIC_MODEL=claude-3-haiku-20240307
LMSTUDIO_MODEL=llama-3.2-8x3b-moe...

# Request Configuration
HTTP_TIMEOUT_SECONDS=120       # Request timeout
HTTP_MAX_RETRIES=3            # Retry attempts
RATE_LIMIT_REQUESTS_PER_MINUTE=60

# MCTS Configuration
MCTS_ITERATIONS=100
MCTS_C=1.414
SEED=42
```

## Troubleshooting

### Common Issues

1. **401 Unauthorized**
   - Check API key is correct
   - Ensure key has proper permissions

2. **404 Model Not Found**
   - Verify model name is correct
   - Check model availability in your region

3. **Rate Limiting**
   - Reduce `RATE_LIMIT_REQUESTS_PER_MINUTE`
   - Implement exponential backoff

4. **Timeout Errors**
   - Increase `HTTP_TIMEOUT_SECONDS`
   - Check network connectivity

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed API requests/responses
```

## Security Best Practices

1. **Never commit `.env` file** - It's in `.gitignore`
2. **Use environment-specific keys** - Dev vs Production
3. **Rotate keys regularly** - Monthly recommended
4. **Monitor usage** - Set up billing alerts
5. **Use least privilege** - Restrict key permissions

## Next Steps

1. Test your configuration: `python scripts/verification/test_api_integrations.py`
2. Try different providers in your workflows
3. Set up Weights & Biases experiment tracking
4. Configure cost monitoring and alerts
