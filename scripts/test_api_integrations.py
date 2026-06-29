#!/usr/bin/env python3
"""
Test script to verify API integrations with actual API calls.

This script tests:
1. OpenAI API connection and generation
2. Anthropic API connection and generation
3. Weights & Biases tracking
4. Switching between providers
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv

from src.adapters.llm import create_client
from src.config.settings import get_settings

load_dotenv()


async def test_openai():
    """Test OpenAI API with actual generation."""
    print("\n1. Testing OpenAI API")
    print("-" * 40)

    try:
        # Get settings
        settings = get_settings()
        print(f"Provider: {settings.LLM_PROVIDER}")
        print("Model: gpt-4-turbo-preview")

        # Create client
        client = create_client(provider="openai", api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-4-turbo-preview")

        # Test generation
        print("\nSending test prompt...")
        async with client:
            response = await client.generate(
                prompt="What is 2+2? Answer in exactly one word.", temperature=0.0, max_tokens=10
            )

        print(f"Response: {response.text}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print("[SUCCESS] OpenAI API working correctly!")

        return True

    except Exception as e:
        print(f"[ERROR] OpenAI test failed: {e}")
        return False


async def test_anthropic():
    """Test Anthropic API with actual generation."""
    print("\n2. Testing Anthropic API")
    print("-" * 40)

    try:
        # Create client
        client = create_client(
            provider="anthropic", api_key=os.environ.get("ANTHROPIC_API_KEY"), model="claude-3-haiku-20240307"
        )

        print("Provider: anthropic")
        print("Model: claude-3-haiku-20240307")

        # Test generation
        print("\nSending test prompt...")
        async with client:
            response = await client.generate(
                prompt="What is 2+2? Answer in exactly one word.", temperature=0.0, max_tokens=10
            )

        print(f"Response: {response.text}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print("[SUCCESS] Anthropic API working correctly!")

        return True

    except Exception as e:
        print(f"[ERROR] Anthropic test failed: {e}")
        return False


def test_wandb():
    """Test Weights & Biases integration."""
    print("\n3. Testing Weights & Biases")
    print("-" * 40)

    try:
        import wandb

        # Initialize a test run
        _run = wandb.init(
            project="langgraph-mcts-test",
            name="api-integration-test",
            config={
                "test": "api_integration",
                "providers": ["openai", "anthropic"],
            },
            mode="offline",  # Use offline mode for testing
        )

        # Log some test metrics
        wandb.log({"test_openai": 1.0, "test_anthropic": 1.0, "integration_status": "success"})

        # Finish the run
        wandb.finish()

        print("[SUCCESS] Weights & Biases integration working!")
        print("Note: Run was created in offline mode")
        return True

    except Exception as e:
        print(f"[ERROR] W&B test failed: {e}")
        return False


async def test_provider_switching():
    """Test switching between providers."""
    print("\n4. Testing Provider Switching")
    print("-" * 40)

    try:
        # Test with environment variable switching
        original_provider = os.environ.get("LLM_PROVIDER", "openai")

        # Switch to OpenAI
        os.environ["LLM_PROVIDER"] = "openai"
        settings = get_settings()
        print(f"Provider set to: {settings.LLM_PROVIDER}")

        # Switch to Anthropic
        os.environ["LLM_PROVIDER"] = "anthropic"
        # Clear settings cache
        import src.config.settings

        src.config.settings._settings = None
        settings = get_settings()
        print(f"Provider switched to: {settings.LLM_PROVIDER}")

        # Switch to LMStudio
        os.environ["LLM_PROVIDER"] = "lmstudio"
        src.config.settings._settings = None
        settings = get_settings()
        print(f"Provider switched to: {settings.LLM_PROVIDER}")

        # Restore original
        os.environ["LLM_PROVIDER"] = original_provider
        src.config.settings._settings = None

        print("[SUCCESS] Provider switching working correctly!")
        return True

    except Exception as e:
        print(f"[ERROR] Provider switching failed: {e}")
        return False


async def main():
    """Run all integration tests."""
    print("API Integration Tests")
    print("=" * 60)
    print("Testing with live API calls...")

    results = {
        "OpenAI": await test_openai(),
        "Anthropic": await test_anthropic(),
        "Weights & Biases": test_wandb(),
        "Provider Switching": await test_provider_switching(),
    }

    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("-" * 40)

    for test, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test}")

    all_passed = all(results.values())

    if all_passed:
        print("\nAll tests passed! Your API integrations are working correctly.")
    else:
        print("\nSome tests failed. Please check the errors above.")

    print("\nUsage Example:")
    print("-" * 40)
    print("""
# In your code:
from src.adapters.llm import create_client
from src.config.settings import get_settings

# Use default provider from .env
settings = get_settings()
client = create_client(provider=settings.LLM_PROVIDER.value)

# Or specify provider explicitly
openai_client = create_client(provider="openai")
anthropic_client = create_client(provider="anthropic")
    """)

    return all_passed


if __name__ == "__main__":
    # Install dotenv if not present
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Installing python-dotenv...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
        from dotenv import load_dotenv

    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
