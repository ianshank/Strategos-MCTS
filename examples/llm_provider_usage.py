"""
Example usage of the LLM Provider Abstraction Layer.

Demonstrates how to:
1. Create clients for different providers
2. Switch between providers
3. Use the async agent base class
4. Integrate with the existing LangGraphMultiAgentFramework
"""

import asyncio

# Import the LLM client factory
from src.adapters.llm import (
    LLMClientError,
    LLMRateLimitError,
    create_anthropic_client,
    create_client,
    create_client_from_config,
    create_openai_client,
    list_providers,
)

# Import the async agent base
from src.framework.agents import (
    AgentContext,
    AgentResult,
    AsyncAgentBase,
)

# ============================================================================
# Example 1: Basic Client Usage
# ============================================================================


async def basic_client_usage():
    """Demonstrate basic client creation and usage."""
    print("=" * 60)
    print("Example 1: Basic Client Usage")
    print("=" * 60)

    # List available providers
    print(f"Available providers: {list_providers()}")

    # Create OpenAI client
    client = create_client(
        "openai",
        model="gpt-4-turbo-preview",
        timeout=60.0,
    )

    try:
        # Simple prompt-based generation
        response = await client.generate(
            prompt="What is the capital of France?",
            temperature=0.3,
            max_tokens=100,
        )
        print(f"Response: {response.text}")
        print(f"Model: {response.model}")
        print(f"Tokens used: {response.total_tokens}")

    except LLMClientError as e:
        print(f"Error: {e}")
    finally:
        await client.close()


async def message_based_usage():
    """Demonstrate message-based conversation."""
    print("\n" + "=" * 60)
    print("Example 2: Message-based Conversation")
    print("=" * 60)

    client = create_anthropic_client(model="sonnet")

    messages = [
        {"role": "system", "content": "You are a helpful tactical advisor."},
        {"role": "user", "content": "What are the key principles of defensive positioning?"},
    ]

    try:
        response = await client.generate(
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        print(f"Claude's response:\n{response.text}")
        print(f"\nTokens: {response.usage}")

    except LLMClientError as e:
        print(f"Error: {e}")
    finally:
        await client.close()


# ============================================================================
# Example 2: Switching Providers
# ============================================================================


async def provider_switching():
    """Demonstrate switching between different providers."""
    print("\n" + "=" * 60)
    print("Example 3: Provider Switching")
    print("=" * 60)

    # Configuration-driven provider selection
    configs = [
        {
            "provider": "openai",
            "model": "gpt-4-turbo-preview",
            "timeout": 60.0,
        },
        {
            "provider": "anthropic",
            "model": "claude-3.5-sonnet",
            "timeout": 120.0,
        },
        {
            "provider": "lmstudio",
            "base_url": "http://localhost:1234/v1",
            "timeout": 300.0,
        },
    ]

    prompt = "Explain MCTS in one sentence."

    for config in configs:
        provider = config["provider"]
        print(f"\n--- Using {provider.upper()} ---")

        try:
            client = create_client_from_config(config)

            # Check if local server is available
            if provider == "lmstudio" and hasattr(client, "check_health"):
                is_healthy = await client.check_health()
                if not is_healthy:
                    print(f"  Skipping {provider}: Server not running")
                    continue

            response = await client.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=100,
            )
            print(f"  Response: {response.text}")
            print(f"  Model: {response.model}")

        except LLMClientError as e:
            print(f"  Error: {e}")
        except Exception as e:
            print(f"  Skipped: {e}")
        finally:
            if "client" in locals():
                await client.close()


# ============================================================================
# Example 3: Streaming Responses
# ============================================================================


async def streaming_example():
    """Demonstrate streaming responses."""
    print("\n" + "=" * 60)
    print("Example 4: Streaming Response")
    print("=" * 60)

    client = create_openai_client(model="gpt-4-turbo-preview")

    try:
        stream = await client.generate(
            prompt="Write a haiku about artificial intelligence.",
            temperature=0.9,
            max_tokens=50,
            stream=True,
        )

        print("Streaming response: ", end="", flush=True)
        async for chunk in stream:
            print(chunk, end="", flush=True)
        print("\n")

    except LLMClientError as e:
        print(f"Error: {e}")
    finally:
        await client.close()


# ============================================================================
# Example 4: Custom Agent Implementation
# ============================================================================


class SimpleHRMAgent(AsyncAgentBase):
    """
    Example HRM agent using the new async agent base.

    This demonstrates backward compatibility with the existing framework.
    """

    async def _process_impl(self, context: AgentContext) -> AgentResult:
        """Implement hierarchical reasoning."""
        # Step 1: Decompose the problem
        decomposition_prompt = f"""Analyze this query and break it down into key components:

Query: {context.query}

{f"Context: {context.rag_context}" if context.rag_context else ""}

List the key sub-problems that need to be addressed:"""

        decomposition_response = await self.generate_llm_response(
            prompt=decomposition_prompt,
            temperature=0.3,
            max_tokens=500,
        )

        # Step 2: Generate solution
        solution_prompt = f"""Based on this decomposition:

{decomposition_response.text}

Original Query: {context.query}

Provide a comprehensive solution that addresses each component:"""

        solution_response = await self.generate_llm_response(
            prompt=solution_prompt,
            temperature=context.temperature,
            max_tokens=1000,
        )

        # Calculate confidence based on response completeness
        confidence = min(0.9, len(solution_response.text) / 1000)

        return AgentResult(
            response=solution_response.text,
            confidence=confidence,
            metadata={
                "decomposition": decomposition_response.text,
                "decomposition_quality_score": confidence,
            },
            token_usage={
                "total_tokens": (decomposition_response.total_tokens + solution_response.total_tokens),
            },
            intermediate_steps=[
                {"step": "decomposition", "output": decomposition_response.text},
                {"step": "solution", "output": solution_response.text},
            ],
        )


async def custom_agent_example():
    """Demonstrate custom agent implementation."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Agent Implementation")
    print("=" * 60)

    # Create LLM client
    client = create_openai_client(model="gpt-4-turbo-preview")

    # Create agent with the client
    agent = SimpleHRMAgent(
        model_adapter=client,
        name="ExampleHRM",
    )

    try:
        # Process using backward-compatible interface
        result = await agent.process(
            query="What defensive strategy should be employed for a night operation?",
            rag_context="Night operations require enhanced communication and coordination.",
        )

        print(f"Response: {result['response'][:500]}...")
        print(f"\nConfidence: {result['metadata']['confidence']:.2f}")
        print(f"Processing time: {result['metadata']['processing_time_ms']:.2f}ms")
        print(f"Total tokens: {result['metadata']['token_usage']}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()


# ============================================================================
# Example 5: Integration with LangGraphMultiAgentFramework
# ============================================================================


async def framework_integration_example():
    """
    Show how to integrate the new LLM clients with the existing framework.

    The existing LangGraphMultiAgentFramework expects a model_adapter with:
    - async generate(prompt=..., temperature=...) -> response with .text attribute

    Our new clients provide exactly this interface!
    """
    print("\n" + "=" * 60)
    print("Example 6: Framework Integration")
    print("=" * 60)

    # The new client works directly with the existing framework
    # Example of how to use it:

    code_example = """
# In your main application:

from src.adapters.llm import create_client
from langgraph_multi_agent_mcts import LangGraphMultiAgentFramework
import logging

# Create a provider-agnostic client
model_adapter = create_client(
    "openai",  # or "anthropic", "lmstudio"
    model="gpt-4-turbo-preview",
    timeout=60.0,
)

# Use it with the existing framework
framework = LangGraphMultiAgentFramework(
    model_adapter=model_adapter,
    logger=logging.getLogger("framework"),
    mcts_iterations=100,
    max_iterations=2,
)

# Process queries as before
result = await framework.process(
    query="Recommend defensive positions",
    use_rag=True,
    use_mcts=False,
)

print(result["response"])
"""

    print("Code example for integration:")
    print(code_example)

    # Demonstrate that interface is compatible
    client = create_openai_client(model="gpt-4-turbo-preview")

    # This is exactly what the framework does internally
    try:
        response = await client.generate(
            prompt="Test compatibility",
            temperature=0.5,
        )
        print(f"\nVerification - Response type: {type(response)}")
        print(f"Has .text attribute: {hasattr(response, 'text')}")
        print(f"Has .usage attribute: {hasattr(response, 'usage')}")
        print("Interface is fully compatible!")

    except LLMClientError as e:
        print(f"Error (expected if no API key): {e}")
    finally:
        await client.close()


# ============================================================================
# Example 6: Retry and Error Handling
# ============================================================================


async def retry_and_error_handling():
    """Demonstrate retry logic and error handling."""
    print("\n" + "=" * 60)
    print("Example 7: Retry and Error Handling")
    print("=" * 60)

    # The client automatically retries on:
    # - Rate limits (429)
    # - Server errors (5xx)
    # - Connection errors

    client = create_openai_client(
        model="gpt-4-turbo-preview",
        max_retries=3,  # Will retry up to 3 times
        timeout=30.0,
    )

    try:
        response = await client.generate(
            prompt="Test with automatic retry",
            temperature=0.5,
        )
        print(f"Success: {response.text[:100]}...")

    except LLMRateLimitError as e:
        print(f"Rate limited: {e}")
        if e.retry_after:
            print(f"Retry after: {e.retry_after} seconds")

    except LLMClientError as e:
        print(f"Client error: {e}")
        print(f"Provider: {e.provider}")
        print(f"Status code: {e.status_code}")

    finally:
        await client.close()


# ============================================================================
# Example 7: Registering Custom Providers
# ============================================================================


def custom_provider_registration():
    """Demonstrate registering a custom provider."""
    print("\n" + "=" * 60)
    print("Example 8: Custom Provider Registration")
    print("=" * 60)

    # You can register custom providers:
    # register_provider(
    #     "azure",
    #     "src.adapters.llm.azure_client",
    #     "AzureOpenAIClient"
    # )

    # Then use it:
    # client = create_client("azure", deployment_name="my-gpt4")

    print("To register a custom provider:")
    print("""
    from src.adapters.llm import register_provider

    register_provider(
        "azure",
        "src.adapters.llm.azure_client",
        "AzureOpenAIClient"
    )

    # Now you can use it:
    client = create_client("azure", deployment_name="my-deployment")
    """)

    print(f"Currently available: {list_providers()}")


# ============================================================================
# Main
# ============================================================================


async def main():
    """Run all examples."""
    print("LLM Provider Abstraction Layer - Usage Examples")
    print("=" * 60)

    # Note: Some examples require API keys to be set:
    # - OPENAI_API_KEY for OpenAI
    # - ANTHROPIC_API_KEY for Anthropic
    # - LM Studio server running locally

    # Run examples that demonstrate concepts
    custom_provider_registration()
    await framework_integration_example()

    # Uncomment these to run live API calls:
    # await basic_client_usage()
    # await message_based_usage()
    # await provider_switching()
    # await streaming_example()
    # await custom_agent_example()
    # await retry_and_error_handling()


if __name__ == "__main__":
    asyncio.run(main())
