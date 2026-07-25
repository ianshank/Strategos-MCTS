#!/usr/bin/env python3
"""
Test script for LM Studio connection with liquid/lfm2-1.2b model.

Verifies:
1. Connection to LM Studio endpoint
2. Model availability
3. Basic inference capability
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_connection():
    """Test LM Studio connection and basic inference."""
    print("=" * 60)
    print("LM Studio Connection Test")
    print("=" * 60)

    # Load configuration
    print("\n1. Loading configuration...")
    try:
        from src.config.settings import get_settings

        settings = get_settings()
        print(f"   Provider: {settings.LLM_PROVIDER}")
        print(f"   Base URL: {settings.LMSTUDIO_BASE_URL}")
        print(f"   Model: {settings.LMSTUDIO_MODEL}")
        print("   ✓ Configuration loaded successfully")
    except Exception as e:
        print(f"   ✗ Configuration error: {e}")
        return False

    # Test HTTP connection
    print("\n2. Testing HTTP connection...")
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test models endpoint
            response = await client.get(f"{settings.LMSTUDIO_BASE_URL}/models")
            if response.status_code == 200:
                models_data = response.json()
                print("   ✓ Connected to LM Studio")
                print("   Available models:")
                if "data" in models_data:
                    for model in models_data["data"]:
                        print(f"      - {model.get('id', 'unknown')}")
                else:
                    print(f"      {models_data}")
            else:
                print(f"   ✗ HTTP {response.status_code}: {response.text}")
                return False
    except httpx.ConnectError as e:
        print(f"   ✗ Connection failed: {e}")
        print(f"   Make sure LM Studio is running at {settings.LMSTUDIO_BASE_URL}")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test LLM client creation
    print("\n3. Creating LLM client...")
    try:
        from src.adapters.llm import create_client

        client = create_client(
            provider="lmstudio",
            base_url=settings.LMSTUDIO_BASE_URL,
            model=settings.LMSTUDIO_MODEL or "liquid/lfm2-1.2b",
            timeout=120.0,
            max_retries=3,
        )
        print("   ✓ LLM client created")
        print(f"   Model: {client.model}")
    except Exception as e:
        print(f"   ✗ Client creation error: {e}")
        return False

    # Test basic inference
    print("\n4. Testing inference...")
    try:
        response = await client.generate(
            prompt="Hello, please respond with a brief greeting.",
            temperature=0.7,
            max_tokens=50,
        )
        print("   ✓ Inference successful")
        print(f"   Model: {response.model}")
        print(f"   Response: {response.text[:200]}...")
        print(f"   Usage: {response.usage}")
    except Exception as e:
        print(f"   ✗ Inference error: {e}")
        return False

    # Test MCP server initialization
    print("\n5. Testing MCP server...")
    try:
        from tools.mcp.server import MCPServer

        mcp_server = MCPServer()
        init_result = await mcp_server.initialize()
        print("   ✓ MCP server initialized")
        print(f"   Status: {init_result}")

        # List available tools
        tools = mcp_server.get_tools()
        print(f"   Available tools ({len(tools)}):")
        for tool in tools:
            print(f"      - {tool['name']}: {tool['description'][:50]}...")
    except Exception as e:
        print(f"   ✗ MCP server error: {e}")
        return False

    # Test MCTS with LM Studio
    print("\n6. Testing MCTS engine...")
    try:
        from src.framework.mcts.config import FAST_CONFIG
        from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
        from src.framework.mcts.policies import RandomRolloutPolicy

        config = FAST_CONFIG.copy(seed=42)
        engine = MCTSEngine(
            seed=config.seed,
            exploration_weight=config.exploration_weight,
        )

        def action_generator(state):
            depth = len(state.state_id.split("_")) - 1
            if depth == 0:
                return ["analyze", "decompose", "evaluate"]
            elif depth < 2:
                return ["refine", "expand"]
            return []

        def state_transition(state, action):
            return MCTSState(state_id=f"{state.state_id}_{action}", features=state.features.copy())

        root = MCTSNode(
            state=MCTSState(state_id="root", features={"test": True}),
            rng=engine.rng,
        )

        rollout_policy = RandomRolloutPolicy()

        best_action, stats = await engine.search(
            root=root,
            num_iterations=10,  # Quick test
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=rollout_policy,
        )

        print("   ✓ MCTS engine working")
        print(f"   Best action: {best_action}")
        print(f"   Iterations: {stats.get('total_iterations', 0)}")
        print(f"   Seed: {stats.get('seed', 'N/A')}")
    except Exception as e:
        print(f"   ✗ MCTS engine error: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed! LM Studio integration is working.")
    print("=" * 60)

    print("\nQuick Start:")
    print("  1. Run MCP server:")
    print("     python3 tools/mcp/server.py")
    print("")
    print("  2. Or use in code:")
    print("     from src.adapters.llm import create_client")
    print("     client = create_client('lmstudio')")
    print("     response = await client.generate(prompt='Your query')")
    print("")
    print(f"  3. Model: {settings.LMSTUDIO_MODEL or 'liquid/lfm2-1.2b'}")
    print(f"  4. Endpoint: {settings.LMSTUDIO_BASE_URL}")

    return True


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)
