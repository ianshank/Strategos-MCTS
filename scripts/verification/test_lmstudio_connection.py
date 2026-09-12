#!/usr/bin/env python3
"""
Test script for LM Studio connection with a locally served OpenAI-compat model.

Verifies:
1. Connection to LM Studio ``/v1`` endpoint
2. Model availability
3. Basic inference (twice at temperature=0)
MCP/MCTS sections are advisory and do not fail the adapter lane.
"""

import asyncio
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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
        print("   [ok] Configuration loaded successfully")
    except Exception as e:
        print(f"   [FAIL] Configuration error: {e}")
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
                print("   [ok] Connected to LM Studio")
                print("   Available models:")
                if "data" in models_data:
                    for model in models_data["data"]:
                        print(f"      - {model.get('id', 'unknown')}")
                else:
                    print(f"      {models_data}")
            else:
                print(f"   [FAIL] HTTP {response.status_code}: {response.text}")
                return False
    except httpx.ConnectError as e:
        print(f"   [FAIL] Connection failed: {e}")
        print(f"   Make sure LM Studio is running at {settings.LMSTUDIO_BASE_URL}")
        return False
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        return False

    # Test LLM client creation
    print("\n3. Creating LLM client...")
    try:
        from src.adapters.llm import create_client

        client = create_client(
            provider="lmstudio",
            base_url=settings.LMSTUDIO_BASE_URL,
            model=settings.LMSTUDIO_MODEL or "local-model",
            timeout=settings.LMSTUDIO_TIMEOUT,
            max_retries=3,
        )
        print("   [ok] LLM client created")
        print(f"   Model: {client.model}")
    except Exception as e:
        print(f"   [FAIL] Client creation error: {e}")
        return False

    # Test basic inference
    print("\n4. Testing inference (temperature=0, twice)...")
    try:
        first = await client.generate(
            prompt="Reply with the single word ping.",
            temperature=0.0,
            max_tokens=128,
        )
        second = await client.generate(
            prompt="Reply with the single word ping.",
            temperature=0.0,
            max_tokens=128,
        )
        if not first.text.strip() or not second.text.strip():
            print("   [FAIL] Empty generate() text")
            return False
        print("   [ok] Inference successful (2 calls)")
        print(f"   Model: {first.model}")
        print(f"   Response 1: {first.text[:200]}")
        print(f"   Response 2: {second.text[:200]}")
    except Exception as e:
        print(f"   [FAIL] Inference error: {e}")
        return False

    # Test MCP server initialization (advisory; not adapter QA)
    print("\n5. Testing MCP server (advisory)...")
    try:
        from tools.mcp.server import MCPServer

        mcp_server = MCPServer()
        init_result = await mcp_server.initialize()
        print("   [ok] MCP server initialized")
        print(f"   Status: {init_result}")

        # List available tools
        tools = mcp_server.get_tools()
        print(f"   Available tools ({len(tools)}):")
        for tool in tools:
            print(f"      - {tool['name']}: {tool['description'][:50]}...")
    except Exception as e:
        print(f"   ! MCP server skipped: {e}")

    # Test MCTS with LM Studio
    print("\n6. Testing MCTS engine (advisory)...")
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

        print("   [ok] MCTS engine working")
        print(f"   Best action: {best_action}")
        print(f"   Iterations: {stats.get('total_iterations', 0)}")
        print(f"   Seed: {stats.get('seed', 'N/A')}")
    except Exception as e:
        print(f"   ! MCTS engine skipped: {e}")

    print("\n" + "=" * 60)
    print("[ok] Adapter checks passed. LM Studio integration is working.")
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
    print(f"  3. Model: {settings.LMSTUDIO_MODEL or 'local-model'}")
    print(f"  4. Endpoint: {settings.LMSTUDIO_BASE_URL}")

    return True


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)
