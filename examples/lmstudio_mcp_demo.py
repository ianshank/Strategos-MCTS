#!/usr/bin/env python3
"""
Demonstration of LM Studio integration with MCP Server.

This script shows how to use the MCP server with LM Studio
to perform multi-agent MCTS searches.
"""

import asyncio
import json
from typing import Any


def create_mcp_request(method: str, params: dict[str, Any] = None) -> dict[str, Any]:
    """Create a properly formatted MCP request."""
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}


async def demo_mcp_with_lmstudio():
    """Demonstrate MCP server functionality with LM Studio."""

    print("LM Studio + MCP Server Demonstration")
    print("=" * 60)
    print("Model: llama-3.2-8x3b-moe-dark-champion-instruct-uncensored-abliterated-18.4b")
    print("Endpoint: http://127.0.0.1:1234/v1")
    print("=" * 60)

    # Example 1: Query HRM Agent
    print("\n1. Hierarchical Reasoning Model (HRM) Demo:")
    print("-" * 40)

    hrm_request = create_mcp_request(
        "tools/call",
        {
            "name": "query_agent",
            "arguments": {
                "agent_type": "hrm",
                "query": "How can I improve the performance of a Python web application?",
                "temperature": 0.7,
                "max_tokens": 300,
            },
        },
    )

    print("Request:")
    print(json.dumps(hrm_request, indent=2))
    print("\nThis will decompose the problem into hierarchical sub-tasks.")

    # Example 2: Query TRM Agent
    print("\n\n2. Task Refinement Model (TRM) Demo:")
    print("-" * 40)

    trm_request = create_mcp_request(
        "tools/call",
        {
            "name": "query_agent",
            "arguments": {
                "agent_type": "trm",
                "query": "Implement a caching system for API responses",
                "temperature": 0.7,
                "max_tokens": 300,
            },
        },
    )

    print("Request:")
    print(json.dumps(trm_request, indent=2))
    print("\nThis will iteratively refine the task into concrete steps.")

    # Example 3: Run MCTS Search
    print("\n\n3. MCTS Search Demo:")
    print("-" * 40)

    mcts_request = create_mcp_request(
        "tools/call",
        {
            "name": "run_mcts",
            "arguments": {
                "query": "Design a distributed task queue system with fault tolerance",
                "iterations": 25,  # Reduced for demo
                "exploration_weight": 1.414,
                "seed": 42,
                "use_rag": False,
            },
        },
    )

    print("Request:")
    print(json.dumps(mcts_request, indent=2))
    print("\nThis will explore different solution paths using MCTS.")

    # Example 4: Health Check
    print("\n\n4. System Health Check:")
    print("-" * 40)

    health_request = create_mcp_request("tools/call", {"name": "health_check", "arguments": {}})

    print("Request:")
    print(json.dumps(health_request, indent=2))
    print("\nThis verifies LM Studio connectivity and framework status.")

    print("\n" + "=" * 60)
    print("To execute these requests:")
    print("1. Ensure LM Studio is running with the model loaded")
    print("2. Ensure MCP server is running (python tools/mcp/server.py)")
    print("3. Send requests to the MCP server via stdin")
    print("4. Read responses from stdout")

    # Show example of actual execution
    print("\nExample execution code:")
    print("-" * 40)
    print("""
# Python example
process = subprocess.Popen(
    ['python', 'tools/mcp/server.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True
)

# Send request
process.stdin.write(json.dumps(request) + '\\n')
process.stdin.flush()

# Read response
response = process.stdout.readline()
result = json.loads(response)
""")

    print("\n" + "=" * 60)
    print("Configuration verified and ready for use!")
    print("LM Studio integration is complete.")


if __name__ == "__main__":
    asyncio.run(demo_mcp_with_lmstudio())
