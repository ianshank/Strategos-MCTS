#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server for Multi-Agent MCTS Framework.

Exposes tools for running MCTS searches, agent queries, and retrieving artifacts.
All tools are async and use Pydantic validation for inputs.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import BaseModel, Field, field_validator

# Import centralized validation from framework
from src.models.validation import QueryInput

# MCP Protocol imports (simplified implementation)
# In production, use the official MCP SDK


class MCPRequest(BaseModel):
    """Base MCP request model."""

    jsonrpc: str = "2.0"
    id: int | str
    method: str
    params: dict = Field(default_factory=dict)


class MCPResponse(BaseModel):
    """Base MCP response model."""

    jsonrpc: str = "2.0"
    id: int | str
    result: Any = None
    error: dict | None = None


# Tool Input Models with Pydantic Validation


class RunMCTSInput(BaseModel):
    """Input for running MCTS search."""

    query: str = Field(..., min_length=1, max_length=10000, description="The query to process")
    iterations: int = Field(default=100, ge=1, le=10000, description="Number of MCTS iterations")
    exploration_weight: float = Field(default=1.414, ge=0.0, le=10.0, description="UCB1 exploration constant")
    seed: int | None = Field(default=None, description="Random seed for determinism")
    use_rag: bool = Field(default=False, description="Enable RAG context retrieval")

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """
        Sanitize input query using centralized validation.

        Delegates to QueryInput.sanitize_query for consistent security checks.
        """
        # Use the centralized sanitization from QueryInput
        return QueryInput.sanitize_query(v)


class QueryAgentInput(BaseModel):
    """Input for querying a specific agent."""

    agent_type: str = Field(..., pattern="^(hrm|trm|mcts)$", description="Agent type: hrm, trm, or mcts")
    query: str = Field(..., min_length=1, max_length=10000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=100000)

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """
        Sanitize input query using centralized validation.

        Delegates to QueryInput.sanitize_query for consistent security checks.
        """
        # Use the centralized sanitization from QueryInput
        return QueryInput.sanitize_query(v)


class GetArtifactInput(BaseModel):
    """Input for retrieving stored artifacts."""

    artifact_id: str = Field(..., min_length=1, max_length=256)
    artifact_type: str = Field(default="mcts_stats", pattern="^(mcts_stats|config|trace|log)$")


class ListArtifactsInput(BaseModel):
    """Input for listing available artifacts."""

    artifact_type: str | None = Field(default=None, pattern="^(mcts_stats|config|trace|log)$")
    limit: int = Field(default=50, ge=1, le=1000)


class MCPServer:
    """
    MCP Server exposing Multi-Agent MCTS Framework tools.

    Tools:
    - run_mcts: Execute MCTS search with configurable parameters
    - query_agent: Query individual agents (HRM, TRM, MCTS)
    - get_artifact: Retrieve stored artifacts
    - list_artifacts: List available artifacts
    - get_config: Get current framework configuration
    - health_check: Check framework and LLM provider health
    """

    def __init__(self):
        self.artifacts: dict[str, dict] = {}
        self.run_history: list[dict] = []
        self._llm_client = None
        self._chess_engine = None  # Lazy-initialised by chess tools

    async def initialize(self):
        """Initialize the framework and LLM client."""
        try:
            from src.adapters.llm import create_client
            from src.config.settings import get_settings

            settings = get_settings()

            # Create LLM client based on settings
            self._llm_client = create_client(
                provider=settings.LLM_PROVIDER,
                base_url=settings.LMSTUDIO_BASE_URL if settings.LLM_PROVIDER == "lmstudio" else None,
                timeout=float(settings.HTTP_TIMEOUT_SECONDS),
                max_retries=settings.HTTP_MAX_RETRIES,
            )

            return {"status": "initialized", "provider": settings.LLM_PROVIDER}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_tools(self) -> list[dict]:
        """Return list of available tools with schemas."""
        from src.games.chess.mcp_chess_tools import get_chess_tool_definitions

        base_tools = [
            {
                "name": "run_mcts",
                "description": "Execute MCTS search with the multi-agent framework",
                "inputSchema": RunMCTSInput.model_json_schema(),
            },
            {
                "name": "query_agent",
                "description": "Query a specific agent (HRM, TRM, or MCTS)",
                "inputSchema": QueryAgentInput.model_json_schema(),
            },
            {
                "name": "get_artifact",
                "description": "Retrieve a stored artifact by ID",
                "inputSchema": GetArtifactInput.model_json_schema(),
            },
            {
                "name": "list_artifacts",
                "description": "List available artifacts",
                "inputSchema": ListArtifactsInput.model_json_schema(),
            },
            {
                "name": "get_config",
                "description": "Get current framework configuration",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "health_check",
                "description": "Check health of framework and LLM provider",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
        # Append chess-specific tools
        base_tools.extend(get_chess_tool_definitions())
        return base_tools

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Execute a tool by name with given arguments."""
        from src.games.chess.mcp_chess_tools import CHESS_TOOL_HANDLERS, dispatch_chess_tool

        tools = {
            "run_mcts": self._run_mcts,
            "query_agent": self._query_agent,
            "get_artifact": self._get_artifact,
            "list_artifacts": self._list_artifacts,
            "get_config": self._get_config,
            "health_check": self._health_check,
        }

        if name in tools:
            return await tools[name](arguments)

        # Delegate to chess tools if recognised
        if name in CHESS_TOOL_HANDLERS:
            return await dispatch_chess_tool(name, arguments, engine=self._chess_engine)

        raise ValueError(f"Unknown tool: {name}")

    async def _run_mcts(self, args: dict) -> dict:
        """Run MCTS search."""
        input_data = RunMCTSInput(**args)

        try:
            from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
            from src.framework.mcts.policies import RandomRolloutPolicy

            # Create engine with seed
            engine = MCTSEngine(
                seed=input_data.seed if input_data.seed is not None else 42,
                exploration_weight=input_data.exploration_weight,
            )

            # Simple action generator for demo
            def action_generator(state: MCTSState) -> list[str]:
                depth = len(state.state_id.split("_")) - 1
                if depth == 0:
                    return ["analyze", "decompose", "simulate", "evaluate"]
                elif depth < 3:
                    return ["refine", "expand", "consolidate"]
                return []

            def state_transition(state: MCTSState, action: str) -> MCTSState:
                new_id = f"{state.state_id}_{action}"
                features = state.features.copy()
                features["last_action"] = action
                features["query"] = input_data.query[:100]
                return MCTSState(state_id=new_id, features=features)

            # Create root node
            root = MCTSNode(
                state=MCTSState(
                    state_id="root", features={"query": input_data.query[:100], "use_rag": input_data.use_rag}
                ),
                rng=engine.rng,
            )

            # Create rollout policy
            rollout_policy = RandomRolloutPolicy()

            # Run search
            best_action, stats = await engine.search(
                root=root,
                num_iterations=input_data.iterations,
                action_generator=action_generator,
                state_transition=state_transition,
                rollout_policy=rollout_policy,
            )

            # Store artifact
            artifact_id = f"mcts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{stats.get('seed', 0)}"
            self.artifacts[artifact_id] = {
                "type": "mcts_stats",
                "query": input_data.query,
                "stats": stats,
                "best_action": best_action,
                "timestamp": datetime.utcnow().isoformat(),
            }

            return {
                "success": True,
                "best_action": best_action,
                "stats": stats,
                "artifact_id": artifact_id,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def _query_agent(self, args: dict) -> dict:
        """Query a specific agent."""
        input_data = QueryAgentInput(**args)

        if not self._llm_client:
            await self.initialize()

        if not self._llm_client:
            return {"success": False, "error": "LLM client not initialized"}

        try:
            # Create agent-specific prompt
            prompts = {
                "hrm": f"As a Hierarchical Reasoning Model (HRM), decompose this query into sub-problems:\n\n{input_data.query}",
                "trm": f"As a Task Refinement Model (TRM), iteratively refine this query:\n\n{input_data.query}",
                "mcts": f"As an MCTS planner, suggest possible actions for:\n\n{input_data.query}",
            }

            prompt = prompts[input_data.agent_type]

            response = await self._llm_client.generate(
                prompt=prompt,
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens,
            )

            return {
                "success": True,
                "agent_type": input_data.agent_type,
                "response": response.text,
                "model": response.model,
                "usage": response.usage,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def _get_artifact(self, args: dict) -> dict:
        """Retrieve a stored artifact."""
        input_data = GetArtifactInput(**args)

        if input_data.artifact_id in self.artifacts:
            artifact = self.artifacts[input_data.artifact_id]
            if artifact.get("type") == input_data.artifact_type:
                return {"success": True, "artifact": artifact}
            else:
                return {
                    "success": False,
                    "error": f"Artifact type mismatch. Expected {input_data.artifact_type}, got {artifact.get('type')}",
                }

        return {"success": False, "error": f"Artifact {input_data.artifact_id} not found"}

    async def _list_artifacts(self, args: dict) -> dict:
        """List available artifacts."""
        input_data = ListArtifactsInput(**args)

        artifacts = []
        for artifact_id, artifact in self.artifacts.items():
            if input_data.artifact_type is None or artifact.get("type") == input_data.artifact_type:
                artifacts.append(
                    {
                        "id": artifact_id,
                        "type": artifact.get("type"),
                        "timestamp": artifact.get("timestamp"),
                    }
                )

        # Sort by timestamp descending and limit
        artifacts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        artifacts = artifacts[: input_data.limit]

        return {"success": True, "artifacts": artifacts, "total": len(artifacts)}

    async def _get_config(self, args: dict) -> dict:
        """Get current framework configuration."""
        try:
            from src.config.settings import get_settings

            settings = get_settings()
            return {
                "success": True,
                "config": settings.safe_dict(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _health_check(self, args: dict) -> dict:
        """Check health of framework components."""
        health = {
            "framework": "unknown",
            "llm_provider": "unknown",
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Check framework imports
        try:
            from src.config.settings import get_settings

            settings = get_settings()
            health["framework"] = "healthy"
            health["provider"] = settings.LLM_PROVIDER
            health["lmstudio_url"] = settings.LMSTUDIO_BASE_URL
        except Exception as e:
            health["framework"] = f"error: {str(e)}"

        # Check LLM provider connection
        if self._llm_client:
            try:
                # Try a simple health check if available
                if hasattr(self._llm_client, "health_check"):
                    is_healthy = await self._llm_client.health_check()
                    health["llm_provider"] = "healthy" if is_healthy else "unhealthy"
                else:
                    health["llm_provider"] = "client_initialized"
            except Exception as e:
                health["llm_provider"] = f"error: {str(e)}"
        else:
            health["llm_provider"] = "not_initialized"

        return {"success": True, "health": health}

    async def handle_request(self, request_data: dict) -> dict:
        """Handle incoming MCP request."""
        try:
            request = MCPRequest(**request_data)

            if request.method == "initialize":
                result = await self.initialize()
            elif request.method == "tools/list":
                result = {"tools": self.get_tools()}
            elif request.method == "tools/call":
                tool_name = request.params.get("name")
                if not isinstance(tool_name, str):
                    raise ValueError("Tool name must be a string")

                tool_args = request.params.get("arguments", {})
                if not isinstance(tool_args, dict):
                    raise ValueError("Tool arguments must be a dictionary")

                result = await self.call_tool(tool_name, tool_args)
            else:
                return MCPResponse(
                    id=request.id, error={"code": -32601, "message": f"Method not found: {request.method}"}
                ).model_dump()

            return MCPResponse(id=request.id, result=result).model_dump()

        except Exception as e:
            return MCPResponse(id=request_data.get("id", 0), error={"code": -32603, "message": str(e)}).model_dump()

    async def run_stdio(self):
        """Run MCP server over stdio (standard input/output)."""
        print(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {"serverInfo": {"name": "mcts-framework", "version": "0.1.0"}},
                }
            ),
            flush=True,
        )

        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break

                request_data = json.loads(line.strip())
                response = await self.handle_request(request_data)
                print(json.dumps(response), flush=True)

            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                }
                print(json.dumps(error_response), flush=True)
            except KeyboardInterrupt:
                break
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                }
                print(json.dumps(error_response), flush=True)


async def main():
    """Main entry point for MCP server."""
    server = MCPServer()
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
