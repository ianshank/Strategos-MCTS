"""
LangGraph Integration Module - Extract graph building with new MCTS core integration.

Provides:
- Graph building extracted from LangGraphMultiAgentFramework
- Integration with new deterministic MCTS core
- Backward compatibility with original process() signature
- Support for parallel HRM/TRM execution
"""

from __future__ import annotations

from typing import Any

# LangGraph imports (these would be installed dependencies)
try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
except ImportError:
    # Stubs for development without LangGraph installed.
    # END must match langgraph.graph.END's actual value so visualization
    # comparisons against "__end__" still match in the stubbed path.
    StateGraph = None  # type: ignore[assignment,misc]
    END = "__end__"
    MemorySaver = None  # type: ignore[assignment,misc]

from ..mcts.config import MCTSConfig
from ..mcts.experiments import ExperimentTracker
from .builder import GraphBuilder


class IntegratedFramework:
    """
    Integrated multi-agent framework with new MCTS core.

    Maintains backward compatibility with original process() signature.
    """

    def __init__(
        self,
        model_adapter,
        logger,
        vector_store=None,
        _embedding_model=None,
        hrm_config: dict | None = None,
        trm_config: dict | None = None,
        mcts_config: MCTSConfig | None = None,
        top_k_retrieval: int = 5,
        max_iterations: int = 3,
        consensus_threshold: float = 0.75,
        enable_parallel_agents: bool = True,
        adk_agents: dict[str, Any] | None = None,
    ):
        """
        Initialize integrated framework.

        Backward compatible with LangGraphMultiAgentFramework.
        """
        self.model_adapter = model_adapter
        self.logger = logger
        self.vector_store = vector_store
        self.adk_agents = adk_agents or {}

        # Import LLM-backed agents for graph processing
        try:
            from src.framework.agents.llm_hrm import LLMHRMAgent
            from src.framework.agents.llm_trm import LLMTRMAgent

            self.hrm_agent: LLMHRMAgent | None = LLMHRMAgent(
                model_adapter=model_adapter,
                logger=logger,
                **(hrm_config or {}),
            )
            self.trm_agent: LLMTRMAgent | None = LLMTRMAgent(
                model_adapter=model_adapter,
                logger=logger,
                **(trm_config or {}),
            )
        except ImportError:
            self.hrm_agent = None
            self.trm_agent = None
            self.logger.warning("Could not import LLM HRM/TRM agents")

        # Build graph
        self.graph_builder = GraphBuilder(
            hrm_agent=self.hrm_agent,
            trm_agent=self.trm_agent,
            model_adapter=model_adapter,
            logger=logger,
            vector_store=vector_store,
            mcts_config=mcts_config,
            top_k_retrieval=top_k_retrieval,
            max_iterations=max_iterations,
            consensus_threshold=consensus_threshold,
            enable_parallel_agents=enable_parallel_agents,
            adk_agents=self.adk_agents,
        )

        # Compile graph
        if StateGraph is not None:
            self.graph = self.graph_builder.build_graph()
            self.memory = MemorySaver() if MemorySaver else None
            self.app = self.graph.compile(checkpointer=self.memory) if self.memory else self.graph.compile()
        else:
            self.graph = None
            self.app = None

        self.logger.info("Integrated framework initialized with new MCTS core")

    async def process(
        self,
        query: str,
        use_rag: bool = True,
        use_mcts: bool = False,
        config: dict | None = None,
    ) -> dict:
        """
        Process query through LangGraph.

        Backward compatible with original signature.

        Args:
            query: User query to process
            use_rag: Enable RAG context retrieval
            use_mcts: Enable MCTS simulation
            config: Optional LangGraph config

        Returns:
            Dictionary with response, metadata, and state
        """
        if self.app is None:
            raise RuntimeError("LangGraph not available. Install with: pip install langgraph")

        initial_state = {
            "query": query,
            "use_rag": use_rag,
            "use_mcts": use_mcts,
            "iteration": 0,
            "max_iterations": self.graph_builder.max_iterations,
            "agent_outputs": [],
        }

        config = config or {"configurable": {"thread_id": "default"}}

        result = await self.app.ainvoke(initial_state, config=config)

        return {
            "response": result.get("final_response", ""),
            "metadata": result.get("metadata", {}),
            "state": result,
        }

    async def astream(
        self,
        query: str,
        use_rag: bool = True,
        use_mcts: bool = False,
        config: dict | None = None,
    ):
        """
        Stream node-level state updates through LangGraph.

        Yields state updates as each node completes execution.

        Args:
            query: User query to process
            use_rag: Enable RAG context retrieval
            use_mcts: Enable MCTS simulation
            config: Optional LangGraph config

        Yields:
            Tuple of (node_name, state_update) for each node completion
        """
        if self.app is None:
            raise RuntimeError("LangGraph not available. Install with: pip install langgraph")

        initial_state = {
            "query": query,
            "use_rag": use_rag,
            "use_mcts": use_mcts,
            "iteration": 0,
            "max_iterations": self.graph_builder.max_iterations,
            "agent_outputs": [],
        }

        config = config or {"configurable": {"thread_id": "default"}}

        self.logger.debug(f"Starting streaming execution for query: {query[:100]}")

        async for event in self.app.astream(initial_state, config=config):
            for node_name, state_update in event.items():
                self.logger.debug(f"Node '{node_name}' completed with keys: {list(state_update.keys())}")
                yield node_name, state_update

    async def astream_events(
        self,
        query: str,
        use_rag: bool = True,
        use_mcts: bool = False,
        config: dict | None = None,
        include_types: list[str] | None = None,
    ):
        """
        Stream detailed events from LangGraph execution.

        Provides fine-grained streaming including token-level events when
        LLM calls support streaming, and node lifecycle events.

        Args:
            query: User query to process
            use_rag: Enable RAG context retrieval
            use_mcts: Enable MCTS simulation
            config: Optional LangGraph config
            include_types: Event types to include (default: all)
                Supported: ['on_llm_start', 'on_llm_stream', 'on_llm_end',
                           'on_chain_start', 'on_chain_end', 'on_tool_start', 'on_tool_end']

        Yields:
            StreamEvent dict with event type, name, data, and metadata
        """
        if self.app is None:
            raise RuntimeError("LangGraph not available. Install with: pip install langgraph")

        initial_state = {
            "query": query,
            "use_rag": use_rag,
            "use_mcts": use_mcts,
            "iteration": 0,
            "max_iterations": self.graph_builder.max_iterations,
            "agent_outputs": [],
        }

        config = config or {"configurable": {"thread_id": "default"}}

        # Default to all event types if not specified
        include_types = include_types or [
            "on_llm_start",
            "on_llm_stream",
            "on_llm_end",
            "on_chain_start",
            "on_chain_end",
        ]

        self.logger.debug(f"Starting event streaming for query: {query[:100]}, event_types: {include_types}")

        try:
            async for event in self.app.astream_events(
                initial_state,
                config=config,
                version="v2",
            ):
                event_type = event.get("event", "")

                # Filter by event type if specified
                if include_types and event_type not in include_types:
                    continue

                # Construct standardized event format
                stream_event = {
                    "event_type": event_type,
                    "name": event.get("name", ""),
                    "run_id": event.get("run_id", ""),
                    "data": event.get("data", {}),
                    "metadata": event.get("metadata", {}),
                    "tags": event.get("tags", []),
                }

                # For LLM streaming events, extract token content
                if event_type == "on_llm_stream":
                    chunk = event.get("data", {}).get("chunk", {})
                    if hasattr(chunk, "content"):
                        stream_event["token"] = chunk.content
                    elif isinstance(chunk, dict):
                        stream_event["token"] = chunk.get("content", "")

                self.logger.debug(f"Event: {event_type}, name: {stream_event['name']}")
                yield stream_event

        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            # Yield error event
            yield {
                "event_type": "on_error",
                "name": "streaming_error",
                "run_id": "",
                "data": {"error": str(e)},
                "metadata": {},
                "tags": [],
            }

    def get_experiment_tracker(self) -> ExperimentTracker:
        """Get the experiment tracker for analysis."""
        return self.graph_builder.experiment_tracker

    def set_mcts_seed(self, seed: int) -> None:
        """Set MCTS seed for deterministic behavior."""
        self.graph_builder.mcts_engine.reset_seed(seed)
        self.graph_builder.mcts_config.seed = seed

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def get_graph(self):
        """
        Get the compiled LangGraph application.

        Returns:
            The compiled LangGraph app object.
        """
        return self.app

    def get_graph_structure(self) -> dict:
        """
        Extract the graph structure for visualization and introspection.

        Returns:
            Dict containing nodes, edges, and conditional routing info.
        """
        # Core node definitions with descriptions
        nodes = [
            {"id": "entry", "label": "Entry", "description": "Input validation", "type": "start"},
            {"id": "retrieve_context", "label": "Retrieve Context", "description": "RAG retrieval", "type": "process"},
            {"id": "route_decision", "label": "Route Decision", "description": "Agent selection", "type": "branch"},
            {
                "id": "parallel_agents",
                "label": "Parallel Agents",
                "description": "HRM + TRM parallel",
                "type": "process",
            },
            {"id": "hrm_agent", "label": "HRM Agent", "description": "Hierarchical reasoning", "type": "agent"},
            {"id": "trm_agent", "label": "TRM Agent", "description": "Task refinement", "type": "agent"},
            {"id": "mcts_simulator", "label": "MCTS Simulator", "description": "Monte Carlo search", "type": "agent"},
            {
                "id": "aggregate_results",
                "label": "Aggregate Results",
                "description": "Combine outputs",
                "type": "process",
            },
            {
                "id": "evaluate_consensus",
                "label": "Evaluate Consensus",
                "description": "Check agreement",
                "type": "branch",
            },
            {"id": "synthesize", "label": "Synthesize", "description": "Final response", "type": "end"},
        ]

        # Add symbolic agent if enabled
        if self.graph_builder.use_symbolic_reasoning:
            nodes.insert(
                7,
                {
                    "id": "symbolic_agent",
                    "label": "Symbolic Agent",
                    "description": "Neuro-symbolic reasoning",
                    "type": "agent",
                },
            )

        # Add ADK agent nodes
        for name in self.graph_builder.adk_agents:
            nodes.append(
                {
                    "id": f"adk_{name}",
                    "label": f"ADK: {name}",
                    "description": f"Google ADK agent: {name}",
                    "type": "agent",
                }
            )

        # Sequential edges
        edges = [
            {"source": "__start__", "target": "entry"},
            {"source": "entry", "target": "retrieve_context"},
            {"source": "retrieve_context", "target": "route_decision"},
            {"source": "parallel_agents", "target": "aggregate_results"},
            {"source": "hrm_agent", "target": "aggregate_results"},
            {"source": "trm_agent", "target": "aggregate_results"},
            {"source": "mcts_simulator", "target": "aggregate_results"},
            {"source": "aggregate_results", "target": "evaluate_consensus"},
            {"source": "synthesize", "target": END},
        ]

        # Add symbolic agent edge
        if self.graph_builder.use_symbolic_reasoning:
            edges.append({"source": "symbolic_agent", "target": "aggregate_results"})

        # Add ADK agent edges
        for name in self.graph_builder.adk_agents:
            edges.append({"source": f"adk_{name}", "target": "aggregate_results"})

        # Conditional routing
        routing_map = {
            "parallel": "parallel_agents",
            "hrm": "hrm_agent",
            "trm": "trm_agent",
            "mcts": "mcts_simulator",
            "aggregate": "aggregate_results",
        }

        if self.graph_builder.use_symbolic_reasoning:
            routing_map["symbolic"] = "symbolic_agent"

        for name in self.graph_builder.adk_agents:
            routing_map[f"adk_{name}"] = f"adk_{name}"

        conditional_edges = {
            "route_decision": {
                "condition": "_route_to_agents",
                "routes": routing_map,
            },
            "evaluate_consensus": {
                "condition": "_check_consensus",
                "routes": {
                    "synthesize": "synthesize",
                    "iterate": "route_decision",
                },
            },
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "conditional_edges": conditional_edges,
            "entry_point": "entry",
            "terminal_node": "synthesize",
        }

    def get_graph_mermaid(
        self,
        include_descriptions: bool = True,
        theme: str = "default",
    ) -> str:
        """
        Generate a Mermaid flowchart diagram of the graph.

        Args:
            include_descriptions: Include node descriptions in labels.
            theme: Mermaid theme (default, forest, dark, neutral).

        Returns:
            Mermaid diagram source code.
        """
        structure = self.get_graph_structure()
        lines = []

        # Add theme configuration
        if theme != "default":
            lines.append(f"%%{{init: {{'theme': '{theme}'}}}}%%")

        lines.append("flowchart TD")

        # Define node styles by type
        lines.append("    %% Node style definitions")
        lines.append("    classDef startNode fill:#90EE90,stroke:#228B22")
        lines.append("    classDef endNode fill:#FFB6C1,stroke:#DC143C")
        lines.append("    classDef processNode fill:#87CEEB,stroke:#4169E1")
        lines.append("    classDef branchNode fill:#DDA0DD,stroke:#8B008B")
        lines.append("    classDef agentNode fill:#FFD700,stroke:#DAA520")
        lines.append("")

        # Add nodes
        lines.append("    %% Nodes")
        node_type_map = {}
        for node in structure["nodes"]:
            node_id = node["id"]
            label = node["label"]
            node_type = node["type"]
            node_type_map[node_id] = node_type

            if include_descriptions and node.get("description"):
                label = f"{label}<br/><small>{node['description']}</small>"

            # Use different shapes based on node type
            if node_type == "start" or node_type == "end":
                lines.append(f"    {node_id}([{label}])")
            elif node_type == "branch":
                lines.append(f"    {node_id}{{{label}}}")
            elif node_type == "agent":
                lines.append(f"    {node_id}[/{label}/]")
            else:
                lines.append(f"    {node_id}[{label}]")

        lines.append("")

        # Add sequential edges
        lines.append("    %% Sequential edges")
        for edge in structure["edges"]:
            source = edge["source"]
            target = edge["target"]

            # Handle special start/end nodes
            if source == "__start__":
                lines.append(f"    START(( )) --> {target}")
            elif target == END:
                lines.append(f"    {source} --> END(( ))")
            else:
                lines.append(f"    {source} --> {target}")

        lines.append("")

        # Add conditional edges
        lines.append("    %% Conditional edges")
        for source_node, edge_info in structure["conditional_edges"].items():
            routes = edge_info["routes"]
            for label, target in routes.items():
                lines.append(f"    {source_node} -->|{label}| {target}")

        lines.append("")

        # Apply styles
        lines.append("    %% Apply styles")
        for node_id, node_type in node_type_map.items():
            if node_type == "start":
                lines.append(f"    class {node_id} startNode")
            elif node_type == "end":
                lines.append(f"    class {node_id} endNode")
            elif node_type == "process":
                lines.append(f"    class {node_id} processNode")
            elif node_type == "branch":
                lines.append(f"    class {node_id} branchNode")
            elif node_type == "agent":
                lines.append(f"    class {node_id} agentNode")

        return "\n".join(lines)

    def draw_mermaid(
        self,
        output_file: str | None = None,
        format: str = "png",
        include_descriptions: bool = True,
    ) -> str:
        """
        Generate and optionally render a Mermaid diagram of the graph.

        This method generates the Mermaid source code and can optionally
        render it to PNG via the Kroki API.

        Args:
            output_file: Optional file path to save PNG (requires network).
            format: Output format ('png', 'svg').
            include_descriptions: Include node descriptions.

        Returns:
            Mermaid diagram source code.

        Raises:
            RuntimeError: If rendering fails and output_file is specified.
        """
        import base64
        import zlib

        mermaid_code = self.get_graph_mermaid(include_descriptions=include_descriptions)

        # If no output file requested, just return the code
        if not output_file:
            return mermaid_code

        # Render via Kroki API
        try:
            import httpx

            # Compress and encode for Kroki
            compressed = zlib.compress(mermaid_code.encode("utf-8"), 9)
            encoded = base64.urlsafe_b64encode(compressed).decode("ascii")

            # Build Kroki URL
            kroki_url = f"https://kroki.io/mermaid/{format}/{encoded}"

            # Fetch rendered diagram
            with httpx.Client(timeout=30.0) as client:
                response = client.get(kroki_url)
                response.raise_for_status()

            # Save to file
            with open(output_file, "wb") as f:
                f.write(response.content)

            self.logger.info(f"Saved graph diagram to {output_file}")

        except ImportError:
            self.logger.warning("httpx not available for Kroki rendering. Install with: pip install httpx")
        except Exception as e:
            self.logger.error(f"Failed to render diagram: {e}")
            raise RuntimeError(f"Diagram rendering failed: {e}") from e

        return mermaid_code

    def get_execution_trace_mermaid(
        self,
        execution_path: list[str],
        timings: dict[str, float] | None = None,
    ) -> str:
        """
        Generate a Mermaid diagram showing an execution trace.

        Args:
            execution_path: List of node IDs in execution order.
            timings: Optional dict of node_id -> execution_time_ms.

        Returns:
            Mermaid sequence diagram source code.
        """
        lines = ["sequenceDiagram"]
        lines.append("    autonumber")
        lines.append("")

        # Define participants
        unique_nodes = []
        for node in execution_path:
            if node not in unique_nodes:
                unique_nodes.append(node)

        for node in unique_nodes:
            # Format label
            label = node.replace("_", " ").title()
            lines.append(f"    participant {node} as {label}")

        lines.append("")

        # Add execution arrows
        for i, node in enumerate(execution_path):
            if i == 0:
                lines.append(f"    Note over {node}: Start")
            elif i < len(execution_path) - 1:
                prev_node = execution_path[i - 1]
                timing_label = ""
                if timings and prev_node in timings:
                    timing_label = f" [{timings[prev_node]:.1f}ms]"
                lines.append(f"    {prev_node}->>+{node}: Execute{timing_label}")

            if i == len(execution_path) - 1:
                lines.append(f"    Note over {node}: End")

        return "\n".join(lines)
