"""
LangGraph extension for enterprise use cases.

Integrates enterprise use cases with the existing GraphBuilder pattern,
enabling MCTS-guided multi-agent orchestration for enterprise domains.
"""

from __future__ import annotations

import logging
import operator
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, TypedDict

if TYPE_CHECKING:
    from src.framework.graph import GraphBuilder

from ..base.domain_detector import DomainDetector, get_domain_detector
from ..base.use_case import UseCaseProtocol
from ..config.enterprise_settings import EnterpriseDomain
from ..factories.use_case_factory import EnterpriseUseCaseFactory


class EnterpriseAgentState(TypedDict, total=False):
    """Extended agent state with enterprise use case support."""

    # Standard agent state fields
    query: str
    use_mcts: bool
    use_rag: bool
    hrm_results: dict[str, Any]
    trm_results: dict[str, Any]
    agent_outputs: Annotated[list[dict], operator.add]
    confidence_scores: dict[str, float]
    consensus_reached: bool
    iteration: int
    max_iterations: int

    # Enterprise-specific fields
    enterprise_domain: NotRequired[str]
    domain_state: NotRequired[dict[str, Any]]
    domain_agents_results: NotRequired[dict[str, Any]]
    use_case_metadata: NotRequired[dict[str, Any]]
    enterprise_result: NotRequired[dict[str, Any]]


class EnterpriseGraphBuilder:
    """
    Extends GraphBuilder with enterprise use case support.

    Provides:
    - Dynamic use case loading based on configuration
    - Domain-specific agent routing
    - Specialized MCTS integration per use case
    - Seamless integration with existing graph infrastructure

    Example:
        >>> builder = EnterpriseGraphBuilder(base_builder)
        >>> graph = builder.build_enterprise_graph()
        >>> result = await graph.ainvoke({"query": "Analyze acquisition"})
    """

    def __init__(
        self,
        base_graph_builder: GraphBuilder | None = None,
        use_case_factory: EnterpriseUseCaseFactory | None = None,
        domain_detector: DomainDetector | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the enterprise graph builder.

        Args:
            base_graph_builder: Existing graph builder to extend
            use_case_factory: Factory for creating use cases
            domain_detector: Optional domain detector (uses singleton if not provided)
            logger: Optional logger instance
        """
        self._base_builder = base_graph_builder
        self._factory = use_case_factory or EnterpriseUseCaseFactory()
        self._detector = domain_detector or get_domain_detector()
        self._logger = logger or logging.getLogger(__name__)
        self._use_cases: dict[EnterpriseDomain, UseCaseProtocol] = {}

        # Load enabled use cases
        self._load_use_cases()

    def _load_use_cases(self) -> None:
        """Load all enabled enterprise use cases."""
        self._use_cases = self._factory.create_all_enabled()
        self._logger.info(f"Loaded {len(self._use_cases)} enterprise use cases: {[d.value for d in self._use_cases]}")

    @property
    def use_cases(self) -> dict[EnterpriseDomain, UseCaseProtocol]:
        """Get loaded use cases."""
        return self._use_cases

    def get_use_case(self, domain: EnterpriseDomain) -> UseCaseProtocol | None:
        """Get a specific use case by domain."""
        return self._use_cases.get(domain)

    async def process_enterprise_query(
        self,
        state: EnterpriseAgentState,
    ) -> dict[str, Any]:
        """
        Process a query through enterprise use cases.

        This can be used as a node in the LangGraph workflow.

        Args:
            state: Current agent state

        Returns:
            Updated state with enterprise results
        """
        query = state.get("query", "")

        # Auto-detect domain from query
        use_case = self._factory.create_from_query(query)

        if use_case is None:
            self._logger.info("No enterprise domain detected for query")
            return {
                "enterprise_domain": None,
                "enterprise_result": None,
            }

        self._logger.info(f"Processing with enterprise use case: {use_case.name}")

        # Process query
        result = await use_case.process(
            query=query,
            context={
                "rag_context": state.get("hrm_results", {}).get("context", ""),
                **state.get("use_case_metadata", {}),
            },
            use_mcts=state.get("use_mcts", True),
        )

        return {
            "enterprise_domain": use_case.domain,
            "domain_state": result.get("domain_state", {}),
            "domain_agents_results": result.get("agent_results", {}),
            "enterprise_result": result,
            "agent_outputs": [
                {
                    "agent": f"enterprise_{use_case.name}",
                    "response": result.get("result", ""),
                    "confidence": result.get("confidence", 0.0),
                }
            ],
            "use_case_metadata": {
                "use_case": use_case.name,
                "domain": use_case.domain,
                "mcts_stats": result.get("mcts_stats", {}),
            },
        }

    def create_enterprise_node(self, domain: EnterpriseDomain):
        """
        Create a LangGraph node for a specific enterprise domain.

        Args:
            domain: Enterprise domain to create node for

        Returns:
            Async function suitable for use as a LangGraph node
        """
        use_case = self._use_cases.get(domain)
        if use_case is None:
            raise ValueError(f"Use case not loaded for domain: {domain.value}")

        async def node_handler(state: EnterpriseAgentState) -> dict:
            self._logger.info(f"Executing enterprise node: {domain.value}")

            result = await use_case.process(
                query=state["query"],
                context={
                    "rag_context": state.get("hrm_results", {}).get("context", ""),
                },
                use_mcts=state.get("use_mcts", True),
            )

            return {
                "enterprise_domain": domain.value,
                "domain_state": result.get("domain_state", {}),
                "domain_agents_results": result.get("agent_results", {}),
                "agent_outputs": [
                    {
                        "agent": f"enterprise_{use_case.name}",
                        "response": result.get("result", ""),
                        "confidence": result.get("confidence", 0.0),
                    }
                ],
                "use_case_metadata": result.get("mcts_stats", {}),
            }

        return node_handler

    def should_route_to_enterprise(self, state: EnterpriseAgentState) -> bool:
        """
        Determine if query should be routed to enterprise processing.

        Uses the centralized DomainDetector for consistent domain detection.

        Args:
            state: Current agent state

        Returns:
            True if enterprise routing is appropriate
        """
        query = state.get("query", "")
        result = self._detector.detect(query)
        return result.is_detected

    def get_enterprise_route(self, state: EnterpriseAgentState) -> str:
        """
        Determine which enterprise route to take.

        Args:
            state: Current agent state

        Returns:
            Route name (enterprise domain or 'standard')
        """
        if not self.should_route_to_enterprise(state):
            return "standard"

        use_case = self._factory.create_from_query(state.get("query", ""))
        if use_case:
            return f"enterprise_{use_case.name}"

        return "standard"
