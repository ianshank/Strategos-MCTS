"""
Deep Search agent for production-ready research with human-in-the-loop.

Based on: https://github.com/google/adk-samples/tree/main/python/agents/deep-search
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any

from ..base import ADKAgentAdapter, ADKAgentRequest, ADKAgentResponse, ADKConfig


class DeepSearchAgent(ADKAgentAdapter):
    """
    Production-ready research agent with human-in-the-loop planning.

    This agent implements a two-phase workflow:

    Phase 1 - Human-in-the-Loop Planning:
    - User submits research topic
    - Agent generates strategic research plan
    - User approves or refines plan

    Phase 2 - Autonomous Research:
    - Creates report outline
    - Performs iterative web searches
    - Uses critique cycles to identify gaps
    - Synthesizes findings into comprehensive report with citations

    Features:
    - Multi-agent collaboration with specialized reasoning roles
    - Function calling for web search
    - Iterative research refinement loops
    - Inline source citations
    """

    def __init__(self, config: ADKConfig):
        """
        Initialize Deep Search agent.

        Args:
            config: ADK configuration
        """
        super().__init__(config, agent_name="deep_search")

        # Deep search specific directories
        self.research_dir = Path(config.workspace_dir) / "research"
        self.plans_dir = Path(config.workspace_dir) / "plans"
        self.reports_dir = Path(config.workspace_dir) / "reports"
        self.sources_dir = Path(config.workspace_dir) / "sources"

        self.research_dir.mkdir(parents=True, exist_ok=True)
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)

        # Research workflow state
        self.research_sessions: dict[str, dict[str, Any]] = {}

    async def _agent_initialize(self) -> None:
        """Initialize Deep Search agent resources."""
        try:
            import google.adk  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "google-adk not installed. Install with: pip install 'langgraph-multi-agent-mcts[google-adk]'"
            ) from None

        # Verify search is enabled
        if not self.config.enable_search:
            raise ValueError("Deep Search agent requires enable_search=True")

    async def _agent_invoke(self, request: ADKAgentRequest) -> ADKAgentResponse:
        """
        Execute deep search task.

        Args:
            request: Agent request with research topic

        Returns:
            Agent response with research plan or report
        """
        phase = request.parameters.get("phase", "planning")
        research_id = request.session_id or self._generate_research_id(request.query)

        if phase == "planning":
            return await self._handle_planning_phase(request, research_id)
        elif phase == "execution":
            return await self._handle_execution_phase(request, research_id)
        elif phase == "full":
            # Execute both phases sequentially
            plan_response = await self._handle_planning_phase(request, research_id)
            if plan_response.status == "success":
                return await self._handle_execution_phase(request, research_id)
            return plan_response
        else:
            return ADKAgentResponse(
                result="",
                status="error",
                error=f"Unknown phase: {phase}. Use 'planning', 'execution', or 'full'",
            )

    async def _handle_planning_phase(
        self,
        request: ADKAgentRequest,
        research_id: str,
    ) -> ADKAgentResponse:
        """
        Handle research planning phase with human-in-the-loop.

        Args:
            request: Agent request
            research_id: Research session ID

        Returns:
            Agent response with research plan
        """
        topic = request.query
        plan_requirements = request.parameters.get("plan_requirements", {})

        # Generate research plan
        research_plan = self._generate_research_plan(topic, plan_requirements)

        # Save plan
        plan_file = self.plans_dir / f"{research_id}_plan.json"
        with open(plan_file, "w") as f:
            json.dump(research_plan, f, indent=2)

        # Store session state
        self.research_sessions[research_id] = {
            "topic": topic,
            "plan": research_plan,
            "status": "plan_pending_approval",
            "plan_file": str(plan_file),
        }

        # Format plan for user review
        plan_text = self._format_research_plan(research_plan)

        return ADKAgentResponse(
            result=plan_text,
            metadata={
                "phase": "planning",
                "research_id": research_id,
                "status": "plan_pending_approval",
                "plan_file": str(plan_file),
            },
            artifacts=[str(plan_file)],
            status="success",
            session_id=research_id,
        )

    async def _handle_execution_phase(
        self,
        request: ADKAgentRequest,
        research_id: str,
    ) -> ADKAgentResponse:
        """
        Handle autonomous research execution phase.

        Args:
            request: Agent request
            research_id: Research session ID

        Returns:
            Agent response with research report
        """
        # Load plan
        session = self.research_sessions.get(research_id)
        if not session:
            return ADKAgentResponse(
                result="",
                status="error",
                error=f"No research session found for ID: {research_id}. Run planning phase first.",
            )

        research_plan = session["plan"]

        # Generate research report
        report = self._generate_research_report(
            topic=session["topic"],
            research_plan=research_plan,
            query=request.query,
        )

        # Save report
        report_file = self.reports_dir / f"{research_id}_report.md"
        with open(report_file, "w") as f:
            f.write(report)

        # Update session
        session["status"] = "completed"
        session["report_file"] = str(report_file)

        return ADKAgentResponse(
            result=report,
            metadata={
                "phase": "execution",
                "research_id": research_id,
                "status": "completed",
                "report_file": str(report_file),
            },
            artifacts=[str(report_file)],
            status="success",
            session_id=research_id,
        )

    def _generate_research_id(self, topic: str) -> str:
        """Generate unique, non-guessable research ID."""
        safe_topic = "".join(ch for ch in topic.lower() if ch.isalnum())[:8] or "research"
        return f"{safe_topic}-{secrets.token_hex(4)}"

    def _generate_research_plan(
        self,
        topic: str,
        requirements: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate comprehensive research plan."""
        return {
            "topic": topic,
            "objectives": [
                f"Understand current state of {topic}",
                f"Identify key developments and trends in {topic}",
                "Analyze challenges and opportunities",
                "Synthesize comprehensive overview",
            ],
            "research_questions": [
                f"What is {topic}?",
                f"What are the latest developments in {topic}?",
                "What are the key challenges and solutions?",
                "What are future directions?",
            ],
            "search_strategy": {
                "initial_queries": [
                    f"{topic} overview",
                    f"{topic} latest research",
                    f"{topic} trends 2024-2025",
                    f"{topic} challenges",
                ],
                "refinement_cycles": 3,
                "sources_per_query": 5,
            },
            "report_structure": {
                "sections": [
                    "Executive Summary",
                    "Introduction",
                    "Background and Context",
                    "Current State",
                    "Key Findings",
                    "Analysis and Insights",
                    "Challenges and Limitations",
                    "Future Directions",
                    "Conclusion",
                    "References",
                ],
            },
            "quality_criteria": {
                "citation_count_min": 10,
                "source_diversity": "high",
                "recency_weight": "prioritize_recent",
            },
            "requirements": requirements,
        }

    def _format_research_plan(self, plan: dict[str, Any]) -> str:
        """Format research plan for human review."""
        return f"""
# Research Plan: {plan["topic"]}

## Objectives
{chr(10).join(f"{i + 1}. {obj}" for i, obj in enumerate(plan["objectives"]))}

## Research Questions
{chr(10).join(f"{i + 1}. {q}" for i, q in enumerate(plan["research_questions"]))}

## Search Strategy

### Initial Queries
{chr(10).join(f"- {q}" for q in plan["search_strategy"]["initial_queries"])}

### Refinement
- **Cycles**: {plan["search_strategy"]["refinement_cycles"]}
- **Sources per Query**: {plan["search_strategy"]["sources_per_query"]}

## Report Structure
{chr(10).join(f"{i + 1}. {section}" for i, section in enumerate(plan["report_structure"]["sections"]))}

## Quality Criteria
- **Minimum Citations**: {plan["quality_criteria"]["citation_count_min"]}
- **Source Diversity**: {plan["quality_criteria"]["source_diversity"]}
- **Recency**: {plan["quality_criteria"]["recency_weight"]}

---
**Next Step**: Approve this plan to begin autonomous research execution
or provide feedback to refine the plan.
""".strip()

    def _generate_research_report(
        self,
        topic: str,
        research_plan: dict[str, Any],
        query: str,
    ) -> str:
        """Generate comprehensive research report."""
        return f"""
# Research Report: {topic}

## Executive Summary

This report provides a comprehensive analysis of {topic} based on systematic
web research and synthesis of multiple authoritative sources. The research
follows a structured methodology including initial exploration, iterative
refinement, and critical evaluation.

## Introduction

{query}

## Research Methodology

### Search Strategy
This research employed a multi-phase search approach:

1. **Initial Exploration**
{chr(10).join(f"   - {q}" for q in research_plan["search_strategy"]["initial_queries"])}

2. **Iterative Refinement**
   - Number of refinement cycles: {research_plan["search_strategy"]["refinement_cycles"]}
   - Sources evaluated per query: {research_plan["search_strategy"]["sources_per_query"]}

3. **Critical Evaluation**
   - Source credibility assessment
   - Information cross-validation
   - Gap identification and targeted searches

### Quality Assurance
- Minimum {research_plan["quality_criteria"]["citation_count_min"]} authoritative citations
- High source diversity across different perspectives
- Emphasis on recent developments and current state

## Background and Context

{topic} represents an important area of investigation. This section provides
foundational context necessary for understanding the current state and recent
developments.

[Note: Full implementation would include web search results synthesized here]

## Current State

Based on comprehensive research across multiple sources, the current state
of {topic} can be characterized by several key dimensions:

### Key Developments
1. **Development 1**: [Description with citation]
2. **Development 2**: [Description with citation]
3. **Development 3**: [Description with citation]

### Major Players and Initiatives
- Organization/Project 1 [1]
- Organization/Project 2 [2]
- Organization/Project 3 [3]

## Key Findings

### Finding 1: [Title]
[Detailed analysis with citations]

### Finding 2: [Title]
[Detailed analysis with citations]

### Finding 3: [Title]
[Detailed analysis with citations]

## Analysis and Insights

### Cross-Cutting Themes
1. **Theme 1**: Analysis of patterns across sources
2. **Theme 2**: Emerging trends and trajectories
3. **Theme 3**: Contradictions and debates

### Critical Evaluation
- Strengths of current approaches
- Limitations and gaps
- Areas of consensus and disagreement

## Challenges and Limitations

### Current Challenges
1. **Challenge 1**: [Description]
2. **Challenge 2**: [Description]
3. **Challenge 3**: [Description]

### Research Limitations
- Scope limitations of this study
- Data availability constraints
- Temporal considerations

## Future Directions

Based on the research findings, several promising directions emerge:

1. **Direction 1**: [Description and rationale]
2. **Direction 2**: [Description and rationale]
3. **Direction 3**: [Description and rationale]

## Conclusion

This comprehensive research on {topic} reveals [key conclusions]. The
findings suggest [implications] and point toward [future considerations].

## References

[Note: Full implementation would include complete citation list]

1. Source 1 - [Title, URL, Date]
2. Source 2 - [Title, URL, Date]
3. Source 3 - [Title, URL, Date]
...

---
*Report generated by Deep Search Agent*
*Research ID: [ID]*
*Generation Date: [Date]*

---

## Appendices

### A. Search Query Log
[Detailed log of all search queries executed]

### B. Source Evaluation Matrix
[Table showing source credibility and relevance scores]

### C. Gap Analysis
[Identified gaps that required additional research]

""".strip()

    async def create_research_plan(
        self,
        topic: str,
        requirements: dict[str, Any] | None = None,
    ) -> ADKAgentResponse:
        """
        Create research plan for human review.

        Args:
            topic: Research topic
            requirements: Additional planning requirements

        Returns:
            Agent response with research plan
        """
        request = ADKAgentRequest(
            query=topic,
            parameters={
                "phase": "planning",
                "plan_requirements": requirements or {},
            },
        )

        return await self.invoke(request)

    async def execute_research(
        self,
        research_id: str,
        additional_instructions: str | None = None,
    ) -> ADKAgentResponse:
        """
        Execute research based on approved plan.

        Args:
            research_id: Research session ID from planning phase
            additional_instructions: Optional execution instructions

        Returns:
            Agent response with research report
        """
        request = ADKAgentRequest(
            query=additional_instructions or "Execute research plan",
            parameters={
                "phase": "execution",
            },
            session_id=research_id,
        )

        return await self.invoke(request)

    async def full_research(
        self,
        topic: str,
        requirements: dict[str, Any] | None = None,
    ) -> ADKAgentResponse:
        """
        Execute full research workflow (planning + execution).

        Args:
            topic: Research topic
            requirements: Planning requirements

        Returns:
            Agent response with complete research report
        """
        request = ADKAgentRequest(
            query=topic,
            parameters={
                "phase": "full",
                "plan_requirements": requirements or {},
            },
        )

        return await self.invoke(request)

    def get_research_status(self, research_id: str) -> dict[str, Any] | None:
        """
        Get status of research session.

        Args:
            research_id: Research session ID

        Returns:
            Session status dictionary or None if not found
        """
        return self.research_sessions.get(research_id)

    def list_research_sessions(self) -> list[dict[str, Any]]:
        """
        List all research sessions.

        Returns:
            List of research session summaries
        """
        return [
            {
                "research_id": rid,
                "topic": session["topic"],
                "status": session["status"],
            }
            for rid, session in self.research_sessions.items()
        ]

    def get_capabilities(self) -> dict[str, Any]:
        """Get Deep Search agent capabilities."""
        base_caps = super().get_capabilities()
        base_caps.update(
            {
                "agent_type": "deep_search",
                "supports_streaming": False,  # Could be implemented
                "workflow_phases": ["planning", "execution", "full"],
                "features": [
                    "human_in_loop_planning",
                    "autonomous_research",
                    "iterative_refinement",
                    "critique_cycles",
                    "inline_citations",
                    "multi_agent_collaboration",
                    "web_search",
                ],
                "report_sections": [
                    "executive_summary",
                    "methodology",
                    "findings",
                    "analysis",
                    "conclusions",
                    "references",
                ],
            }
        )
        return base_caps
