"""
Simplified agent implementations for the e2e workflow demo (scripts/run_e2e_workflow.py).
"""

import asyncio
from typing import Any


class HRMAgent:
    """Hierarchical Reasoning Module - breaks down complex queries."""

    def __init__(self, llm_client):
        """Initialize with an LLM client.

        Args:
            llm_client: LLM client (mock or real, duck-typed: needs an async generate())
        """
        self.llm_client = llm_client
        self.name = "HRM (Hierarchical Reasoning)"

    async def process(self, query: str) -> dict[str, Any]:
        """Process query using hierarchical decomposition.

        Args:
            query: Input query to process

        Returns:
            Dictionary with response, confidence, and reasoning steps
        """
        # Step 1: Decompose the query
        decomposition_steps = await self._decompose_query(query)

        # Step 2: Analyze each component
        analysis_results = await self._analyze_components(decomposition_steps)

        # Step 3: Synthesize hierarchical response
        llm_result = await self.llm_client.generate(
            prompt=f"Hierarchical analysis of: {query}", context=f"Components: {', '.join(decomposition_steps)}"
        )

        # Compile reasoning steps
        reasoning_steps = [
            f"1. Query decomposition: Identified {len(decomposition_steps)} key components",
            f"2. Component analysis: {analysis_results}",
            "3. Hierarchical synthesis: Combined insights from all levels",
            f"4. Confidence assessment: {llm_result['confidence']:.1%} based on component clarity",
        ]

        return {
            "response": llm_result["response"],
            "confidence": llm_result["confidence"],
            "steps": reasoning_steps,
            "components": decomposition_steps,
            "tokens_used": llm_result.get("tokens_used", 0),
        }

    async def _decompose_query(self, query: str) -> list[str]:
        """Decompose query into hierarchical components."""
        # Simulate decomposition based on query structure
        await asyncio.sleep(0.05)  # Simulate processing

        # Simple heuristic decomposition
        components = []

        # Extract key phrases
        query_lower = query.lower()

        if "?" in query:
            components.append("Question type: Analytical")
        else:
            components.append("Question type: Directive")

        if "how" in query_lower:
            components.append("Focus: Methodology/Process")
        elif "what" in query_lower:
            components.append("Focus: Definition/Identification")
        elif "why" in query_lower:
            components.append("Focus: Causation/Reasoning")
        elif "should" in query_lower or "best" in query_lower:
            components.append("Focus: Decision/Recommendation")
        else:
            components.append("Focus: General inquiry")

        # Domain detection
        if any(term in query_lower for term in ["database", "sql", "nosql", "storage"]):
            components.append("Domain: Data Management")
        elif any(term in query_lower for term in ["architecture", "design", "pattern"]):
            components.append("Domain: System Architecture")
        elif any(term in query_lower for term in ["performance", "optimization", "speed"]):
            components.append("Domain: Performance Engineering")
        elif any(term in query_lower for term in ["scale", "distributed", "cluster"]):
            components.append("Domain: Distributed Systems")
        else:
            components.append("Domain: Software Engineering")

        # Complexity assessment
        word_count = len(query.split())
        if word_count > 20:
            components.append("Complexity: High (detailed query)")
        elif word_count > 10:
            components.append("Complexity: Medium")
        else:
            components.append("Complexity: Low (concise query)")

        return components

    async def _analyze_components(self, components: list[str]) -> str:
        """Analyze the decomposed components."""
        await asyncio.sleep(0.03)  # Simulate processing

        # Generate analysis summary
        analysis_parts = []

        for component in components:
            if "Focus:" in component:
                focus = component.split(":")[1].strip()
                analysis_parts.append(f"requires {focus.lower()} approach")
            elif "Domain:" in component:
                domain = component.split(":")[1].strip()
                analysis_parts.append(f"applies to {domain}")
            elif "Complexity:" in component:
                complexity = component.split(":")[1].strip().split()[0]
                analysis_parts.append(f"{complexity.lower()} complexity level")

        return "; ".join(analysis_parts) if analysis_parts else "Standard analysis"


class TRMAgent:
    """Tree Reasoning Module - iterative refinement of responses."""

    def __init__(self, llm_client):
        """Initialize with an LLM client.

        Args:
            llm_client: LLM client (mock or real, duck-typed: needs an async generate())
        """
        self.llm_client = llm_client
        self.name = "TRM (Iterative Refinement)"
        self.max_iterations = 3

    async def process(self, query: str) -> dict[str, Any]:
        """Process query using iterative refinement.

        Args:
            query: Input query to process

        Returns:
            Dictionary with response, confidence, and reasoning steps
        """
        reasoning_steps = []
        current_response = ""
        current_confidence = 0.0

        # Iterative refinement loop
        for iteration in range(self.max_iterations):
            step_num = iteration + 1

            # Generate or refine response
            if iteration == 0:
                # Initial response
                result = await self.llm_client.generate(prompt=query, context="")
                current_response = result["response"]
                current_confidence = result["confidence"]
                reasoning_steps.append(
                    f"Iteration {step_num}: Initial response generated (confidence: {current_confidence:.1%})"
                )
            else:
                # Refinement iteration
                refinement_result = await self._refine_response(query, current_response, iteration)
                current_response = refinement_result["response"]

                # Confidence typically improves with refinement
                confidence_improvement = min(0.1, (1 - current_confidence) * 0.3)
                current_confidence = min(0.95, current_confidence + confidence_improvement)

                reasoning_steps.append(
                    f"Iteration {step_num}: {refinement_result['refinement_type']} "
                    f"(confidence: {current_confidence:.1%})"
                )

            # Check if confidence is high enough to stop
            if current_confidence > 0.85:
                reasoning_steps.append(f"Early termination: High confidence ({current_confidence:.1%}) achieved")
                break

        # Final reasoning step
        reasoning_steps.append(f"Final: Response refined through {len(reasoning_steps)} iterations")

        return {
            "response": current_response,
            "confidence": round(current_confidence, 3),
            "steps": reasoning_steps,
            "iterations_used": min(iteration + 1, self.max_iterations),
            "refinement_history": reasoning_steps,
        }

    async def _refine_response(self, query: str, current_response: str, iteration: int) -> dict[str, Any]:
        """Refine the current response."""
        await asyncio.sleep(0.05)  # Simulate refinement processing

        # Different refinement strategies based on iteration
        refinement_strategies = [
            ("Clarity enhancement", "improve clarity and precision"),
            ("Detail expansion", "add technical depth and specifics"),
            ("Validation check", "verify accuracy and completeness"),
        ]

        strategy_name, strategy_action = refinement_strategies[iteration % len(refinement_strategies)]

        # Generate refined response
        refinement_prompt = f"""
        Original query: {query}
        Current response: {current_response}
        Refinement task: {strategy_action}
        """

        result = await self.llm_client.generate(
            prompt=refinement_prompt, context=f"Refinement iteration {iteration + 1}"
        )

        # Enhance the response based on strategy
        enhanced_response = current_response
        if strategy_name == "Clarity enhancement":
            enhanced_response = f"{current_response}. {result['response']}"
        elif strategy_name == "Detail expansion":
            enhanced_response = f"{current_response}. Furthermore, {result['response']}"
        else:  # Validation
            enhanced_response = f"{current_response}. Validated: {result['response']}"

        # Truncate if too long
        if len(enhanced_response) > 300:
            enhanced_response = enhanced_response[:297] + "..."

        return {"response": enhanced_response, "refinement_type": strategy_name, "strategy_action": strategy_action}
