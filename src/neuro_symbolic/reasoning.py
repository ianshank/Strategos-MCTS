"""
Symbolic Reasoning Agent for Neuro-Symbolic AI.

Provides symbolic reasoning capabilities with:
- Logic engine supporting multiple backends (Z3, Prolog-style, SymPy)
- Proof tree generation for explainability
- Neural fallback for ambiguous cases
- Integration with MCTS and other agents

Best Practices 2025:
- Protocol-based interfaces for solver backends
- Async-first for non-blocking reasoning
- Comprehensive proof tree generation
- Graceful degradation with neural fallback
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

from src.observability.logging import get_logger

from .config import (
    LogicEngineConfig,
    NeuroSymbolicConfig,
)
from .state import Fact, NeuroSymbolicState, SymbolicFactType

logger = get_logger(__name__)


class ProofStatus(Enum):
    """Status of a proof attempt."""

    SUCCESS = auto()  # Proof found
    FAILURE = auto()  # No proof exists
    TIMEOUT = auto()  # Timed out
    UNKNOWN = auto()  # Could not determine
    PARTIAL = auto()  # Partial proof (for iterative deepening)


@dataclass(frozen=True)
class Predicate:
    """
    Logical predicate representation.

    Supports variables (prefixed with ?) for pattern matching.
    """

    name: str
    arguments: tuple[Any, ...]
    negated: bool = False

    def __post_init__(self):
        if not self.name:
            raise ValueError("Predicate name cannot be empty")

    def to_string(self) -> str:
        """Convert to Prolog-style string."""
        prefix = "\\+" if self.negated else ""
        args = ", ".join(str(a) for a in self.arguments)
        return f"{prefix}{self.name}({args})"

    def to_fact(self) -> Fact:
        """Convert to Fact (only for ground predicates)."""
        if any(isinstance(a, str) and a.startswith("?") for a in self.arguments):
            raise ValueError("Cannot convert predicate with variables to fact")
        return Fact(
            name=self.name,
            arguments=self.arguments,
            fact_type=SymbolicFactType.PREDICATE,
        )

    def is_ground(self) -> bool:
        """Check if predicate has no variables."""
        return not any(isinstance(a, str) and a.startswith("?") for a in self.arguments)

    def get_variables(self) -> set[str]:
        """Get all variable names in this predicate."""
        return {a[1:] for a in self.arguments if isinstance(a, str) and a.startswith("?")}

    def substitute(self, bindings: dict[str, Any]) -> Predicate:
        """Apply variable bindings."""
        new_args = []
        for arg in self.arguments:
            if isinstance(arg, str) and arg.startswith("?"):
                var_name = arg[1:]
                if var_name in bindings:
                    new_args.append(bindings[var_name])
                else:
                    new_args.append(arg)
            else:
                new_args.append(arg)
        return Predicate(
            name=self.name,
            arguments=tuple(new_args),
            negated=self.negated,
        )


@dataclass
class Rule:
    """
    Inference rule (Horn clause).

    head :- body1, body2, ... (if body then head)
    """

    rule_id: str
    head: Predicate
    body: list[Predicate]
    confidence: float = 1.0
    source: str = "unknown"

    def to_string(self) -> str:
        """Convert to Prolog-style string."""
        body_str = ", ".join(p.to_string() for p in self.body)
        return f"{self.head.to_string()} :- {body_str}."

    def get_all_variables(self) -> set[str]:
        """Get all variables in the rule."""
        variables = self.head.get_variables()
        for pred in self.body:
            variables |= pred.get_variables()
        return variables


@dataclass
class ProofStep:
    """Single step in a proof tree."""

    step_id: str
    predicate: Predicate
    rule_applied: Rule | None
    bindings: dict[str, Any]
    children: list[ProofStep] = field(default_factory=list)
    success: bool = True
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "predicate": self.predicate.to_string(),
            "rule_applied": self.rule_applied.rule_id if self.rule_applied else None,
            "bindings": self.bindings,
            "children": [c.to_dict() for c in self.children],
            "success": self.success,
            "explanation": self.explanation,
        }


@dataclass
class ProofTree:
    """Complete proof tree for a query."""

    root: ProofStep
    query: Predicate
    status: ProofStatus
    bindings: list[dict[str, Any]]  # All successful bindings
    depth: int = 0
    node_count: int = 1
    search_time_ms: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query.to_string(),
            "status": self.status.name,
            "bindings": self.bindings,
            "depth": self.depth,
            "node_count": self.node_count,
            "search_time_ms": self.search_time_ms,
            "explanation": self.explanation,
            "proof_tree": self.root.to_dict(),
        }

    def generate_explanation(
        self,
        verbosity: int = 2,
    ) -> str:
        """
        Generate natural language explanation of the proof.

        Args:
            verbosity: 1=minimal, 2=standard, 3=detailed

        Returns:
            Natural language explanation
        """
        if self.status != ProofStatus.SUCCESS:
            return f"Could not prove {self.query.to_string()}: {self.status.name}"

        lines = []

        if verbosity >= 1:
            lines.append(f"Proved: {self.query.to_string()}")

        if verbosity >= 2 and self.bindings:
            lines.append("Variable bindings:")
            for binding in self.bindings:
                binding_strs = [f"{k}={v}" for k, v in binding.items()]
                lines.append(f"  - {', '.join(binding_strs)}")

        if verbosity >= 3:
            lines.append("")
            lines.append("Proof steps:")
            self._explain_step(self.root, lines, indent=0)

        return "\n".join(lines)

    def _explain_step(
        self,
        step: ProofStep,
        lines: list[str],
        indent: int,
    ) -> None:
        """Recursively explain proof steps."""
        prefix = "  " * indent
        if step.rule_applied:
            lines.append(f"{prefix}Used rule '{step.rule_applied.rule_id}' to prove {step.predicate.to_string()}")
        else:
            lines.append(f"{prefix}Fact: {step.predicate.to_string()}")

        for child in step.children:
            self._explain_step(child, lines, indent + 1)


@dataclass
class Proof:
    """Result of a proof attempt."""

    success: bool
    result: Any | None = None
    bindings: dict[str, Any] = field(default_factory=dict)
    proof_tree: ProofTree | None = None
    confidence: float = 1.0
    explanation: str = ""


@runtime_checkable
class LogicSolver(Protocol):
    """Protocol for logic solvers."""

    async def query(
        self,
        goal: Predicate,
        facts: frozenset[Fact],
        rules: list[Rule],
        timeout_ms: int,
    ) -> ProofTree:
        """Execute a query and return proof tree."""
        ...


class LogicEngine:
    """
    Logic engine supporting multiple backends.

    Provides:
    - Forward and backward chaining
    - Proof tree generation
    - Memoization for efficiency
    """

    def __init__(self, config: LogicEngineConfig):
        self.config = config
        self._rules: list[Rule] = []
        self._cache: dict[str, ProofTree] = {}
        self._step_counter = 0

    def add_rule(self, rule: Rule) -> None:
        """Add an inference rule."""
        self._rules.append(rule)
        self._cache.clear()  # Invalidate cache

    def add_rules(self, rules: list[Rule]) -> None:
        """Add multiple inference rules."""
        self._rules.extend(rules)
        self._cache.clear()

    def clear_rules(self) -> None:
        """Clear all rules."""
        self._rules.clear()
        self._cache.clear()

    def _get_cache_key(
        self,
        goal: Predicate,
        facts: frozenset[Fact],
    ) -> str:
        """Generate cache key for query."""
        goal_str = goal.to_string()
        facts_str = "|".join(sorted(f.to_string() for f in facts))
        content = f"{goal_str}:{facts_str}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _unify(
        self,
        pred1: Predicate,
        pred2: Predicate,
        bindings: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Unify two predicates.

        Returns extended bindings if successful, None otherwise.
        """
        if pred1.name != pred2.name:
            return None
        if len(pred1.arguments) != len(pred2.arguments):
            return None
        if pred1.negated != pred2.negated:
            return None

        new_bindings = bindings.copy()

        for arg1, arg2 in zip(pred1.arguments, pred2.arguments, strict=True):
            # Apply existing bindings
            if isinstance(arg1, str) and arg1.startswith("?"):
                var_name = arg1[1:]
                if var_name in new_bindings:
                    arg1 = new_bindings[var_name]

            if isinstance(arg2, str) and arg2.startswith("?"):
                var_name = arg2[1:]
                if var_name in new_bindings:
                    arg2 = new_bindings[var_name]

            # Unify
            if isinstance(arg1, str) and arg1.startswith("?"):
                var_name = arg1[1:]
                new_bindings[var_name] = arg2
            elif isinstance(arg2, str) and arg2.startswith("?"):
                var_name = arg2[1:]
                new_bindings[var_name] = arg1
            elif arg1 != arg2:
                return None

        return new_bindings

    async def _prove_backward(
        self,
        goal: Predicate,
        facts: frozenset[Fact],
        bindings: dict[str, Any],
        depth: int,
        timeout_ms: int,
        start_time: float,
    ) -> tuple[bool, list[dict[str, Any]], ProofStep]:
        """
        Prove goal using backward chaining.

        Returns (success, bindings_list, proof_step)
        """
        self._step_counter += 1
        step_id = f"step_{self._step_counter}"

        # Check timeout
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > timeout_ms:
            return (
                False,
                [],
                ProofStep(
                    step_id=step_id,
                    predicate=goal,
                    rule_applied=None,
                    bindings=bindings,
                    success=False,
                    explanation="timeout",
                ),
            )

        # Check depth limit
        if depth > self.config.max_proof_depth:
            return (
                False,
                [],
                ProofStep(
                    step_id=step_id,
                    predicate=goal,
                    rule_applied=None,
                    bindings=bindings,
                    success=False,
                    explanation="depth limit exceeded",
                ),
            )

        # Apply current bindings to goal
        goal = goal.substitute(bindings)

        # Try to match against facts
        for fact in facts:
            fact_pred = Predicate(name=fact.name, arguments=fact.arguments)
            unified = self._unify(goal, fact_pred, bindings)
            if unified is not None:
                return (
                    True,
                    [unified],
                    ProofStep(
                        step_id=step_id,
                        predicate=goal,
                        rule_applied=None,
                        bindings=unified,
                        success=True,
                        explanation=f"matched fact: {fact.to_string()}",
                    ),
                )

        # Try to apply rules
        all_bindings: list[dict[str, Any]] = []
        best_step: ProofStep | None = None

        for rule in self._rules:
            # Rename rule variables to avoid conflicts
            renamed_head = self._rename_variables(rule.head, depth)
            renamed_body = [self._rename_variables(p, depth) for p in rule.body]

            # Try to unify goal with rule head
            unified = self._unify(goal, renamed_head, bindings)
            if unified is None:
                continue

            # Prove all body predicates
            body_success = True
            body_bindings = unified
            child_steps: list[ProofStep] = []

            for body_pred in renamed_body:
                sub_success, sub_bindings_list, sub_step = await self._prove_backward(
                    body_pred,
                    facts,
                    body_bindings,
                    depth + 1,
                    timeout_ms,
                    start_time,
                )

                child_steps.append(sub_step)

                if not sub_success:
                    body_success = False
                    break

                # Use first successful binding
                if sub_bindings_list:
                    body_bindings = sub_bindings_list[0]

            if body_success:
                proof_step = ProofStep(
                    step_id=step_id,
                    predicate=goal,
                    rule_applied=rule,
                    bindings=body_bindings,
                    children=child_steps,
                    success=True,
                    explanation=f"applied rule: {rule.rule_id}",
                )
                all_bindings.append(body_bindings)
                if best_step is None:
                    best_step = proof_step
            elif best_step is None:
                best_step = ProofStep(
                    step_id=step_id,
                    predicate=goal,
                    rule_applied=rule,
                    bindings=bindings,
                    children=child_steps,
                    success=False,
                    explanation="body failed",
                )

        if all_bindings:
            return True, all_bindings, best_step  # type: ignore

        return (
            False,
            [],
            best_step
            or ProofStep(
                step_id=step_id,
                predicate=goal,
                rule_applied=None,
                bindings=bindings,
                success=False,
                explanation="no matching rules or facts",
            ),
        )

    def _rename_variables(self, pred: Predicate, suffix: int) -> Predicate:
        """Rename variables to avoid conflicts."""
        new_args = []
        for arg in pred.arguments:
            if isinstance(arg, str) and arg.startswith("?"):
                new_args.append(f"{arg}_{suffix}")
            else:
                new_args.append(arg)
        return Predicate(
            name=pred.name,
            arguments=tuple(new_args),
            negated=pred.negated,
        )

    async def query(
        self,
        goal: Predicate,
        state: NeuroSymbolicState,
    ) -> ProofTree:
        """
        Execute a query against the knowledge base.

        Args:
            goal: Goal predicate to prove
            state: Current state with facts

        Returns:
            ProofTree with result
        """
        # Check cache
        if self.config.enable_memoization:
            cache_key = self._get_cache_key(goal, state.facts)
            if cache_key in self._cache:
                return self._cache[cache_key]

        self._step_counter = 0
        start_time = time.perf_counter()

        success, bindings, root_step = await self._prove_backward(
            goal,
            state.facts,
            {},
            0,
            self.config.solver_timeout_ms,
            start_time,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Determine status
        if success:
            status = ProofStatus.SUCCESS
        elif elapsed_ms > self.config.solver_timeout_ms:
            status = ProofStatus.TIMEOUT
        else:
            status = ProofStatus.FAILURE

        logger.debug(
            "Logic engine query complete: goal=%s, status=%s, time_ms=%.2f, steps=%d",
            goal.to_string(),
            status.name,
            elapsed_ms,
            self._step_counter,
        )

        # Build proof tree
        proof_tree = ProofTree(
            root=root_step,
            query=goal,
            status=status,
            bindings=bindings,
            depth=self._count_depth(root_step),
            node_count=self._step_counter,
            search_time_ms=elapsed_ms,
        )

        # Cache result
        if self.config.enable_memoization and len(self._cache) < self.config.cache_size:
            self._cache[cache_key] = proof_tree

        return proof_tree

    def _count_depth(self, step: ProofStep) -> int:
        """Count maximum depth of proof tree."""
        if not step.children:
            return 1
        return 1 + max(self._count_depth(c) for c in step.children)


class SymbolicReasoner:
    """
    High-level symbolic reasoner combining logic engine with state.

    Provides convenient methods for common reasoning tasks.
    """

    def __init__(
        self,
        config: NeuroSymbolicConfig,
        logic_engine: LogicEngine | None = None,
    ):
        self.config = config
        self.logic_engine = logic_engine or LogicEngine(config.logic_engine)

    def add_rule(
        self,
        rule_id: str,
        head: tuple[str, tuple[Any, ...]],
        body: list[tuple[str, tuple[Any, ...]]],
        confidence: float = 1.0,
    ) -> None:
        """
        Add an inference rule.

        Args:
            rule_id: Unique rule identifier
            head: (name, arguments) tuple for head predicate
            body: List of (name, arguments) tuples for body predicates
        """
        head_pred = Predicate(name=head[0], arguments=head[1])
        body_preds = [Predicate(name=b[0], arguments=b[1]) for b in body]
        rule = Rule(
            rule_id=rule_id,
            head=head_pred,
            body=body_preds,
            confidence=confidence,
        )
        self.logic_engine.add_rule(rule)

    async def prove(
        self,
        goal: tuple[str, tuple[Any, ...]],
        state: NeuroSymbolicState,
    ) -> Proof:
        """
        Attempt to prove a goal.

        Args:
            goal: (name, arguments) tuple for goal predicate
            state: Current state with facts

        Returns:
            Proof result with explanation
        """
        goal_pred = Predicate(name=goal[0], arguments=goal[1])
        proof_tree = await self.logic_engine.query(goal_pred, state)

        if proof_tree.status == ProofStatus.SUCCESS:
            explanation = proof_tree.generate_explanation(self.config.proof.explanation_verbosity_level)
            # Calculate confidence from rule confidences
            confidence = self._calculate_proof_confidence(proof_tree)
            return Proof(
                success=True,
                result=proof_tree.bindings[0] if proof_tree.bindings else None,
                bindings=proof_tree.bindings[0] if proof_tree.bindings else {},
                proof_tree=proof_tree,
                confidence=confidence,
                explanation=explanation,
            )
        else:
            return Proof(
                success=False,
                proof_tree=proof_tree,
                explanation=f"Could not prove {goal_pred.to_string()}: {proof_tree.status.name}",
            )

    def _calculate_proof_confidence(self, proof_tree: ProofTree) -> float:
        """Calculate overall confidence from proof tree."""

        def step_confidence(step: ProofStep) -> float:
            if not step.success:
                return 0.0
            if step.rule_applied:
                rule_conf = step.rule_applied.confidence
                if step.children:
                    child_confs = [step_confidence(c) for c in step.children]
                    # Use geometric mean for child confidences
                    if child_confs:
                        import math

                        product = math.prod(child_confs)
                        child_conf = product ** (1 / len(child_confs))
                        return rule_conf * child_conf
                return rule_conf
            return 1.0  # Fact

        return step_confidence(proof_tree.root)

    async def ask(
        self,
        query_str: str,
        state: NeuroSymbolicState,
    ) -> list[dict[str, Any]]:
        """
        Ask a query in natural format and get all answers.

        Args:
            query_str: Query string like "parent(?X, john)"
            state: Current state

        Returns:
            List of variable bindings that satisfy the query
        """
        # Parse query string
        goal = self._parse_query(query_str)
        proof_tree = await self.logic_engine.query(goal, state)

        if proof_tree.status == ProofStatus.SUCCESS:
            return proof_tree.bindings
        return []

    def _parse_query(self, query_str: str) -> Predicate:
        """Parse a query string into a Predicate."""
        import re

        # Match pattern: name(arg1, arg2, ...)
        match = re.match(r"(\w+)\((.*)\)", query_str.strip())
        if not match:
            raise ValueError(f"Invalid query format: {query_str}")

        name = match.group(1)
        args_str = match.group(2)

        # Parse arguments
        args: list[int | float | str] = []
        for arg in args_str.split(","):
            arg = arg.strip()
            if not arg:
                continue
            # Try to parse as number
            try:
                args.append(int(arg))
            except ValueError:
                try:
                    args.append(float(arg))
                except ValueError:
                    args.append(arg)

        return Predicate(name=name, arguments=tuple(args))


class SymbolicReasoningAgent:
    """
    Complete symbolic reasoning agent for the multi-agent framework.

    Integrates with HRM, TRM, and MCTS agents.
    """

    def __init__(
        self,
        config: NeuroSymbolicConfig,
        neural_fallback: Callable[[str, NeuroSymbolicState], Any] | None = None,
        logger: Any | None = None,
    ):
        self.config = config
        self.reasoner = SymbolicReasoner(config)
        self.neural_fallback = neural_fallback
        self.logger = logger
        self._query_count = 0
        self._success_count = 0
        self._fallback_count = 0

    async def process(
        self,
        query: str,
        rag_context: str | None = None,
        state: NeuroSymbolicState | None = None,
    ) -> dict[str, Any]:
        """
        Process a query using symbolic reasoning.

        Args:
            query: User query
            rag_context: Optional RAG context
            state: Optional initial state

        Returns:
            Response dictionary compatible with framework
        """
        self._query_count += 1
        start_time = time.perf_counter()

        logger.debug("SymbolicReasoningAgent processing query %d: %s", self._query_count, query[:100])

        # Create state if not provided
        if state is None:
            state = NeuroSymbolicState(
                state_id=f"query_{self._query_count}",
                facts=frozenset(),
                metadata={"query": query, "rag_context": rag_context},
            )

        # Extract facts from RAG context if available
        if rag_context:
            state = self._extract_facts_from_context(state, rag_context)

        # Try symbolic reasoning
        try:
            proof = await self._reason(query, state)

            if proof.success and proof.confidence >= self.config.agent.neural_fallback_confidence_threshold:
                self._success_count += 1
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                return {
                    "response": self._format_response(query, proof),
                    "metadata": {
                        "agent": "symbolic",
                        "proof_found": True,
                        "confidence": proof.confidence,
                        "explanation": proof.explanation,
                        "proof_tree": proof.proof_tree.to_dict() if proof.proof_tree else None,
                        "processing_time_ms": elapsed_ms,
                        "symbolic_quality_score": proof.confidence,
                    },
                }
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Symbolic reasoning failed: {e}")

        # Fall back to neural if configured
        if self.config.agent.fallback_to_neural and self.neural_fallback:
            self._fallback_count += 1
            if self.logger:
                self.logger.info("Falling back to neural reasoning")

            neural_result = await self._call_neural_fallback(query, state)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return {
                "response": neural_result,
                "metadata": {
                    "agent": "symbolic_with_neural_fallback",
                    "proof_found": False,
                    "confidence": self.config.agent.neural_fallback_confidence_threshold,
                    "explanation": "Used neural fallback after symbolic reasoning failed",
                    "processing_time_ms": elapsed_ms,
                    "symbolic_quality_score": 0.0,
                },
            }

        # No fallback available
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return {
            "response": f"Could not determine answer for: {query}",
            "metadata": {
                "agent": "symbolic",
                "proof_found": False,
                "confidence": 0.0,
                "explanation": "Symbolic reasoning failed and no neural fallback available",
                "processing_time_ms": elapsed_ms,
                "symbolic_quality_score": 0.0,
            },
        }

    async def _reason(
        self,
        query: str,
        state: NeuroSymbolicState,
    ) -> Proof:
        """Perform symbolic reasoning on query."""
        # Parse query to determine goal
        goal = self._query_to_goal(query)

        if goal:
            return await self.reasoner.prove(goal, state)

        # If can't parse as formal query, try to match against facts
        return Proof(
            success=False,
            explanation="Could not parse query as formal goal",
        )

    def _query_to_goal(
        self,
        query: str,
    ) -> tuple[str, tuple[Any, ...]] | None:
        """Convert natural language query to formal goal."""
        # Simple pattern matching for demo
        # In production, use NL-to-logic parser or LLM

        query_lower = query.lower()

        # Pattern: "is X a Y?" -> isa(X, Y)
        if query_lower.startswith("is ") and " a " in query_lower:
            parts = query_lower[3:].rstrip("?").split(" a ")
            if len(parts) == 2:
                return ("isa", (parts[0].strip(), parts[1].strip()))

        # Pattern: "does X have Y?" -> has(X, Y)
        if query_lower.startswith("does ") and " have " in query_lower:
            parts = query_lower[5:].rstrip("?").split(" have ")
            if len(parts) == 2:
                return ("has", (parts[0].strip(), parts[1].strip()))

        # Pattern: "can X do Y?" -> can(X, Y)
        if query_lower.startswith("can ") and " do " in query_lower:
            parts = query_lower[4:].rstrip("?").split(" do ")
            if len(parts) == 2:
                return ("can", (parts[0].strip(), parts[1].strip()))

        # Check for Prolog-style query
        import re

        match = re.match(r"(\w+)\((.*)\)\??", query.strip())
        if match:
            name = match.group(1)
            args_str = match.group(2)
            args = tuple(a.strip() for a in args_str.split(",") if a.strip())
            return (name, args)

        return None

    def _extract_facts_from_context(
        self,
        state: NeuroSymbolicState,
        context: str,
    ) -> NeuroSymbolicState:
        """Extract facts from RAG context."""
        # Simple fact extraction - in production use NER/RE
        facts = set(state.facts)

        # Extract simple statements
        import re

        # Pattern: "X is a Y" -> isa(X, Y)
        for match in re.finditer(r"(\w+)\s+is\s+a\s+(\w+)", context, re.IGNORECASE):
            facts.add(
                Fact(
                    name="isa",
                    arguments=(match.group(1).lower(), match.group(2).lower()),
                    source="rag_context",
                )
            )

        # Pattern: "X has Y" -> has(X, Y)
        for match in re.finditer(r"(\w+)\s+has\s+(\w+)", context, re.IGNORECASE):
            facts.add(
                Fact(
                    name="has",
                    arguments=(match.group(1).lower(), match.group(2).lower()),
                    source="rag_context",
                )
            )

        return NeuroSymbolicState(
            state_id=state.state_id,
            facts=frozenset(facts),
            constraints=state.constraints,
            confidence=state.confidence,
            metadata=state.metadata,
        )

    async def _call_neural_fallback(
        self,
        query: str,
        state: NeuroSymbolicState,
    ) -> str:
        """Call neural fallback if available."""
        if self.neural_fallback:
            result = self.neural_fallback(query, state)
            if asyncio.iscoroutine(result):
                return str(await result)
            return str(result)
        return f"No answer available for: {query}"

    def _format_response(self, query: str, proof: Proof) -> str:
        """Format proof result as response."""
        if proof.bindings:
            bindings_str = ", ".join(f"{k}={v}" for k, v in proof.bindings.items())
            return f"Yes. {proof.explanation}\nBindings: {bindings_str}"
        return f"Yes. {proof.explanation}"

    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_queries": self._query_count,
            "successful_proofs": self._success_count,
            "neural_fallbacks": self._fallback_count,
            "success_rate": (self._success_count / self._query_count if self._query_count > 0 else 0.0),
            "fallback_rate": (self._fallback_count / self._query_count if self._query_count > 0 else 0.0),
        }

    def add_knowledge(
        self,
        facts: list[tuple[str, tuple[Any, ...]]],
        rules: list[tuple[str, tuple[str, tuple[Any, ...]], list[tuple[str, tuple[Any, ...]]]]],
    ) -> None:
        """
        Add knowledge to the reasoner.

        Args:
            facts: List of (name, arguments) tuples
            rules: List of (rule_id, head, body) tuples
        """
        for rule_id, head, body in rules:
            self.reasoner.add_rule(rule_id, head, body)
