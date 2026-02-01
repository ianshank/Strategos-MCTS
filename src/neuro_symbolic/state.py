"""
Unified Neuro-Symbolic State Representation.

Provides a state representation that combines:
- Symbolic facts: Ground truth logical predicates
- Neural embeddings: Learned vector representations
- Constraints: Hard and soft rules that must hold
- Confidence: Uncertainty quantification

This enables reasoning about states both symbolically (logical entailment)
and neurally (similarity, interpolation).

Best Practices 2025:
- Protocol-based interfaces for extensibility
- Immutable state for thread safety
- Efficient hashing for caching
- Lazy evaluation of expensive properties
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

import numpy as np

# Optional torch import for environments without GPU support
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


class SymbolicFactType(Enum):
    """Types of symbolic facts."""

    PREDICATE = auto()  # Logical predicate (e.g., parent(john, mary))
    RELATION = auto()  # Binary relation (e.g., greater_than(x, y))
    ATTRIBUTE = auto()  # Unary attribute (e.g., is_valid(x))
    FUNCTION = auto()  # Function application (e.g., add(x, y) = z)
    CONSTRAINT = auto()  # Constraint (e.g., x < 10)
    RULE = auto()  # Inference rule (e.g., if A then B)


@dataclass(frozen=True)
class Fact:
    """
    Immutable representation of a logical fact.

    Facts are ground predicates that represent known truths.
    """

    name: str
    arguments: tuple[Any, ...]
    fact_type: SymbolicFactType = SymbolicFactType.PREDICATE
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Validate fact after initialization."""
        if not self.name:
            raise ValueError("Fact name cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            # Use object.__setattr__ for frozen dataclass
            object.__setattr__(self, "confidence", max(0.0, min(1.0, self.confidence)))

    def to_string(self) -> str:
        """Convert fact to Prolog-style string representation."""
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.name}({args_str})"

    def matches(self, other: Fact) -> bool:
        """Check if this fact matches another (same name and arguments)."""
        return self.name == other.name and self.arguments == other.arguments

    def unify(self, template: Fact) -> dict[str, Any] | None:
        """
        Attempt to unify this fact with a template.

        Returns variable bindings if successful, None otherwise.
        """
        if self.name != template.name:
            return None

        if len(self.arguments) != len(template.arguments):
            return None

        bindings: dict[str, Any] = {}
        for self_arg, template_arg in zip(self.arguments, template.arguments, strict=True):
            if isinstance(template_arg, str) and template_arg.startswith("?"):
                # Variable in template
                var_name = template_arg[1:]
                if var_name in bindings:
                    if bindings[var_name] != self_arg:
                        return None  # Conflicting bindings
                else:
                    bindings[var_name] = self_arg
            elif self_arg != template_arg:
                return None

        return bindings

    def substitute(self, bindings: dict[str, Any]) -> Fact:
        """Apply variable bindings to create a new fact."""
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

        return Fact(
            name=self.name,
            arguments=tuple(new_args),
            fact_type=self.fact_type,
            confidence=self.confidence,
            source=self.source,
        )

    def to_hash_key(self) -> str:
        """Generate deterministic hash key for caching."""
        content = f"{self.name}:{':'.join(str(a) for a in self.arguments)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# Type alias for backward compatibility
SymbolicFact = Fact


@runtime_checkable
class StateEncoder(Protocol):
    """Protocol for encoding states to neural embeddings."""

    def encode(self, state: NeuroSymbolicState) -> Any:  # torch.Tensor when available
        """Encode a state to a tensor representation."""
        ...

    def decode(self, embedding: Any) -> dict[str, Any]:  # torch.Tensor when available
        """Decode a tensor representation to state features."""
        ...


@dataclass
class NeuroSymbolicState:
    """
    Unified state representation combining symbolic and neural components.

    Attributes:
        state_id: Unique identifier for this state
        facts: Set of ground truth symbolic facts
        neural_embedding: Learned vector representation (lazy computed)
        constraints: Set of constraints that must hold
        confidence: Overall confidence in this state
        metadata: Additional state metadata
    """

    state_id: str
    facts: frozenset[Fact] = field(default_factory=frozenset)
    constraints: frozenset[str] = field(default_factory=frozenset)
    neural_embedding: Any | None = None  # torch.Tensor when available
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Caching
    _hash_key: str | None = field(default=None, repr=False)
    _fact_index: dict[str, list[Fact]] | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate state after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            self.confidence = max(0.0, min(1.0, self.confidence))

    @property
    def hash_key(self) -> str:
        """Compute or return cached hash key."""
        if self._hash_key is None:
            # Deterministic hash based on facts and constraints
            fact_strs = sorted(f.to_string() for f in self.facts)
            constraint_strs = sorted(self.constraints)
            content = {
                "state_id": self.state_id,
                "facts": fact_strs,
                "constraints": constraint_strs,
            }
            content_str = json.dumps(content, sort_keys=True)
            self._hash_key = hashlib.sha256(content_str.encode()).hexdigest()
        return self._hash_key

    @property
    def fact_index(self) -> dict[str, list[Fact]]:
        """Build or return cached index of facts by name."""
        if self._fact_index is None:
            self._fact_index = {}
            for fact in self.facts:
                if fact.name not in self._fact_index:
                    self._fact_index[fact.name] = []
                self._fact_index[fact.name].append(fact)
        return self._fact_index

    def add_fact(self, fact: Fact) -> NeuroSymbolicState:
        """Create new state with an additional fact."""
        new_facts = frozenset(self.facts | {fact})
        return NeuroSymbolicState(
            state_id=self.state_id,
            facts=new_facts,
            constraints=self.constraints,
            neural_embedding=None,  # Invalidate embedding
            confidence=self.confidence,
            metadata=self.metadata.copy(),
        )

    def remove_fact(self, fact: Fact) -> NeuroSymbolicState:
        """Create new state without the specified fact."""
        new_facts = frozenset(f for f in self.facts if not f.matches(fact))
        return NeuroSymbolicState(
            state_id=self.state_id,
            facts=new_facts,
            constraints=self.constraints,
            neural_embedding=None,
            confidence=self.confidence,
            metadata=self.metadata.copy(),
        )

    def add_constraint(self, constraint: str) -> NeuroSymbolicState:
        """Create new state with an additional constraint."""
        new_constraints = frozenset(self.constraints | {constraint})
        return NeuroSymbolicState(
            state_id=self.state_id,
            facts=self.facts,
            constraints=new_constraints,
            neural_embedding=self.neural_embedding,
            confidence=self.confidence,
            metadata=self.metadata.copy(),
        )

    def query_facts(self, name: str, **kwargs: Any) -> list[Fact]:
        """
        Query facts by name and optional argument filters.

        Args:
            name: Fact name to search for
            **kwargs: Argument filters (position=value)

        Returns:
            List of matching facts
        """
        candidates = self.fact_index.get(name, [])

        if not kwargs:
            return candidates

        results = []
        for fact in candidates:
            match = True
            for key, value in kwargs.items():
                if key.startswith("arg"):
                    try:
                        idx = int(key[3:])
                        if idx >= len(fact.arguments) or fact.arguments[idx] != value:
                            match = False
                            break
                    except (ValueError, IndexError):
                        match = False
                        break
            if match:
                results.append(fact)

        return results

    def has_fact(self, name: str, *args: Any) -> bool:
        """Check if state contains a specific fact."""
        target = Fact(name=name, arguments=tuple(args))
        return any(f.matches(target) for f in self.facts)

    def get_embedding(self, encoder: StateEncoder) -> Any:  # torch.Tensor when available
        """Get or compute neural embedding."""
        if self.neural_embedding is None:
            self.neural_embedding = encoder.encode(self)
        return self.neural_embedding

    def similarity(
        self,
        other: NeuroSymbolicState,
        encoder: StateEncoder | None = None,
        symbolic_weight: float = 0.5,
    ) -> float:
        """
        Compute similarity between two states.

        Combines symbolic Jaccard similarity with neural cosine similarity.

        Args:
            other: State to compare with
            encoder: Encoder for neural embeddings (optional if torch unavailable)
            symbolic_weight: Weight for symbolic similarity (0-1)

        Returns:
            Combined similarity score (0-1)
        """
        # Symbolic similarity (Jaccard index of facts)
        self_fact_strs = {f.to_string() for f in self.facts}
        other_fact_strs = {f.to_string() for f in other.facts}
        intersection = len(self_fact_strs & other_fact_strs)
        union = len(self_fact_strs | other_fact_strs)
        symbolic_sim = intersection / union if union > 0 else 1.0

        # If torch not available or no encoder, use only symbolic similarity
        if not TORCH_AVAILABLE or encoder is None:
            return symbolic_sim

        # Neural similarity (cosine similarity of embeddings)
        self_emb = self.get_embedding(encoder)
        other_emb = other.get_embedding(encoder)

        neural_sim = torch.nn.functional.cosine_similarity(
            self_emb.flatten().unsqueeze(0),
            other_emb.flatten().unsqueeze(0),
        ).item()

        # Normalize neural similarity to [0, 1]
        neural_sim = (neural_sim + 1) / 2

        # Combine
        neural_weight = 1.0 - symbolic_weight
        return symbolic_weight * symbolic_sim + neural_weight * neural_sim

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "state_id": self.state_id,
            "facts": [
                {
                    "name": f.name,
                    "arguments": f.arguments,
                    "type": f.fact_type.name,
                    "confidence": f.confidence,
                    "source": f.source,
                }
                for f in self.facts
            ],
            "constraints": list(self.constraints),
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NeuroSymbolicState:
        """Create state from dictionary."""
        facts = frozenset(
            Fact(
                name=f["name"],
                arguments=tuple(f["arguments"]),
                fact_type=SymbolicFactType[f.get("type", "PREDICATE")],
                confidence=f.get("confidence", 1.0),
                source=f.get("source", "unknown"),
            )
            for f in data.get("facts", [])
        )

        return cls(
            state_id=data["state_id"],
            facts=facts,
            constraints=frozenset(data.get("constraints", [])),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )

    def __hash__(self) -> int:
        """Hash based on state_id for set membership."""
        return hash(self.state_id)

    def __eq__(self, other: object) -> bool:
        """Equality based on facts and constraints."""
        if not isinstance(other, NeuroSymbolicState):
            return False
        return self.facts == other.facts and self.constraints == other.constraints


@dataclass
class StateTransition:
    """Represents a state transition with symbolic and neural components."""

    from_state: NeuroSymbolicState
    to_state: NeuroSymbolicState
    action: str
    preconditions: frozenset[Fact] = field(default_factory=frozenset)
    postconditions: frozenset[Fact] = field(default_factory=frozenset)
    probability: float = 1.0
    cost: float = 0.0

    def is_valid(self) -> bool:
        """Check if transition preconditions are satisfied."""
        return all(any(f.matches(precondition) for f in self.from_state.facts) for precondition in self.preconditions)

    def apply(self) -> NeuroSymbolicState:
        """Apply transition to create new state."""
        if not self.is_valid():
            raise ValueError("Transition preconditions not satisfied")

        # Remove negated facts and add postconditions
        new_facts = set(self.from_state.facts)
        for post in self.postconditions:
            # Handle negation (facts starting with 'not_')
            if post.name.startswith("not_"):
                positive_name = post.name[4:]
                to_remove = [f for f in new_facts if f.name == positive_name]
                for f in to_remove:
                    new_facts.discard(f)
            else:
                new_facts.add(post)

        return NeuroSymbolicState(
            state_id=f"{self.from_state.state_id}_{self.action}",
            facts=frozenset(new_facts),
            constraints=self.from_state.constraints,
            confidence=self.from_state.confidence * self.probability,
            metadata={
                **self.from_state.metadata,
                "last_action": self.action,
                "transition_cost": self.cost,
            },
        )


class SimpleStateEncoder:
    """
    Simple state encoder for testing and lightweight use cases.

    Encodes facts as bag-of-words style embeddings.
    """

    def __init__(self, embedding_dim: int, vocab_size: int = 10000, seed: int = 42):
        """Initialize encoder with random projection matrix."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("SimpleStateEncoder requires torch to be installed")

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)

        # Random projection matrix for hashing trick
        self.projection = torch.tensor(
            self.rng.normal(0, 1 / embedding_dim, (vocab_size, embedding_dim)),
            dtype=torch.float32,
        )

    def _hash_string(self, s: str) -> int:
        """Hash string to vocabulary index."""
        return int(hashlib.md5(s.encode()).hexdigest(), 16) % self.vocab_size

    def encode(self, state: NeuroSymbolicState) -> Any:  # torch.Tensor
        """Encode state to embedding vector."""
        embedding = torch.zeros(self.embedding_dim)

        for fact in state.facts:
            # Hash fact to vocabulary index
            idx = self._hash_string(fact.to_string())
            # Add weighted projection
            embedding += fact.confidence * self.projection[idx]

        for constraint in state.constraints:
            idx = self._hash_string(f"constraint:{constraint}")
            embedding += self.projection[idx]

        # Normalize
        norm = torch.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def decode(self, embedding: Any) -> dict[str, Any]:  # torch.Tensor
        """Decode embedding (returns metadata only for simple encoder)."""
        return {
            "embedding_norm": torch.norm(embedding).item(),
            "embedding_dim": len(embedding),
        }
