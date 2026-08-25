"""Construction-time validation for the LangGraph orchestration graph.

Pure standard-library helpers (no langgraph dependency) so they are importable and
unit-testable even when the optional graph runtime is absent. Three responsibilities:

* ``validate_state_schema`` — assert a ``TypedDict`` state schema is well-formed
  (annotations resolvable, reducer channels carry a callable) at graph build time.
* ``validate_graph_topology`` — assert every wired edge / conditional target refers
  to a registered node (or the terminal sentinel) at graph build time.
* ``validate_initial_state`` — assert a caller-supplied initial state has the required
  keys, no stray keys, and shallowly type-correct values *before* the graph runs.

The design goal is that shape errors surface at construction / invocation boundaries
with named exceptions, never mid-execution.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
import types
from typing import Any, NotRequired, Required, Union, get_args, get_origin, get_type_hints

# The terminal sentinel used by langgraph.graph.END. Duplicated here as a plain string so
# this module never imports langgraph; callers pass the real END and it compares equal.
END_SENTINEL = "__end__"


class GraphConstructionError(ValueError):
    """Raised when the graph's schema or topology is invalid at construction time."""


class StateValidationError(ValueError):
    """Raised when an initial state does not conform to its schema before execution."""


# Cache resolved hints per schema type (types are hashable; a plain dict sidesteps the
# lru_cache/type[Any] Hashable typing friction under mypy --strict).
_STATE_HINTS_CACHE: dict[type[Any], dict[str, Any]] = {}


def resolve_state_hints(schema: type[Any]) -> dict[str, Any]:
    """Return resolved type hints for a TypedDict schema (cached per type).

    Raises:
        GraphConstructionError: if the annotations cannot be resolved.
    """
    cached = _STATE_HINTS_CACHE.get(schema)
    if cached is not None:
        return cached
    try:
        hints = dict(get_type_hints(schema, include_extras=True))
    except Exception as exc:  # noqa: BLE001 - surface any resolution failure uniformly
        raise GraphConstructionError(f"Cannot resolve annotations for state schema {schema!r}: {exc}") from exc
    _STATE_HINTS_CACHE[schema] = hints
    return hints


def _strip_qualifiers(hint: Any) -> Any:
    """Unwrap ``Required``/``NotRequired`` and ``Annotated`` down to the payload type."""
    origin = get_origin(hint)
    if origin is Required or origin is NotRequired:
        args = get_args(hint)
        if args:
            hint = args[0]
    # Annotated payloads expose the wrapped type via ``__origin__`` and metadata via
    # ``__metadata__``; using the dunders is stable across Python versions.
    if hasattr(hint, "__metadata__"):
        hint = hint.__origin__
    return hint


def _reducer_metadata(hint: Any) -> tuple[Any, ...] | None:
    """Return the ``Annotated`` metadata tuple for a channel hint, or None."""
    inner = hint
    origin = get_origin(inner)
    if origin is Required or origin is NotRequired:
        args = get_args(inner)
        if args:
            inner = args[0]
    metadata = getattr(inner, "__metadata__", None)
    if metadata is None:
        return None
    return tuple(metadata)


def required_keys(schema: type[Any]) -> set[str]:
    """Return the required keys of a TypedDict schema, derived from resolved hints.

    ``__required_keys__`` / ``__optional_keys__`` are unreliable here: modules under
    ``from __future__ import annotations`` (PEP 563) stringize their annotations, so at
    class-creation time the TypedDict machinery cannot see the ``NotRequired[...]`` wrapper
    and marks every key required. ``get_type_hints(..., include_extras=True)`` evaluates the
    strings and preserves ``Required``/``NotRequired``, so required-ness is computed from the
    resolved hints instead, honoring the schema's ``total`` flag:

    * ``Required[...]``  -> always required,
    * ``NotRequired[...]`` -> always optional,
    * bare field        -> required only when the TypedDict is ``total=True`` (the default);
      for ``total=False`` schemas a bare field is optional.
    """
    total = bool(getattr(schema, "__total__", True))
    hints = resolve_state_hints(schema)
    required: set[str] = set()
    for key, hint in hints.items():
        origin = get_origin(hint)
        if origin is Required:
            required.add(key)
        elif origin is NotRequired:
            continue
        elif total:
            required.add(key)
    return required


def validate_state_schema(schema: type[Any]) -> None:
    """Validate that ``schema`` is a well-formed TypedDict state schema.

    Checks performed:
    * the schema is a TypedDict (exposes ``__annotations__`` and ``__total__``),
    * every annotation resolves,
    * every reducer channel (an ``Annotated`` field) carries at least one callable.

    Raises:
        GraphConstructionError: on any structural problem.
    """
    if not hasattr(schema, "__annotations__") or not hasattr(schema, "__total__"):
        raise GraphConstructionError(f"State schema {schema!r} is not a TypedDict")

    hints = resolve_state_hints(schema)
    if not hints:
        raise GraphConstructionError(f"State schema {schema!r} declares no fields")

    for name, hint in hints.items():
        metadata = _reducer_metadata(hint)
        if metadata is not None and not any(callable(meta) for meta in metadata):
            raise GraphConstructionError(
                f"Reducer channel '{name}' in {schema!r} carries no callable reducer in its Annotated metadata"
            )


def validate_graph_topology(
    *,
    nodes: Collection[str],
    edges: Iterable[tuple[str, str]],
    conditional_targets: Iterable[str],
    entry_point: str,
    terminal: str = END_SENTINEL,
) -> None:
    """Validate that every edge and conditional target refers to a known node.

    Args:
        nodes: registered node names.
        edges: ``(source, destination)`` static edges.
        conditional_targets: destination node names reachable via conditional edges.
        entry_point: the graph's entry node.
        terminal: the END sentinel value (destinations equal to it are always valid).

    Raises:
        GraphConstructionError: if a source/destination/entry point is not registered.
    """
    node_set = set(nodes)
    if entry_point not in node_set:
        raise GraphConstructionError(f"Entry point '{entry_point}' is not a registered node")

    for source, destination in edges:
        if source not in node_set:
            raise GraphConstructionError(f"Edge source '{source}' is not a registered node")
        if destination != terminal and destination not in node_set:
            raise GraphConstructionError(f"Edge destination '{destination}' is not a registered node or END")

    for target in conditional_targets:
        if target != terminal and target not in node_set:
            raise GraphConstructionError(f"Conditional routing target '{target}' is not a registered node or END")


def _matches_type(value: Any, hint: Any) -> bool:
    """Best-effort shallow type check of ``value`` against a (qualifier-stripped) hint."""
    hint = _strip_qualifiers(hint)
    if hint is Any or hint is object:
        return True

    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        return any(_matches_type(value, arg) for arg in get_args(hint))
    if origin is not None:
        # Parameterized generic (list[...], dict[...], ...) — check the container only.
        hint = origin

    if hint is None or hint is type(None):
        return value is None
    if not isinstance(hint, type):
        # Unresolvable to a concrete runtime type; do not reject.
        return True
    # ``isinstance`` already respects that ``bool`` is a subclass of ``int`` (an int
    # annotation accepts a bool), while a bool annotation rejects a plain int.
    return isinstance(value, hint)


def validate_initial_state(
    state: Mapping[str, Any],
    schema: type[Any] | None = None,
    *,
    allow_extra_keys: bool = False,
) -> None:
    """Validate an initial state mapping against a TypedDict schema before execution.

    Args:
        state: the initial state mapping to validate.
        schema: the TypedDict schema (defaults to :class:`AgentState`).
        allow_extra_keys: when True, keys outside the schema are permitted (escape
            hatch for callers that legitimately smuggle extra state).

    Raises:
        StateValidationError: on missing required keys, unknown keys, or a value whose
            type does not match its annotation.
    """
    if schema is None:
        from .state import AgentState

        schema = AgentState

    if not isinstance(state, Mapping):
        raise StateValidationError(f"Initial state must be a mapping, got {type(state).__name__}")

    hints = resolve_state_hints(schema)
    required = required_keys(schema)

    missing = required - set(state)
    if missing:
        raise StateValidationError(f"Initial state is missing required key(s): {sorted(missing)}")

    if not allow_extra_keys:
        unknown = set(state) - set(hints)
        if unknown:
            raise StateValidationError(f"Initial state has unknown key(s): {sorted(unknown)}")

    for key, value in state.items():
        hint = hints.get(key)
        if hint is None:
            continue  # unknown key already allowed (allow_extra_keys); nothing to check
        if not _matches_type(value, hint):
            raise StateValidationError(
                f"Initial state key '{key}' expected type compatible with {hint!r}, got {type(value).__name__}"
            )
