"""
LangSmith E2E Workflow Tracing Utilities.

Provides decorators and helpers for tracing E2E test workflows with LangSmith.
Automatically integrates with LangChain/LangGraph's built-in tracing while adding
test-specific metadata and tags.

Usage:
    @trace_e2e_test("e2e_complete_query_flow", phase="validation")
    async def test_tactical_analysis_flow(...):
        ...

    # Or use context manager
    with trace_e2e_workflow("e2e_mcts_simulation", tags=["mcts", "performance"]):
        run_simulation()
"""

from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
import functools
import inspect
import os
from typing import Any

from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree, tracing_context


def get_test_metadata() -> dict[str, Any]:
    """
    Extract common test metadata from environment and context.

    Returns:
        Dictionary with test environment metadata.
    """
    return {
        "test_timestamp": datetime.now().isoformat(),
        "ci_branch": os.getenv("GITHUB_REF_NAME", os.getenv("CI_BRANCH", "local")),
        "ci_commit": os.getenv("GITHUB_SHA", os.getenv("CI_COMMIT_SHA", "unknown")),
        "ci_run_id": os.getenv("GITHUB_RUN_ID", os.getenv("CI_RUN_ID", "local")),
        "environment": "ci" if os.getenv("CI") else "local",
        "python_version": os.getenv("PYTHON_VERSION", "3.11"),
        "langsmith_project": os.getenv("LANGSMITH_PROJECT", "langgraph-multi-agent-mcts"),
    }


def _build_trace_metadata(
    test_name: str,
    phase: str | None,
    scenario_type: str | None,
    provider: str | None,
    use_mcts: bool,
    mcts_iterations: int | None,
    tags: list[str] | None,
    metadata: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """
    Build metadata and tags for tracing (DRY helper for decorators).

    Args:
        test_name: Name of the test
        phase: Test phase
        scenario_type: Scenario being tested
        provider: LLM provider
        use_mcts: Whether MCTS is enabled
        mcts_iterations: Number of MCTS iterations
        tags: Additional tags
        metadata: Additional metadata

    Returns:
        Tuple of (run_metadata, run_tags)
    """
    run_metadata = get_test_metadata()

    # Add test-specific metadata
    run_metadata.update(
        {
            "test_suite": "e2e",
            "test_name": test_name,
            "phase": phase or "unknown",
            "scenario_type": scenario_type,
            "provider": provider,
            "use_mcts": use_mcts,
            "mcts_iterations": mcts_iterations,
        }
    )

    # Merge custom metadata
    if metadata:
        run_metadata.update(metadata)

    # Build tags list
    run_tags = ["e2e", "test"]
    if tags:
        run_tags.extend(tags)
    if phase:
        run_tags.append(f"phase:{phase}")
    if scenario_type:
        run_tags.append(f"scenario:{scenario_type}")
    if provider:
        run_tags.append(f"provider:{provider}")
    if use_mcts:
        run_tags.append("mcts")

    return run_metadata, run_tags


def trace_e2e_test(
    test_name: str,
    *,
    phase: str | None = None,
    scenario_type: str | None = None,
    provider: str | None = None,
    use_mcts: bool = False,
    mcts_iterations: int | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """
    Decorator for tracing E2E test functions with LangSmith.

    Creates a root-level trace for the test with comprehensive metadata,
    while allowing nested LangChain/LangGraph operations to auto-trace.

    Args:
        test_name: Name of the E2E test (e.g., "e2e_complete_query_flow")
        phase: Test phase (e.g., "validation", "processing", "integration")
        scenario_type: Scenario being tested (e.g., "tactical", "cybersecurity")
        provider: LLM provider being tested (e.g., "openai", "anthropic")
        use_mcts: Whether MCTS is enabled in this test
        mcts_iterations: Number of MCTS iterations (if applicable)
        tags: Additional tags for filtering in LangSmith UI
        metadata: Additional metadata to attach to the trace

    Example:
        @trace_e2e_test(
            "e2e_tactical_analysis",
            phase="complete_flow",
            scenario_type="tactical",
            use_mcts=False,
            tags=["hrm", "trm", "consensus"]
        )
        async def test_tactical_analysis_flow(mock_llm_client, tactical_query):
            # Test implementation
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            run_metadata, run_tags = _build_trace_metadata(
                test_name,
                phase,
                scenario_type,
                provider,
                use_mcts,
                mcts_iterations,
                tags,
                metadata,
            )

            # Use LangSmith's traceable decorator for the actual tracing
            traced_func = traceable(
                run_type="chain",  # E2E tests are like chains
                name=test_name,
                tags=run_tags,
                metadata=run_metadata,
            )(func)

            return await traced_func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            run_metadata, run_tags = _build_trace_metadata(
                test_name,
                phase,
                scenario_type,
                provider,
                use_mcts,
                mcts_iterations,
                tags,
                metadata,
            )

            # Use LangSmith's traceable decorator for the actual tracing
            traced_func = traceable(
                run_type="chain",
                name=test_name,
                tags=run_tags,
                metadata=run_metadata,
            )(func)

            return traced_func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def trace_e2e_workflow(
    workflow_name: str,
    *,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """
    Context manager for tracing E2E workflows.

    Use this for sections of code that should be traced as a single unit
    but aren't necessarily entire test functions.

    Args:
        workflow_name: Name of the workflow (e.g., "mcts_simulation", "api_request")
        tags: Tags for filtering in LangSmith
        metadata: Additional metadata

    Example:
        with trace_e2e_workflow("multi_provider_test", tags=["openai", "anthropic"]):
            results = run_multi_provider_test()
    """
    run_metadata = get_test_metadata()
    if metadata:
        run_metadata.update(metadata)

    run_tags = ["e2e", "workflow"]
    if tags:
        run_tags.extend(tags)

    with tracing_context(
        project_name=os.getenv("LANGSMITH_PROJECT", "langgraph-multi-agent-mcts"),
        tags=run_tags,
        metadata=run_metadata,
    ):
        yield


def trace_api_endpoint(
    endpoint: str,
    method: str = "POST",
    *,
    use_mcts: bool = False,
    use_rag: bool = False,
    tags: list[str] | None = None,
):
    """
    Decorator for tracing API endpoint tests.

    Args:
        endpoint: API endpoint path (e.g., "/query", "/health")
        method: HTTP method (default: "POST")
        use_mcts: Whether MCTS is enabled
        use_rag: Whether RAG is enabled
        tags: Additional tags

    Example:
        @trace_api_endpoint("/query", use_mcts=True, tags=["performance"])
        async def test_query_with_mcts():
            ...
    """

    def decorator(func: Callable) -> Callable:
        test_name = f"api_{endpoint.strip('/').replace('/', '_')}"
        run_tags = ["api", f"method:{method.lower()}"]

        if tags:
            run_tags.extend(tags)
        if use_mcts:
            run_tags.append("mcts")
        if use_rag:
            run_tags.append("rag")

        return trace_e2e_test(
            test_name,
            phase="api",
            tags=run_tags,
            metadata={
                "endpoint": endpoint,
                "http_method": method,
                "use_mcts": use_mcts,
                "use_rag": use_rag,
            },
        )(func)

    return decorator


def trace_mcts_simulation(
    iterations: int,
    *,
    scenario_type: str,
    seed: int | None = None,
    max_depth: int | None = None,
    tags: list[str] | None = None,
):
    """
    Decorator for tracing MCTS simulation tests.

    Args:
        iterations: Number of MCTS iterations
        scenario_type: Type of scenario (e.g., "tactical", "cybersecurity")
        seed: Random seed for reproducibility
        max_depth: Maximum tree depth
        tags: Additional tags

    Example:
        @trace_mcts_simulation(
            iterations=200,
            scenario_type="tactical",
            seed=42,
            tags=["performance"]
        )
        def test_200_iterations_latency():
            ...
    """

    def decorator(func: Callable) -> Callable:
        test_name = f"mcts_simulation_{iterations}_iterations"
        run_tags = ["mcts", "simulation", f"iterations:{iterations}"]

        if tags:
            run_tags.extend(tags)

        return trace_e2e_test(
            test_name,
            phase="mcts_simulation",
            scenario_type=scenario_type,
            use_mcts=True,
            mcts_iterations=iterations,
            tags=run_tags,
            metadata={
                "iterations": iterations,
                "seed": seed,
                "max_depth": max_depth,
            },
        )(func)

    return decorator


def update_run_metadata(additional_metadata: dict[str, Any]) -> None:
    """
    Update the current run's metadata dynamically.

    Useful for adding metadata during test execution based on runtime conditions.

    Args:
        additional_metadata: Metadata to add to the current run

    Example:
        update_run_metadata({
            "actual_latency_ms": 1234,
            "consensus_score": 0.85,
        })
    """
    try:
        current_run = get_current_run_tree()
        if current_run:
            current_run.extra = {**(current_run.extra or {}), **additional_metadata}
    except Exception:
        # Silently fail if we're not in a traced context
        pass


def add_run_tag(tag: str) -> None:
    """
    Add a tag to the current run dynamically.

    Args:
        tag: Tag to add

    Example:
        if error_occurred:
            add_run_tag("error")
    """
    try:
        current_run = get_current_run_tree()
        if current_run:
            if current_run.tags:
                current_run.tags.append(tag)
            else:
                current_run.tags = [tag]
    except Exception:
        # Silently fail if we're not in a traced context
        pass


def get_langsmith_client() -> Client:
    """
    Get configured LangSmith client.

    Returns:
        LangSmith Client instance

    Example:
        client = get_langsmith_client()
        runs = client.list_runs(project_name="langgraph-multi-agent-mcts")
    """
    return Client(
        api_key=os.getenv("LANGSMITH_API_KEY"),
        api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    )


def create_test_dataset(
    dataset_name: str,
    examples: list[dict[str, Any]],
    description: str | None = None,
) -> str:
    """
    Create a dataset in LangSmith for evaluation.

    Args:
        dataset_name: Name of the dataset
        examples: List of examples (each with 'inputs' and 'outputs')
        description: Dataset description

    Returns:
        Dataset ID

    Example:
        examples = [
            {
                "inputs": {"query": "Analyze tactical situation..."},
                "outputs": {"recommendation": "Secure Alpha position"}
            }
        ]
        dataset_id = create_test_dataset("tactical_scenarios", examples)
    """
    client = get_langsmith_client()
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description or f"E2E test dataset: {dataset_name}",
    )

    for example in examples:
        client.create_example(
            dataset_id=dataset.id,
            inputs=example.get("inputs", {}),
            outputs=example.get("outputs"),
        )

    return str(dataset.id)


# Convenience function for checking if tracing is enabled
def is_tracing_enabled() -> bool:
    """
    Check if LangSmith tracing is enabled.

    Returns:
        True if tracing is enabled, False otherwise
    """
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true"
