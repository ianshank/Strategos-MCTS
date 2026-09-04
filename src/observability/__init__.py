# Observability Module
"""
Comprehensive observability infrastructure for multi-agent MCTS framework.

Includes:
- JSON structured logging with correlation IDs
- OpenTelemetry tracing with automatic span creation
- Metrics collection for MCTS and agent performance
- Debug utilities for MCTS tree visualization
- Performance profiling tools
"""

from .debug import MCTSDebugger, export_tree_to_dot, visualize_mcts_tree
from .logging import CorrelationIdFilter, get_logger, setup_logging
from .metrics import MetricsCollector, agent_metrics, mcts_metrics
from .profiling import AsyncProfiler, MemoryProfiler, generate_performance_report, profile_block
from .tracing import TracingManager, get_tracer, trace_operation

# Braintrust integration (optional)
try:
    from .braintrust_tracker import (  # noqa: F401
        BRAINTRUST_AVAILABLE,
        BraintrustContextManager,
        BraintrustTracker,
        create_training_tracker,
    )

    _braintrust_exports = [
        "BraintrustTracker",
        "BraintrustContextManager",
        "create_training_tracker",
        "BRAINTRUST_AVAILABLE",
    ]
except ImportError:
    _braintrust_exports = []

__all__ = [
    # Logging
    "setup_logging",
    "configure_cli_logging",
    "get_logger",
    "CorrelationIdFilter",
    # Tracing
    "TracingManager",
    "trace_operation",
    "get_tracer",
    # Metrics
    "MetricsCollector",
    "mcts_metrics",
    "agent_metrics",
    # Debug
    "MCTSDebugger",
    "export_tree_to_dot",
    "visualize_mcts_tree",
    # Profiling
    "profile_block",
    "AsyncProfiler",
    "MemoryProfiler",
    "generate_performance_report",
] + _braintrust_exports
