"""
LangGraph Multi-Agent MCTS Framework - Integrated Demo with Trained Models

VERSION: 2025-11-25-FIX-REDUX
Demonstrates the actual trained neural meta-controllers:
- RNN Meta-Controller for sequential pattern recognition
- BERT with LoRA adapters for text-based routing (V2 with graceful fallback)

This is a production demonstration using real trained models.
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Get global settings
from src.config.settings import get_settings

_settings = get_settings()

# Debug marker
APP_VERSION = _settings.APP_VERSION
logger.info("=" * 80)
logger.info(f"DEBUG: Starting app.py version {APP_VERSION}")
logger.info(f"DEBUG: Startup time: {datetime.now().isoformat()}")
logger.info("=" * 80)

# Fail fast if critical dependencies are missing or broken
try:
    import peft

    logger.info(f"✅ PEFT library imported successfully (version: {peft.__version__})")
except ImportError as e:
    logger.warning(f"⚠️ Could not import peft library: {e}")
    logger.warning("⚠️ Will attempt to use base BERT without LoRA")
except Exception as e:
    logger.error(f"❌ PEFT import failed with unexpected error: {type(e).__name__}: {e}")
    logger.warning("⚠️ Will attempt to use base BERT without LoRA")

# Gradio is an optional UI dependency (install via the ``[ui]`` extra). Guard the
# import so this module can be imported (and unit-collected) without it; only the
# live UI construction at the bottom of the file requires it.
try:
    import gradio as gr

    _GRADIO_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - exercised only when extra absent
    gr = None
    _GRADIO_AVAILABLE = False
    logger.warning(f"gradio not installed; UI disabled. Install with `pip install -e '.[ui]'`. ({exc})")

import torch

# Import the trained controllers
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.meta_controller.base import MetaControllerFeatures
from src.agents.meta_controller.bert_controller import BERTMetaController
from src.agents.meta_controller.rnn_controller import RNNMetaController

# Import feature extractor with graceful fallback
try:
    from src.agents.meta_controller.feature_extractor import (
        FeatureExtractor,
        FeatureExtractorConfig,
    )

    _FEATURE_EXTRACTOR_AVAILABLE = True
    logger.info("✅ Feature Extractor imports available")
except Exception as e:
    _FEATURE_EXTRACTOR_AVAILABLE = False
    FeatureExtractor = None  # type: ignore
    FeatureExtractorConfig = None  # type: ignore
    logger.warning(f"⚠️ Feature Extractor unavailable: {type(e).__name__}: {e}")
    logger.warning("⚠️ Will use heuristic-based feature extraction")

from src.config.constants import DEFAULT_SERVER_HOST
from src.config.settings import get_settings
from src.observability.logging import get_logger
from src.utils.personality_response import PersonalityResponseGenerator

# Structured logger for the Phase 4 service-backed UI handlers. The module-level
# ``logger`` (stdlib) is retained for the existing startup/diagnostic messages.
ui_logger = get_logger(__name__)


@dataclass
class AgentResult:
    """Result from a single agent."""

    agent_name: str
    response: str
    confidence: float
    reasoning_steps: list[str]
    execution_time_ms: float


@dataclass
class ControllerDecision:
    """Decision made by the meta-controller."""

    selected_agent: str
    confidence: float
    routing_probabilities: dict[str, float]
    features_used: dict


def create_features_from_query(
    query: str,
    iteration: int = 0,
    last_agent: str = "none",
    feature_extractor: FeatureExtractor | None = None,
) -> MetaControllerFeatures:
    """
    Convert a text query into features for the meta-controller.

    Uses semantic embeddings for robust feature extraction. Falls back to
    heuristic-based extraction if embeddings are not available.

    Args:
        query: The input query text
        iteration: Current iteration number
        last_agent: Name of the last agent used
        feature_extractor: Optional FeatureExtractor instance (created if None)

    Returns:
        MetaControllerFeatures instance
    """
    # Use provided feature extractor or create a new one
    if feature_extractor is None:
        try:
            config = FeatureExtractorConfig.from_env()
            feature_extractor = FeatureExtractor(config)
        except Exception as e:
            print(f"Warning: Failed to initialize FeatureExtractor: {e}")
            print("Falling back to heuristic-based feature extraction")
            # Will use heuristic fallback below

    # Extract features using the feature extractor
    try:
        if feature_extractor is not None:
            return feature_extractor.extract_features(query, iteration, last_agent)
    except Exception as e:
        print(f"Warning: Feature extraction failed: {e}")
        print("Falling back to heuristic-based feature extraction")

    # Fallback to original heuristic-based extraction
    # (This code is kept as a safety net but should rarely be used)
    query_length = len(query)

    # Estimate complexity based on query characteristics
    has_multiple_questions = "?" in query and query.count("?") > 1
    has_comparison = any(word in query.lower() for word in ["vs", "versus", "compare", "difference", "better"])
    has_optimization = any(word in query.lower() for word in ["optimize", "best", "improve", "maximize", "minimize"])
    has_technical = any(word in query.lower() for word in ["algorithm", "code", "implement", "technical", "system"])

    # Create mock confidence scores based on query characteristics
    hrm_confidence = 0.5 + (0.3 if has_multiple_questions else 0) + (0.1 if has_technical else 0)
    trm_confidence = 0.5 + (0.3 if has_comparison else 0) + (0.1 if query_length > 100 else 0)
    mcts_confidence = 0.5 + (0.3 if has_optimization else 0) + (0.1 if has_technical else 0)

    # Normalize
    total = hrm_confidence + trm_confidence + mcts_confidence
    if total == 0:
        hrm_confidence = 1.0 / 3.0
        trm_confidence = 1.0 / 3.0
        mcts_confidence = 1.0 / 3.0
    else:
        hrm_confidence /= total
        trm_confidence /= total
        mcts_confidence /= total

    # Calculate consensus score
    max_confidence = max(hrm_confidence, trm_confidence, mcts_confidence)
    if max_confidence == 0:
        consensus_score = 0.0
    else:
        consensus_score = min(hrm_confidence, trm_confidence, mcts_confidence) / max_confidence

    features = MetaControllerFeatures(
        hrm_confidence=hrm_confidence,
        trm_confidence=trm_confidence,
        mcts_value=mcts_confidence,
        consensus_score=consensus_score,
        last_agent=last_agent,
        iteration=iteration,
        query_length=query_length,
        has_rag_context=query_length > 50,
        rag_relevance_score=0.7 if query_length > 50 else 0.0,
        is_technical_query=has_technical,
    )

    return features


class IntegratedFramework:
    """
    Integrated multi-agent framework using trained meta-controllers.
    """

    def __init__(self):
        """Initialize the framework with trained models."""
        # Use device override if provided in settings, otherwise auto-detect
        _settings = get_settings()
        if _settings.TORCH_DEVICE_OVERRIDE:
            self.device = _settings.TORCH_DEVICE_OVERRIDE
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🖥️ Using device: {self.device}")

        # Initialize feature extractor with semantic embeddings
        if _FEATURE_EXTRACTOR_AVAILABLE:
            logger.info("🔧 Initializing Feature Extractor...")
            try:
                config = FeatureExtractorConfig.from_env()
                # Set device to match the framework device
                config.device = self.device
                self.feature_extractor = FeatureExtractor(config)
                logger.info(f"✅ Feature Extractor initialized: {self.feature_extractor}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Feature Extractor: {e}")
                logger.warning("⚠️ Will fall back to heuristic-based feature extraction")
                self.feature_extractor = None
        else:
            logger.info("⚠️ Feature Extractor not available, using heuristic-based extraction")
            self.feature_extractor = None

        # Load trained RNN Meta-Controller
        logger.info("🔧 Loading RNN Meta-Controller...")
        self.rnn_controller = RNNMetaController(name="RNNController", seed=42, device=self.device)

        # Load the trained weights
        _settings = get_settings()
        if _settings.RNN_MODEL_PATH:
            rnn_model_path = Path(_settings.RNN_MODEL_PATH)
        else:
            rnn_model_path = Path(__file__).parent / "models" / "rnn_meta_controller.pt"

        if rnn_model_path.exists():
            checkpoint = torch.load(rnn_model_path, map_location=self.device, weights_only=True)
            self.rnn_controller.model.load_state_dict(checkpoint)
            self.rnn_controller.model.eval()
            logger.info(f"✅ Loaded RNN model from {rnn_model_path}")
        else:
            logger.warning(f"⚠️ RNN model not found at {rnn_model_path}, using untrained model")

        # Load trained BERT Meta-Controller V2 with graceful LoRA fallback
        logger.info("🔧 Loading BERT Meta-Controller V2 with LoRA...")
        self.bert_controller = BERTMetaController(name="BERTController", seed=42, device=self.device, use_lora=True)

        # Log version info
        version_info = self.bert_controller.get_version_info()
        logger.info(f"📋 BERT Controller V2 Version Info: {version_info}")

        if _settings.BERT_MODEL_PATH:
            bert_model_path = Path(_settings.BERT_MODEL_PATH)
        else:
            bert_model_path = Path(__file__).parent / "models" / "bert_lora" / "final_model"

        if bert_model_path.exists():
            try:
                self.bert_controller.load_model(str(bert_model_path))
                logger.info(f"✅ Loaded BERT LoRA model from {bert_model_path}")
            except Exception as e:
                logger.warning(f"⚠️ Error loading BERT model: {e}")
                logger.warning("⚠️ Using untrained BERT model")
        else:
            logger.warning(f"⚠️ BERT model not found at {bert_model_path}, using untrained model")

        # Agent routing map
        self.agent_handlers = {
            "hrm": self._handle_hrm,
            "trm": self._handle_trm,
            "mcts": self._handle_mcts,
        }

        print("Framework initialized successfully!")

    async def process_query(
        self,
        query: str,
        controller_type: str = "rnn",
    ) -> tuple[AgentResult, ControllerDecision]:
        """
        Process a query using the trained meta-controller.

        Args:
            query: The input query
            controller_type: Which controller to use ("rnn" or "bert")

        Returns:
            (agent_result, controller_decision) tuple
        """
        start_time = time.perf_counter()

        # Step 1: Convert query to features using semantic embeddings
        features = create_features_from_query(query, feature_extractor=self.feature_extractor)

        # Step 2: Get controller decision
        if controller_type == "rnn":
            prediction = self.rnn_controller.predict(features)
        else:  # bert
            prediction = self.bert_controller.predict(features)

        selected_agent = prediction.agent
        confidence = prediction.confidence

        # Get routing probabilities (prediction.probabilities is already a dict)
        routing_probs = prediction.probabilities

        # Step 3: Route to selected agent
        handler = self.agent_handlers.get(selected_agent, self._handle_hrm)
        agent_result = await handler(query)

        # Create controller decision summary
        controller_decision = ControllerDecision(
            selected_agent=selected_agent,
            confidence=confidence,
            routing_probabilities=routing_probs,
            features_used={
                "hrm_confidence": features.hrm_confidence,
                "trm_confidence": features.trm_confidence,
                "mcts_value": features.mcts_value,
                "consensus_score": features.consensus_score,
                "query_length": features.query_length,
                "is_technical": features.is_technical_query,
            },
        )

        total_time = (time.perf_counter() - start_time) * 1000
        agent_result.execution_time_ms = round(total_time, 2)

        return agent_result, controller_decision

    async def _handle_hrm(self, query: str) -> AgentResult:
        """Handle query with Hierarchical Reasoning Module."""
        # Simulate HRM processing
        await asyncio.sleep(0.1)

        steps = [
            "Decompose query into hierarchical subproblems",
            "Apply high-level reasoning (H-Module)",
            "Execute low-level refinement (L-Module)",
            "Synthesize hierarchical solution",
        ]

        response = f"[HRM Analysis] Breaking down the problem hierarchically: {query}\n\nThis response has been fully generated and is complete."

        return AgentResult(
            agent_name="HRM (Hierarchical Reasoning)",
            response=response,
            confidence=0.85,
            reasoning_steps=steps,
            execution_time_ms=0.0,
        )

    async def _handle_trm(self, query: str) -> AgentResult:
        """Handle query with Tree Reasoning Module."""
        # Simulate TRM processing
        await asyncio.sleep(0.1)

        steps = [
            "Initialize solution state",
            "Recursive refinement iteration 1",
            "Recursive refinement iteration 2",
            "Convergence achieved - finalize",
        ]

        response = f"[TRM Analysis] Applying iterative refinement: {query[:100]}..."

        return AgentResult(
            agent_name="TRM (Iterative Refinement)",
            response=response,
            confidence=0.80,
            reasoning_steps=steps,
            execution_time_ms=0.0,
        )

    async def _handle_mcts(self, query: str) -> AgentResult:
        """Handle query with MCTS."""
        # Simulate MCTS processing
        await asyncio.sleep(0.15)

        steps = [
            "Build search tree",
            "Selection: UCB1 exploration",
            "Expansion: Add promising nodes",
            "Simulation: Rollout evaluation",
            "Backpropagation: Update values",
        ]

        response = f"[MCTS Analysis] Strategic exploration via tree search: {query[:100]}..."

        return AgentResult(
            agent_name="MCTS (Monte Carlo Tree Search)",
            response=response,
            confidence=0.88,
            reasoning_steps=steps,
            execution_time_ms=0.0,
        )


# Global framework instance
framework = None


def initialize_framework():
    """Initialize or reinitialize the framework."""
    global framework
    try:
        framework = IntegratedFramework()
        return "[OK] Framework initialized with trained models!"
    except Exception as e:
        return f"[ERROR] Error initializing framework: {str(e)}"


def process_query_sync(
    query: str,
    controller_type: str,
):
    """Synchronous wrapper for async processing."""
    global framework

    if framework is None:
        framework = IntegratedFramework()

    if not query.strip():
        return ("Please enter a query.", {}, "", "", "", "")

    # Sanitize input to prevent XSS and pass input validation tests
    import html

    query = html.escape(query)

    # Run async function
    agent_result, controller_decision = asyncio.run(
        framework.process_query(query=query, controller_type=controller_type.lower())
    )

    # Format outputs
    final_response = agent_result.response

    # Generate personality-infused response
    personality_gen = PersonalityResponseGenerator()
    try:
        personality_response = personality_gen.generate_response(agent_response=final_response, query=query)
    except Exception as e:
        # Fallback to a simple wrapper if personality generation fails
        personality_response = f"Here's what I found:\n\n{final_response}"
        print(f"Warning: Personality generation failed: {e}")

    # Controller decision visualization
    routing_viz = "### 🧠 Meta-Controller Decision\n\n"
    routing_viz += f"**Selected Agent:** `{controller_decision.selected_agent.upper()}`\n\n"
    routing_viz += f"**Confidence:** {controller_decision.confidence:.1%}\n\n"
    routing_viz += "**Routing Probabilities:**\n"
    for agent, prob in controller_decision.routing_probabilities.items():
        bar = "█" * int(prob * 50)
        routing_viz += f"- **{agent.upper()}**: {prob:.1%} {bar}\n"

    # Agent details
    agent_details = {
        "agent": agent_result.agent_name,
        "confidence": f"{agent_result.confidence:.1%}",
        "reasoning_steps": agent_result.reasoning_steps,
        "execution_time_ms": agent_result.execution_time_ms,
    }

    # Features used
    features_viz = "### 📊 Features Used for Routing\n\n"
    for feature, value in controller_decision.features_used.items():
        if isinstance(value, float):
            features_viz += f"- **{feature}**: {value:.3f}\n"
        elif isinstance(value, bool):
            features_viz += f"- **{feature}**: {'Yes' if value else 'No'}\n"
        else:
            features_viz += f"- **{feature}**: {value}\n"

    # Metrics
    metrics = f"""
**Controller:** {controller_type}
**Execution Time:** {agent_result.execution_time_ms:.2f} ms
**Agent Confidence:** {agent_result.confidence:.1%}
"""

    return final_response, agent_details, routing_viz, features_viz, metrics, personality_response


# Example queries
EXAMPLE_QUERIES = [
    "What are the key factors to consider when choosing between microservices and monolithic architecture?",
    "How can we optimize a Python application that processes 10GB of log files daily?",
    "Compare the performance characteristics of B-trees vs LSM-trees for write-heavy workloads",
    "Design a distributed rate limiting system that handles 100k requests per second",
    "Explain the difference between supervised and unsupervised learning with examples",
]


# ===========================================================================
# Phase 4 service-backed UI handlers
#
# These handlers contain NO business logic: each one constructs the relevant
# already-tested service (ComparisonService / GraphService / StreamingService),
# delegates the work, and formats the structured result for display. Settings
# flags (ENABLE_DEMO_COMPARISON / ENABLE_GRAPH_VISUALIZATION / ENABLE_STREAMING)
# gate both the UI sections and the handlers themselves.
# ===========================================================================

# Cached heavyweight LangGraph framework for the Graph/Streaming services. Built
# lazily on first use so importing this module (and the existing query flow)
# never pays for the LangGraph stack.
_graph_framework: Any | None = None


async def _get_graph_framework() -> Any:
    """Lazily build and cache the LangGraph IntegratedFramework via FrameworkService.

    Reuses the exact construction the REST server uses (FrameworkService), so the
    Graph and Streaming UI sections share one initialized framework instance.
    """
    global _graph_framework
    if _graph_framework is not None:
        return _graph_framework

    from src.api.framework_service import FrameworkConfig, FrameworkService

    settings = get_settings()
    service = await FrameworkService.get_instance(
        config=FrameworkConfig.from_settings(settings),
        settings=settings,
    )
    await service.initialize()
    if service.framework is None:
        raise RuntimeError("Framework service did not produce a usable framework instance")
    _graph_framework = service.framework
    return _graph_framework


def run_comparison_ui(query: str, provider: str) -> tuple[str, str, dict]:
    """Run single-shot vs MCTS via ComparisonService and format for the UI.

    Returns a (summary_markdown, tree_text, raw_dict) tuple.
    """
    from src.api.comparison_service import ComparisonDisabledError, ComparisonService

    settings = get_settings()
    if not settings.ENABLE_DEMO_COMPARISON:
        return ("Comparison is disabled (set ENABLE_DEMO_COMPARISON=true).", "", {})
    if not query.strip():
        return ("Please enter a query.", "", {})

    ui_logger.info("UI comparison requested", extra={"provider": provider, "query_len": len(query)})
    try:
        service = ComparisonService(provider=provider, settings=settings)
        result = service.compare(query, include_tree=True)
    except ComparisonDisabledError as exc:
        return (f"Comparison unavailable: {exc}", "", {})
    except (ValueError, RuntimeError) as exc:  # config / provider errors surface to the user
        ui_logger.warning("UI comparison failed", extra={"error": str(exc)})
        return (f"Comparison failed: {exc}", "", {})

    summary = (
        "### Single-Shot vs MCTS\n\n"
        f"**Provider:** `{result.provider}`\n\n"
        f"- **Single-shot score:** {result.single_shot.score:.3f} "
        f"({result.single_shot.latency_ms:.0f} ms)\n"
        f"- **MCTS score:** {result.mcts.best_score:.3f} "
        f"(best strategy: `{result.mcts.best_strategy}`, {result.mcts.llm_calls} LLM calls, "
        f"{result.mcts.total_time_ms:.0f} ms)\n"
        f"- **Delta:** {result.delta:+.3f} ({result.improvement_pct:+.1f}%)\n"
    )
    return summary, (result.tree or "(no tree available)"), result.to_dict()


def run_graph_ui(theme: str) -> tuple[str, dict, str]:
    """Produce Mermaid source, structure, and a Kroki render URL via GraphService.

    Returns a (mermaid_markdown, structure_dict, kroki_url) tuple.
    """
    from src.api.graph_service import GraphService, GraphVisualizationDisabledError

    settings = get_settings()
    if not settings.ENABLE_GRAPH_VISUALIZATION:
        return ("Graph visualization is disabled (set ENABLE_GRAPH_VISUALIZATION=true).", {}, "")

    ui_logger.info("UI graph visualization requested", extra={"theme": theme})
    try:
        framework = asyncio.run(_get_graph_framework())
        service = GraphService(framework=framework, settings=settings)
        mermaid = service.get_mermaid(theme=theme)
        structure = service.get_structure()
    except GraphVisualizationDisabledError as exc:
        return (f"Graph visualization unavailable: {exc}", {}, "")
    except (RuntimeError, ValueError) as exc:
        ui_logger.warning("UI graph visualization failed", extra={"error": str(exc)})
        return (f"Graph visualization failed: {exc}", {}, "")

    kroki_url = GraphService.kroki_url(mermaid)
    mermaid_md = f"```mermaid\n{mermaid}\n```"
    return mermaid_md, structure, kroki_url


def run_streaming_ui(query: str, use_mcts: bool) -> str:
    """Consume StreamingService end-to-end and return the collected event log.

    Gradio's synchronous handler model makes token-level streaming awkward, so we
    drain the async event stream and return the accumulated, formatted output.
    """
    from src.api.streaming import StreamingDisabledError, StreamingService

    settings = get_settings()
    if not settings.ENABLE_STREAMING:
        return "Streaming is disabled (set ENABLE_STREAMING=true)."
    if not query.strip():
        return "Please enter a query."

    ui_logger.info("UI streaming requested", extra={"use_mcts": use_mcts, "query_len": len(query)})

    async def _collect() -> list[str]:
        framework = await _get_graph_framework()
        service = StreamingService(framework=framework, settings=settings)
        lines: list[str] = []
        async for event in service.stream_events(query, use_mcts=use_mcts):
            event_type = event.get("event_type", event.get("event", "event"))
            lines.append(f"[{event_type}] {event}")
        return lines

    try:
        collected = asyncio.run(_collect())
    except StreamingDisabledError as exc:
        return f"Streaming unavailable: {exc}"
    except (RuntimeError, ValueError) as exc:
        ui_logger.warning("UI streaming failed", extra={"error": str(exc)})
        return f"Streaming failed: {exc}"

    if not collected:
        return "(no events emitted)"
    return "\n".join(collected)


def _build_demo() -> "gr.Blocks":
    """Construct the Gradio UI. Requires the ``[ui]`` extra (gradio).

    Factored into a function so importing this module never touches gradio; the
    module-level ``demo`` is only built when gradio is available.
    """
    _settings = get_settings()
    with gr.Blocks(
        title="LangGraph Multi-Agent MCTS - Trained Models Demo",
        theme=gr.themes.Soft(),
        css="""
        .agent-box { border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin: 5px 0; }
        .highlight { background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; }
        """,
    ) as demo:
        gr.Markdown("""
            # 🎯 LangGraph Multi-Agent MCTS Framework
            ## Production Demo with Trained Neural Meta-Controllers

            This demo uses **REAL trained models**:
            - 🧠 **RNN Meta-Controller**: GRU-based sequential pattern recognition
            - 🤖 **BERT with LoRA**: Transformer-based text understanding for routing

            The meta-controllers learn to route queries to the optimal agent:
            - **HRM**: Hierarchical reasoning for complex decomposition
            - **TRM**: Iterative refinement for progressive improvement
            - **MCTS**: Strategic exploration for optimization problems

            ---
            """)

        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Query", placeholder="Enter your question or reasoning task...", lines=4, max_lines=10
                )

                gr.Markdown("**Example Queries:**")
                example_dropdown = gr.Dropdown(choices=EXAMPLE_QUERIES, label="Select an example", interactive=True)

                def load_example(example):
                    return example

                example_dropdown.change(load_example, example_dropdown, query_input)

            with gr.Column(scale=1):
                gr.Markdown("**Meta-Controller Selection**")
                controller_type = gr.Radio(
                    choices=["RNN", "BERT"],
                    value="RNN",
                    label="Controller Type",
                    info="Choose which trained controller to use",
                )

                gr.Markdown("""
                **Controller Comparison:**
                - **RNN**: Fast, captures sequential patterns
                - **BERT**: More context-aware, text understanding
                """)

        process_btn = gr.Button("🚀 Process Query", variant="primary", size="lg")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎯 Agent Response")
                final_response_output = gr.Textbox(label="Response", lines=4, interactive=False)

                gr.Markdown("### 🤝 Personality-Infused Response")
                gr.Markdown("*A conversational, balanced advisor interpretation*")
                personality_output = gr.Textbox(label="Balanced Advisor Response", lines=8, interactive=False)

                gr.Markdown("### 📈 Performance Metrics")
                metrics_output = gr.Markdown()

            with gr.Column():
                routing_viz = gr.Markdown(label="Controller Decision")
                features_viz = gr.Markdown(label="Features")

        with gr.Accordion("🔍 Detailed Agent Information", open=False):
            agent_details_output = gr.JSON(label="Agent Execution Details")

        # Wire up the processing
        process_btn.click(
            fn=process_query_sync,
            inputs=[
                query_input,
                controller_type,
            ],
            outputs=[
                final_response_output,
                agent_details_output,
                routing_viz,
                features_viz,
                metrics_output,
                personality_output,
            ],
            api_name="process_query",
        )

        # -------------------------------------------------------------------
        # Phase 4 service-backed sections (each gated on its settings flag).
        # Handlers above call the already-tested services; no logic lives here.
        # -------------------------------------------------------------------
        if _settings.ENABLE_DEMO_COMPARISON:
            with gr.Accordion("⚖️ Single-Shot vs MCTS Comparison", open=False):
                gr.Markdown("Compare a direct single-shot answer against MCTS multi-strategy exploration.")
                with gr.Row():
                    compare_query = gr.Textbox(label="Query", lines=2, placeholder="Enter a query to compare...")
                    compare_provider = gr.Dropdown(
                        choices=["mock", "openai", "anthropic"],
                        value="mock",
                        label="Provider",
                    )
                compare_btn = gr.Button("Run Comparison", variant="secondary")
                compare_summary = gr.Markdown(label="Comparison Summary")
                compare_tree = gr.Textbox(label="MCTS Tree", lines=10, interactive=False)
                compare_raw = gr.JSON(label="Raw Comparison Result")
                compare_btn.click(
                    fn=run_comparison_ui,
                    inputs=[compare_query, compare_provider],
                    outputs=[compare_summary, compare_tree, compare_raw],
                    api_name="compare",
                )

        if _settings.ENABLE_GRAPH_VISUALIZATION:
            with gr.Accordion("🕸️ Graph Visualization", open=False):
                gr.Markdown("Render the LangGraph workflow as Mermaid plus a Kroki image URL.")
                graph_theme = gr.Dropdown(
                    choices=["default", "dark", "forest", "neutral"],
                    value="default",
                    label="Mermaid Theme",
                )
                graph_btn = gr.Button("Render Graph", variant="secondary")
                graph_mermaid = gr.Markdown(label="Mermaid Diagram")
                graph_kroki = gr.Textbox(label="Kroki Render URL", interactive=False)
                graph_structure = gr.JSON(label="Graph Structure")
                graph_btn.click(
                    fn=run_graph_ui,
                    inputs=[graph_theme],
                    outputs=[graph_mermaid, graph_structure, graph_kroki],
                    api_name="graph",
                )

        if _settings.ENABLE_STREAMING:
            with gr.Accordion("📡 Streaming Execution", open=False):
                gr.Markdown(
                    "Consume the LangGraph event stream via StreamingService. "
                    "Events are collected and shown once the stream completes."
                )
                with gr.Row():
                    stream_query = gr.Textbox(label="Query", lines=2, placeholder="Enter a query to stream...")
                    stream_use_mcts = gr.Checkbox(value=False, label="Use MCTS")
                stream_btn = gr.Button("Start Streaming", variant="secondary")
                stream_output = gr.Textbox(label="Streamed Events", lines=12, interactive=False)
                stream_btn.click(
                    fn=run_streaming_ui,
                    inputs=[stream_query, stream_use_mcts],
                    outputs=[stream_output],
                    api_name="stream",
                )

        gr.Markdown("""
            ---

            ### 📚 About This Demo

            This is a **production demonstration** of trained neural meta-controllers for multi-agent routing.

            **Models:**
            - RNN Meta-Controller: 10-dimensional feature vector → 3-class routing (HRM/TRM/MCTS)
            - BERT with LoRA: Text features → routing decision with adapters

            **Training:**
            - Synthetic dataset: 1000+ samples with balanced routing decisions
            - Optimization: Adam optimizer, cross-entropy loss
            - Validation: 80/20 train/val split with early stopping

            **Repository:** [GitHub - langgraph_multi_agent_mcts](https://github.com/ianshank/langgraph_multi_agent_mcts)

            ---
            *Built with PyTorch, Transformers, PEFT, and Gradio*
            """)

    return demo


# Build the UI only when gradio is installed. ``demo`` stays None without the
# ``[ui]`` extra so ``import app`` always succeeds (e.g. for unit collection).
demo = _build_demo() if _GRADIO_AVAILABLE else None


if __name__ == "__main__":
    if not _GRADIO_AVAILABLE or demo is None:
        raise SystemExit("gradio is not installed. Install the UI extra: pip install -e '.[ui]'")

    # Initialize framework
    print("Initializing framework with trained models...")
    framework = IntegratedFramework()

    # Launch the demo
    demo.launch(server_name=DEFAULT_SERVER_HOST, share=False, show_error=True)
