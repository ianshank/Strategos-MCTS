import socket
import time
import urllib.request

import pytest

# Optional gradio_client import - skip tests if not available
gradio_client = pytest.importorskip("gradio_client", reason="Gradio client required for UI tests")
Client = gradio_client.Client

import app


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def gradio_client():
    """Launch the Gradio demo once for all UI tests and provide a client."""
    from dataclasses import dataclass
    from unittest.mock import AsyncMock, MagicMock

    @dataclass
    class MockAgentResult:
        agent_name: str = "MCTS (Monte Carlo Tree Search)"
        response: str = (
            "[MCTS Analysis] This is a complete mock response for testing purposes. It contains enough content to pass character limit assertions."
        )
        confidence: float = 0.88
        reasoning_steps: list = None
        execution_time_ms: float = 125.5

        def __post_init__(self):
            if self.reasoning_steps is None:
                self.reasoning_steps = ["Step 1", "Step 2", "Step 3"]

    @dataclass
    class MockControllerDecision:
        selected_agent: str = "mcts"
        confidence: float = 0.73
        routing_probabilities: dict = None
        features_used: dict = None

        def __post_init__(self):
            if self.routing_probabilities is None:
                self.routing_probabilities = {"hrm": 0.2, "trm": 0.1, "mcts": 0.7}
            if self.features_used is None:
                self.features_used = {
                    "hrm_confidence": 0.35,
                    "trm_confidence": 0.25,
                    "mcts_value": 0.40,
                    "consensus_score": 0.65,
                    "query_length": 75,
                    "is_technical": True,
                }

    mock = MagicMock()
    mock.process_query = AsyncMock(return_value=(MockAgentResult(), MockControllerDecision()))
    mock.device = "cpu"

    app.framework = mock

    port = _get_free_port()
    _, _, _ = app.demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        show_error=True,
    )

    local_url = f"http://127.0.0.1:{port}"

    for _ in range(120):
        try:
            urllib.request.urlopen(f"{local_url}/", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        app.demo.close()
        raise RuntimeError("Gradio demo did not become ready in time")

    client = Client(local_url)
    yield client
    app.demo.close()


@pytest.mark.e2e
@pytest.mark.ui
def test_rnn_controller_flow_returns_complete_response(gradio_client):
    response, agent_details, routing_viz, features_viz, metrics, personality = gradio_client.predict(
        "Explain the difference between supervised and unsupervised learning with concrete examples.",
        "RNN",
        api_name="/process_query",
    )

    # App returns mocked responses with specific tags
    assert "[HRM Analysis]" in response or "[TRM Analysis]" in response or "[MCTS Analysis]" in response
    assert len(response) > 50

    assert isinstance(agent_details, dict)
    assert "agent" in agent_details
    assert len(agent_details.get("reasoning_steps", [])) >= 3

    assert "Selected Agent" in routing_viz
    assert "query_length" in features_viz
    assert "Execution Time" in metrics
    assert len(personality) > 50


@pytest.mark.e2e
@pytest.mark.ui
def test_bert_controller_flow_infers_personality_response(gradio_client):
    response, agent_details, routing_viz, features_viz, metrics, personality = gradio_client.predict(
        "Design a distributed rate limiting service for 100k requests per second and explain trade-offs.",
        "BERT",
        api_name="/process_query",
    )

    assert "[HRM Analysis]" in response or "[TRM Analysis]" in response or "[MCTS Analysis]" in response
    assert len(response) > 50

    assert isinstance(agent_details, dict)
    assert agent_details.get("agent")
    assert agent_details.get("confidence")

    assert "Routing Probabilities" in routing_viz
    assert "is_technical" in features_viz
    assert "Controller:" in metrics
    assert "Balanced" in personality or len(personality) > 50
