"""
End-to-End integration tests for live inference API.
Spins up the inference_server.py process and validates endpoints.
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import pytest

# Constants
TEST_PORT = 8123
HOST = f"http://127.0.0.1:{TEST_PORT}"
CHECKPOINT_PATH = "artifacts/trainings/unified_orchestrator_checkpoint.pt"


@pytest.fixture(scope="module")
def live_server():
    """Fixture to start and stop the inference server as a background process."""
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(f"E2E test requires {CHECKPOINT_PATH} to be present.")

    # Start the server
    cmd = [
        "python",
        "-m",
        "src.api.inference_server",
        "--checkpoint",
        CHECKPOINT_PATH,
        "--port",
        str(TEST_PORT),
        "--device",
        "cpu"  # Force CPU for test stability across environments
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for server to become healthy
    max_retries = 30
    ready = False

    for _ in range(max_retries):
        try:
            req = urllib.request.Request(f"{HOST}/health")
            with urllib.request.urlopen(req) as response:
                if response.getcode() == 200:
                    data = json.loads(response.read().decode())
                    if data.get("status") == "healthy":
                        ready = True
                        break
        except urllib.error.URLError:
            pass
        time.sleep(1.0)

    if not ready:
        process.terminate()
        stdout, stderr = process.communicate()
        raise RuntimeError(f"Server failed to start. stderr: {stderr.decode()}")

    yield HOST

    # Teardown
    process.terminate()
    process.wait(timeout=5)


@pytest.mark.e2e
def test_health_endpoint(live_server):
    """Test that the /health endpoint returns valid status."""
    req = urllib.request.Request(f"{live_server}/health")
    with urllib.request.urlopen(req) as response:
        assert response.getcode() == 200
        data = json.loads(response.read().decode())
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


@pytest.mark.e2e
def test_inference_endpoint(live_server):
    """Test the /inference endpoint with a dummy state."""
    # Assuming board_size 6x7 for connect four, action_size=7, state=42
    # but the neural_net config dynamically defines it.
    # For a generic state, let's just pass a generic tensor shape of what is expected
    # The default board_size=19 implies 19x19 input if Go, or maybe flat 361.
    # The model expects [batch_size, input_channels, rows, cols].
    # If the unified checkpoint was Connect Four, state is [17, 6, 7] flattened = 714
    # Wait, the request parses state as a list of floats.
    # Let's send a minimal valid state shape.
    state = [[[0.0]*7 for _ in range(6)] for _ in range(17)]

    payload = {
        "state": state,
        "use_hrm_decomposition": True,
        "use_mcts": True,
        "use_trm_refinement": True
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{live_server}/inference", data=data, headers={'Content-Type': 'application/json'})

    try:
        with urllib.request.urlopen(req) as response:
            assert response.getcode() == 200
            res_data = json.loads(response.read().decode())
            assert "best_action" in res_data
            assert "value_estimate" in res_data
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode()
        pytest.fail(f"HTTPError {e.code}: {error_msg}")
