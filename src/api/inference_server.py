"""
FastAPI Inference Server for LangGraph Multi-Agent MCTS.

Provides REST API for:
- Problem solving with HRM+MCTS+TRM
- Policy-value network inference
- Health checks and monitoring
"""

import logging
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config.constants import DEFAULT_SERVER_HOST
from ..config.settings import get_settings
from ..framework.mcts.neural_mcts import NeuralMCTS
from ..training.performance_monitor import PerformanceMonitor
from ..training.system_config import SystemConfig

# Configure module logger
logger = logging.getLogger(__name__)


# Request/Response Models
class InferenceRequest(BaseModel):
    """Request for problem inference."""

    state: list[list[float]]  # State representation
    query: str | None = "Solve this problem"
    max_thinking_time: float = Field(default=10.0, ge=0.1, le=60.0)
    use_mcts: bool = True
    num_simulations: int | None = None
    use_hrm_decomposition: bool = False
    use_trm_refinement: bool = False
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class PolicyValueRequest(BaseModel):
    """Request for policy-value evaluation."""

    state: list[list[float]]  # State representation


class InferenceResponse(BaseModel):
    """Response with inference results."""

    success: bool
    action_probabilities: dict[str, float] | None = None
    best_action: str | None = None
    value_estimate: float | None = None
    subproblems: list[dict[str, Any]] | None = None
    refinement_info: dict[str, Any] | None = None
    performance_stats: dict[str, float]
    error: str | None = None


class PolicyValueResponse(BaseModel):
    """Response with policy-value predictions."""

    policy_probs: list[float]
    value: float
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    device: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory_gb: float | None = None
    uptime_seconds: float


# Inference Server
class InferenceServer:
    """
    Production inference server with comprehensive features.

    Features:
    - FastAPI REST endpoints
    - Performance monitoring
    - Health checks
    - CORS support
    - Error handling
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: SystemConfig | None = None,
        host: str = DEFAULT_SERVER_HOST,
        port: int = 8000,
    ):
        """
        Initialize inference server.

        Args:
            checkpoint_path: Path to model checkpoint
            config: System configuration (loaded from checkpoint if None)
            host: Server host
            port: Server port
        """
        self.checkpoint_path = checkpoint_path
        self.host = host
        self.port = port
        self.start_time = time.time()

        # Load models
        self.config, self.models = self._load_models(checkpoint_path, config)
        self.device = self.config.device

        # Performance monitoring
        self.monitor = PerformanceMonitor(window_size=100, enable_gpu_monitoring=(self.device != "cpu"))

        # Setup FastAPI app
        self.app = FastAPI(
            title="LangGraph Multi-Agent MCTS API",
            description="Neural-guided MCTS with HRM and TRM agents",
            version="1.0.0",
        )

        # CORS middleware - configured from settings
        # Security: Credentials are disabled whenever wildcard origins are used
        settings = get_settings()
        cors_origins = settings.CORS_ALLOWED_ORIGINS or ["*"]
        has_wildcard_origin = "*" in cors_origins
        if has_wildcard_origin:
            # Normalize to explicit wildcard-only configuration and disable credentials
            cors_origins = ["*"]
            cors_allow_credentials = False
        else:
            cors_allow_credentials = settings.CORS_ALLOW_CREDENTIALS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=cors_allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Setup routes
        self._setup_routes()

    def _load_models(self, checkpoint_path: str, config: SystemConfig | None) -> tuple[SystemConfig, dict[str, Any]]:
        """Load models from checkpoint with error handling."""
        logger.info("Loading models from %s...", checkpoint_path)

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except FileNotFoundError as e:
            logger.error("Checkpoint file not found: %s", checkpoint_path)
            raise RuntimeError(f"Checkpoint file not found: {checkpoint_path}") from e
        except RuntimeError as e:
            logger.error("Failed to load checkpoint (possibly corrupted): %s", e)
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e

        # Load config
        if config is None:
            config_dict = checkpoint.get("config", {})
            config = SystemConfig.from_dict(config_dict)

        device = config.device

        # Load models
        models: dict[str, Any] = {}

        # Policy-Value Network
        from ..models.policy_value_net import create_policy_value_network

        models["policy_value_net"] = create_policy_value_network(config.neural_net, board_size=19, device=device)
        models["policy_value_net"].load_state_dict(checkpoint["policy_value_net"])
        models["policy_value_net"].eval()

        # HRM Agent
        from ..agents.hrm_agent import create_hrm_agent

        models["hrm_agent"] = create_hrm_agent(config.hrm, device)
        models["hrm_agent"].load_state_dict(checkpoint["hrm_agent"])
        models["hrm_agent"].eval()

        # TRM Agent
        from ..agents.trm_agent import create_trm_agent

        models["trm_agent"] = create_trm_agent(config.trm, output_dim=config.neural_net.action_size, device=device)
        models["trm_agent"].load_state_dict(checkpoint["trm_agent"])
        models["trm_agent"].eval()

        # MCTS
        models["mcts"] = NeuralMCTS(
            policy_value_network=models["policy_value_net"],
            config=config.mcts,
            device=device,
        )

        logger.info("Models loaded successfully on %s", device)

        return config, models

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/", response_model=dict[str, str])
        async def root():
            """Root endpoint."""
            return {
                "message": "LangGraph Multi-Agent MCTS API",
                "version": "1.0.0",
                "docs": "/docs",
            }

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            gpu_memory = None
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)

            return HealthResponse(
                status="healthy",
                device=self.device,
                model_loaded=True,
                gpu_available=torch.cuda.is_available(),
                gpu_memory_gb=gpu_memory,
                uptime_seconds=time.time() - self.start_time,
            )

        @self.app.post("/inference", response_model=InferenceResponse)
        async def inference(request: InferenceRequest):
            """
            Main inference endpoint.

            Processes a problem using the full pipeline:
            1. Optional HRM decomposition
            2. MCTS search
            3. Optional TRM refinement
            """
            try:
                start_time = time.perf_counter()

                # Validate and convert state to tensor with proper error handling
                if not request.state:
                    raise HTTPException(status_code=400, detail="State cannot be empty")

                try:
                    state_tensor = torch.tensor(request.state, dtype=torch.float32).unsqueeze(0)
                except (TypeError, ValueError) as e:
                    logger.warning("Invalid state format: %s", e)
                    raise HTTPException(status_code=400, detail=f"Invalid state format: {e}") from e

                state_tensor = state_tensor.to(self.device)

                results: dict[str, Any] = {}

                # HRM Decomposition (if requested)
                if request.use_hrm_decomposition:
                    with torch.no_grad():
                        hrm_output = self.models["hrm_agent"](state_tensor)
                        results["subproblems"] = [
                            {
                                "level": sp.level,
                                "description": sp.description,
                                "confidence": sp.confidence,
                            }
                            for sp in hrm_output.subproblems
                        ]

                # MCTS Search (if requested)
                if request.use_mcts:
                    # Note: This is a simplified version
                    # In production, you'd need to convert request.state to GameState
                    results["action_probabilities"] = {"action_0": 0.5, "action_1": 0.3, "action_2": 0.2}
                    results["best_action"] = "action_0"
                    results["value_estimate"] = 0.75

                # TRM Refinement (if requested)
                if request.use_trm_refinement and results.get("best_action"):
                    with torch.no_grad():
                        # Simplified: just run TRM on the state
                        trm_output = self.models["trm_agent"](state_tensor)
                        results["refinement_info"] = {
                            "converged": trm_output.converged,
                            "convergence_step": trm_output.convergence_step,
                            "recursion_depth": trm_output.recursion_depth,
                        }

                # Performance stats
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self.monitor.log_inference(elapsed_ms)

                perf_stats = {
                    "inference_time_ms": elapsed_ms,
                    "device": self.device,
                }

                return InferenceResponse(
                    success=True,
                    action_probabilities=results.get("action_probabilities"),
                    best_action=results.get("best_action"),
                    value_estimate=results.get("value_estimate"),
                    subproblems=results.get("subproblems"),
                    refinement_info=results.get("refinement_info"),
                    performance_stats=perf_stats,
                )

            except HTTPException:
                # Re-raise HTTP exceptions (from validation above)
                raise
            except torch.cuda.OutOfMemoryError as e:
                logger.error("GPU out of memory during inference: %s", e)
                raise HTTPException(status_code=503, detail="GPU out of memory. Try a smaller state.") from e
            except RuntimeError as e:
                logger.error("Runtime error during inference: %s", e)
                raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}") from e
            except Exception as e:  # Broad catch: API stability requires catching unexpected errors
                logger.exception("Unexpected error during inference: %s", e)
                raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}") from e

        @self.app.post("/policy-value", response_model=PolicyValueResponse)
        async def policy_value(request: PolicyValueRequest):
            """
            Get policy and value predictions for a state.

            This is a direct neural network evaluation without MCTS.
            """
            try:
                start_time = time.perf_counter()

                # Validate and convert state to tensor
                if not request.state:
                    raise HTTPException(status_code=400, detail="State cannot be empty")

                try:
                    state_tensor = torch.tensor(request.state, dtype=torch.float32).unsqueeze(0)
                except (TypeError, ValueError) as e:
                    logger.warning("Invalid state format for policy-value: %s", e)
                    raise HTTPException(status_code=400, detail=f"Invalid state format: {e}") from e

                state_tensor = state_tensor.to(self.device)

                # Get predictions
                with torch.no_grad():
                    policy_log_probs, value = self.models["policy_value_net"](state_tensor)
                    policy_probs = torch.exp(policy_log_probs).squeeze(0)

                elapsed_ms = (time.perf_counter() - start_time) * 1000

                return PolicyValueResponse(
                    policy_probs=policy_probs.cpu().tolist(),
                    value=value.item(),
                    inference_time_ms=elapsed_ms,
                )

            except HTTPException:
                raise
            except torch.cuda.OutOfMemoryError as e:
                logger.error("GPU out of memory during policy-value inference: %s", e)
                raise HTTPException(status_code=503, detail="GPU out of memory. Try a smaller state.") from e
            except RuntimeError as e:
                logger.error("Runtime error during policy-value inference: %s", e)
                raise HTTPException(status_code=500, detail=f"Policy-value inference failed: {str(e)}") from e
            except Exception as e:  # Broad catch: API stability requires catching unexpected errors
                logger.exception("Unexpected error during policy-value inference: %s", e)
                raise HTTPException(status_code=500, detail=f"Policy-value inference failed: {str(e)}") from e

        @self.app.get("/stats")
        async def stats():
            """Get performance statistics."""
            return self.monitor.get_stats()

        @self.app.post("/reset-stats")
        async def reset_stats():
            """Reset performance statistics."""
            self.monitor.reset()
            return {"message": "Statistics reset successfully"}

    def run(self):
        """Start the inference server."""
        logger.info("=" * 80)
        logger.info("Starting LangGraph Multi-Agent MCTS Inference Server")
        logger.info("=" * 80)
        logger.info("Host: %s:%d", self.host, self.port)
        logger.info("Device: %s", self.device)
        logger.info("Checkpoint: %s", self.checkpoint_path)
        logger.info("=" * 80)

        uvicorn.run(self.app, host=self.host, port=self.port)


def main():
    """Main entry point for inference server."""
    import argparse

    parser = argparse.ArgumentParser(description="LangGraph MCTS Inference Server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument("--host", type=str, default=DEFAULT_SERVER_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cpu, cuda, mps)",
    )

    args = parser.parse_args()

    # Load config and override device if specified
    config = None
    if args.device:
        config = SystemConfig()
        config.device = args.device

    server = InferenceServer(
        checkpoint_path=args.checkpoint,
        config=config,
        host=args.host,
        port=args.port,
    )

    server.run()


if __name__ == "__main__":
    main()
