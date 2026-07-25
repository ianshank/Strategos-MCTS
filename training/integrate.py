"""
Integration Module

Utilities for integrating trained models with the existing multi-agent MCTS system.
Handles model loading, hot-swapping, configuration management, and rollback.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    import torch

    from training.utils.checkpoint_loader import load_checkpoint_safe

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class ModelIntegrator:
    """Integrate trained models into production system."""

    def __init__(self, config_path: str = "training/config.yaml"):
        """
        Initialize model integrator.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.models_dir = Path("training/models")
        self.production_dir = Path("models")
        self.backup_dir = Path("training/models/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.loaded_models = {}
        self.model_versions = {}

        logger.info("ModelIntegrator initialized")

    def load_trained_model(self, model_type: str, checkpoint_path: str) -> Any:
        """
        Load a trained model from checkpoint.

        Args:
            model_type: Type of model (hrm, trm, mcts, router)
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded model
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available, loading model metadata only")
            return {"type": model_type, "path": checkpoint_path}

        try:
            checkpoint = load_checkpoint_safe(checkpoint_path, map_location="cpu")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            # One last try with weights_only=False for compatibility
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except Exception as final_e:
                logger.error(f"Final attempt to load checkpoint {checkpoint_path} failed: {final_e}")
                return None

        # Extract model state
        model_state = checkpoint.get("model_state_dict", checkpoint)

        # Store model info
        self.loaded_models[model_type] = {
            "state_dict": model_state,
            "config": checkpoint.get("config", {}),
            "epoch": checkpoint.get("current_epoch", 0),
            "metrics": checkpoint.get("training_history", []),
            "loaded_at": datetime.now().isoformat(),
        }

        logger.info(f"Loaded {model_type} model from {checkpoint_path}")
        return self.loaded_models[model_type]

    def inject_into_agent(self, agent_class: type, model_type: str, **kwargs) -> Any:
        """
        Inject trained model into agent class.

        Args:
            agent_class: Agent class to instantiate
            model_type: Type of trained model
            **kwargs: Additional arguments for agent

        Returns:
            Agent instance with trained model
        """
        if model_type not in self.loaded_models:
            raise ValueError(f"Model {model_type} not loaded")

        model_info = self.loaded_models[model_type]

        # Create agent instance
        agent = agent_class(**kwargs)

        # Inject model weights
        if hasattr(agent, "model") and HAS_TORCH:
            try:
                agent.model.load_state_dict(model_info["state_dict"])
                logger.info(f"Injected {model_type} weights into {agent_class.__name__}")
            except Exception as e:
                logger.error(f"Failed to inject model weights: {e}")

        # Store version info
        self.model_versions[agent] = {
            "model_type": model_type,
            "epoch": model_info["epoch"],
            "loaded_at": model_info["loaded_at"],
        }

        return agent

    def update_mcts_components(self, mcts_engine: Any, value_checkpoint: str, policy_checkpoint: str) -> None:
        """
        Update MCTS with trained neural components.

        Args:
            mcts_engine: MCTS engine instance
            value_checkpoint: Path to value network checkpoint
            policy_checkpoint: Path to policy network checkpoint
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available, skipping MCTS update")
            return

        # Load value network
        value_state = torch.load(value_checkpoint, map_location="cpu", weights_only=True)
        if hasattr(mcts_engine, "value_network"):
            mcts_engine.value_network.load_state_dict(value_state["model_state_dict"])

        # Load policy network
        policy_state = torch.load(policy_checkpoint, map_location="cpu", weights_only=True)
        if hasattr(mcts_engine, "policy_network"):
            mcts_engine.policy_network.load_state_dict(policy_state["model_state_dict"])

        logger.info("Updated MCTS neural components")

    def replace_rag_index(self, current_rag: Any, new_index_path: str) -> None:
        """
        Replace RAG index with trained version.

        Args:
            current_rag: Current RAG system
            new_index_path: Path to new trained index
        """
        # Backup current index
        if hasattr(current_rag, "index_path"):
            backup_path = self.backup_dir / f"rag_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if Path(current_rag.index_path).exists():
                shutil.copytree(current_rag.index_path, backup_path)
                logger.info(f"Backed up current RAG index to {backup_path}")

        # Load new index
        if hasattr(current_rag, "load_index"):
            current_rag.load_index(new_index_path)
            logger.info(f"Replaced RAG index with {new_index_path}")

    def wire_meta_controller(
        self, graph_config: dict[str, Any], router_checkpoint: str, aggregator_checkpoint: str
    ) -> dict[str, Any]:
        """
        Wire trained meta-controller into graph configuration.

        Args:
            graph_config: LangGraph configuration
            router_checkpoint: Path to router checkpoint
            aggregator_checkpoint: Path to aggregator checkpoint

        Returns:
            Updated graph configuration
        """
        # Load meta-controller models
        self.load_trained_model("router", router_checkpoint)

        # Update graph config
        updated_config = graph_config.copy()
        updated_config["meta_controller"] = {
            "router_path": router_checkpoint,
            "aggregator_path": aggregator_checkpoint,
            "enabled": True,
        }

        logger.info("Wired meta-controller into graph configuration")
        return updated_config

    def export_production_models(self, export_dir: str) -> dict[str, str]:
        """
        Export all trained models for production deployment.

        Args:
            export_dir: Directory to export models

        Returns:
            Dictionary of exported model paths
        """
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        exported = {}

        for model_type, model_info in self.loaded_models.items():
            # Export model
            model_file = export_path / f"{model_type}_production.pt"

            if HAS_TORCH:
                torch.save(
                    {
                        "model_state_dict": model_info["state_dict"],
                        "config": model_info["config"],
                        "version": self._generate_version(),
                        "exported_at": datetime.now().isoformat(),
                    },
                    model_file,
                )
            else:
                # Save metadata only
                with open(export_path / f"{model_type}_metadata.json", "w") as f:
                    json.dump(model_info, f, indent=2, default=str)

            exported[model_type] = str(model_file)
            logger.info(f"Exported {model_type} to {model_file}")

        return exported

    def _generate_version(self) -> str:
        """Generate version string."""
        return datetime.now().strftime("v%Y.%m.%d-%H%M%S")


class ConfigurationManager:
    """Manage training and production configurations."""

    def __init__(self, config_path: str = "training/config.yaml"):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to main configuration file
        """
        self.config_path = Path(config_path)
        self.configs_dir = Path("training/configs")
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path) as f:
            self.current_config = yaml.safe_load(f)

        self.config_history = []

        logger.info("ConfigurationManager initialized")

    def update_config(self, updates: dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates
        """
        # Save current config to history
        self.config_history.append({"config": self.current_config.copy(), "timestamp": datetime.now().isoformat()})

        # Apply updates
        self._deep_update(self.current_config, updates)

        # Save updated config
        self._save_config()

        logger.info(f"Configuration updated: {list(updates.keys())}")

    def _deep_update(self, base: dict, updates: dict) -> dict:
        """Deep update nested dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
        return base

    def _save_config(self) -> None:
        """Save current configuration."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.current_config, f, default_flow_style=False, sort_keys=False)

        # Also save timestamped version
        timestamped = self.configs_dir / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(timestamped, "w") as f:
            yaml.dump(self.current_config, f, default_flow_style=False, sort_keys=False)

    def rollback_config(self, steps: int = 1) -> None:
        """
        Rollback configuration to previous version.

        Args:
            steps: Number of versions to rollback
        """
        if steps > len(self.config_history):
            raise ValueError(f"Cannot rollback {steps} steps, only {len(self.config_history)} versions in history")

        # Get previous config
        previous = self.config_history[-steps]["config"]

        # Update current
        self.current_config = previous
        self._save_config()

        # Remove from history
        for _ in range(steps):
            self.config_history.pop()

        logger.info(f"Configuration rolled back {steps} steps")

    def validate_config(self) -> tuple[bool, list[str]]:
        """
        Validate current configuration.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields
        required_sections = ["data", "training", "agents", "rag", "meta_controller", "evaluation"]
        for section in required_sections:
            if section not in self.current_config:
                errors.append(f"Missing required section: {section}")

        # Validate training parameters
        training = self.current_config.get("training", {})
        if training.get("batch_size", 0) <= 0:
            errors.append("Invalid batch_size: must be positive")
        if training.get("learning_rate", 0) <= 0:
            errors.append("Invalid learning_rate: must be positive")
        if training.get("epochs", 0) <= 0:
            errors.append("Invalid epochs: must be positive")

        # Validate agent configs
        agents = self.current_config.get("agents", {})
        for agent_name in ["hrm", "trm", "mcts"]:
            if agent_name not in agents:
                errors.append(f"Missing agent configuration: {agent_name}")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("Configuration validation passed")
        else:
            logger.warning(f"Configuration validation failed: {errors}")

        return is_valid, errors

    def create_production_config(self) -> dict[str, Any]:
        """
        Create production-optimized configuration.

        Returns:
            Production configuration
        """
        prod_config = self.current_config.copy()

        # Optimize for production
        prod_config["training"]["fp16"] = True
        prod_config["training"]["gradient_accumulation_steps"] = 1
        prod_config["monitoring"]["profiling"]["enabled"] = False
        prod_config["continual_learning"]["ab_testing"]["enabled"] = True

        # Save production config
        prod_path = self.configs_dir / "production_config.yaml"
        with open(prod_path, "w") as f:
            yaml.dump(prod_config, f, default_flow_style=False)

        logger.info(f"Created production config at {prod_path}")
        return prod_config


class HotSwapper:
    """Hot-swap models in production without downtime."""

    def __init__(self):
        """Initialize hot swapper."""
        self.active_models = {}
        self.standby_models = {}
        self.swap_history = []

        logger.info("HotSwapper initialized")

    def prepare_swap(self, model_name: str, new_model: Any, validation_fn=None) -> bool:
        """
        Prepare a model for hot-swapping.

        Args:
            model_name: Name of model to swap
            new_model: New model to swap in
            validation_fn: Optional validation function

        Returns:
            True if preparation successful
        """
        # Validate new model
        if validation_fn:
            try:
                is_valid = validation_fn(new_model)
                if not is_valid:
                    logger.error(f"New model {model_name} failed validation")
                    return False
            except Exception as e:
                logger.error(f"Validation error: {e}")
                return False

        # Store in standby
        self.standby_models[model_name] = {"model": new_model, "prepared_at": datetime.now().isoformat()}

        logger.info(f"Model {model_name} prepared for hot-swap")
        return True

    def execute_swap(self, model_name: str) -> bool:
        """
        Execute the hot-swap.

        Args:
            model_name: Name of model to swap

        Returns:
            True if swap successful
        """
        if model_name not in self.standby_models:
            logger.error(f"No standby model prepared for {model_name}")
            return False

        # Backup current model
        if model_name in self.active_models:
            old_model = self.active_models[model_name]
            self.swap_history.append(
                {"model_name": model_name, "old_model": old_model, "timestamp": datetime.now().isoformat()}
            )

        # Perform swap
        self.active_models[model_name] = self.standby_models[model_name]["model"]
        del self.standby_models[model_name]

        logger.info(f"Hot-swap executed for {model_name}")
        return True

    def rollback_swap(self, model_name: str) -> bool:
        """
        Rollback to previous model version.

        Args:
            model_name: Name of model to rollback

        Returns:
            True if rollback successful
        """
        # Find last swap for this model
        for i in range(len(self.swap_history) - 1, -1, -1):
            if self.swap_history[i]["model_name"] == model_name:
                old_model = self.swap_history[i]["old_model"]
                self.active_models[model_name] = old_model
                self.swap_history.pop(i)

                logger.info(f"Rolled back {model_name} to previous version")
                return True

        logger.warning(f"No previous version found for {model_name}")
        return False

    def get_swap_status(self) -> dict[str, Any]:
        """Get current swap status."""
        return {
            "active_models": list(self.active_models.keys()),
            "standby_models": list(self.standby_models.keys()),
            "swap_history_count": len(self.swap_history),
            "last_swap": self.swap_history[-1] if self.swap_history else None,
        }

    def create_ab_split(self, model_name: str, model_a: Any, model_b: Any, split_ratio: float = 0.5) -> callable:
        """
        Create A/B split function for gradual rollout.

        Args:
            model_name: Name of model
            model_a: Model A (control)
            model_b: Model B (treatment)
            split_ratio: Ratio of traffic to model B

        Returns:
            Function that selects model based on request
        """
        self.active_models[f"{model_name}_a"] = model_a
        self.active_models[f"{model_name}_b"] = model_b

        def select_model(request_id: str) -> Any:
            # Hash-based consistent selection
            hash_val = hash(request_id) % 100
            if hash_val < split_ratio * 100:
                return model_b
            return model_a

        logger.info(f"Created A/B split for {model_name} with {split_ratio:.0%} to B")
        return select_model


if __name__ == "__main__":
    # Test integration module
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing Integration Module")

    # Test ModelIntegrator
    integrator = ModelIntegrator()

    # Create mock checkpoint
    mock_checkpoint_path = Path("training/models/checkpoints/test_checkpoint.pt")
    mock_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if HAS_TORCH:
        mock_checkpoint = {
            "model_state_dict": {},
            "config": {"type": "test"},
            "current_epoch": 10,
            "training_history": [],
        }
        torch.save(mock_checkpoint, mock_checkpoint_path)

        # Load model
        model_info = integrator.load_trained_model("hrm", str(mock_checkpoint_path))
        logger.info(f"Loaded model info: {model_info}")

        # Export models
        exported = integrator.export_production_models("training/models/production")
        logger.info(f"Exported models: {exported}")

    # Test ConfigurationManager
    config_manager = ConfigurationManager()

    # Validate config
    is_valid, errors = config_manager.validate_config()
    logger.info(f"Config valid: {is_valid}, errors: {errors}")

    # Update config
    config_manager.update_config({"training": {"batch_size": 64}})
    logger.info(f"Updated batch_size to {config_manager.current_config['training']['batch_size']}")

    # Rollback
    config_manager.rollback_config(steps=1)
    logger.info(f"Rolled back batch_size to {config_manager.current_config['training']['batch_size']}")

    # Create production config
    prod_config = config_manager.create_production_config()
    logger.info(f"Production config created with fp16={prod_config['training']['fp16']}")

    # Test HotSwapper
    hot_swapper = HotSwapper()

    # Prepare swap
    mock_model_a = {"version": "1.0", "weights": "old"}
    mock_model_b = {"version": "2.0", "weights": "new"}

    hot_swapper.active_models["test_model"] = mock_model_a

    # Prepare and execute swap
    success = hot_swapper.prepare_swap("test_model", mock_model_b)
    logger.info(f"Swap preparation: {success}")

    success = hot_swapper.execute_swap("test_model")
    logger.info(f"Swap execution: {success}")

    status = hot_swapper.get_swap_status()
    logger.info(f"Swap status: {status}")

    # Rollback
    success = hot_swapper.rollback_swap("test_model")
    logger.info(f"Rollback: {success}")

    # A/B split
    selector = hot_swapper.create_ab_split("experiment", mock_model_a, mock_model_b, 0.3)
    selections = {"a": 0, "b": 0}
    for i in range(100):
        model = selector(f"request_{i}")
        if model == mock_model_a:
            selections["a"] += 1
        else:
            selections["b"] += 1
    logger.info(f"A/B split results: {selections}")

    logger.info("Integration Module test complete")
