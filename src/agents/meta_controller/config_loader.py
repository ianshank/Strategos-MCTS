"""
Configuration loader for the Neural Meta-Controller framework.

This module provides dataclass-based configuration management for the Meta-Controller,
supporting both RNN and BERT-based neural network controllers with comprehensive
validation and serialization capabilities.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RNNConfig:
    """
    Configuration for RNN-based Meta-Controller.

    Attributes:
        hidden_dim: Hidden dimension size for RNN layers. Default is 64.
        num_layers: Number of RNN layers. Default is 1.
        dropout: Dropout rate for regularization. Default is 0.1.
        model_path: Optional path to a pre-trained model file. None for untrained model.
    """

    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    model_path: str | None = None


@dataclass
class BERTConfig:
    """
    Configuration for BERT-based Meta-Controller with LoRA fine-tuning.

    Attributes:
        model_name: Name of the pre-trained BERT model from HuggingFace.
                   Default is "prajjwal1/bert-mini" for lightweight deployment.
        use_lora: Whether to use LoRA (Low-Rank Adaptation) for efficient fine-tuning.
                  Default is True.
        lora_r: LoRA rank parameter. Controls the rank of the low-rank matrices.
                Default is 4.
        lora_alpha: LoRA alpha parameter. Scaling factor for LoRA weights.
                    Default is 16.
        lora_dropout: Dropout rate for LoRA layers. Default is 0.1.
        model_path: Optional path to a trained LoRA adapter. None for base model only.
    """

    model_name: str = "prajjwal1/bert-mini"
    use_lora: bool = True
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    model_path: str | None = None


@dataclass
class InferenceConfig:
    """
    Configuration for inference settings.

    Attributes:
        device: Device to use for inference ("cpu", "cuda", "cuda:0", etc.).
                None for auto-detection based on available hardware.
        seed: Random seed for reproducibility. Default is 42.
    """

    device: str | None = None
    seed: int = 42


@dataclass
class MetaControllerConfig:
    """
    Main configuration for the Neural Meta-Controller framework.

    This configuration controls the behavior of the Meta-Controller, including
    which type of neural network to use (RNN or BERT), fallback behavior,
    and specific model parameters.

    Attributes:
        enabled: Whether the neural Meta-Controller is enabled. Default is False
                 for backward compatibility with rule-based systems.
        type: Type of neural network controller ("rnn" or "bert"). Default is "rnn".
        fallback_to_rule_based: Whether to fall back to rule-based selection on errors.
                                Default is True for robustness.
        rnn: Configuration for RNN-based controller.
        bert: Configuration for BERT-based controller.
        inference: Configuration for inference settings.
    """

    enabled: bool = False
    type: str = "rnn"  # "rnn" or "bert"
    fallback_to_rule_based: bool = True
    rnn: RNNConfig = field(default_factory=RNNConfig)
    bert: BERTConfig = field(default_factory=BERTConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


class MetaControllerConfigLoader:
    """
    Loader class for Meta-Controller configuration.

    Provides methods for loading configuration from YAML files or dictionaries,
    converting configuration to dictionaries, and validating configuration values.

    Example:
        >>> loader = MetaControllerConfigLoader()
        >>> config = loader.load_from_yaml("config/meta_controller.yaml")
        >>> print(config.type)
        'rnn'
        >>> config.validate()
    """

    @staticmethod
    def load_from_yaml(path: str) -> MetaControllerConfig:
        """
        Load Meta-Controller configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            MetaControllerConfig: Loaded and parsed configuration object.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
            KeyError: If the 'meta_controller' key is missing from the file.

        Example:
            >>> config = MetaControllerConfigLoader.load_from_yaml("config/meta_controller.yaml")
            >>> print(config.enabled)
            False
        """
        yaml_path = Path(path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(yaml_path) as f:
            raw_config = yaml.safe_load(f)

        if "meta_controller" not in raw_config:
            raise KeyError("Configuration file must contain 'meta_controller' key")

        return MetaControllerConfigLoader.load_from_dict(raw_config["meta_controller"])

    @staticmethod
    def load_from_dict(config_dict: dict[str, Any]) -> MetaControllerConfig:
        """
        Load Meta-Controller configuration from a dictionary.

        Args:
            config_dict: Dictionary containing configuration values.

        Returns:
            MetaControllerConfig: Parsed configuration object with defaults
                                  applied for missing values.

        Example:
            >>> config_dict = {
            ...     'enabled': True,
            ...     'type': 'bert',
            ...     'bert': {'model_name': 'bert-base-uncased'}
            ... }
            >>> config = MetaControllerConfigLoader.load_from_dict(config_dict)
            >>> print(config.type)
            'bert'
        """
        # Parse nested configurations
        rnn_config = RNNConfig(**config_dict.get("rnn", {}))
        bert_config = BERTConfig(**config_dict.get("bert", {}))
        inference_config = InferenceConfig(**config_dict.get("inference", {}))

        # Create main config with nested configs
        return MetaControllerConfig(
            enabled=config_dict.get("enabled", False),
            type=config_dict.get("type", "rnn"),
            fallback_to_rule_based=config_dict.get("fallback_to_rule_based", True),
            rnn=rnn_config,
            bert=bert_config,
            inference=inference_config,
        )

    @staticmethod
    def to_dict(config: MetaControllerConfig) -> dict[str, Any]:
        """
        Convert a MetaControllerConfig object to a dictionary.

        Args:
            config: MetaControllerConfig object to convert.

        Returns:
            Dict[str, Any]: Dictionary representation of the configuration.

        Example:
            >>> config = MetaControllerConfig(enabled=True, type='bert')
            >>> config_dict = MetaControllerConfigLoader.to_dict(config)
            >>> print(config_dict['enabled'])
            True
        """
        return asdict(config)

    @staticmethod
    def validate(config: MetaControllerConfig) -> None:
        """
        Validate the Meta-Controller configuration.

        Checks that:
        - The controller type is valid ("rnn" or "bert")
        - Model paths exist if specified
        - Numeric parameters are within valid ranges

        Args:
            config: MetaControllerConfig object to validate.

        Raises:
            ValueError: If the configuration contains invalid values.
            FileNotFoundError: If specified model paths do not exist.

        Example:
            >>> config = MetaControllerConfig(type='invalid')
            >>> MetaControllerConfigLoader.validate(config)
            ValueError: Invalid controller type 'invalid'. Must be 'rnn' or 'bert'.
        """
        # Validate controller type
        valid_types = ["rnn", "bert"]
        if config.type not in valid_types:
            raise ValueError(f"Invalid controller type '{config.type}'. Must be one of: {valid_types}")

        # Validate RNN config
        if config.rnn.hidden_dim <= 0:
            raise ValueError(f"RNN hidden_dim must be positive, got {config.rnn.hidden_dim}")
        if config.rnn.num_layers <= 0:
            raise ValueError(f"RNN num_layers must be positive, got {config.rnn.num_layers}")
        if not 0.0 <= config.rnn.dropout <= 1.0:
            raise ValueError(f"RNN dropout must be between 0 and 1, got {config.rnn.dropout}")
        if config.rnn.model_path is not None:
            rnn_path = Path(config.rnn.model_path)
            if not rnn_path.exists():
                raise FileNotFoundError(f"RNN model path does not exist: {config.rnn.model_path}")

        # Validate BERT config
        if config.bert.lora_r <= 0:
            raise ValueError(f"BERT lora_r must be positive, got {config.bert.lora_r}")
        if config.bert.lora_alpha <= 0:
            raise ValueError(f"BERT lora_alpha must be positive, got {config.bert.lora_alpha}")
        if not 0.0 <= config.bert.lora_dropout <= 1.0:
            raise ValueError(f"BERT lora_dropout must be between 0 and 1, got {config.bert.lora_dropout}")
        if config.bert.model_path is not None:
            bert_path = Path(config.bert.model_path)
            if not bert_path.exists():
                raise FileNotFoundError(f"BERT model path does not exist: {config.bert.model_path}")

        # Validate inference config
        if config.inference.device is not None:
            valid_devices = ["cpu", "cuda", "mps"]
            # Check if device starts with a valid prefix (e.g., "cuda:0", "cuda:1")
            device_base = config.inference.device.split(":")[0]
            if device_base not in valid_devices:
                raise ValueError(f"Invalid device '{config.inference.device}'. Must start with one of: {valid_devices}")

        if not isinstance(config.inference.seed, int) or config.inference.seed < 0:
            raise ValueError(f"Inference seed must be a non-negative integer, got {config.inference.seed}")

    @staticmethod
    def save_to_yaml(config: MetaControllerConfig, path: str) -> None:
        """
        Save a MetaControllerConfig object to a YAML file.

        Args:
            config: MetaControllerConfig object to save.
            path: Path where the YAML file will be saved.

        Example:
            >>> config = MetaControllerConfig(enabled=True)
            >>> MetaControllerConfigLoader.save_to_yaml(config, "my_config.yaml")
        """
        yaml_path = Path(path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {"meta_controller": MetaControllerConfigLoader.to_dict(config)}

        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def get_default_config() -> MetaControllerConfig:
        """
        Get a default MetaControllerConfig with all default values.

        Returns:
            MetaControllerConfig: Configuration object with default values.

        Example:
            >>> config = MetaControllerConfigLoader.get_default_config()
            >>> print(config.enabled)
            False
        """
        return MetaControllerConfig()
