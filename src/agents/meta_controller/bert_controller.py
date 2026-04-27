"""
BERT-based Meta-Controller with LoRA adapters for efficient fine-tuning.

This module provides a BERT-based meta-controller that uses Low-Rank Adaptation (LoRA)
for parameter-efficient fine-tuning. The controller converts agent state features into
text and uses a sequence classification model to predict the optimal agent.
"""

import logging
import os
import warnings
from typing import Any

import torch

from src.agents.meta_controller.base import (
    AbstractMetaController,
    MetaControllerFeatures,
    MetaControllerPrediction,
)
from src.agents.meta_controller.utils import features_to_text

logger = logging.getLogger(__name__)

# Handle optional transformers and peft imports gracefully
_TRANSFORMERS_AVAILABLE = False
_PEFT_AVAILABLE = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    warnings.warn(
        "transformers library not installed. Install it with: pip install transformers",
        ImportWarning,
        stacklevel=2,
    )
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

try:
    from peft import LoraConfig, TaskType, get_peft_model

    _PEFT_AVAILABLE = True
except ImportError:
    # Fallback if peft is missing or broken (e.g. version mismatch with transformers)
    _PEFT_AVAILABLE = False
    LoraConfig = None
    TaskType = None
    get_peft_model = None


class BERTMetaController(AbstractMetaController):
    """
    BERT-based meta-controller with optional LoRA adapters for efficient fine-tuning.

    This controller converts agent state features into structured text and uses
    a pre-trained BERT model (with optional LoRA adapters) to classify which
    agent should handle the current query. LoRA enables parameter-efficient
    fine-tuning by only training low-rank decomposition matrices.

    Attributes:
        DEFAULT_MODEL_NAME: Default BERT model to use.
        NUM_LABELS: Number of output labels (agents to choose from).
        device: PyTorch device for tensor operations.
        model_name: Name of the pre-trained model.
        lora_r: LoRA rank parameter.
        lora_alpha: LoRA alpha scaling parameter.
        lora_dropout: LoRA dropout rate.
        use_lora: Whether to use LoRA adapters.
        tokenizer: BERT tokenizer for text processing.
        model: BERT sequence classification model (with or without LoRA).

    Example:
        >>> controller = BERTMetaController(name="BERTController", seed=42)
        >>> features = MetaControllerFeatures(
        ...     hrm_confidence=0.8,
        ...     trm_confidence=0.6,
        ...     mcts_value=0.75,
        ...     consensus_score=0.7,
        ...     last_agent='hrm',
        ...     iteration=2,
        ...     query_length=150,
        ...     has_rag_context=True
        ... )
        >>> prediction = controller.predict(features)
        >>> prediction.agent in ['hrm', 'trm', 'mcts']
        True
        >>> 0.0 <= prediction.confidence <= 1.0
        True
    """

    DEFAULT_MODEL_NAME = "prajjwal1/bert-mini"
    NUM_LABELS = 3

    def __init__(
        self,
        name: str = "BERTMetaController",
        seed: int = 42,
        model_name: str | None = None,
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        device: str | None = None,
        use_lora: bool = True,
    ) -> None:
        """
        Initialize the BERT meta-controller with optional LoRA adapters.

        Args:
            name: Name identifier for this controller. Defaults to "BERTMetaController".
            seed: Random seed for reproducibility. Defaults to 42.
            model_name: Pre-trained model name from HuggingFace. If None, uses DEFAULT_MODEL_NAME.
            lora_r: LoRA rank parameter (lower = more compression). Defaults to 4.
            lora_alpha: LoRA alpha scaling parameter. Defaults to 16.
            lora_dropout: Dropout rate for LoRA layers. Defaults to 0.1.
            device: Device to run model on ('cpu', 'cuda', 'mps', etc.).
                   If None, auto-detects best available device.
            use_lora: Whether to apply LoRA adapters to the model. Defaults to True.

        Raises:
            ImportError: If transformers library is not installed.
            ImportError: If use_lora is True and peft library is not installed.

        Example:
            >>> controller = BERTMetaController(
            ...     name="CustomBERT",
            ...     seed=123,
            ...     lora_r=8,
            ...     lora_alpha=32,
            ...     use_lora=True
            ... )
        """
        super().__init__(name=name, seed=seed)

        # Check for required dependencies
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for BERTMetaController. Install it with: pip install transformers"
            )

        if use_lora and not _PEFT_AVAILABLE:
            raise ImportError("peft library is required for LoRA support. Install it with: pip install peft")

        # Set random seed for reproducibility
        torch.manual_seed(seed)

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Store configuration parameters
        # Resolution order: explicit arg > BERT_MODEL_NAME env var > class default.
        self.model_name = model_name or os.environ.get("BERT_MODEL_NAME", self.DEFAULT_MODEL_NAME)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_lora = use_lora

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Initialize base model for sequence classification
        base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.NUM_LABELS)

        # Apply LoRA adapters if requested
        if self.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["query", "value"],
            )
            self.model = get_peft_model(base_model, lora_config)
        else:
            self.model = base_model

        # Move model to device
        self.model = self.model.to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Initialize tokenization cache for performance optimization
        self._tokenization_cache: dict[str, Any] = {}

        logger.info(
            "Initialized BERTMetaController",
            extra={
                "model_name": self.model_name,
                "use_lora": self.use_lora,
                "device": str(self.device),
                "lora_r": self.lora_r if self.use_lora else None,
            },
        )

    def predict(self, features: MetaControllerFeatures) -> MetaControllerPrediction:
        """
        Predict which agent should handle the current query.

        Converts features to structured text, tokenizes the text, runs through
        the BERT model, and returns a prediction with confidence scores.

        Args:
            features: Features extracted from the current agent state.

        Returns:
            Prediction containing the selected agent, confidence score,
            and probability distribution over all agents.

        Example:
            >>> controller = BERTMetaController()
            >>> features = MetaControllerFeatures(
            ...     hrm_confidence=0.9,
            ...     trm_confidence=0.3,
            ...     mcts_value=0.5,
            ...     consensus_score=0.8,
            ...     last_agent='none',
            ...     iteration=0,
            ...     query_length=100,
            ...     has_rag_context=False
            ... )
            >>> pred = controller.predict(features)
            >>> isinstance(pred.agent, str)
            True
            >>> isinstance(pred.confidence, float)
            True
            >>> len(pred.probabilities) == 3
            True
        """
        # Convert features to structured text
        text = features_to_text(features)

        # Check cache for tokenized text
        if text in self._tokenization_cache:
            inputs = self._tokenization_cache[text]
        else:
            # Tokenize the text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            # Cache the tokenized result
            self._tokenization_cache[text] = inputs

        # Move inputs to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Perform inference without gradient tracking
        with torch.no_grad():
            # Get logits from model
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Get predicted agent index (argmax)
            predicted_idx: int = int(torch.argmax(probabilities, dim=-1).item())

            # Extract confidence for selected agent
            confidence = probabilities[0, predicted_idx].item()

            # Create probability dictionary
            prob_dict: dict[str, float] = {}
            for i, agent_name in enumerate(self.AGENT_NAMES):
                prob_dict[agent_name] = probabilities[0, i].item()

        # Get agent name
        selected_agent = self.AGENT_NAMES[predicted_idx]

        logger.debug(
            "BERT meta-controller routing decision",
            extra={"selected_agent": selected_agent, "confidence": float(confidence), "probabilities": prob_dict},
        )

        return MetaControllerPrediction(
            agent=selected_agent,
            confidence=float(confidence),
            probabilities=prob_dict,
        )

    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        For LoRA models, loads the PEFT adapter weights. For base models,
        loads the full state dictionary.

        Args:
            path: Path to the saved model file or directory.
                 For LoRA models, this should be a directory containing
                 adapter_config.json and adapter_model.bin.
                 For base models, this should be a .pt or .pth file.

        Raises:
            FileNotFoundError: If the model file or directory does not exist.
            RuntimeError: If the state dict is incompatible with the model.

        Example:
            >>> controller = BERTMetaController(use_lora=True)
            >>> controller.load_model("/path/to/lora_adapter")
            >>> controller = BERTMetaController(use_lora=False)
            >>> controller.load_model("/path/to/model.pt")
        """
        if self.use_lora:
            # Load PEFT adapter weights
            # For PEFT models, the path should be a directory containing adapter files
            from peft import PeftModel

            # Get the base model from the PEFT wrapper
            base_model = self.model.get_base_model()

            # Load the PEFT model from the saved path
            self.model = PeftModel.from_pretrained(base_model, path)
            self.model = self.model.to(self.device)
        else:
            # Load base model state dict
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)

        # Ensure model is in evaluation mode
        self.model.eval()
        logger.info("Loaded BERT meta-controller from %s (lora=%s)", path, self.use_lora)

    def save_model(self, path: str) -> None:
        """
        Save the current model to disk.

        For LoRA models, saves the PEFT adapter weights. For base models,
        saves the full state dictionary.

        Args:
            path: Path where the model should be saved.
                 For LoRA models, this should be a directory path where
                 adapter_config.json and adapter_model.bin will be saved.
                 For base models, this should be a .pt or .pth file path.

        Example:
            >>> controller = BERTMetaController(use_lora=True)
            >>> controller.save_model("/path/to/lora_adapter")
            >>> controller = BERTMetaController(use_lora=False)
            >>> controller.save_model("/path/to/model.pt")
        """
        if self.use_lora:
            # Save PEFT adapter weights
            # This saves only the LoRA adapter weights, not the full model
            self.model.save_pretrained(path)
        else:
            # Save base model state dict
            torch.save(self.model.state_dict(), path)
        logger.info("Saved BERT meta-controller to %s (lora=%s)", path, self.use_lora)

    def clear_cache(self) -> None:
        """
        Clear the tokenization cache.

        This method removes all cached tokenized inputs, freeing memory.
        Useful when processing many different feature combinations or
        when memory usage is a concern.

        Example:
            >>> controller = BERTMetaController()
            >>> # After many predictions...
            >>> controller.clear_cache()
            >>> info = controller.get_cache_info()
            >>> info['cache_size'] == 0
            True
        """
        self._tokenization_cache.clear()

    def get_cache_info(self) -> dict[str, Any]:
        """
        Get information about the current tokenization cache.

        Returns:
            Dictionary containing cache statistics:
            - cache_size: Number of cached tokenizations
            - cache_keys: List of cached text inputs (truncated for display)

        Example:
            >>> controller = BERTMetaController()
            >>> features = MetaControllerFeatures(
            ...     hrm_confidence=0.8,
            ...     trm_confidence=0.6,
            ...     mcts_value=0.75,
            ...     consensus_score=0.7,
            ...     last_agent='hrm',
            ...     iteration=2,
            ...     query_length=150,
            ...     has_rag_context=True
            ... )
            >>> _ = controller.predict(features)
            >>> info = controller.get_cache_info()
            >>> 'cache_size' in info
            True
            >>> info['cache_size'] >= 1
            True
        """
        # Truncate keys for display (first 50 chars)
        truncated_keys = [key[:50] + "..." if len(key) > 50 else key for key in self._tokenization_cache]

        return {
            "cache_size": len(self._tokenization_cache),
            "cache_keys": truncated_keys,
        }

    def get_trainable_parameters(self) -> dict[str, int]:
        """
        Get the number of trainable and total parameters in the model.

        This is particularly useful for LoRA models to see the efficiency
        gains from using low-rank adaptation.

        Returns:
            Dictionary containing:
            - total_params: Total number of parameters in the model
            - trainable_params: Number of trainable parameters
            - trainable_percentage: Percentage of parameters that are trainable

        Example:
            >>> controller = BERTMetaController(use_lora=True)
            >>> params = controller.get_trainable_parameters()
            >>> params['trainable_percentage'] < 10.0  # LoRA trains <10% of params
            True
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0.0

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percentage": round(trainable_percentage, 2),
        }
