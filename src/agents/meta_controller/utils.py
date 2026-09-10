"""
Utility functions for Neural Meta-Controller feature processing.

Provides functions for normalizing, encoding, and converting features
into formats suitable for different neural network architectures.
"""

import torch

from src.agents.meta_controller.base import MetaControllerFeatures


def controller_initialized_extra(*, seed: int, name: str) -> dict[str, object]:
    """Logger extra for controller init. Never use LogRecord-reserved key ``name``."""
    return {"seed": seed, "controller_name": name}


def normalize_features(features: MetaControllerFeatures) -> list[float]:
    """
    Normalize meta-controller features to a 10-dimensional vector in range [0, 1].

    The normalization strategy:
    - Confidence scores (hrm, trm, mcts_value, consensus): Already 0-1, clipped
    - last_agent: Encoded as 3 one-hot values (hrm=0, trm=1, mcts=2)
    - iteration: Normalized to 0-1 assuming max 20 iterations
    - query_length: Normalized to 0-1 assuming max 10000 characters
    - has_rag_context: Binary 0 or 1

    Output vector structure (10 dimensions):
    [hrm_conf, trm_conf, mcts_value, consensus, last_hrm, last_trm, last_mcts,
     iteration_norm, query_length_norm, has_rag]

    Args:
        features: MetaControllerFeatures instance to normalize.

    Returns:
        List of 10 floats, each normalized to range [0, 1].

    Example:
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
        >>> normalized = normalize_features(features)
        >>> len(normalized)
        10
        >>> all(0.0 <= v <= 1.0 for v in normalized)
        True
    """
    # Clip confidence scores to [0, 1]
    hrm_conf = max(0.0, min(1.0, features.hrm_confidence))
    trm_conf = max(0.0, min(1.0, features.trm_confidence))
    mcts_val = max(0.0, min(1.0, features.mcts_value))
    consensus = max(0.0, min(1.0, features.consensus_score))

    # One-hot encode last_agent (3 dimensions)
    last_agent_onehot = one_hot_encode_agent(features.last_agent)

    # Normalize iteration (assuming max 20 iterations)
    max_iterations = 20
    iteration_norm = max(0.0, min(1.0, features.iteration / max_iterations))

    # Normalize query length (assuming max 10000 characters)
    max_query_length = 10000
    query_length_norm = max(0.0, min(1.0, features.query_length / max_query_length))

    # Binary for has_rag_context
    has_rag = 1.0 if features.has_rag_context else 0.0

    # Combine into 10-dimensional vector
    return [
        hrm_conf,
        trm_conf,
        mcts_val,
        consensus,
        last_agent_onehot[0],  # hrm
        last_agent_onehot[1],  # trm
        last_agent_onehot[2],  # mcts
        iteration_norm,
        query_length_norm,
        has_rag,
    ]


def one_hot_encode_agent(agent: str) -> list[float]:
    """
    One-hot encode an agent name into a 3-dimensional vector.

    Encoding:
    - 'hrm' -> [1.0, 0.0, 0.0]
    - 'trm' -> [0.0, 1.0, 0.0]
    - 'mcts' -> [0.0, 0.0, 1.0]
    - 'none' or other -> [0.0, 0.0, 0.0]

    Args:
        agent: Agent name string ('hrm', 'trm', 'mcts', or 'none').

    Returns:
        List of 3 floats representing the one-hot encoding.

    Example:
        >>> one_hot_encode_agent('hrm')
        [1.0, 0.0, 0.0]
        >>> one_hot_encode_agent('trm')
        [0.0, 1.0, 0.0]
        >>> one_hot_encode_agent('mcts')
        [0.0, 0.0, 1.0]
        >>> one_hot_encode_agent('none')
        [0.0, 0.0, 0.0]
    """
    agent_lower = agent.lower()

    if agent_lower == "hrm":  # noqa: SIM116
        return [1.0, 0.0, 0.0]
    elif agent_lower == "trm":
        return [0.0, 1.0, 0.0]
    elif agent_lower == "mcts":
        return [0.0, 0.0, 1.0]
    else:
        # 'none' or unknown agent
        return [0.0, 0.0, 0.0]


def features_to_tensor(features: MetaControllerFeatures) -> torch.Tensor:
    """
    Convert meta-controller features to a PyTorch tensor.

    Uses normalize_features internally to create a normalized 10-dimensional
    tensor suitable for neural network input.

    Args:
        features: MetaControllerFeatures instance to convert.

    Returns:
        PyTorch tensor of shape (10,) with float32 dtype.

    Example:
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
        >>> tensor = features_to_tensor(features)
        >>> tensor.shape
        torch.Size([10])
        >>> tensor.dtype
        torch.float32
    """
    normalized = normalize_features(features)
    return torch.tensor(normalized, dtype=torch.float32)


def features_to_text(features: MetaControllerFeatures) -> str:
    """
    Convert meta-controller features to structured text format.

    Creates a human-readable text representation suitable for text-based
    models like BERT or other language models.

    Args:
        features: MetaControllerFeatures instance to convert.

    Returns:
        Structured text string describing the features.

    Example:
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
        >>> text = features_to_text(features)
        >>> 'HRM confidence: 0.800' in text
        True
    """
    rag_status = "available" if features.has_rag_context else "not available"

    text = (
        f"Agent State Features:\n"
        f"HRM confidence: {features.hrm_confidence:.3f}\n"
        f"TRM confidence: {features.trm_confidence:.3f}\n"
        f"MCTS value: {features.mcts_value:.3f}\n"
        f"Consensus score: {features.consensus_score:.3f}\n"
        f"Last agent used: {features.last_agent}\n"
        f"Current iteration: {features.iteration}\n"
        f"Query length: {features.query_length} characters\n"
        f"RAG context: {rag_status}"
    )

    return text


def convert_features_to_text(features: MetaControllerFeatures) -> str:
    """
    Convert MetaControllerFeatures to a text description for BERT input.

    This is an alias for features_to_text() to maintain backward compatibility.

    Args:
        features: MetaControllerFeatures instance

    Returns:
        Text description of the features
    """
    return features_to_text(features)
