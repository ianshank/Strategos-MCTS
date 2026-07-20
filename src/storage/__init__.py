# Storage Module
"""
Storage infrastructure for multi-agent MCTS framework.

Includes:
- Async S3 client with retry strategies
- Content-hash based idempotent keys
- Compression support
- Pinecone vector storage for agent selection history
"""

# S3 integration (optional — requires tenacity, aioboto3, botocore)
try:
    from .s3_client import S3Config, S3StorageClient  # noqa: F401

    S3_AVAILABLE = True
    _s3_exports = [
        "S3StorageClient",
        "S3Config",
        "S3_AVAILABLE",
    ]
except ImportError:
    S3_AVAILABLE = False
    _s3_exports = ["S3_AVAILABLE"]

# Pinecone integration (optional)
try:
    from .pinecone_store import (  # noqa: F401
        PINECONE_AVAILABLE,
        PineconeVectorStore,
    )

    _pinecone_exports = [
        "PineconeVectorStore",
        "PINECONE_AVAILABLE",
    ]
except ImportError:
    _pinecone_exports = []

__all__ = _s3_exports + _pinecone_exports
