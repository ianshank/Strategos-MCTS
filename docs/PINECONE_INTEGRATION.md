# Pinecone Integration Guide

## Overview

The LangGraph Multi-Agent MCTS framework includes optional Pinecone vector storage integration for the Meta-Controller. This enables:

- **Semantic Search**: Find similar past routing decisions based on feature similarity
- **Pattern Analysis**: Analyze agent selection patterns over time
- **Retrieval-Augmented Routing**: Use historical decisions to improve agent selection
- **Persistent Storage**: Store and retrieve agent selection history across sessions

## Architecture

```
MetaControllerFeatures (10D vector)
    ↓
PineconeVectorStore
    ↓
Pinecone Index (cosine similarity)
    ↓
Similar Decisions Retrieval
```

## Features

### 1. **Vector Representation**
- 10-dimensional normalized feature vectors
- Features include: task complexity, computational intensity, uncertainty, historical accuracy, etc.
- Consistent normalization ensures all values are in [0, 1] range

### 2. **Offline Buffering**
- Operations are buffered when Pinecone is unavailable
- Automatic retry and flush when connection is restored
- Zero data loss during disconnections

### 3. **Namespace Isolation**
- Different experiments can use separate namespaces
- Default namespace: `meta_controller`
- Easy data segregation for A/B testing

### 4. **Batch Operations**
- Efficient batch storage for training data
- Reduced API calls and improved throughput

## Setup Instructions

### Step 1: Install Pinecone Client

```bash
pip install pinecone
```

Or add to your requirements:

```bash
pip install -e ".[pinecone]"
```

### Step 2: Create Pinecone Account

1. Go to [https://app.pinecone.io/](https://app.pinecone.io/)
2. Sign up for a free account (Starter plan is sufficient)
3. Create a new project

### Step 3: Create an Index

In the Pinecone console:

1. Click "Create Index"
2. Configure with these settings:
   - **Name**: Choose any name (e.g., `meta-controller`)
   - **Dimensions**: `10` (required)
   - **Metric**: `cosine` (recommended)
   - **Pod Type**: `starter` (free tier)
   - **Region**: Choose closest to your location

### Step 4: Get Credentials

From the Pinecone console:

1. **API Key**: Found in "API Keys" section
2. **Environment**: Shown with your index (e.g., `us-east-1-aws`)
3. **Index Host**: Format is `https://[index-name].svc.[environment].pinecone.io`

### Step 5: Configure Environment

#### Option A: Environment Variables

**Windows PowerShell:**
```powershell
$env:PINECONE_API_KEY="your-api-key-here"
$env:PINECONE_HOST="https://your-index.svc.environment.pinecone.io"
```

**Windows CMD:**
```cmd
set PINECONE_API_KEY=your-api-key-here
set PINECONE_HOST=https://your-index.svc.environment.pinecone.io
```

**Linux/Mac:**
```bash
export PINECONE_API_KEY="your-api-key-here"
export PINECONE_HOST="https://your-index.svc.environment.pinecone.io"
```

#### Option B: .env File

Create a `.env` file in the project root:

```
PINECONE_API_KEY=your-api-key-here
PINECONE_HOST=https://your-index.svc.environment.pinecone.io
```

### Step 6: Verify Installation

Run the verification script:

```bash
python scripts/verification/verify_pinecone_integration.py
```

Or test with your credentials:

```bash
python scripts/verification/verify_pinecone_integration.py
```

## Usage Examples

### Basic Usage

```python
from src.storage.pinecone_store import PineconeVectorStore
from src.agents.meta_controller.base import MetaControllerFeatures, MetaControllerPrediction

# Initialize store (auto-loads from environment)
store = PineconeVectorStore(namespace="my_experiment")

# Create features
features = MetaControllerFeatures(
    task_complexity=0.8,
    computational_intensity=0.6,
    uncertainty_level=0.5,
    historical_accuracy=0.9,
    last_agent="hrm",
    iteration=1,
    query_length=200,
    has_rag_context=True
)

# Create prediction
prediction = MetaControllerPrediction(
    agent="trm",
    confidence=0.75,
    probabilities={"hrm": 0.20, "trm": 0.75, "mcts": 0.05}
)

# Store the decision
vector_id = store.store_prediction(features, prediction)

# Find similar past decisions
similar = store.find_similar_decisions(features, top_k=5)

# Get agent distribution for similar cases
distribution = store.get_agent_distribution(features, top_k=10)
```

### Offline Buffering

```python
# Works even without credentials
store = PineconeVectorStore(api_key=None, host=None, auto_init=False)

# Operations are buffered
store.store_prediction(features, prediction)

# Check buffer
buffered = store.get_buffered_operations()
print(f"Buffered {len(buffered)} operations")

# Flush when connection available
store.flush_buffer()
```

### Batch Operations

```python
# Store multiple decisions at once
features_list = [features1, features2, features3]
predictions_list = [pred1, pred2, pred3]

count = store.store_batch(
    features_list,
    predictions_list,
    batch_metadata={"experiment": "v1"}
)
```

## Integration with Meta-Controller

The Pinecone integration is optional and the system works without it:

```python
# In your meta-controller configuration
meta_config = {
    "pinecone": {
        "enabled": True,
        "namespace": "production",
        "retrieve_similar": True,
        "top_k": 10
    }
}
```

## Troubleshooting

### Common Issues

1. **"pinecone-client not installed"**
   - Solution: `pip install pinecone`

2. **"Pinecone not configured"**
   - Solution: Set `PINECONE_API_KEY` and `PINECONE_HOST` environment variables

3. **"Index dimension mismatch"**
   - Solution: Ensure your index is configured with dimension=10

4. **"Connection timeout"**
   - Solution: Check your internet connection and Pinecone service status

### Verification Commands

```bash
# Check if package is installed
pip show pinecone

# Test connection
python -c "from src.storage.pinecone_store import PineconeVectorStore; print(PineconeVectorStore().is_available)"

# Run full verification
python scripts/verification/verify_pinecone_integration.py
```

## Performance Considerations

1. **Batch Size**: Use batch operations for > 10 vectors
2. **Namespace Strategy**: Separate namespaces for dev/staging/prod
3. **Query Optimization**: Limit top_k to necessary results
4. **Index Type**: Starter indexes are sufficient for most use cases

## Cost and Limits

### Free Tier (Starter)
- 1 index
- 100K vectors
- 5 million operations/month
- Sufficient for development and small deployments

### Scaling
- Upgrade to Standard/Enterprise for:
  - Multiple indexes
  - Higher throughput
  - Production SLAs

## Project Details

For the default project (ID: `231aa5df-d444-4d97-8776-21751921022f`), ensure you have:

1. Created an index with the correct dimensions
2. Retrieved the API key from the console
3. Configured the environment variables
4. Verified the connection using the test scripts

## Additional Resources

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Vector Database Concepts](https://www.pinecone.io/learn/vector-database/)
- [Similarity Search Guide](https://www.pinecone.io/learn/what-is-similarity-search/)
