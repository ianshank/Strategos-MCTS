"""
Verify Pinecone integration for the Multi-Agent MCTS Framework.

This script tests:
1. Package availability
2. Connection status
3. Vector storage functionality
4. Retrieval capabilities
5. Error handling and buffering
"""

import os
import sys


def check_package_installed():
    """Check if pinecone-client is installed."""
    print("=" * 60)
    print("PINECONE INTEGRATION VERIFICATION")
    print("=" * 60)
    print()

    print("1. Checking package installation...")
    try:
        import pinecone

        print("   [OK] pinecone-client is installed")
        print(f"   Version: {pinecone.__version__ if hasattr(pinecone, '__version__') else 'Unknown'}")
        return True
    except ImportError:
        print("   [FAIL] pinecone-client is NOT installed")
        print("   Install with: pip install pinecone-client")
        return False


def check_environment_variables():
    """Check if required environment variables are set."""
    print("\n2. Checking environment variables...")

    api_key = os.environ.get("PINECONE_API_KEY")
    host = os.environ.get("PINECONE_HOST")

    env_ok = True

    if api_key:
        print(f"   [OK] PINECONE_API_KEY is set (length: {len(api_key)})")
    else:
        print("   [FAIL] PINECONE_API_KEY is NOT set")
        env_ok = False

    if host:
        print(f"   [OK] PINECONE_HOST is set: {host}")
    else:
        print("   [FAIL] PINECONE_HOST is NOT set")
        print("   Example: https://your-index.svc.environment.pinecone.io")
        env_ok = False

    if not env_ok:
        print("\n   To set environment variables:")
        print("   - Create a .env file with:")
        print("     PINECONE_API_KEY=your-api-key")
        print("     PINECONE_HOST=https://your-index.svc.environment.pinecone.io")

    return env_ok


def test_pinecone_store():
    """Test the PineconeVectorStore functionality."""
    print("\n3. Testing PineconeVectorStore...")

    try:
        from src.agents.meta_controller.base import MetaControllerFeatures, MetaControllerPrediction
        from src.storage.pinecone_store import PINECONE_AVAILABLE, PineconeVectorStore

        if not PINECONE_AVAILABLE:
            print("   [FAIL] Pinecone package not available")
            return False

        # Test 1: Create store instance
        print("   Testing store creation...")
        store = PineconeVectorStore(namespace="integration_test", auto_init=True)

        if store.is_available:
            print("   [OK] Pinecone store initialized successfully")

            # Get stats
            stats = store.get_stats()
            print("   Index stats:")
            print(f"     - Total vectors: {stats.get('total_vectors', 0)}")
            print(f"     - Vector dimension: {store.VECTOR_DIMENSION}")
            print(f"     - Available: {stats.get('available', False)}")

            # Test 2: Store a prediction
            print("\n   Testing vector storage...")
            features = MetaControllerFeatures(
                task_complexity=0.7,
                computational_intensity=0.8,
                uncertainty_level=0.6,
                historical_accuracy=0.75,
                last_agent="trm",
                iteration=4,
                query_length=350,
                has_rag_context=True,
            )

            prediction = MetaControllerPrediction(
                agent="hrm", confidence=0.85, probabilities={"hrm": 0.85, "trm": 0.10, "mcts": 0.05}
            )

            vector_id = store.store_prediction(features, prediction, metadata={"test": "integration"})
            if vector_id:
                print(f"   [OK] Vector stored successfully with ID: {vector_id}")
            else:
                print("   [FAIL] Failed to store vector")
                return False

            # Test 3: Find similar decisions
            print("\n   Testing similarity search...")
            similar = store.find_similar_decisions(features, top_k=3)
            print(f"   Found {len(similar)} similar decisions")

            if similar:
                print("   Sample result:")
                result = similar[0]
                print(f"     - Score: {result.get('score', 0):.4f}")
                if "metadata" in result:
                    print(f"     - Selected agent: {result['metadata'].get('selected_agent')}")
                    print(f"     - Confidence: {result['metadata'].get('confidence', 0):.2f}")

            # Test 4: Agent distribution
            print("\n   Testing agent distribution...")
            distribution = store.get_agent_distribution(features, top_k=5)
            print("   Agent selection distribution:")
            for agent, freq in distribution.items():
                print(f"     - {agent}: {freq:.2%}")

            return True

        else:
            print("   [FAIL] Pinecone store not available (check credentials)")

            # Test buffering
            print("\n   Testing offline buffering...")
            test_store = PineconeVectorStore(api_key=None, host=None, auto_init=False)

            features = MetaControllerFeatures(0.5, 0.6, 0.7, 0.8, "hrm", 1, 100, False)
            prediction = MetaControllerPrediction("trm", 0.7, {"trm": 0.7, "hrm": 0.2, "mcts": 0.1})

            test_store.store_prediction(features, prediction)
            buffered = test_store.get_buffered_operations()

            print(f"   [OK] Buffered {len(buffered)} operations")
            if buffered:
                print(f"   Operation type: {buffered[0]['type']}")
                print(f"   Timestamp: {buffered[0]['timestamp']}")

            return False

    except Exception as e:
        print(f"   [FAIL] Error testing PineconeVectorStore: {e}")
        return False


def test_vector_normalization():
    """Test feature normalization for vectors."""
    print("\n4. Testing feature normalization...")

    try:
        from src.agents.meta_controller.base import MetaControllerFeatures
        from src.agents.meta_controller.utils import normalize_features

        # Create sample features
        features = MetaControllerFeatures(
            task_complexity=0.9,
            computational_intensity=0.3,
            uncertainty_level=0.5,
            historical_accuracy=0.95,
            last_agent="mcts",
            iteration=10,
            query_length=500,
            has_rag_context=True,
        )

        # Normalize
        vector = normalize_features(features)

        print(f"   [OK] Normalized feature vector (dimension: {len(vector)})")
        print(f"   Values: [{', '.join(f'{v:.3f}' for v in vector[:5])}, ...]")

        # Verify properties
        assert len(vector) == 10, "Vector should be 10-dimensional"
        assert all(0.0 <= v <= 1.0 for v in vector), "All values should be in [0, 1]"

        print("   [OK] Vector properties verified")
        return True

    except Exception as e:
        print(f"   [FAIL] Error testing normalization: {e}")
        return False


def show_setup_instructions():
    """Show instructions for setting up Pinecone."""
    print("\n" + "=" * 60)
    print("PINECONE SETUP INSTRUCTIONS")
    print("=" * 60)
    print()
    print("1. Install the Pinecone client:")
    print("   pip install pinecone-client")
    print()
    print("2. Create a Pinecone account:")
    print("   https://app.pinecone.io/")
    print()
    print("3. Create an index with:")
    print("   - Dimension: 10")
    print("   - Metric: cosine")
    print("   - Pod type: starter (free tier)")
    print()
    print("4. Get your API key and host from the Pinecone console")
    print()
    print("5. Set environment variables:")
    print("   PINECONE_API_KEY=your-api-key")
    print("   PINECONE_HOST=https://your-index.svc.environment.pinecone.io")
    print()
    print("6. Optional: Add to .env file for persistence")
    print()
    print("Benefits of Pinecone integration:")
    print("- Semantic search for similar past routing decisions")
    print("- Pattern analysis in agent selection")
    print("- Retrieval-augmented routing strategies")
    print("- Historical decision tracking and learning")


def main():
    """Run all verification tests."""
    # Check package
    package_ok = check_package_installed()

    # Check environment
    env_ok = check_environment_variables()

    # Run tests if package is available
    if package_ok:
        store_ok = test_pinecone_store()
        _norm_ok = test_vector_normalization()  # noqa: F841 - test result logged
    else:
        print("\n3. Skipping PineconeVectorStore tests (package not installed)")
        print("4. Skipping normalization tests")
        store_ok = False
        _norm_ok = True  # noqa: F841 - Normalization doesn't require Pinecone

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_ok = package_ok and env_ok and store_ok

    if all_ok:
        print("[OK] Pinecone integration is fully functional!")
        print("  You can use vector storage for agent selection history.")
    elif package_ok and not env_ok:
        print("[WARN] Pinecone is installed but not configured.")
        print("  Set PINECONE_API_KEY and PINECONE_HOST to enable.")
    elif not package_ok:
        print("[WARN] Pinecone client not installed.")
        print("  The system will work without vector storage.")
        print("  Operations will be buffered until Pinecone is available.")

    # Show setup instructions if needed
    if not all_ok:
        show_setup_instructions()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
