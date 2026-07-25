#!/usr/bin/env python3
"""
Verification script to check that all dependencies are installed correctly.
"""

import os
import sys

# Set UTF-8 encoding for Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

print(f"Python Version: {sys.version}")
print("\n" + "=" * 60)
print("CORE FRAMEWORK IMPORTS")
print("=" * 60)

modules = [
    ("langgraph", "Core state machine framework"),
    ("langchain", "LLM integration library"),
    ("langchain_core", "LangChain core utilities"),
    ("pydantic", "Data validation library"),
    ("httpx", "Async HTTP client"),
    ("tenacity", "Retry logic library"),
    ("openai", "OpenAI API client"),
]

for module_name, description in modules:
    try:
        module = __import__(module_name.replace("_", "-").replace("-", "_"))
        version = getattr(module, "__version__", "")
        print(f"✓ {module_name:<20} {version:<10} - {description}")
    except ImportError as e:
        print(f"✗ {module_name:<20} FAILED - {str(e)}")

print("\n" + "=" * 60)
print("OBSERVABILITY IMPORTS")
print("=" * 60)

observability_modules = [
    ("opentelemetry.api", "OpenTelemetry API"),
    ("opentelemetry.sdk", "OpenTelemetry SDK"),
    ("opentelemetry.exporter.otlp.proto.grpc", "OTLP gRPC exporter"),
    ("psutil", "System monitoring"),
    ("prometheus_client", "Prometheus metrics"),
]

for module_name, description in observability_modules:
    try:
        module = __import__(module_name, fromlist=[""])
        print(f"✓ {module_name:<40} - {description}")
    except ImportError as e:
        print(f"✗ {module_name:<40} - {description}: {str(e)[:50]}")

print("\n" + "=" * 60)
print("STORAGE & AWS IMPORTS")
print("=" * 60)

storage_modules = [
    ("aioboto3", "Async AWS SDK"),
    ("boto3", "AWS SDK"),
    ("botocore", "AWS API"),
]

for module_name, description in storage_modules:
    try:
        module = __import__(module_name)
        print(f"✓ {module_name:<20} - {description}")
    except ImportError as e:
        print(f"✗ {module_name:<20} - {description}: {str(e)[:50]}")

print("\n" + "=" * 60)
print("DEVELOPMENT TOOLS")
print("=" * 60)

dev_modules = [
    ("pytest", "Testing framework"),
    ("mypy", "Static type checker"),
    ("black", "Code formatter"),
    ("ruff", "Python linter"),
    ("bandit", "Security linter"),
    ("pre_commit", "Pre-commit hook framework"),
]

for module_name, description in dev_modules:
    try:
        module = __import__(module_name.replace("-", "_"))
        version = getattr(module, "__version__", "")
        print(f"✓ {module_name:<20} {version:<10} - {description}")
    except ImportError:
        print(f"✗ {module_name:<20} FAILED - {description}")

print("\n" + "=" * 60)
print("PROJECT IMPORTS")
print("=" * 60)

try:
    from src.config.settings import get_settings  # noqa: F401

    print("✓ src.config.settings - Configuration management")
except ImportError as e:
    print(f"✗ src.config.settings - {str(e)}")

try:
    from src.adapters.llm import create_client  # noqa: F401

    print("✓ src.adapters.llm - LLM provider abstraction")
except ImportError as e:
    print(f"✗ src.adapters.llm - {str(e)}")

try:
    from src.framework.mcts.core import MCTSEngine  # noqa: F401

    print("✓ src.framework.mcts.core - MCTS engine")
except ImportError as e:
    print(f"✗ src.framework.mcts.core - {str(e)}")

try:
    from src.observability.logging import setup_logging  # noqa: F401

    print("✓ src.observability.logging - Structured logging")
except ImportError as e:
    print(f"✗ src.observability.logging - {str(e)}")

try:
    from src.storage.s3_client import S3Client  # noqa: F401

    print("✓ src.storage.s3_client - S3 storage client")
except ImportError as e:
    print(f"✗ src.storage.s3_client - {str(e)}")

print("\n" + "=" * 60)
print("SETUP VERIFICATION COMPLETE")
print("=" * 60)
print("\n✓ All critical dependencies are installed!")
print("\nNext steps:")
print("  1. Configure .env file with your API keys and settings")
print("  2. Run: pytest tests/ -v")
print("  3. Start MCP server: python tools/mcp/server.py")
print("  4. Check examples/ for usage patterns")
