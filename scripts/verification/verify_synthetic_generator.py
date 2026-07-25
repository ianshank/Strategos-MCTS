#!/usr/bin/env python3
"""
Verify Synthetic Knowledge Generator Installation and Setup.

This script tests that all components are properly installed and configured.

Usage:
    python scripts/verify_synthetic_generator.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check required dependencies."""
    logger.info("Checking dependencies...")

    required = [
        ("yaml", "PyYAML"),
        ("httpx", "httpx"),
        ("tqdm", "tqdm"),
    ]

    missing = []
    for module, package in required:
        try:
            __import__(module)
            logger.info(f"  ✓ {package}")
        except ImportError:
            logger.error(f"  ✗ {package} - NOT INSTALLED")
            missing.append(package)

    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False

    return True


def check_llm_adapters():
    """Check LLM adapter imports."""
    logger.info("\nChecking LLM adapters...")

    try:
        from src.adapters.llm import list_providers

        providers = list_providers()
        logger.info(f"  ✓ Available providers: {', '.join(providers)}")
        return True

    except ImportError as e:
        logger.error(f"  ✗ Failed to import LLM adapters: {e}")
        return False


def check_generator_import():
    """Check generator import."""
    logger.info("\nChecking synthetic generator...")

    try:
        from training.synthetic_knowledge_generator import (
            QUESTION_TEMPLATES,
            QAPair,
            QualityValidator,
            SyntheticKnowledgeGenerator,
        )

        # Verify all classes are importable
        classes_to_verify = [QAPair, QualityValidator, SyntheticKnowledgeGenerator]

        num_categories = len(QUESTION_TEMPLATES)
        num_templates = sum(len(t) for t in QUESTION_TEMPLATES.values())

        logger.info(f"  ✓ Generator module loaded ({len(classes_to_verify)} classes verified)")
        logger.info(f"  ✓ {num_categories} categories available")
        logger.info(f"  ✓ {num_templates} total templates")

        return True

    except ImportError as e:
        logger.error(f"  ✗ Failed to import generator: {e}")
        return False


def check_api_keys():
    """Check API key configuration."""
    logger.info("\nChecking API keys...")

    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY"),
    }

    has_llm_key = False

    for key_name, key_value in keys.items():
        if key_value:
            logger.info(f"  ✓ {key_name} is set")
            if key_name != "LANGSMITH_API_KEY":
                has_llm_key = True
        else:
            logger.warning(f"  ⚠ {key_name} not set")

    if not has_llm_key:
        logger.warning("\n  No LLM API key found!")
        logger.warning("  Set either OPENAI_API_KEY or ANTHROPIC_API_KEY")
        logger.warning("  Or use local LM Studio (no key needed)")

    return has_llm_key


def check_file_structure():
    """Check required files exist."""
    logger.info("\nChecking file structure...")

    required_files = [
        "training/synthetic_knowledge_generator.py",
        "training/synthetic_generator_config.yaml",
        "scripts/generate_synthetic_training_data.py",
        "examples/synthetic_data_generation_example.py",
        "tests/integration/test_synthetic_knowledge_generator.py",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            logger.info(f"  ✓ {file_path}")
        else:
            logger.error(f"  ✗ {file_path} - NOT FOUND")
            all_exist = False

    return all_exist


async def test_basic_functionality():
    """Test basic generator functionality."""
    logger.info("\nTesting basic functionality...")

    try:
        from training.synthetic_knowledge_generator import (
            QAPair,
            QualityValidator,
        )

        # Test QAPair creation
        pair = QAPair(
            question="What is Monte Carlo Tree Search?",
            answer="MCTS is a heuristic search algorithm...",
            contexts=["Context 1", "Context 2"],
            metadata={"category": "test"},
        )

        logger.info("  ✓ QAPair creation")

        # Test LangSmith format conversion
        langsmith = pair.to_langsmith_format()
        assert "inputs" in langsmith
        assert "outputs" in langsmith
        logger.info("  ✓ LangSmith format conversion")

        # Test validation
        validator = QualityValidator()
        is_valid, errors = validator.validate(pair)
        logger.info(f"  ✓ Quality validation (valid={is_valid})")

        # Test quality scoring
        score = validator.score_quality(pair)
        logger.info(f"  ✓ Quality scoring (score={score:.2f})")

        return True

    except Exception as e:
        logger.error(f"  ✗ Functionality test failed: {e}")
        return False


async def test_mock_generation():
    """Test generation with mock client."""
    logger.info("\nTesting mock generation...")

    try:
        from src.adapters.llm.base import LLMResponse
        from training.synthetic_knowledge_generator import SyntheticKnowledgeGenerator

        # Create mock client
        class MockClient:
            async def generate(self, **_kwargs):
                return LLMResponse(
                    text="This is a test answer.",
                    usage={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
                    model="mock",
                )

        # Create generator
        generator = SyntheticKnowledgeGenerator(
            llm_client=MockClient(),
            output_dir="/tmp/synthetic_test",
        )

        logger.info("  ✓ Generator initialized")

        # Test template filling
        template = "What is {algorithm}?"
        filled = generator._fill_template(template)
        assert "{algorithm}" not in filled
        logger.info(f"  ✓ Template filling: {filled}")

        # Test hash generation
        qa_hash = generator._hash_qa("Test question?")
        assert len(qa_hash) == 32  # MD5 hash
        logger.info(f"  ✓ Deduplication hash: {qa_hash[:8]}...")

        return True

    except Exception as e:
        logger.error(f"  ✗ Mock generation test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for user."""
    logger.info("\n" + "=" * 70)
    logger.info("Next Steps")
    logger.info("=" * 70)

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        logger.info("\n1. Set up API key:")
        logger.info("   export OPENAI_API_KEY='sk-...'")
        logger.info("   # or")
        logger.info("   export ANTHROPIC_API_KEY='sk-ant-...'")
        logger.info("")

    logger.info("2. Test with small generation:")
    logger.info("   python scripts/generate_synthetic_training_data.py \\")
    logger.info("       --num-samples 10 \\")
    logger.info("       --model gpt-3.5-turbo")
    logger.info("")

    logger.info("3. Review generated data:")
    logger.info("   cat training/synthetic_data/*.json | jq '.[0]'")
    logger.info("")

    logger.info("4. Scale to production:")
    logger.info("   python scripts/generate_synthetic_training_data.py \\")
    logger.info("       --num-samples 1000 \\")
    logger.info("       --min-quality 0.6 \\")
    logger.info("       --upload-langsmith")
    logger.info("")

    logger.info("5. Run examples:")
    logger.info("   python examples/synthetic_data_generation_example.py")
    logger.info("")

    logger.info("Documentation:")
    logger.info("  - Quick Start: training/README_SYNTHETIC_GENERATOR.md")
    logger.info("  - Full Guide: training/SYNTHETIC_DATA_GENERATION_GUIDE.md")
    logger.info("=" * 70)


async def main():
    """Main verification function."""
    logger.info("=" * 70)
    logger.info("Synthetic Knowledge Generator - Installation Verification")
    logger.info("=" * 70)
    logger.info("")

    checks = []

    # Run checks
    checks.append(("Dependencies", check_dependencies()))
    checks.append(("LLM Adapters", check_llm_adapters()))
    checks.append(("Generator Module", check_generator_import()))
    checks.append(("File Structure", check_file_structure()))
    checks.append(("API Keys", check_api_keys()))
    checks.append(("Basic Functionality", await test_basic_functionality()))
    checks.append(("Mock Generation", await test_mock_generation()))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Verification Summary")
    logger.info("=" * 70)

    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")

    logger.info("")
    logger.info(f"Passed: {passed}/{total}")

    if passed == total:
        logger.info("\n✓ All checks passed! Installation is ready.")
        print_next_steps()
        return 0
    else:
        logger.error("\n✗ Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
