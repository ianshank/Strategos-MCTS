"""
Example: Using the Academic Research Agent

This example demonstrates paper analysis, citation discovery, and research synthesis.
"""

import asyncio

from src.integrations.google_adk import AcademicResearchAgent
from src.integrations.google_adk.base import ADKBackend, ADKConfig


async def example_paper_analysis():
    """Example: Analyze a research paper."""
    print("=" * 80)
    print("Example 1: Paper Analysis")
    print("=" * 80)

    config = ADKConfig(
        backend=ADKBackend.LOCAL,
        workspace_dir="./workspace/adk_examples/academic_research",
        enable_search=True,
    )

    agent = AcademicResearchAgent(config)
    await agent.initialize()

    # Analyze a paper by title
    response = await agent.analyze_paper(
        paper_title="Attention Is All You Need",
        query="Analyze this seminal transformer paper and its impact on NLP",
    )

    print(f"\nStatus: {response.status}")
    print(f"\nAnalysis:\n{response.result[:500]}...")  # First 500 chars
    print(f"\nFull analysis saved to: {response.metadata.get('analysis_file')}")

    await agent.cleanup()


async def example_citation_discovery():
    """Example: Find recent citations."""
    print("\n" + "=" * 80)
    print("Example 2: Citation Discovery")
    print("=" * 80)

    config = ADKConfig(
        backend=ADKBackend.LOCAL,
        workspace_dir="./workspace/adk_examples/academic_research",
        enable_search=True,
    )

    agent = AcademicResearchAgent(config)
    await agent.initialize()

    # Find recent citations
    response = await agent.find_citations(
        paper_title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    )

    print(f"\nStatus: {response.status}")
    print(f"\nCitation Discovery Plan:\n{response.result[:500]}...")
    print(f"\nResults saved to: {response.metadata.get('citation_file')}")

    await agent.cleanup()


async def example_future_directions():
    """Example: Suggest future research directions."""
    print("\n" + "=" * 80)
    print("Example 3: Future Research Directions")
    print("=" * 80)

    config = ADKConfig(
        backend=ADKBackend.LOCAL,
        workspace_dir="./workspace/adk_examples/academic_research",
        enable_search=True,
    )

    agent = AcademicResearchAgent(config)
    await agent.initialize()

    # Generate research directions
    response = await agent.suggest_future_research(
        paper_title="AlphaFold: Protein Structure Prediction",
        query="Suggest novel research directions building on AlphaFold's success",
    )

    print(f"\nStatus: {response.status}")
    print(f"\nFuture Directions:\n{response.result[:500]}...")

    await agent.cleanup()


async def example_training_corpus_building():
    """Example: Building training corpus from research papers."""
    print("\n" + "=" * 80)
    print("Example 4: Training Corpus Building for MCTS Framework")
    print("=" * 80)

    config = ADKConfig(
        backend=ADKBackend.LOCAL,
        workspace_dir="./workspace/adk_examples/academic_research",
        enable_search=True,
    )

    agent = AcademicResearchAgent(config)
    await agent.initialize()

    # Research papers for your training corpus
    research_topics = [
        "Monte Carlo Tree Search algorithms",
        "Hierarchical Reasoning Models",
        "Multi-agent reinforcement learning",
        "LangGraph agent frameworks",
    ]

    print("\nBuilding research corpus for MCTS framework training...\n")

    corpus_data = []
    for topic in research_topics:
        print(f"Researching: {topic}")

        # Find citations for this topic
        response = await agent.find_citations(paper_title=topic)

        if response.status == "success":
            corpus_data.append(
                {
                    "topic": topic,
                    "citation_file": response.metadata.get("citation_file"),
                }
            )
            print("  ✓ Citations collected")

    print("\nCorpus building complete!")
    print(f"Collected data for {len(corpus_data)} research topics")
    print("\nThis data can be used to:")
    print("- Train your HRM/TRM agents")
    print("- Improve MCTS search strategies")
    print("- Enhance agent decision-making")

    await agent.cleanup()


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Google ADK Academic Research Agent Examples")
    print("=" * 80)

    await example_paper_analysis()
    await example_citation_discovery()
    await example_future_directions()
    await example_training_corpus_building()

    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nIntegration with Your Framework:")
    print("1. Use Academic Research agent to build training corpus")
    print("2. Feed research insights into your HRM/TRM training")
    print("3. Enhance MCTS search with academic best practices")
    print("4. Keep agents updated with latest research")


if __name__ == "__main__":
    asyncio.run(main())
