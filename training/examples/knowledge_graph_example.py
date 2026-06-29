#!/usr/bin/env python3
"""
Knowledge Graph System - Example Usage

Demonstrates how to build and query a knowledge graph for MCTS/AI concepts.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.knowledge_graph import (
    ConceptNode,
    GraphQA,
    GraphQueryEngine,
    KnowledgeExtractor,
    KnowledgeGraphBuilder,
    RelationType,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def example_1_manual_graph_construction():
    """Example 1: Manually construct a knowledge graph."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 1: Manual Graph Construction")
    logger.info("=" * 80)

    config = {
        "backend": "networkx",
        "storage": "./cache/knowledge_graph_examples",
    }

    builder = KnowledgeGraphBuilder(config)

    # Add MCTS-related concepts
    concepts = [
        ConceptNode(
            id="mcts",
            name="MCTS",
            type="algorithm",
            description="Monte Carlo Tree Search - probabilistic search algorithm for decision processes",
            aliases=["Monte Carlo Tree Search", "monte carlo tree search"],
            properties={"complexity": "O(n)", "domain": "game playing, planning"},
        ),
        ConceptNode(
            id="ucb1",
            name="UCB1",
            type="algorithm",
            description="Upper Confidence Bound - algorithm for multi-armed bandit problem",
            aliases=["Upper Confidence Bound", "UCB"],
            properties={"complexity": "O(1)", "domain": "exploration-exploitation"},
        ),
        ConceptNode(
            id="puct",
            name="PUCT",
            type="algorithm",
            description="Polynomial Upper Confidence Trees - UCB variant used in AlphaGo",
            aliases=["Polynomial UCT"],
            properties={"improved_by": "AlphaGo team"},
        ),
        ConceptNode(
            id="alphazero",
            name="AlphaZero",
            type="architecture",
            description="General game-playing system using self-play and neural networks",
            aliases=["Alpha Zero"],
            properties={"year": 2017, "company": "DeepMind"},
        ),
        ConceptNode(
            id="muzero",
            name="MuZero",
            type="architecture",
            description="Model-based RL algorithm that learns environment dynamics",
            aliases=["Mu Zero"],
            properties={"year": 2019, "company": "DeepMind"},
        ),
        ConceptNode(
            id="neural_network",
            name="Neural Network",
            type="technique",
            description="Computational model inspired by biological neural networks",
            aliases=["NN", "neural net", "deep neural network"],
        ),
        ConceptNode(
            id="self_play",
            name="Self-Play",
            type="technique",
            description="Training technique where agent plays against itself",
            aliases=["self play"],
        ),
        ConceptNode(
            id="value_network",
            name="Value Network",
            type="component",
            description="Neural network that evaluates board positions",
            aliases=["value net"],
        ),
        ConceptNode(
            id="policy_network",
            name="Policy Network",
            type="component",
            description="Neural network that suggests promising moves",
            aliases=["policy net"],
        ),
    ]

    for concept in concepts:
        builder.add_concept(concept)
        logger.info(f"Added concept: {concept.name}")

    # Add relationships
    relationships = [
        ("mcts", "ucb1", RelationType.USES, "MCTS uses UCB1 for node selection"),
        ("puct", "ucb1", RelationType.IMPROVES, "PUCT improves upon UCB1"),
        ("alphazero", "mcts", RelationType.USES, "AlphaZero uses MCTS for search"),
        ("alphazero", "neural_network", RelationType.USES, "AlphaZero uses neural networks for evaluation"),
        ("alphazero", "self_play", RelationType.REQUIRES, "AlphaZero requires self-play for training"),
        ("alphazero", "puct", RelationType.USES, "AlphaZero uses PUCT selection"),
        ("alphazero", "value_network", RelationType.USES, "AlphaZero uses value network"),
        ("alphazero", "policy_network", RelationType.USES, "AlphaZero uses policy network"),
        ("muzero", "alphazero", RelationType.EXTENDS, "MuZero extends AlphaZero"),
        ("muzero", "neural_network", RelationType.USES, "MuZero uses neural networks"),
        ("value_network", "neural_network", RelationType.IS_A, "Value network is a neural network"),
        ("policy_network", "neural_network", RelationType.IS_A, "Policy network is a neural network"),
    ]

    for source, target, rel_type, desc in relationships:
        builder.add_relationship(source, target, rel_type, properties={"description": desc}, confidence=0.95)
        logger.info(f"Added relationship: {source} --[{rel_type.value}]--> {target}")

    # Save the graph
    builder.save()
    logger.info(f"\nSaved graph to {config['storage']}")

    return builder


def example_2_query_graph(builder: KnowledgeGraphBuilder):
    """Example 2: Query the knowledge graph."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 2: Querying the Knowledge Graph")
    logger.info("=" * 80)

    query_engine = GraphQueryEngine(builder)

    # Query 1: Find a concept
    logger.info("\n--- Query 1: Find concept 'AlphaZero' ---")
    concept = query_engine.find_concept("AlphaZero")
    if concept:
        logger.info(f"Name: {concept.name}")
        logger.info(f"Type: {concept.type}")
        logger.info(f"Description: {concept.description}")
        logger.info(f"Properties: {concept.properties}")

    # Query 2: Get all relationships for AlphaZero
    logger.info("\n--- Query 2: All relationships for AlphaZero ---")
    rels = query_engine.get_relationships("alphazero", direction="both")
    for rel in rels:
        logger.info(f"  {rel['source']} --[{rel['relation']}]--> {rel['target']}")

    # Query 3: Find only "USES" relationships
    logger.info("\n--- Query 3: What does AlphaZero use? ---")
    uses_rels = query_engine.get_relationships("alphazero", relation_type=RelationType.USES)
    for rel in uses_rels:
        target_concept = query_engine.find_concept(rel["target"])
        logger.info(f"  - {target_concept.name}: {target_concept.description}")

    # Query 4: Find path between concepts
    logger.info("\n--- Query 4: Path from MuZero to UCB1 ---")
    paths = query_engine.find_path("muzero", "ucb1")
    if paths:
        logger.info(f"Found {len(paths)} path(s)")
        shortest = paths[0]
        path_str = " -> ".join([query_engine.concepts[node].name for node in shortest])
        logger.info(f"Shortest path: {path_str}")

    # Query 5: Get related concepts
    logger.info("\n--- Query 5: Concepts related to MCTS (depth=2) ---")
    related = query_engine.get_related_concepts("mcts", depth=2)
    logger.info(f"Root: {related['root']['name']}")
    for depth, concepts in related["related"].items():
        logger.info(f"\nDepth {depth}:")
        for item in concepts:
            logger.info(f"  - {item['concept']['name']} ({item['relation']})")

    # Query 6: Get statistics
    logger.info("\n--- Query 6: Graph Statistics ---")
    stats = query_engine.get_statistics()
    logger.info(f"Total concepts: {stats['total_concepts']}")
    logger.info(f"Total relationships: {stats['total_relationships']}")
    logger.info(f"Concept types: {dict(stats['concept_types'])}")
    logger.info(f"Relation types: {dict(stats['relation_types'])}")
    logger.info(f"Average degree: {stats['avg_degree']:.2f}")


async def example_3_qa_system(builder: KnowledgeGraphBuilder):
    """Example 3: Question answering with the knowledge graph."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 3: Graph-based Question Answering")
    logger.info("=" * 80)

    query_engine = GraphQueryEngine(builder)
    qa = GraphQA(query_engine)

    questions = [
        "What is AlphaZero?",
        "How does AlphaZero differ from MCTS?",
        "What is the relationship between MuZero and UCB1?",
        "What does AlphaZero use?",
    ]

    for question in questions:
        logger.info(f"\nQuestion: {question}")
        result = await qa.answer(question)
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Confidence: {result['confidence']}")
        logger.info(f"Entities identified: {result['entities']}")
        logger.info(f"Question type: {result['question_type']}")


async def example_4_extract_from_paper():
    """Example 4: Extract knowledge from a research paper."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 4: Extract Knowledge from Research Paper")
    logger.info("=" * 80)

    # Note: Requires OPENAI_API_KEY environment variable
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set, skipping extraction example")
        return None

    config = {
        "llm_model": "gpt-4-turbo-preview",
        "confidence_threshold": 0.7,
    }

    extractor = KnowledgeExtractor(config)

    # Sample paper (simplified AlphaZero abstract)
    paper = {
        "id": "arxiv:1712.01815",
        "title": "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm",
        "abstract": """
        The game of chess is the most widely-studied domain in the history of artificial intelligence.
        The strongest programs are based on a combination of sophisticated search techniques,
        domain-specific adaptations, and handcrafted evaluation functions. In this paper, we introduce
        AlphaZero, which achieves superhuman performance in chess using only self-play reinforcement
        learning starting from random play, without any domain knowledge except the game rules.
        AlphaZero uses a general-purpose Monte Carlo tree search (MCTS) algorithm combined with a
        deep neural network that has been trained by self-play reinforcement learning. The neural
        network takes the board position as input and outputs move probabilities and a position
        evaluation. AlphaZero learns through self-play, starting from completely random play and
        gradually improving through repeated games against itself.
        """,
    }

    logger.info(f"Extracting from paper: {paper['title']}")

    concepts, relationships = await extractor.extract_from_paper(paper["id"], paper["title"], paper["abstract"])

    logger.info(f"\nExtracted {len(concepts)} concepts:")
    for concept in concepts:
        logger.info(f"  - {concept.name} ({concept.type}): {concept.description[:100]}...")

    logger.info(f"\nExtracted {len(relationships)} relationships:")
    for rel in relationships:
        logger.info(f"  - {rel.source} --[{rel.relation_type.value}]--> {rel.target} (conf: {rel.confidence:.2f})")

    return concepts, relationships


def example_5_build_from_papers():
    """Example 5: Build graph from multiple papers."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 5: Build Graph from Multiple Papers")
    logger.info("=" * 80)

    config = {
        "backend": "networkx",
        "storage": "./cache/knowledge_graph_papers",
        "extraction": {
            "llm_model": "gpt-4-turbo-preview",
            "confidence_threshold": 0.7,
        },
    }

    # Sample papers
    papers = [
        {
            "id": "arxiv:1712.01815",
            "title": "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm",
            "abstract": "AlphaZero uses Monte Carlo Tree Search combined with deep neural networks trained by self-play.",
        },
        {
            "id": "arxiv:1911.08265",
            "title": "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model",
            "abstract": "MuZero extends AlphaZero by learning a model of the environment without explicit knowledge of game rules.",
        },
    ]

    # Build graph
    builder = KnowledgeGraphBuilder(config)
    extractor = KnowledgeExtractor(config.get("extraction", {}))

    stats = builder.build_from_corpus(extractor, papers)

    logger.info("\nGraph building complete:")
    logger.info(f"  Concepts: {stats['total_concepts']}")
    logger.info(f"  Relationships: {stats['total_relationships']}")
    logger.info(f"  Papers processed: {stats['papers_processed']}")

    builder.save()
    return builder


def example_6_hybrid_retrieval():
    """Example 6: Hybrid retrieval combining vector search and graph traversal."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 6: Hybrid Retrieval (Vector + Graph)")
    logger.info("=" * 80)

    # Note: This example requires integration with vector index from rag_builder.py
    logger.info("This example requires a vector index from rag_builder.py")
    logger.info("For a complete implementation, integrate with VectorIndexBuilder:")

    logger.info("""
    from training.rag_builder import VectorIndexBuilder
    from training.knowledge_graph import HybridKnowledgeRetriever

    # Initialize components
    vector_index = VectorIndexBuilder(rag_config)
    query_engine = GraphQueryEngine(graph_builder)

    # Create hybrid retriever
    hybrid_config = {
        "expansion_depth": 2,
        "vector_weight": 0.6,
        "graph_weight": 0.4,
    }
    retriever = HybridKnowledgeRetriever(query_engine, vector_index, hybrid_config)

    # Retrieve with graph expansion
    results = retriever.retrieve("How does AlphaZero work?", k=10)

    # Results include both vector-similar documents and graph-related concepts
    for result in results:
        print(f"Score: {result['score']:.3f}")
        print(f"Text: {result['text'][:200]}...")
        print(f"Relationships: {len(result['relationships'])}")
    """)


def example_7_advanced_queries():
    """Example 7: Advanced graph queries and analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 7: Advanced Graph Queries")
    logger.info("=" * 80)

    # Load existing graph
    config = {"backend": "networkx", "storage": "./cache/knowledge_graph_examples"}
    builder = KnowledgeGraphBuilder(config)
    builder.load()

    query_engine = GraphQueryEngine(builder)

    # Query 1: Find all variants of MCTS
    logger.info("\n--- Find all algorithms that extend or improve MCTS ---")
    mcts_variants = []

    # Find concepts that have IS_A or EXTENDS relationship to MCTS
    for concept_id, concept in query_engine.concepts.items():
        rels = query_engine.get_relationships(concept_id, direction="outgoing")
        for rel in rels:
            if rel["target"] == "mcts" and rel["relation"] in ["extends", "improves", "is_a"]:
                mcts_variants.append(concept.name)

    logger.info(f"MCTS variants: {mcts_variants}")

    # Query 2: Find all components of AlphaZero
    logger.info("\n--- What components does AlphaZero use? ---")
    alphazero_components = query_engine.get_relationships("alphazero", relation_type=RelationType.USES)

    for rel in alphazero_components:
        component = query_engine.find_concept(rel["target"])
        if component:
            logger.info(f"  - {component.name}: {component.description}")

    # Query 3: Compare AlphaZero and MuZero
    logger.info("\n--- Compare AlphaZero and MuZero ---")

    az_rels = {(r["relation"], r["target"]) for r in query_engine.get_relationships("alphazero", direction="outgoing")}
    mz_rels = {(r["relation"], r["target"]) for r in query_engine.get_relationships("muzero", direction="outgoing")}

    common = az_rels & mz_rels
    az_only = az_rels - mz_rels
    mz_only = mz_rels - az_rels

    logger.info(f"Common: {len(common)} relationships")
    for _rel, target in list(common)[:5]:
        concept = query_engine.find_concept(target)
        if concept:
            logger.info(f"  Both use: {concept.name}")

    logger.info(f"\nAlphaZero unique: {len(az_only)} relationships")
    logger.info(f"MuZero unique: {len(mz_only)} relationships")


def example_8_export_visualization():
    """Example 8: Export graph for visualization."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 8: Export Graph for Visualization")
    logger.info("=" * 80)

    config = {"backend": "networkx", "storage": "./cache/knowledge_graph_examples"}
    builder = KnowledgeGraphBuilder(config)
    builder.load()

    # Export to GraphML (can be opened in Gephi, yEd, etc.)
    import networkx as nx

    output_path = Path("./cache/knowledge_graph_examples/graph_export.graphml")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nx.write_graphml(builder.graph, str(output_path))
    logger.info(f"Exported graph to GraphML: {output_path}")

    # Export to JSON for D3.js visualization
    graph_data = nx.node_link_data(builder.graph)

    json_path = Path("./cache/knowledge_graph_examples/graph_export.json")
    import json

    with open(json_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    logger.info(f"Exported graph to JSON: {json_path}")

    # Generate simple DOT format for Graphviz
    dot_path = Path("./cache/knowledge_graph_examples/graph_export.dot")
    with open(dot_path, "w") as f:
        f.write("digraph KnowledgeGraph {\n")
        f.write("  rankdir=LR;\n")
        f.write("  node [shape=box, style=rounded];\n\n")

        # Write nodes
        for node_id, node_data in builder.graph.nodes(data=True):
            label = node_data.get("name", node_id)
            f.write(f'  "{node_id}" [label="{label}"];\n')

        f.write("\n")

        # Write edges
        for source, target, edge_data in builder.graph.edges(data=True):
            relation = edge_data.get("relation", "related")
            f.write(f'  "{source}" -> "{target}" [label="{relation}"];\n')

        f.write("}\n")

    logger.info(f"Exported graph to DOT: {dot_path}")
    logger.info("You can visualize with: dot -Tpng graph_export.dot -o graph.png")


async def main():
    """Run all examples."""
    logger.info("=" * 80)
    logger.info("Knowledge Graph System - Examples")
    logger.info("=" * 80)

    # Example 1: Manual construction
    builder = example_1_manual_graph_construction()

    # Example 2: Querying
    example_2_query_graph(builder)

    # Example 3: Question answering
    await example_3_qa_system(builder)

    # Example 4: Extract from paper (requires OpenAI API key)
    await example_4_extract_from_paper()

    # Example 5: Build from multiple papers (requires OpenAI API key)
    # Uncomment to run:
    # example_5_build_from_papers()

    # Example 6: Hybrid retrieval
    example_6_hybrid_retrieval()

    # Example 7: Advanced queries
    example_7_advanced_queries()

    # Example 8: Export for visualization
    example_8_export_visualization()

    logger.info("\n" + "=" * 80)
    logger.info("All examples completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
