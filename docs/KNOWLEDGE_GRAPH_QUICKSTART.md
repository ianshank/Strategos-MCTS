# Knowledge Graph System - Quick Start Guide

## Overview

This guide will help you get started with the Knowledge Graph System for MCTS/AI concepts in minutes.

## Installation

1. **Install Dependencies**

```bash
cd training
pip install -r requirements.txt
```

Required packages:
- `networkx>=3.0` - Graph data structures
- `openai>=1.0.0` - LLM for knowledge extraction
- `neo4j>=5.0.0` - Optional Neo4j driver
- `arxiv>=2.0.0` - arXiv API client

2. **Set Environment Variables**

For knowledge extraction from papers:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

For Neo4j backend (optional):
```bash
export NEO4J_PASSWORD="your-password"
```

## Quick Start Examples

### Example 1: Build a Simple Graph

```python
from training.knowledge_graph import (
    ConceptNode,
    KnowledgeGraphBuilder,
    GraphQueryEngine,
    RelationType,
)

# Initialize graph builder
config = {
    "backend": "networkx",
    "storage": "./cache/knowledge_graph",
}
builder = KnowledgeGraphBuilder(config)

# Add concepts
mcts = ConceptNode(
    id="mcts",
    name="MCTS",
    type="algorithm",
    description="Monte Carlo Tree Search algorithm",
    aliases=["Monte Carlo Tree Search"],
)

alphazero = ConceptNode(
    id="alphazero",
    name="AlphaZero",
    type="architecture",
    description="Game-playing AI using MCTS and neural networks",
)

builder.add_concept(mcts)
builder.add_concept(alphazero)

# Add relationships
builder.add_relationship(
    "alphazero",
    "mcts",
    RelationType.USES,
    confidence=0.95
)

# Save graph
builder.save()
print("Graph created and saved!")
```

### Example 2: Query the Graph

```python
from training.knowledge_graph import GraphQueryEngine

# Load graph
builder.load()
query_engine = GraphQueryEngine(builder)

# Find a concept
concept = query_engine.find_concept("AlphaZero")
print(f"Found: {concept.name} - {concept.description}")

# Get relationships
relationships = query_engine.get_relationships("alphazero")
for rel in relationships:
    print(f"{rel['source']} --[{rel['relation']}]--> {rel['target']}")

# Find path between concepts
paths = query_engine.find_path("alphazero", "mcts")
if paths:
    print(f"Path found: {paths[0]}")
```

### Example 3: Question Answering

```python
import asyncio
from training.knowledge_graph import GraphQA

# Initialize QA system
qa = GraphQA(query_engine)

# Ask questions
async def ask_questions():
    questions = [
        "What is AlphaZero?",
        "How does AlphaZero relate to MCTS?",
        "What are MCTS variants?",
    ]

    for question in questions:
        result = await qa.answer(question)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['confidence']}")

asyncio.run(ask_questions())
```

### Example 4: Extract from Papers

```python
from training.knowledge_graph import KnowledgeExtractor

# Initialize extractor
config = {
    "llm_model": "gpt-4-turbo-preview",
    "confidence_threshold": 0.7,
}
extractor = KnowledgeExtractor(config)

# Sample paper
paper = {
    "id": "arxiv:1712.01815",
    "title": "Mastering Chess and Shogi by Self-Play",
    "abstract": "AlphaZero uses Monte Carlo Tree Search...",
}

# Extract knowledge
concepts, relationships = await extractor.extract_from_paper(
    paper['id'],
    paper['title'],
    paper['abstract']
)

print(f"Extracted {len(concepts)} concepts")
print(f"Extracted {len(relationships)} relationships")
```

### Example 5: Hybrid Retrieval (Vector + Graph)

```python
from training.rag_builder import VectorIndexBuilder
from training.knowledge_graph import HybridKnowledgeRetriever

# Initialize vector index and query engine
vector_index = VectorIndexBuilder(rag_config)
vector_index.load_index()

# Create hybrid retriever
retriever = HybridKnowledgeRetriever(
    query_engine=query_engine,
    vector_index=vector_index,
    config={
        "expansion_depth": 2,
        "vector_weight": 0.6,
        "graph_weight": 0.4,
    }
)

# Retrieve with graph expansion
results = retriever.retrieve("How does AlphaZero work?", k=10)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text'][:200]}...")
    print(f"Relationships: {len(result['relationships'])}")
```

### Example 6: Build from Multiple Papers

```python
from training.knowledge_graph import KnowledgeGraphSystem

# Initialize system
kg_system = KnowledgeGraphSystem(config_path="training/config.yaml")

# Sample papers
papers = [
    {
        "id": "arxiv:1712.01815",
        "title": "Mastering Chess with AlphaZero",
        "abstract": "AlphaZero uses MCTS...",
    },
    {
        "id": "arxiv:1911.08265",
        "title": "MuZero: Mastering Atari",
        "abstract": "MuZero extends AlphaZero...",
    },
]

# Build knowledge graph
stats = kg_system.build_from_papers(papers)
print(f"Built graph with {stats['total_concepts']} concepts")

# Save
kg_system.save()
```

## Running the Examples

Run the comprehensive examples:

```bash
cd training/examples
python knowledge_graph_example.py
```

This will run all examples demonstrating:
1. Manual graph construction
2. Querying the graph
3. Question answering
4. Knowledge extraction (requires OpenAI API key)
5. Hybrid retrieval
6. Advanced queries
7. Export for visualization

## Configuration

Edit `training/config.yaml` to customize:

```yaml
knowledge_graph:
  backend: "networkx"  # or "neo4j"
  storage: "./cache/knowledge_graph"

  extraction:
    llm_model: "gpt-4-turbo-preview"
    confidence_threshold: 0.7

  hybrid_retrieval:
    expansion_depth: 2
    vector_weight: 0.6
    graph_weight: 0.4
```

## Common Use Cases

### 1. Build Graph from arXiv Papers

```python
from training.research_corpus_builder import ResearchCorpusBuilder
from training.knowledge_graph import KnowledgeGraphSystem

# Fetch papers
corpus_builder = ResearchCorpusBuilder(config_path="training/config.yaml")
papers = []

for paper_metadata in corpus_builder.fetcher.fetch_papers_by_keywords():
    papers.append({
        "id": paper_metadata.arxiv_id,
        "title": paper_metadata.title,
        "abstract": paper_metadata.abstract,
    })

# Build knowledge graph
kg_system = KnowledgeGraphSystem(config_path="training/config.yaml")
stats = kg_system.build_from_papers(papers)
kg_system.save()
```

### 2. Query Relationships

```python
# Find all variants of MCTS
variants = []
for concept_id, concept in query_engine.concepts.items():
    rels = query_engine.get_relationships(concept_id)
    for rel in rels:
        if rel['target'] == 'mcts' and rel['relation'] in ['extends', 'improves']:
            variants.append(concept.name)

print(f"MCTS variants: {variants}")
```

### 3. Export for Visualization

```python
import networkx as nx

# Export to GraphML (for Gephi, yEd)
nx.write_graphml(builder.graph, "knowledge_graph.graphml")

# Export to JSON (for D3.js)
import json
graph_data = nx.node_link_data(builder.graph)
with open("knowledge_graph.json", "w") as f:
    json.dump(graph_data, f, indent=2)

# Export to DOT (for Graphviz)
with open("knowledge_graph.dot", "w") as f:
    f.write("digraph KnowledgeGraph {\n")
    for node_id, data in builder.graph.nodes(data=True):
        f.write(f'  "{node_id}" [label="{data.get("name", node_id)}"];\n')
    for source, target, data in builder.graph.edges(data=True):
        f.write(f'  "{source}" -> "{target}" [label="{data.get("relation")}"];\n')
    f.write("}\n")

# Render: dot -Tpng knowledge_graph.dot -o knowledge_graph.png
```

### 4. Integrate with RAG Pipeline

```python
# Build knowledge graph from RAG documents
from training.rag_builder import VectorIndexBuilder

# Load RAG index
vector_index = VectorIndexBuilder(rag_config)
vector_index.load_index()

# Extract concepts from RAG documents
extractor = KnowledgeExtractor(extraction_config)
for chunk in vector_index.chunk_store[:100]:  # Sample first 100
    concepts, rels = extractor.extract_from_code(
        chunk['doc_id'],
        chunk['text']
    )
    for concept in concepts:
        builder.add_concept(concept)
    for rel in rels:
        builder.add_relationship(
            rel.source,
            rel.target,
            rel.relation_type,
            rel.properties,
            rel.confidence
        )

builder.save()
```

## Testing

Run the test suite:

```bash
cd training
pytest tests/test_knowledge_graph.py -v
```

Test specific functionality:

```bash
# Test concept creation
pytest tests/test_knowledge_graph.py::TestConceptNode -v

# Test graph building
pytest tests/test_knowledge_graph.py::TestKnowledgeGraphBuilder -v

# Test queries
pytest tests/test_knowledge_graph.py::TestGraphQueryEngine -v

# Test QA
pytest tests/test_knowledge_graph.py::TestGraphQA -v
```

## Troubleshooting

### Issue: OpenAI API errors

**Error**: `OpenAI API key not found`

**Solution**:
```bash
export OPENAI_API_KEY="your-api-key"
```

### Issue: Neo4j connection errors

**Error**: `Could not connect to Neo4j`

**Solution**:
1. Ensure Neo4j is running: `systemctl status neo4j`
2. Check credentials in config.yaml
3. Verify URI is correct

### Issue: Memory errors with large graphs

**Error**: `MemoryError when loading graph`

**Solution**:
1. Use Neo4j backend instead of NetworkX
2. Enable graph partitioning
3. Increase system memory
4. Process papers in smaller batches

### Issue: Low extraction quality

**Problem**: Extracted concepts are too generic

**Solution**:
1. Increase `confidence_threshold` in config
2. Improve extraction prompts in `KnowledgeExtractor`
3. Use domain-specific extractors
4. Fine-tune LLM on domain data

## Next Steps

1. **Read the full documentation**: `training/docs/KNOWLEDGE_GRAPH.md`
2. **Run examples**: `python training/examples/knowledge_graph_example.py`
3. **Integrate with your pipeline**: See integration examples above
4. **Customize extraction**: Modify prompts in `KnowledgeExtractor`
5. **Scale to production**: Switch to Neo4j backend

## Resources

- **Main Implementation**: `training/knowledge_graph.py`
- **Examples**: `training/examples/knowledge_graph_example.py`
- **Documentation**: `training/docs/KNOWLEDGE_GRAPH.md`
- **Tests**: `training/tests/test_knowledge_graph.py`
- **Configuration**: `training/config.yaml` (knowledge_graph section)

## Support

For issues or questions:
- Check the documentation in `training/docs/KNOWLEDGE_GRAPH.md`
- Review example code in `training/examples/knowledge_graph_example.py`
- Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
- Run tests to verify installation: `pytest tests/test_knowledge_graph.py`

## Performance Tips

1. **Batch Processing**: Process papers in batches of 10-50
2. **Caching**: Enable query caching for frequent queries
3. **Parallel Extraction**: Use `max_workers` setting for parallel processing
4. **Incremental Building**: Build graph incrementally, save frequently
5. **Memory Management**: Clear cache periodically for long-running processes
6. **Neo4j for Scale**: Switch to Neo4j for > 10K nodes

## Example Output

```
Building knowledge graph from 100 papers...
Extracted 523 concepts
Extracted 1,247 relationships
Graph building complete

Graph statistics:
  Total concepts: 523
  Total relationships: 1,247
  Concept types: {'algorithm': 234, 'technique': 156, 'architecture': 89, ...}
  Relation types: {'uses': 445, 'extends': 203, 'improves': 178, ...}
  Average degree: 4.77
  Connected components: 12

Graph saved to ./cache/knowledge_graph/knowledge_graph.pkl
```

## License

Part of the LangGraph Multi-Agent MCTS training system.
