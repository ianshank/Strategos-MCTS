# Knowledge Graph System Documentation

## Overview

The Knowledge Graph System provides structured relationship management for AI/ML concepts in the MCTS training platform. It extracts entities and relationships from research papers and code, builds a queryable graph structure, and enables graph-based question answering and hybrid retrieval.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Knowledge Graph System                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐      ┌──────────────────┐             │
│  │   Knowledge    │─────>│  Graph Builder   │             │
│  │   Extractor    │      │   (NetworkX/     │             │
│  │   (LLM-based)  │      │    Neo4j)        │             │
│  └────────────────┘      └──────────────────┘             │
│          │                        │                         │
│          │                        │                         │
│          v                        v                         │
│  ┌────────────────┐      ┌──────────────────┐             │
│  │  Papers/Code   │      │  Graph Query     │             │
│  │  Processing    │      │  Engine          │             │
│  └────────────────┘      └──────────────────┘             │
│                                   │                         │
│                                   │                         │
│          ┌────────────────────────┼────────────┐           │
│          │                        │            │           │
│          v                        v            v           │
│  ┌──────────────┐      ┌─────────────┐  ┌──────────────┐ │
│  │   Hybrid     │      │   Graph QA  │  │  Export &    │ │
│  │   Retriever  │      │   System    │  │  Visualize   │ │
│  └──────────────┘      └─────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Schema Definitions

#### ConceptNode

Represents an AI/ML concept in the knowledge graph.

```python
@dataclass
class ConceptNode:
    id: str                      # Unique identifier (normalized)
    name: str                    # Display name
    type: str                    # algorithm, technique, architecture, metric
    description: str             # Brief description
    aliases: list[str]           # Alternative names
    properties: dict             # Additional metadata
    source_papers: list[str]     # arXiv IDs where concept appears
    code_references: list[str]   # File paths with implementations
    confidence: float            # Extraction confidence (0-1)
```

**Example:**
```python
alphazero = ConceptNode(
    id="alphazero",
    name="AlphaZero",
    type="architecture",
    description="Game-playing AI using MCTS with deep neural networks",
    aliases=["Alpha Zero"],
    properties={"year": 2017, "company": "DeepMind"},
    source_papers=["arxiv:1712.01815"],
    confidence=0.95
)
```

#### RelationType

Enum defining relationship types between concepts:

- **IS_A**: Inheritance/specialization (e.g., "PUCT IS_A UCB variant")
- **USES**: Component usage (e.g., "AlphaZero USES neural networks")
- **IMPROVES**: Enhancement (e.g., "PUCT IMPROVES UCB1")
- **EXTENDS**: Extension/building upon (e.g., "MuZero EXTENDS AlphaZero")
- **IMPLEMENTED_IN**: Code implementation (e.g., "UCB1 IMPLEMENTED_IN paper X")
- **COMPARED_TO**: Comparison relationship
- **REQUIRES**: Dependency (e.g., "AlphaZero REQUIRES self-play")
- **PART_OF**: Component relationship
- **RELATED_TO**: Generic relationship
- **INFLUENCES**: Influence relationship
- **PRECEDES**: Temporal/historical ordering

#### Relationship

```python
@dataclass
class Relationship:
    source: str                  # Source concept ID
    target: str                  # Target concept ID
    relation_type: RelationType  # Type of relationship
    properties: dict             # Additional metadata
    evidence: list[str]          # Supporting evidence (paper IDs)
    confidence: float            # Confidence score (0-1)
```

### 2. Knowledge Extractor

Extracts entities and relationships from research papers and code using LLM.

```python
extractor = KnowledgeExtractor({
    "llm_model": "gpt-4-turbo-preview",
    "confidence_threshold": 0.7,
    "api_key": "your-api-key"  # or set OPENAI_API_KEY env var
})

# Extract from paper
concepts, relationships = await extractor.extract_from_paper(
    paper_id="arxiv:1712.01815",
    title="Mastering Chess by Self-Play",
    abstract="AlphaZero uses MCTS with neural networks..."
)

# Extract from code
concepts, relationships = extractor.extract_from_code(
    file_path="mcts.py",
    code=source_code
)
```

**Features:**
- LLM-powered entity recognition
- Relationship extraction with confidence scores
- Pattern-based code analysis
- Automatic ID normalization
- Deduplication and merging

### 3. Graph Builder

Builds and manages the knowledge graph using NetworkX or Neo4j backend.

```python
builder = KnowledgeGraphBuilder({
    "backend": "networkx",  # or "neo4j"
    "storage": "./cache/knowledge_graph",
    "neo4j": {  # if using Neo4j
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "password"
    }
})

# Add concepts
builder.add_concept(concept_node)

# Add relationships
builder.add_relationship(
    source="alphazero",
    target="mcts",
    relation_type=RelationType.USES,
    properties={"description": "Uses for search"},
    confidence=0.95
)

# Build from corpus
stats = builder.build_from_corpus(extractor, papers)

# Save/load
builder.save()
builder.load()
```

**Backend Options:**

#### NetworkX (Development)
- In-memory graph
- Fast queries
- Easy serialization
- Good for < 100K nodes

#### Neo4j (Production)
- Persistent storage
- Cypher query language
- Scalable to millions of nodes
- Advanced graph analytics

### 4. Graph Query Engine

Query and traverse the knowledge graph.

```python
query_engine = GraphQueryEngine(builder)

# Find concept by name or alias
concept = query_engine.find_concept("AlphaZero")

# Get relationships
relationships = query_engine.get_relationships(
    "alphazero",
    relation_type=RelationType.USES,
    direction="outgoing"  # or "incoming", "both"
)

# Find path between concepts
paths = query_engine.find_path("muzero", "ucb1", max_depth=5)

# Get related concepts (BFS traversal)
related = query_engine.get_related_concepts(
    "mcts",
    depth=2,
    relation_filter=[RelationType.USES, RelationType.EXTENDS]
)

# Get statistics
stats = query_engine.get_statistics()
```

**Query Examples:**

```python
# 1. Find all MCTS variants
for concept in query_engine.concepts.values():
    rels = query_engine.get_relationships(concept.id)
    for rel in rels:
        if rel['target'] == 'mcts' and rel['relation'] in ['is_a', 'extends']:
            print(f"{concept.name} is a variant of MCTS")

# 2. What does AlphaZero use?
uses = query_engine.get_relationships("alphazero", RelationType.USES)
for rel in uses:
    component = query_engine.find_concept(rel['target'])
    print(f"Uses: {component.name}")

# 3. Find papers implementing UCB1
for concept in query_engine.concepts.values():
    if concept.id == "ucb1":
        print(f"Papers: {concept.source_papers}")
        print(f"Code: {concept.code_references}")

# 4. Compare two methods
az_rels = set(query_engine.get_relationships("alphazero"))
mz_rels = set(query_engine.get_relationships("muzero"))
common = az_rels & mz_rels
print(f"Common components: {common}")
```

### 5. Hybrid Retrieval

Combines vector search with graph traversal for enhanced retrieval.

```python
from training.rag_builder import VectorIndexBuilder

# Initialize components
vector_index = VectorIndexBuilder(rag_config)
query_engine = GraphQueryEngine(graph_builder)

# Create hybrid retriever
retriever = HybridKnowledgeRetriever(
    query_engine=query_engine,
    vector_index=vector_index,
    config={
        "expansion_depth": 2,
        "vector_weight": 0.6,
        "graph_weight": 0.4
    }
)

# Retrieve with graph expansion
results = retriever.retrieve("How does AlphaZero work?", k=10)
```

**How it works:**

1. **Vector Search**: Find initial relevant documents using semantic similarity
2. **Concept Extraction**: Identify AI/ML concepts mentioned in results
3. **Graph Expansion**: Traverse graph to find related concepts (up to `expansion_depth`)
4. **Re-ranking**: Combine vector similarity and graph relevance scores
5. **Enrichment**: Add relationship context to results

**Benefits:**
- Discovers related concepts not explicitly mentioned in query
- Provides structured relationship context
- Improves recall through graph traversal
- Better handles technical terminology

### 6. Graph-based Question Answering

Answer questions using graph reasoning.

```python
qa = GraphQA(query_engine)

result = await qa.answer("How does AlphaZero differ from MCTS?")

print(result['answer'])
print(f"Confidence: {result['confidence']}")
print(f"Entities: {result['entities']}")
print(f"Question type: {result['question_type']}")
```

**Supported Question Types:**

1. **Relationship Questions**
   - "What is the relationship between X and Y?"
   - "How does X connect to Y?"

2. **Property Questions**
   - "What is AlphaZero?"
   - "Define MCTS"

3. **Comparison Questions**
   - "How does X differ from Y?"
   - "Compare X and Y"

4. **General Questions**
   - "What are MCTS variants?"
   - "What does AlphaZero use?"

**Example Answers:**

```
Q: "How does AlphaZero differ from MCTS?"
A: AlphaZero extends MCTS by adding:
   - Neural networks for position evaluation
   - Self-play training
   - Policy and value networks
   - PUCT selection algorithm

Q: "What is the relationship between MuZero and UCB1?"
A: The connection is: MuZero → extends → AlphaZero → uses → MCTS → uses → UCB1

Q: "What is MCTS?"
A: MCTS (Monte Carlo Tree Search) is a probabilistic search algorithm
   for decision processes, commonly used in game playing and planning.
```

## Configuration

Add to `training/config.yaml`:

```yaml
knowledge_graph:
  backend: "networkx"  # or "neo4j"
  storage: "./cache/knowledge_graph"

  # Neo4j configuration (if using Neo4j backend)
  neo4j:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: "${NEO4J_PASSWORD}"

  # Extraction configuration
  extraction:
    llm_model: "gpt-4-turbo-preview"
    confidence_threshold: 0.7
    api_key: "${OPENAI_API_KEY}"  # or set as environment variable

  # Hybrid retrieval configuration
  hybrid_retrieval:
    expansion_depth: 2
    vector_weight: 0.6
    graph_weight: 0.4
```

## Integration Examples

### Integration with Research Corpus Builder

```python
from training.research_corpus_builder import ResearchCorpusBuilder
from training.knowledge_graph import KnowledgeGraphSystem

# Build corpus
corpus_builder = ResearchCorpusBuilder(config_path="training/config.yaml")

# Initialize knowledge graph
kg_system = KnowledgeGraphSystem(config_path="training/config.yaml")

# Extract papers
papers = []
for paper_metadata in corpus_builder.fetcher.fetch_papers_by_keywords():
    papers.append({
        "id": paper_metadata.arxiv_id,
        "title": paper_metadata.title,
        "abstract": paper_metadata.abstract,
    })

# Build knowledge graph
stats = kg_system.build_from_papers(papers)
kg_system.save()

print(f"Built graph with {stats['total_concepts']} concepts")
```

### Integration with RAG Builder

```python
from training.rag_builder import VectorIndexBuilder
from training.knowledge_graph import KnowledgeGraphSystem, HybridKnowledgeRetriever

# Load RAG index
vector_index = VectorIndexBuilder(rag_config)
vector_index.load_index()

# Load knowledge graph
kg_system = KnowledgeGraphSystem()
kg_system.load()

# Create hybrid retriever
retriever = HybridKnowledgeRetriever(
    query_engine=kg_system.query_engine,
    vector_index=vector_index,
    config=hybrid_config
)

# Enhanced retrieval
results = retriever.retrieve(
    "Explain how PUCT improves upon UCB1 in AlphaZero",
    k=10
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text'][:200]}...")
    print(f"Relationships: {len(result['relationships'])}")
```

### Integration with Training Pipeline

```python
from training.agent_trainer import AgentTrainer
from training.knowledge_graph import KnowledgeGraphSystem

# Initialize components
trainer = AgentTrainer(config_path="training/config.yaml")
kg_system = KnowledgeGraphSystem(config_path="training/config.yaml")

# Use knowledge graph to enhance training examples
def augment_training_examples(examples):
    augmented = []

    for example in examples:
        # Extract concepts from example
        concepts = extract_concepts(example['text'])

        # Get related concepts from graph
        related = []
        for concept in concepts:
            rel = kg_system.query_engine.get_related_concepts(concept, depth=1)
            related.extend(rel.get('related', {}).get(1, []))

        # Add relationship context to example
        example['graph_context'] = {
            'concepts': concepts,
            'related': related
        }

        augmented.append(example)

    return augmented

# Augment and train
augmented_examples = augment_training_examples(training_examples)
trainer.train(augmented_examples)
```

## Visualization

### Export for Gephi/yEd

```python
import networkx as nx

# Export to GraphML
nx.write_graphml(builder.graph, "knowledge_graph.graphml")
```

Open in:
- **Gephi**: Advanced visualization and analysis
- **yEd**: Automatic layout algorithms
- **Cytoscape**: Biological network visualization

### Export for D3.js

```python
import json
import networkx as nx

# Export to JSON
graph_data = nx.node_link_data(builder.graph)

with open("knowledge_graph.json", "w") as f:
    json.dump(graph_data, f, indent=2)
```

Use with D3.js force-directed graph layout.

### Export to Graphviz

```python
# Generate DOT file
with open("knowledge_graph.dot", "w") as f:
    f.write("digraph KnowledgeGraph {\n")
    f.write("  rankdir=LR;\n")

    for node_id, data in builder.graph.nodes(data=True):
        label = data.get('name', node_id)
        f.write(f'  "{node_id}" [label="{label}"];\n')

    for source, target, data in builder.graph.edges(data=True):
        relation = data.get('relation', 'related')
        f.write(f'  "{source}" -> "{target}" [label="{relation}"];\n')

    f.write("}\n")

# Render with Graphviz
# dot -Tpng knowledge_graph.dot -o knowledge_graph.png
```

## Performance Considerations

### NetworkX Backend

**Strengths:**
- Fast in-memory operations
- Simple API
- No external dependencies
- Easy serialization

**Limitations:**
- Memory bound (< 100K nodes recommended)
- No persistence
- Limited concurrent access

**Best for:**
- Development and prototyping
- Small to medium graphs
- Single-user applications

### Neo4j Backend

**Strengths:**
- Persistent storage
- Scales to millions of nodes
- Advanced query capabilities (Cypher)
- Concurrent access
- Built-in graph algorithms

**Limitations:**
- Requires Neo4j server
- More complex setup
- Higher latency for small queries

**Best for:**
- Production deployments
- Large-scale graphs
- Multi-user applications
- Complex graph analytics

### Optimization Tips

1. **Batch Operations**: Add concepts and relationships in batches
2. **Caching**: Cache frequently accessed concepts
3. **Indexing**: Create indices on frequently queried properties (Neo4j)
4. **Pruning**: Remove low-confidence relationships
5. **Lazy Loading**: Load graph subsets on demand

## Advanced Features

### Custom Extractors

Create domain-specific extractors:

```python
class CustomExtractor(KnowledgeExtractor):
    def extract_from_domain_text(self, text: str) -> tuple:
        # Custom extraction logic
        concepts = self._extract_domain_concepts(text)
        relationships = self._extract_domain_relations(text)
        return concepts, relationships
```

### Graph Analytics

Compute graph metrics:

```python
import networkx as nx

# Centrality measures
betweenness = nx.betweenness_centrality(builder.graph)
pagerank = nx.pagerank(builder.graph)

# Find important concepts
important = sorted(
    pagerank.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

# Community detection
communities = nx.community.greedy_modularity_communities(
    builder.graph.to_undirected()
)
```

### Temporal Analysis

Track concept evolution:

```python
# Filter relationships by date
def get_temporal_graph(builder, start_date, end_date):
    filtered = nx.MultiDiGraph()

    for source, target, data in builder.graph.edges(data=True):
        created = data.get('created_at', '')
        if start_date <= created <= end_date:
            filtered.add_edge(source, target, **data)

    return filtered

# Analyze concept emergence
yearly_graphs = {
    year: get_temporal_graph(builder, f"{year}-01-01", f"{year}-12-31")
    for year in range(2015, 2024)
}
```

## Troubleshooting

### Common Issues

**1. OpenAI API errors**
```
Error: OpenAI API key not found
Solution: Set OPENAI_API_KEY environment variable
```

**2. Neo4j connection errors**
```
Error: Could not connect to Neo4j
Solution: Ensure Neo4j is running and credentials are correct
```

**3. Memory errors with large graphs**
```
Error: MemoryError when loading graph
Solution: Use Neo4j backend or implement graph partitioning
```

**4. Low extraction quality**
```
Issue: Extracted concepts are too generic
Solution:
- Increase confidence_threshold
- Improve extraction prompts
- Use domain-specific extractors
```

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('training.knowledge_graph')
```

## Best Practices

1. **Start Small**: Build graph incrementally, test with small corpus first
2. **Validate Extractions**: Manually review initial extractions for quality
3. **Version Control**: Save graph snapshots with timestamps
4. **Monitor Confidence**: Track and visualize confidence scores
5. **Iterative Refinement**: Continuously improve extraction prompts
6. **Documentation**: Document custom relationships and properties
7. **Backup**: Regularly backup graph data
8. **Testing**: Write tests for critical graph operations

## Future Enhancements

- [ ] Multi-modal extraction (images, equations)
- [ ] Automatic relation discovery
- [ ] Temporal knowledge graphs
- [ ] Federated knowledge graphs
- [ ] Graph neural networks integration
- [ ] Real-time graph updates
- [ ] Collaborative graph building
- [ ] Version control for graphs

## References

- NetworkX Documentation: https://networkx.org/
- Neo4j Documentation: https://neo4j.com/docs/
- Knowledge Graphs: https://arxiv.org/abs/2003.02320
- Graph Neural Networks: https://arxiv.org/abs/1901.00596

## Support

For issues or questions:
- Check existing issues on GitHub
- Review example code in `examples/knowledge_graph_example.py`
- Enable debug logging for detailed error messages
- Consult the documentation above

## License

Part of the LangGraph Multi-Agent MCTS training system.
