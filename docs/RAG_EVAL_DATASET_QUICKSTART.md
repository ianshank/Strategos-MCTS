# RAG Evaluation Dataset - Quick Start Guide

## What Was Created

A LangSmith dataset named **`rag-eval-dataset`** with **26 comprehensive Q&A examples** about MCTS, AlphaZero, and game tree search.

## Dataset ID
```
2c72f9fe-ca6d-497c-ba89-d689ea6afb17
```

## Quick Run

### Prerequisites
```bash
# Ensure you have the API key set
export LANGSMITH_API_KEY=your_key_here
```

### Run the Script
```bash
python scripts/create_rag_eval_datasets.py
```

## What's in the Dataset

Each of the 26 examples includes:
- **Question**: A specific MCTS/AlphaZero/game tree question
- **Contexts**: 2-4 relevant text snippets with answer information
- **Ground Truth**: The complete, accurate answer

### Example Entry
```json
{
    "inputs": {
        "question": "How does UCB1 balance exploration and exploitation in MCTS?",
        "contexts": [
            "UCB1 (Upper Confidence Bound 1) is a key formula in MCTS...",
            "The UCB1 formula is: UCB1 = Q/N + C * sqrt(ln(N_parent)/N)...",
            "The first term (Q/N) represents exploitation...",
            "As a node is visited more, the exploration term decreases..."
        ]
    },
    "outputs": {
        "ground_truth": "UCB1 balances exploration and exploitation through..."
    }
}
```

## Topics Covered

1. **Basic MCTS** (6 questions)
   - What is MCTS?
   - Four phases of MCTS
   - Game trees
   - State/action spaces

2. **Exploration/Exploitation** (5 questions)
   - UCB1 algorithm
   - PUCT formula
   - Exploration constant
   - First Play Urgency

3. **AlphaZero** (8 questions)
   - Architecture overview
   - Policy/value networks
   - Self-play training
   - Neural network integration

4. **Advanced Techniques** (7 questions)
   - RAVE
   - Progressive widening
   - Virtual loss
   - Regret minimization

## Use Cases

### 1. Test Your RAG System
```python
from langsmith import Client

client = Client()
dataset = client.read_dataset(dataset_name="rag-eval-dataset")

for example in dataset.examples:
    question = example.inputs["question"]
    contexts = example.inputs["contexts"]
    ground_truth = example.outputs["ground_truth"]

    # Run your RAG system
    generated_answer = your_rag_system(question, contexts)

    # Compare with ground truth
    evaluate(generated_answer, ground_truth)
```

### 2. Benchmark Retrieval Quality
```python
# Test if your retriever finds the right contexts
for example in dataset.examples:
    question = example.inputs["question"]
    expected_contexts = example.inputs["contexts"]

    # Run retrieval
    retrieved = your_retriever.search(question, k=4)

    # Measure precision/recall
    evaluate_retrieval(retrieved, expected_contexts)
```

### 3. Run LangSmith Evaluation
```python
from langsmith import evaluate

def rag_pipeline(inputs):
    question = inputs["question"]
    # Your RAG implementation
    answer = generate_answer(question)
    return {"answer": answer}

results = evaluate(
    rag_pipeline,
    data="rag-eval-dataset",
    evaluators=[correctness, relevancy, faithfulness],
    experiment_prefix="rag-eval"
)
```

## View in LangSmith UI

1. Go to https://smith.langchain.com/
2. Navigate to "Datasets"
3. Find "rag-eval-dataset"
4. Explore the 26 examples with their questions, contexts, and ground truth answers

## Key Features

- **High-Quality Ground Truth**: All answers are accurate and comprehensive
- **Multi-Context Scenarios**: Tests ability to synthesize information from multiple sources
- **Diverse Question Types**: Definitional, comparative, explanatory, procedural
- **Technical Depth**: From basic concepts to advanced algorithms
- **Real-World Relevance**: Based on actual MCTS and AlphaZero implementations

## Files

- **Script**: `scripts/create_rag_eval_datasets.py`
- **Full Documentation**: `docs/RAG_EVAL_DATASET_SUMMARY.md`
- **Quick Start**: `docs/RAG_EVAL_DATASET_QUICKSTART.md` (this file)

## Next Steps

1. **View the dataset** in LangSmith UI
2. **Implement your RAG pipeline** for MCTS knowledge
3. **Run evaluations** to measure performance
4. **Iterate** on retrieval and generation strategies
5. **Benchmark** different embedding models and LLMs

## Sample Questions

Here are a few questions from the dataset:

1. What is Monte Carlo Tree Search?
2. How does UCB1 balance exploration and exploitation in MCTS?
3. What is the difference between MCTS and minimax?
4. What is AlphaZero and how does it differ from traditional MCTS?
5. What are the four phases of MCTS and what happens in each?
6. What is PUCT and how is it used in AlphaZero?
7. What is virtual loss in parallel MCTS and why is it important?
8. How does self-play training work in AlphaZero?
9. What are policy and value networks in AlphaZero?
10. What is RAVE (Rapid Action Value Estimation) in MCTS?

... and 16 more!

## Support

For issues or questions about the dataset:
1. Check the full documentation: `docs/RAG_EVAL_DATASET_SUMMARY.md`
2. Review the script: `scripts/create_rag_eval_datasets.py`
3. View examples in LangSmith UI

## Success Metrics

When evaluating your RAG system with this dataset, aim for:
- **Answer Correctness**: >85% factual accuracy
- **Context Precision**: >90% relevant contexts retrieved
- **Context Recall**: >80% necessary contexts found
- **Faithfulness**: >95% answer grounded in contexts
- **Answer Similarity**: >0.8 cosine similarity to ground truth

Happy evaluating!
