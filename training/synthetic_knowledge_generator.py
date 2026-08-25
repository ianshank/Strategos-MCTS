"""
Synthetic Knowledge Generator for Multi-Agent MCTS Training.

This module uses LLMs to generate high-quality training data at scale for:
- MCTS algorithm questions (UCB1, PUCT, exploration/exploitation)
- LangGraph workflow questions
- Multi-agent coordination scenarios
- Code debugging and implementation questions
- System design questions

Features:
- Async/parallel generation for efficiency
- Quality control and validation
- Cost tracking for API calls
- Progress tracking and resumability
- Configurable via training/config.yaml
- LangSmith dataset format output
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
from typing import Any

from tqdm import tqdm
import yaml

# Import LLM adapters
from src.adapters.llm import create_client, create_client_from_config
from src.adapters.llm.base import LLMResponse
from src.adapters.llm.exceptions import (
    LLMClientError,
    LLMQuotaExceededError,
    LLMRateLimitError,
)

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """Represents a generated question-answer pair."""

    question: str
    answer: str
    contexts: list[str]
    metadata: dict[str, Any]
    quality_score: float = 0.0
    reasoning_paths: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_langsmith_format(self) -> dict[str, Any]:
        """Convert to LangSmith dataset format."""
        return {
            "inputs": {
                "question": self.question,
                "contexts": self.contexts,
            },
            "outputs": {
                "ground_truth": self.answer,
            },
            "metadata": {
                **self.metadata,
                "quality_score": self.quality_score,
                "generated_at": self.generated_at,
                "num_reasoning_paths": len(self.reasoning_paths),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "metadata": self.metadata,
            "quality_score": self.quality_score,
            "reasoning_paths": self.reasoning_paths,
            "generated_at": self.generated_at,
        }


# Comprehensive question templates organized by category
QUESTION_TEMPLATES = {
    "mcts_algorithms": [
        "Explain {algorithm} step by step with examples",
        "How does {algorithm} work in the context of {domain}?",
        "What are the key differences between {method_a} and {method_b}?",
        "Implement {algorithm} in Python with {constraints}",
        "What are the theoretical guarantees of {algorithm}?",
        "How do you tune the hyperparameters of {algorithm}?",
        "What are the computational complexity and memory requirements of {algorithm}?",
        "How does {algorithm} handle {specific_challenge}?",
        "Compare and contrast {algorithm} with traditional approaches to {problem}",
        "What are the advantages and limitations of using {algorithm} for {application}?",
    ],
    "exploration_exploitation": [
        "How does UCB1 balance exploration and exploitation in {context}?",
        "Explain the exploration constant C in {algorithm} and how to choose it",
        "What is the regret bound for {algorithm} and what does it mean?",
        "How does {method} ensure sufficient exploration while maximizing reward?",
        "Compare exploration strategies in {method_a} vs {method_b}",
        "What role does the visit count play in {algorithm}'s exploration strategy?",
        "How does temperature affect exploration in {algorithm}?",
        "Explain the exploration-exploitation tradeoff in multi-armed bandits",
    ],
    "alphazero_neural": [
        "How does AlphaZero combine MCTS with neural networks?",
        "Explain the policy and value networks in AlphaZero",
        "What is PUCT and how is it different from UCB1?",
        "How does AlphaZero's self-play training work?",
        "What is the role of Dirichlet noise in AlphaZero?",
        "How does AlphaZero handle {game_specific_feature}?",
        "Explain knowledge distillation from MCTS to neural networks in AlphaZero",
        "What neural network architecture does AlphaZero use and why?",
        "How does AlphaZero's training differ from supervised learning?",
        "What is the replay buffer in AlphaZero and why is it important?",
    ],
    "langgraph_workflows": [
        "How do you implement a {workflow_type} workflow in LangGraph?",
        "What are the key components of a LangGraph state machine?",
        "How does LangGraph handle conditional edges and branching?",
        "Explain how to use {feature} in LangGraph",
        "What are the best practices for error handling in LangGraph workflows?",
        "How do you implement cycles and loops in LangGraph?",
        "Compare LangGraph with {alternative_framework} for {use_case}",
        "How do you optimize LangGraph performance for {scenario}?",
        "What are the state management patterns in LangGraph?",
    ],
    "multi_agent_coordination": [
        "Design a multi-agent system for {task} using {framework}",
        "How do agents communicate and share state in {system}?",
        "What coordination patterns are used in {system_type} systems?",
        "How do you handle conflicts between agents in {scenario}?",
        "Explain the hierarchical reasoning model (HRM) approach to {problem}",
        "What is task refinement in multi-agent systems and why is it useful?",
        "How do you implement a meta-controller for agent routing?",
        "What are the tradeoffs between centralized vs decentralized coordination?",
        "How do you ensure consensus in multi-agent decision making?",
    ],
    "code_implementation": [
        "Implement {algorithm} in Python with proper error handling",
        "Write a {component} class that supports {features}",
        "Debug this {language} code that implements {algorithm}: {code_snippet}",
        "Optimize this implementation of {algorithm} for {constraint}",
        "Add {feature} to this existing {component} implementation",
        "Write unit tests for a {component} that {functionality}",
        "Refactor this code to follow {pattern} design pattern",
    ],
    "system_design": [
        "Design a scalable {system_type} system for {use_case}",
        "What are the architectural components of {system}?",
        "How would you scale {system} to handle {scale}?",
        "What data structures are appropriate for {problem}?",
        "Design the API for a {service} that {functionality}",
        "How do you handle fault tolerance in {system}?",
        "What monitoring and observability features should {system} have?",
    ],
    "advanced_mcts": [
        "What is RAVE and how does it improve MCTS performance?",
        "Explain progressive widening and when to use it",
        "How does virtual loss work in parallel MCTS?",
        "What is First Play Urgency (FPU) and why is it needed?",
        "How do transposition tables work with MCTS?",
        "Explain the difference between tree policy and rollout policy",
        "What are the convergence properties of MCTS?",
        "How do you adapt MCTS for continuous action spaces?",
    ],
    "practical_applications": [
        "How would you apply MCTS to {real_world_problem}?",
        "What are the challenges of using {algorithm} in production?",
        "How do you monitor and debug {system} in production?",
        "What metrics should you track for {algorithm}?",
        "How do you A/B test improvements to {system}?",
        "What are common failure modes of {algorithm} and how to mitigate them?",
    ],
}

# Domain-specific vocabularies for template filling
DOMAIN_VOCABULARIES = {
    "algorithm": [
        "UCB1",
        "PUCT",
        "MCTS",
        "AlphaZero",
        "Monte Carlo Tree Search",
        "Upper Confidence Bound",
        "progressive widening",
        "RAVE",
    ],
    "method_a": [
        "UCB1",
        "MCTS with random rollouts",
        "minimax search",
        "alpha-beta pruning",
        "neural MCTS",
    ],
    "method_b": [
        "PUCT",
        "MCTS with neural networks",
        "MCTS",
        "beam search",
        "traditional MCTS",
    ],
    "domain": [
        "game playing",
        "reinforcement learning",
        "decision making",
        "planning under uncertainty",
        "multi-agent systems",
    ],
    "constraints": [
        "memory-efficient data structures",
        "time limits under 100ms",
        "support for parallel execution",
        "GPU acceleration",
        "streaming input data",
    ],
    "specific_challenge": [
        "very large branching factors",
        "continuous action spaces",
        "partial observability",
        "sparse rewards",
        "adversarial environments",
    ],
    "problem": [
        "exploration-exploitation tradeoff",
        "search tree traversal",
        "value estimation",
        "policy improvement",
    ],
    "application": [
        "board game AI",
        "robotics control",
        "resource allocation",
        "automated planning",
        "dialogue systems",
    ],
    "context": [
        "tree search",
        "multi-armed bandits",
        "game playing",
        "reinforcement learning",
    ],
    "game_specific_feature": [
        "game symmetries",
        "opening book knowledge",
        "endgame tablebases",
        "ko rules in Go",
    ],
    "workflow_type": [
        "conditional branching",
        "human-in-the-loop",
        "parallel execution",
        "cyclic reasoning",
    ],
    "feature": [
        "state persistence",
        "error recovery",
        "streaming outputs",
        "tool calling",
    ],
    "framework": [
        "LangGraph",
        "LangChain",
        "CrewAI",
        "AutoGen",
    ],
    "task": [
        "document analysis",
        "code generation",
        "research synthesis",
        "multi-step reasoning",
    ],
    "system": [
        "hierarchical multi-agent MCTS",
        "distributed planning system",
        "collaborative problem solver",
    ],
    "system_type": [
        "multi-agent",
        "hierarchical",
        "distributed",
        "federated",
    ],
    "component": [
        "MCTSNode",
        "PolicyNetwork",
        "ValueNetwork",
        "ReplayBuffer",
        "StateGraph",
    ],
    "use_case": [
        "real-time decision making",
        "batch processing",
        "online learning",
        "production inference",
    ],
    "language": [
        "Python",
        "TypeScript",
        "Rust",
        "Go",
    ],
    "real_world_problem": [
        "supply chain optimization",
        "traffic routing",
        "resource scheduling",
        "portfolio optimization",
    ],
    # Additional vocabulary keys for template coverage
    "method": [
        "UCB1 selection",
        "PUCT with neural guidance",
        "progressive widening",
        "virtual loss parallelization",
        "Dirichlet noise injection",
    ],
    "alternative_framework": [
        "LangChain",
        "CrewAI",
        "AutoGen",
        "Haystack",
        "semantic-kernel",
    ],
    "scenario": [
        "high-throughput inference",
        "real-time decision making",
        "distributed training",
        "edge deployment",
        "multi-tenant serving",
    ],
    "features": [
        "caching and batching",
        "async execution and streaming",
        "retry logic and error handling",
        "logging and observability",
        "type safety and validation",
    ],
    "code_snippet": [
        "# Implementation code here",
        "def process(data): return transform(data)",
        "async def fetch(): return await client.get()",
        "class Handler: pass",
    ],
    "constraint": [
        "memory efficiency",
        "latency requirements",
        "throughput optimization",
        "resource limits",
        "concurrent access",
    ],
    "functionality": [
        "processes input data",
        "manages state transitions",
        "handles error recovery",
        "coordinates multi-agent workflows",
        "optimizes search strategies",
    ],
    "pattern": [
        "singleton",
        "factory",
        "observer",
        "strategy",
        "decorator",
    ],
    "scale": [
        "1000 requests per second",
        "millions of daily active users",
        "petabytes of data",
        "global distribution",
        "real-time streaming",
    ],
    "service": [
        "inference API",
        "training orchestrator",
        "model registry",
        "experiment tracker",
        "feature store",
    ],
    "metrics": [
        "latency percentiles (p50, p95, p99)",
        "throughput and error rates",
        "resource utilization",
        "model accuracy and drift",
        "business KPIs",
    ],
}


class QualityValidator:
    """Validates and scores generated Q&A pairs."""

    def __init__(self, min_question_length: int = 20, min_answer_length: int = 100):
        self.min_question_length = min_question_length
        self.min_answer_length = min_answer_length

    def validate(self, qa_pair: QAPair) -> tuple[bool, list[str]]:
        """
        Validate a Q&A pair.

        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []

        # Length checks
        if len(qa_pair.question) < self.min_question_length:
            errors.append(f"Question too short: {len(qa_pair.question)} < {self.min_question_length}")

        if len(qa_pair.answer) < self.min_answer_length:
            errors.append(f"Answer too short: {len(qa_pair.answer)} < {self.min_answer_length}")

        # Content quality checks
        if not qa_pair.question.strip().endswith("?"):
            errors.append("Question should end with '?'")

        # Check for placeholder text
        placeholders = ["{", "}", "[placeholder]", "[TODO]", "XXX"]
        for placeholder in placeholders:
            if placeholder in qa_pair.question or placeholder in qa_pair.answer:
                errors.append(f"Contains placeholder text: {placeholder}")

        # Check for duplicate content (question in answer)
        # This is often okay, but flag very short answers that just repeat the question
        if qa_pair.question.lower() in qa_pair.answer.lower() and len(qa_pair.answer) < len(qa_pair.question) * 2:
            errors.append("Answer mostly repeats the question")

        # Check contexts
        if not qa_pair.contexts or len(qa_pair.contexts) == 0:
            errors.append("No contexts provided")

        return len(errors) == 0, errors

    def score_quality(self, qa_pair: QAPair) -> float:
        """
        Score the quality of a Q&A pair (0.0 to 1.0).

        Factors:
        - Length and detail of answer
        - Presence of examples
        - Structured formatting
        - Technical depth
        - Context relevance
        """
        score = 0.0

        # Answer length score (up to 0.2)
        answer_length = len(qa_pair.answer)
        if answer_length >= 500:
            score += 0.2
        elif answer_length >= 300:
            score += 0.15
        elif answer_length >= 200:
            score += 0.1

        # Examples and code blocks (up to 0.2)
        if "```" in qa_pair.answer:
            score += 0.15
        if any(marker in qa_pair.answer.lower() for marker in ["for example", "e.g.", "such as"]):
            score += 0.05

        # Structured content (up to 0.2)
        structure_markers = [
            r"\n\d+\.",  # Numbered lists
            r"\n[-*]",  # Bullet points
            r"\n#{1,3} ",  # Headings
        ]
        for marker in structure_markers:
            if re.search(marker, qa_pair.answer):
                score += 0.07
                break

        # Technical terms (up to 0.2)
        technical_terms = [
            "algorithm",
            "implementation",
            "complexity",
            "optimization",
            "performance",
            "tradeoff",
            "convergence",
        ]
        term_count = sum(1 for term in technical_terms if term in qa_pair.answer.lower())
        score += min(0.2, term_count * 0.05)

        # Context quality (up to 0.2)
        if len(qa_pair.contexts) >= 3:
            score += 0.15
        elif len(qa_pair.contexts) >= 2:
            score += 0.1
        elif len(qa_pair.contexts) >= 1:
            score += 0.05

        # Multiple reasoning paths (bonus)
        if len(qa_pair.reasoning_paths) >= 2:
            score += 0.1

        return min(1.0, score)


class SyntheticKnowledgeGenerator:
    """Generate synthetic Q&A training data using LLMs."""

    def __init__(
        self,
        llm_client,
        output_dir: str = "training/synthetic_data",
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the generator.

        Args:
            llm_client: LLM client for generation
            output_dir: Directory to save generated data
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or {}
        self.validator = QualityValidator(
            min_question_length=self.config.get("min_question_length", 20),
            min_answer_length=self.config.get("min_answer_length", 100),
        )

        # Statistics tracking
        self.stats = {
            "total_generated": 0,
            "valid_pairs": 0,
            "invalid_pairs": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "api_calls": 0,
            "start_time": datetime.utcnow().isoformat(),
        }

        # Progress tracking
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.generated_pairs: list[QAPair] = []
        self.generated_hashes: set[str] = set()

        # Load existing checkpoint if resuming
        self._load_checkpoint()

        logger.info(f"Initialized SyntheticKnowledgeGenerator with output dir: {output_dir}")

    def _load_checkpoint(self) -> None:
        """Load checkpoint to resume generation."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    checkpoint = json.load(f)

                self.stats = checkpoint.get("stats", self.stats)
                self.generated_hashes = set(checkpoint.get("generated_hashes", []))

                logger.info(
                    f"Loaded checkpoint: {self.stats['valid_pairs']} valid pairs, "
                    f"{self.stats['invalid_pairs']} invalid pairs"
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

    def _save_checkpoint(self) -> None:
        """Save checkpoint for resumability."""
        checkpoint = {
            "stats": self.stats,
            "generated_hashes": list(self.generated_hashes),
            "last_updated": datetime.utcnow().isoformat(),
        }

        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _hash_qa(self, question: str) -> str:
        """Generate hash for duplicate detection."""
        normalized = re.sub(r"\s+", " ", question.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _fill_template(self, template: str) -> str:
        """Fill a template with random vocabulary."""
        filled = template

        # Extract all placeholders
        placeholders = re.findall(r"\{(\w+)\}", template)

        for placeholder in placeholders:
            if placeholder in DOMAIN_VOCABULARIES:
                replacement = random.choice(DOMAIN_VOCABULARIES[placeholder])
                filled = filled.replace(f"{{{placeholder}}}", replacement, 1)

        return filled

    def _generate_contexts(self, question: str, answer: str) -> list[str]:
        """Extract key contexts from the answer."""
        contexts = []

        # Split answer into sentences
        sentences = re.split(r"[.!?]\s+", answer)

        # Take first 2-3 sentences as primary context
        if len(sentences) >= 2:
            contexts.append(". ".join(sentences[:2]) + ".")

        # Extract any code blocks as context
        code_blocks = re.findall(r"```[\w]*\n(.*?)\n```", answer, re.DOTALL)
        if code_blocks:
            contexts.append(f"Code example:\n{code_blocks[0][:200]}")

        # Extract numbered/bulleted points
        points = re.findall(r"(?:\d+\.|[-*])\s+([^\n]+)", answer)
        if points and len(points) >= 2:
            contexts.append("Key points: " + "; ".join(points[:3]))

        # If we don't have enough contexts, split the answer
        if len(contexts) < 2:
            mid_point = len(answer) // 2
            contexts.append(answer[:mid_point].strip())
            contexts.append(answer[mid_point:].strip())

        return contexts[:4]  # Limit to 4 contexts

    async def _generate_question_answer(
        self, template: str, category: str, difficulty: str = "medium"
    ) -> QAPair | None:
        """
        Generate a single Q&A pair from a template.

        Args:
            template: Question template
            category: Category of question
            difficulty: Difficulty level (easy, medium, hard)

        Returns:
            QAPair or None if generation failed
        """
        try:
            # Fill template
            question = self._fill_template(template)

            # Check for duplicates
            qa_hash = self._hash_qa(question)
            if qa_hash in self.generated_hashes:
                logger.debug(f"Skipping duplicate question: {question[:50]}")
                return None

            # Generate detailed answer
            prompt = self._create_answer_prompt(question, difficulty)

            response: LLMResponse = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2000,
            )

            answer = response.text.strip()

            # Update stats
            self.stats["api_calls"] += 1
            self.stats["total_tokens"] += response.total_tokens
            self.stats["total_cost"] += self._estimate_cost(response)

            # Extract contexts
            contexts = self._generate_contexts(question, answer)

            # Create Q&A pair
            qa_pair = QAPair(
                question=question,
                answer=answer,
                contexts=contexts,
                metadata={
                    "category": category,
                    "difficulty": difficulty,
                    "template": template,
                    "model": response.model,
                },
            )

            # Validate
            is_valid, errors = self.validator.validate(qa_pair)

            if not is_valid:
                logger.warning(f"Invalid Q&A pair: {errors}")
                self.stats["invalid_pairs"] += 1
                return None

            # Score quality
            qa_pair.quality_score = self.validator.score_quality(qa_pair)

            # Mark as generated
            self.generated_hashes.add(qa_hash)
            self.stats["valid_pairs"] += 1
            self.stats["total_generated"] += 1

            return qa_pair

        except (LLMRateLimitError, LLMQuotaExceededError) as e:
            logger.warning(f"Rate limit/quota error: {e}")
            await asyncio.sleep(60)  # Wait before retrying
            return None

        except LLMClientError as e:
            logger.error(f"LLM error generating Q&A: {e}")
            self.stats["invalid_pairs"] += 1
            return None

    async def _generate_reasoning_paths(self, question: str, num_paths: int = 3) -> list[str]:
        """
        Generate multiple reasoning paths for a question.

        Args:
            question: The question to reason about
            num_paths: Number of different reasoning paths to generate

        Returns:
            List of reasoning path strings
        """
        paths = []

        for i in range(num_paths):
            try:
                prompt = f"""Think step-by-step about how to answer this question. Provide a detailed reasoning path.

Question: {question}

Reasoning Path #{i + 1}:"""

                response: LLMResponse = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.8,  # Higher temperature for diversity
                    max_tokens=500,
                )

                paths.append(response.text.strip())

                self.stats["api_calls"] += 1
                self.stats["total_tokens"] += response.total_tokens
                self.stats["total_cost"] += self._estimate_cost(response)

            except Exception as e:
                logger.warning(f"Failed to generate reasoning path {i + 1}: {e}")

        return paths

    def _create_answer_prompt(self, question: str, difficulty: str) -> str:
        """Create a prompt for generating a detailed answer."""
        difficulty_instructions = {
            "easy": "Provide a clear, beginner-friendly explanation with simple examples.",
            "medium": "Provide a detailed explanation with examples and some technical depth.",
            "hard": "Provide an in-depth, technical explanation with advanced examples and edge cases.",
        }

        instruction = difficulty_instructions.get(difficulty, difficulty_instructions["medium"])

        return f"""You are an expert in Monte Carlo Tree Search, LangGraph, and multi-agent systems.

Question: {question}

{instruction}

Your answer should:
1. Be technically accurate and well-structured
2. Include specific examples where appropriate
3. Use proper terminology
4. Be comprehensive but focused
5. Include code snippets if relevant (use Python)

Answer:"""

    def _estimate_cost(self, response: LLMResponse) -> float:
        """
        Estimate API cost based on token usage.

        This is a rough estimate - adjust based on actual pricing.
        """
        # Rough estimates (update with actual pricing)
        cost_per_1k_tokens = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "claude-3-opus": 0.015,
            "claude-3-sonnet": 0.003,
            "claude-3-haiku": 0.00025,
        }

        model = response.model.lower()

        # Match model to pricing
        for model_key, price in cost_per_1k_tokens.items():
            if model_key in model:
                return (response.total_tokens / 1000.0) * price

        # Default estimate
        return (response.total_tokens / 1000.0) * 0.002

    async def generate_variations(self, base_template: str, category: str, num_variations: int = 5) -> list[QAPair]:
        """
        Generate multiple variations of a question template.

        Args:
            base_template: Base question template
            category: Question category
            num_variations: Number of variations to generate

        Returns:
            List of generated QAPairs
        """
        variations = []

        difficulties = ["easy", "medium", "hard"]

        for i in range(num_variations):
            difficulty = difficulties[i % len(difficulties)]

            qa_pair = await self._generate_question_answer(base_template, category, difficulty)

            if qa_pair:
                # Optionally generate reasoning paths for higher quality pairs
                if qa_pair.quality_score >= 0.7:
                    qa_pair.reasoning_paths = await self._generate_reasoning_paths(qa_pair.question, num_paths=2)

                variations.append(qa_pair)

            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)

        return variations

    async def generate_batch(
        self,
        num_samples: int = 100,
        categories: list[str] | None = None,
        batch_size: int = 10,
    ) -> list[QAPair]:
        """
        Generate a batch of Q&A pairs.

        Args:
            num_samples: Target number of samples to generate
            categories: List of categories to include (None = all)
            batch_size: Number of concurrent generations

        Returns:
            List of generated QAPairs
        """
        if categories is None:
            categories = list(QUESTION_TEMPLATES.keys())

        all_pairs = []

        # Create task list
        category_templates = []

        for category in categories:
            templates = QUESTION_TEMPLATES[category]
            for template in templates:
                category_templates.append((category, template))

        # Shuffle for diversity
        random.shuffle(category_templates)

        # Generate samples
        pbar = tqdm(total=num_samples, desc="Generating Q&A pairs")

        for i in range(0, len(category_templates), batch_size):
            batch = category_templates[i : i + batch_size]

            # Create async tasks for batch
            batch_tasks = []
            for category, template in batch:
                # Generate 2-3 variations per template
                num_variations = random.randint(2, 3)
                task = self.generate_variations(template, category, num_variations=num_variations)
                batch_tasks.append(task)

            # Execute batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Collect results
            for result in batch_results:
                if isinstance(result, list):
                    for qa_pair in result:
                        all_pairs.append(qa_pair)
                        pbar.update(1)

                        if len(all_pairs) >= num_samples:
                            break

            # Save checkpoint periodically
            if i % (batch_size * 5) == 0:
                self._save_checkpoint()

            if len(all_pairs) >= num_samples:
                break

        pbar.close()

        self.generated_pairs.extend(all_pairs)
        return all_pairs[:num_samples]

    def save_dataset(
        self,
        pairs: list[QAPair],
        output_file: str = "synthetic_qa_dataset.json",
        format: str = "langsmith",
    ) -> None:
        """
        Save generated Q&A pairs to file.

        Args:
            pairs: List of Q&A pairs to save
            output_file: Output filename
            format: Output format ('langsmith' or 'raw')
        """
        output_path = self.output_dir / output_file

        if format == "langsmith":
            data = [pair.to_langsmith_format() for pair in pairs]
        else:
            data = [pair.to_dict() for pair in pairs]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(pairs)} Q&A pairs to {output_path}")

        # Save statistics
        stats_file = self.output_dir / "generation_stats.json"
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Saved generation statistics to {stats_file}")

    def filter_by_quality(self, pairs: list[QAPair], min_score: float = 0.5) -> list[QAPair]:
        """Filter Q&A pairs by minimum quality score."""
        filtered = [pair for pair in pairs if pair.quality_score >= min_score]
        logger.info(f"Filtered {len(pairs)} pairs to {len(filtered)} with min_score={min_score}")
        return filtered

    def get_statistics(self) -> dict[str, Any]:
        """Get generation statistics."""
        return {
            **self.stats,
            "pairs_in_memory": len(self.generated_pairs),
            "unique_questions": len(self.generated_hashes),
            "avg_quality_score": (
                sum(p.quality_score for p in self.generated_pairs) / len(self.generated_pairs)
                if self.generated_pairs
                else 0
            ),
        }


async def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic Q&A training data")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of Q&A pairs to generate")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/synthetic_data",
        help="Output directory",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "lmstudio"],
        help="LLM provider",
    )
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--batch-size", type=int, default=10, help="Concurrent generation batch size")
    parser.add_argument("--min-quality", type=float, default=0.5, help="Minimum quality score filter")
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="Categories to generate (default: all)",
    )
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Create LLM client
    logger.info(f"Creating {args.provider} client...")

    if args.config and "llm" in config:
        llm_client = create_client_from_config(config["llm"])
    else:
        llm_client = create_client(
            provider=args.provider,
            model=args.model,
            rate_limit_per_minute=60,  # Rate limiting
        )

    # Create generator
    generator = SyntheticKnowledgeGenerator(
        llm_client=llm_client,
        output_dir=args.output_dir,
        config=config.get("generation", {}),
    )

    # Generate data
    logger.info(f"Generating {args.num_samples} Q&A pairs...")
    pairs = await generator.generate_batch(
        num_samples=args.num_samples,
        categories=args.categories,
        batch_size=args.batch_size,
    )

    # Filter by quality
    filtered_pairs = generator.filter_by_quality(pairs, min_score=args.min_quality)

    # Save datasets
    generator.save_dataset(filtered_pairs, "synthetic_qa_langsmith.json", format="langsmith")
    generator.save_dataset(filtered_pairs, "synthetic_qa_raw.json", format="raw")

    # Print statistics
    stats = generator.get_statistics()
    logger.info("\n" + "=" * 70)
    logger.info("Generation Complete!")
    logger.info("=" * 70)
    logger.info(f"Total generated: {stats['total_generated']}")
    logger.info(f"Valid pairs: {stats['valid_pairs']}")
    logger.info(f"Invalid pairs: {stats['invalid_pairs']}")
    logger.info(f"High quality (>= {args.min_quality}): {len(filtered_pairs)}")
    logger.info(f"Average quality score: {stats['avg_quality_score']:.3f}")
    logger.info(f"Total API calls: {stats['api_calls']}")
    logger.info(f"Total tokens: {stats['total_tokens']:,}")
    logger.info(f"Estimated cost: ${stats['total_cost']:.2f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
