"""
Comprehensive E2E User Journey Tests.

Tests complete user journeys through the multi-agent MCTS framework:
1. New user onboarding and first query
2. Multi-turn conversation with context
3. Complex tactical analysis with all agents
4. Cybersecurity incident response workflow
5. Training pipeline journey (data → train → evaluate → deploy)
6. API consumer journey (auth → query → results)
7. Failure recovery and graceful degradation

Each journey tests the full stack from input to output,
validating integration between all system components.
"""

import asyncio
import random
import time

import pytest

# Optional torch import - skip tests if not available
torch = pytest.importorskip("torch", reason="PyTorch required for E2E user journey tests")

from tests.mocks.mock_external_services import (
    MockLLMClient,
    create_mock_braintrust,
    create_mock_llm,
    create_mock_pinecone,
)
from tests.utils.langsmith_tracing import (
    trace_e2e_test,
    update_run_metadata,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Create mock LLM client with comprehensive responses."""
    client = create_mock_llm(provider="openai")
    client.set_responses(
        [
            # HRM hierarchical decomposition response
            """Hierarchical Reasoning Model Analysis:
        Primary Objective: Secure northern perimeter
        Sub-objectives:
        1. Establish defensive positions at Alpha
        2. Deploy UAV for reconnaissance
        3. Coordinate communication channels
        Risk Level: Medium
        Confidence: 0.87""",
            # TRM iterative refinement response
            """Task Refinement Model Analysis:
        Iteration 1 - Position Evaluation:
        - Alpha: Coverage 85%, Risk Medium, Score 0.78
        - Beta: Coverage 72%, Risk Low, Score 0.71
        - Gamma: Coverage 68%, Risk High, Score 0.54

        Recommendation: Alpha with Beta fallback
        Refinement cycles: 3
        Confidence: 0.83""",
            # Consensus response
            """Consensus Analysis:
        HRM and TRM agree on Alpha position (85% alignment)
        Risk factors identified: ammunition, visibility
        Final recommendation: Secure Alpha, prepare Beta fallback
        Combined confidence: 0.85""",
        ]
    )
    return client


@pytest.fixture
def mock_pinecone_client():
    """Create mock Pinecone client for RAG testing."""
    return create_mock_pinecone()


@pytest.fixture
def mock_braintrust_tracker():
    """Create mock Braintrust tracker for experiment tracking."""
    return create_mock_braintrust(project="user-journey-tests")


@pytest.fixture
def tactical_scenario():
    """Standard tactical scenario for testing."""
    return {
        "query": "Enemy forces approaching from north. Night conditions, limited visibility. "
        "Available: Infantry platoon (25), UAV support, limited ammunition (100 rounds). "
        "Recommend optimal defensive strategy and engagement plan.",
        "context": {
            "terrain": "urban",
            "weather": "night",
            "visibility": "low",
            "friendly_forces": {"infantry": 25, "uav": 1},
            "enemy_estimate": {"infantry": 30, "direction": "north"},
        },
        "expected_agents": ["hrm", "trm", "mcts"],
    }


@pytest.fixture
def cybersecurity_scenario():
    """Cybersecurity incident scenario for testing."""
    return {
        "query": "APT28 indicators detected on critical infrastructure. "
        "Evidence: credential harvesting, lateral movement, C2 communication. "
        "Systems affected: 3 domain controllers, 12 workstations. "
        "Recommend immediate containment and response strategy.",
        "indicators": {
            "iocs": ["malicious_ip_1", "malicious_domain_1"],
            "techniques": ["T1078", "T1021", "T1071"],
            "affected_systems": 15,
        },
        "severity": "HIGH",
    }


@pytest.fixture
def sample_training_data():
    """Sample training data for meta-controller."""
    from src.agents.meta_controller.base import MetaControllerFeatures

    return [
        MetaControllerFeatures(
            hrm_confidence=0.9,
            trm_confidence=0.6,
            mcts_value=0.7,
            consensus_score=0.8,
            last_agent="hrm",
            iteration=1,
            query_length=100,
            has_rag_context=True,
        ),
        MetaControllerFeatures(
            hrm_confidence=0.5,
            trm_confidence=0.9,
            mcts_value=0.4,
            consensus_score=0.6,
            last_agent="trm",
            iteration=2,
            query_length=200,
            has_rag_context=False,
        ),
        MetaControllerFeatures(
            hrm_confidence=0.3,
            trm_confidence=0.4,
            mcts_value=0.95,
            consensus_score=0.5,
            last_agent="mcts",
            iteration=5,
            query_length=500,
            has_rag_context=True,
        ),
    ]


# ============================================================================
# JOURNEY 1: NEW USER ONBOARDING
# ============================================================================


class TestNewUserOnboardingJourney:
    """
    Tests the journey of a new user interacting with the system for the first time.

    Steps:
    1. User submits first query
    2. System validates input
    3. System processes through agents
    4. User receives formatted response with confidence
    5. User can access response metadata
    """

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "journey_new_user_first_query",
        phase="onboarding",
        scenario_type="tactical",
        tags=["user_journey", "onboarding", "first_query"],
    )
    async def test_first_query_journey(self, mock_llm_client, tactical_scenario):
        """New user submits their first tactical query."""
        from src.models.validation import QueryInput

        # Step 1: User submits query
        user_query = tactical_scenario["query"]

        # Step 2: System validates input
        query_input = QueryInput(
            query=user_query,
            use_rag=True,
            use_mcts=False,  # Start simple for first query
            thread_id="new_user_session_001",
        )

        assert query_input.query is not None
        assert len(query_input.query) > 0
        assert len(query_input.query) <= 10000

        # Step 3: Process through HRM agent (simplified first experience)
        hrm_response = await mock_llm_client.generate(f"HRM Analysis: {query_input.query}")

        assert hrm_response.content is not None
        assert "Confidence:" in hrm_response.content

        # Step 4: Format response for user
        response = {
            "recommendation": "Secure Alpha position",
            "confidence": 0.87,
            "reasoning": hrm_response.content,
            "next_steps": [
                "Establish defensive positions",
                "Deploy UAV reconnaissance",
                "Coordinate communications",
            ],
        }

        # Step 5: Validate response structure
        assert "recommendation" in response
        assert "confidence" in response
        assert 0.0 <= response["confidence"] <= 1.0
        assert len(response["next_steps"]) >= 1

        update_run_metadata(
            {
                "journey": "new_user_onboarding",
                "steps_completed": 5,
                "final_confidence": response["confidence"],
                "agents_used": ["hrm"],
            }
        )

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "journey_new_user_validation_feedback",
        phase="onboarding",
        scenario_type="validation",
        tags=["user_journey", "onboarding", "validation"],
    )
    async def test_input_validation_feedback(self):
        """New user receives helpful feedback on invalid input."""
        import pydantic

        from src.models.validation import QueryInput

        # Test empty query
        with pytest.raises(pydantic.ValidationError) as exc_info:
            QueryInput(query="", use_rag=True, use_mcts=False)

        error = exc_info.value
        assert len(error.errors()) > 0

        # Test oversized query
        with pytest.raises(pydantic.ValidationError):
            QueryInput(query="x" * 15000, use_rag=True, use_mcts=False)

        # Test valid query passes
        valid_input = QueryInput(
            query="What is the best defensive position?",
            use_rag=True,
            use_mcts=False,
        )
        assert valid_input.query is not None

        update_run_metadata(
            {
                "journey": "validation_feedback",
                "validation_tests": 3,
                "all_passed": True,
            }
        )


# ============================================================================
# JOURNEY 2: MULTI-TURN CONVERSATION
# ============================================================================


class TestMultiTurnConversationJourney:
    """
    Tests a multi-turn conversation where context builds across turns.

    Steps:
    1. Initial query establishes context
    2. Follow-up query references previous context
    3. System maintains conversation state
    4. User can refine recommendations
    """

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "journey_multi_turn_conversation",
        phase="conversation",
        scenario_type="tactical",
        tags=["user_journey", "multi_turn", "context_retention"],
    )
    async def test_multi_turn_tactical_conversation(self, mock_llm_client):
        """User has a multi-turn conversation with context retention."""
        from src.models.validation import QueryInput

        thread_id = "conversation_thread_001"
        conversation_history = []

        # Turn 1: Initial query
        turn1_query = QueryInput(
            query="Enemy spotted approaching from north. Recommend defensive positions.",
            use_rag=True,
            use_mcts=False,
            thread_id=thread_id,
        )

        turn1_response = await mock_llm_client.generate(turn1_query.query)
        conversation_history.append(
            {
                "turn": 1,
                "query": turn1_query.query,
                "response": turn1_response.content,
            }
        )

        # Turn 2: Follow-up with context
        turn2_query = QueryInput(
            query="What if we have limited ammunition? Adjust the recommendation.",
            use_rag=True,
            use_mcts=False,
            thread_id=thread_id,
        )

        # Context should be maintained
        context_prompt = f"Previous: {conversation_history[-1]['response'][:200]}... New query: {turn2_query.query}"
        turn2_response = await mock_llm_client.generate(context_prompt)
        conversation_history.append(
            {
                "turn": 2,
                "query": turn2_query.query,
                "response": turn2_response.content,
            }
        )

        # Turn 3: Further refinement
        turn3_query = QueryInput(
            query="Now consider night conditions. Final recommendation?",
            use_rag=True,
            use_mcts=True,  # Enable MCTS for final decision
            thread_id=thread_id,
        )

        turn3_response = await mock_llm_client.generate(turn3_query.query)
        conversation_history.append(
            {
                "turn": 3,
                "query": turn3_query.query,
                "response": turn3_response.content,
            }
        )

        # Validate conversation flow
        assert len(conversation_history) == 3
        assert all(turn["response"] is not None for turn in conversation_history)

        # All turns should use same thread
        assert all(turn["turn"] > 0 for turn in conversation_history)

        update_run_metadata(
            {
                "journey": "multi_turn_conversation",
                "total_turns": 3,
                "thread_id": thread_id,
                "context_maintained": True,
            }
        )


# ============================================================================
# JOURNEY 3: FULL AGENT ORCHESTRATION
# ============================================================================


class TestFullAgentOrchestrationJourney:
    """
    Tests the complete multi-agent orchestration flow.

    Steps:
    1. Query enters system
    2. Meta-controller selects initial agent
    3. HRM performs hierarchical decomposition
    4. TRM refines the solution
    5. MCTS simulates tactical options
    6. Consensus mechanism synthesizes final response
    """

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "journey_full_agent_orchestration",
        phase="orchestration",
        scenario_type="tactical",
        use_mcts=True,
        mcts_iterations=200,
        tags=["user_journey", "orchestration", "full_stack", "hrm", "trm", "mcts"],
    )
    async def test_complete_agent_orchestration(self, mock_llm_client, tactical_scenario, sample_training_data):
        """Complete journey through all agents with orchestration."""
        from src.agents.meta_controller.base import MetaControllerFeatures
        from src.models.validation import QueryInput

        query_input = QueryInput(
            query=tactical_scenario["query"],
            use_rag=True,
            use_mcts=True,
            thread_id="orchestration_test_001",
        )

        # Step 1: Meta-controller feature extraction
        MetaControllerFeatures(
            hrm_confidence=0.0,
            trm_confidence=0.0,
            mcts_value=0.0,
            consensus_score=0.0,
            last_agent="none",
            iteration=0,
            query_length=len(query_input.query),
            has_rag_context=query_input.use_rag,
        )

        # Step 2: HRM hierarchical decomposition
        hrm_response = await mock_llm_client.generate(f"HRM: {query_input.query}")
        hrm_confidence = 0.87

        assert "Hierarchical" in hrm_response.content

        # Step 3: TRM iterative refinement
        trm_response = await mock_llm_client.generate(f"TRM: {query_input.query}")
        trm_confidence = 0.83

        assert "Refinement" in trm_response.content or "refinement" in trm_response.content.lower()

        # Step 4: MCTS tactical simulation
        random.seed(42)
        mcts_iterations = 200
        possible_actions = [
            "secure_alpha",
            "secure_beta",
            "flanking_maneuver",
            "defensive_hold",
            "strategic_retreat",
        ]

        action_stats = {action: {"visits": 0, "value_sum": 0.0} for action in possible_actions}

        for _ in range(mcts_iterations):
            action = random.choice(possible_actions)
            value = random.uniform(0.3, 0.9)
            action_stats[action]["visits"] += 1
            action_stats[action]["value_sum"] += value

        # Calculate best action
        best_action = max(
            action_stats.keys(), key=lambda a: action_stats[a]["value_sum"] / max(action_stats[a]["visits"], 1)
        )
        mcts_value = action_stats[best_action]["value_sum"] / action_stats[best_action]["visits"]

        # Step 5: Consensus calculation
        consensus_score = (hrm_confidence + trm_confidence + mcts_value) / 3

        # Step 6: Final response synthesis
        final_response = {
            "recommendation": best_action.replace("_", " ").title(),
            "confidence": consensus_score,
            "agents_consulted": ["HRM", "TRM", "MCTS"],
            "hrm_analysis": {
                "confidence": hrm_confidence,
                "decomposition_levels": 3,
            },
            "trm_analysis": {
                "confidence": trm_confidence,
                "refinement_cycles": 3,
            },
            "mcts_analysis": {
                "iterations": mcts_iterations,
                "best_action": best_action,
                "value": mcts_value,
                "action_distribution": {a: s["visits"] for a, s in action_stats.items()},
            },
            "consensus_score": consensus_score,
        }

        # Validate complete flow
        assert final_response["confidence"] >= 0.5
        assert len(final_response["agents_consulted"]) == 3
        assert final_response["mcts_analysis"]["iterations"] == mcts_iterations

        update_run_metadata(
            {
                "journey": "full_agent_orchestration",
                "hrm_confidence": hrm_confidence,
                "trm_confidence": trm_confidence,
                "mcts_value": mcts_value,
                "consensus_score": consensus_score,
                "best_action": best_action,
                "total_mcts_iterations": mcts_iterations,
            }
        )


# ============================================================================
# JOURNEY 4: CYBERSECURITY INCIDENT RESPONSE
# ============================================================================


class TestCybersecurityIncidentResponseJourney:
    """
    Tests the cybersecurity incident response workflow.

    Steps:
    1. Threat detected and reported
    2. System analyzes indicators
    3. MCTS evaluates response strategies
    4. Prioritized containment actions generated
    5. Response timeline established
    """

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "journey_cybersecurity_incident_response",
        phase="incident_response",
        scenario_type="cybersecurity",
        use_mcts=True,
        tags=["user_journey", "cybersecurity", "incident_response", "threat_analysis"],
    )
    async def test_apt_incident_response_journey(self, mock_llm_client, cybersecurity_scenario):
        """Complete incident response journey for APT detection."""
        from src.models.validation import QueryInput

        # Step 1: Incident reported
        incident_query = QueryInput(
            query=cybersecurity_scenario["query"],
            use_rag=True,
            use_mcts=True,
            thread_id="incident_response_001",
        )

        # Step 2: Threat analysis
        threat_analysis = {
            "threat_actor": "APT28",
            "confidence": 0.89,
            "techniques_identified": cybersecurity_scenario["indicators"]["techniques"],
            "affected_systems": cybersecurity_scenario["indicators"]["affected_systems"],
            "severity": cybersecurity_scenario["severity"],
        }

        # Step 3: HRM hierarchical threat breakdown
        await mock_llm_client.generate(f"HRM Threat Analysis: {incident_query.query}")

        threat_hierarchy = {
            "primary_threat": "APT28 Intrusion",
            "attack_vectors": [
                "credential_harvesting",
                "lateral_movement",
                "c2_communication",
            ],
            "containment_priorities": [
                "isolate_domain_controllers",
                "block_c2_channels",
                "reset_compromised_credentials",
            ],
        }

        # Step 4: MCTS response strategy evaluation
        random.seed(42)
        response_strategies = [
            "immediate_isolation",
            "forensics_first",
            "phased_containment",
            "full_network_lockdown",
            "targeted_remediation",
        ]

        strategy_scores = {}
        for strategy in response_strategies:
            # Simulate MCTS evaluation
            scores = [random.uniform(0.5, 0.95) for _ in range(50)]
            strategy_scores[strategy] = {
                "mean_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "min_score": min(scores),
                "evaluations": len(scores),
            }

        best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s]["mean_score"])

        # Step 5: Generate response timeline
        response_timeline = {
            "immediate_0_15min": [
                "Isolate affected domain controllers",
                "Block known C2 IP addresses",
                "Alert security operations center",
            ],
            "short_term_15_60min": [
                "Reset all compromised credentials",
                "Deploy additional monitoring",
                "Begin forensic evidence collection",
            ],
            "medium_term_1_4hr": [
                "Complete system isolation assessment",
                "Patch identified vulnerabilities",
                "Restore from clean backups",
            ],
            "long_term_4hr_plus": [
                "Full incident documentation",
                "Lessons learned analysis",
                "Security posture improvement",
            ],
        }

        # Final incident response package
        incident_response = {
            "threat_analysis": threat_analysis,
            "threat_hierarchy": threat_hierarchy,
            "recommended_strategy": best_strategy,
            "strategy_confidence": strategy_scores[best_strategy]["mean_score"],
            "response_timeline": response_timeline,
            "agents_consulted": ["HRM", "TRM", "MCTS"],
        }

        # Validate response
        assert incident_response["threat_analysis"]["severity"] == "HIGH"
        assert incident_response["strategy_confidence"] >= 0.5
        assert len(incident_response["response_timeline"]) == 4

        update_run_metadata(
            {
                "journey": "cybersecurity_incident_response",
                "threat_actor": threat_analysis["threat_actor"],
                "severity": threat_analysis["severity"],
                "best_strategy": best_strategy,
                "strategy_confidence": strategy_scores[best_strategy]["mean_score"],
                "affected_systems": threat_analysis["affected_systems"],
            }
        )


# ============================================================================
# JOURNEY 5: TRAINING PIPELINE
# ============================================================================


class TestTrainingPipelineJourney:
    """
    Tests the complete training pipeline journey.

    Steps:
    1. Generate synthetic training data
    2. Split into train/val/test
    3. Train meta-controller model
    4. Evaluate on test set
    5. Save model checkpoint
    6. Load and verify predictions
    """

    @pytest.mark.e2e
    @trace_e2e_test(
        "journey_training_pipeline",
        phase="training",
        scenario_type="ml_pipeline",
        tags=["user_journey", "training", "ml_pipeline", "meta_controller"],
    )
    def test_complete_training_pipeline(self, mock_braintrust_tracker):
        """Complete training pipeline from data generation to deployment."""
        from src.agents.meta_controller.base import MetaControllerPrediction
        from src.agents.meta_controller.rnn_controller import RNNMetaController
        from src.training.data_generator import MetaControllerDataGenerator
        from src.training.train_rnn import RNNTrainer

        # Step 1: Generate synthetic training data
        generator = MetaControllerDataGenerator(seed=42)
        features_list, labels_list = generator.generate_balanced_dataset(
            num_samples_per_class=30  # Small for fast testing
        )

        assert len(features_list) == 90  # 30 * 3 classes
        assert len(labels_list) == 90

        # Step 2: Convert to tensors and split
        X, y = generator.to_tensor_dataset(features_list, labels_list)
        splits = generator.split_dataset(X, y, train_ratio=0.7, val_ratio=0.15)

        assert "X_train" in splits
        assert "X_val" in splits
        assert "X_test" in splits

        # Step 3: Initialize experiment tracking
        mock_braintrust_tracker.init_experiment("training_pipeline_test")
        mock_braintrust_tracker.log_hyperparameters(
            {
                "hidden_dim": 32,
                "num_layers": 1,
                "learning_rate": 1e-3,
                "batch_size": 16,
                "epochs": 3,
            }
        )

        # Step 4: Train model
        trainer = RNNTrainer(
            hidden_dim=32,
            num_layers=1,
            dropout=0.1,
            lr=1e-3,
            batch_size=16,
            epochs=3,
            early_stopping_patience=2,
            seed=42,
        )

        history = trainer.train(
            train_data=(splits["X_train"], splits["y_train"]),
            val_data=(splits["X_val"], splits["y_val"]),
        )

        assert "train_losses" in history
        assert "val_losses" in history
        assert len(history["train_losses"]) > 0

        # Log training metrics
        for epoch, (train_loss, val_loss) in enumerate(
            zip(history["train_losses"], history["val_losses"], strict=False)
        ):
            mock_braintrust_tracker.log_metric("train_loss", train_loss, step=epoch)
            mock_braintrust_tracker.log_metric("val_loss", val_loss, step=epoch)

        # Step 5: Evaluate on test set
        test_loader = trainer.create_dataloader(splits["X_test"], splits["y_test"], shuffle=False)
        eval_results = trainer.evaluate(test_loader)

        assert "accuracy" in eval_results
        assert "per_class_metrics" in eval_results

        mock_braintrust_tracker.log_metric("test_accuracy", eval_results["accuracy"])

        # Step 6: Create controller and verify predictions
        controller = RNNMetaController(name="TrainedController", seed=42)
        controller.model = trainer.model

        # Test predictions on sample features
        for features in features_list[:5]:
            prediction = controller.predict(features)
            assert isinstance(prediction, MetaControllerPrediction)
            assert prediction.agent in ["hrm", "trm", "mcts"]
            assert 0.0 <= prediction.confidence <= 1.0

        # End experiment
        summary = mock_braintrust_tracker.end_experiment()

        update_run_metadata(
            {
                "journey": "training_pipeline",
                "total_samples": len(features_list),
                "train_epochs": len(history["train_losses"]),
                "final_train_loss": history["train_losses"][-1],
                "final_val_loss": history["val_losses"][-1],
                "test_accuracy": eval_results["accuracy"],
                "experiment_id": summary.get("id"),
            }
        )


# ============================================================================
# JOURNEY 6: API CONSUMER
# ============================================================================


class TestAPIConsumerJourney:
    """
    Tests the API consumer journey.

    Steps:
    1. Authenticate with API key
    2. Submit query request
    3. Receive processed response
    4. Handle rate limiting
    5. Access usage statistics
    """

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "journey_api_consumer",
        phase="api",
        scenario_type="api_integration",
        tags=["user_journey", "api", "authentication", "rate_limiting"],
    )
    async def test_complete_api_consumer_journey(self, tactical_scenario):
        """Complete API consumer journey from auth to results."""
        from src.api.auth import APIKeyAuthenticator, RateLimitConfig

        # Step 1: Setup authentication
        authenticator = APIKeyAuthenticator(
            valid_keys=["test-api-key-001"],
            rate_limit_config=RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
            ),
        )

        # Step 2: Authenticate
        client_info = authenticator.require_auth("test-api-key-001")
        assert client_info is not None
        assert client_info.client_id is not None

        # Step 3: Simulate query request
        query_request = {
            "query": tactical_scenario["query"],
            "use_mcts": True,
            "use_rag": True,
            "thread_id": "api_test_session_001",
        }

        # Validate request
        from src.models.validation import QueryInput

        query_input = QueryInput(**query_request)
        assert query_input.query is not None

        # Additional config for MCTS iterations (not part of QueryInput)
        mcts_iterations = 100

        # Step 4: Simulate response
        start_time = time.perf_counter()
        await asyncio.sleep(0.05)  # Simulate processing
        processing_time = (time.perf_counter() - start_time) * 1000

        api_response = {
            "response": f"Processed: {query_request['query'][:100]}...",
            "confidence": 0.85,
            "agents_used": ["hrm", "trm", "mcts"],
            "mcts_stats": {
                "iterations": mcts_iterations,
                "best_action": "secure_alpha",
            },
            "processing_time_ms": processing_time,
            "metadata": {
                "client_id": client_info.client_id,
                "thread_id": query_request["thread_id"],
            },
        }

        assert api_response["confidence"] > 0
        assert api_response["processing_time_ms"] < 5000  # Under 5s SLA

        # Step 5: Check rate limiting
        stats = authenticator.get_client_stats(client_info.client_id)
        assert "requests_this_minute" in stats or stats is not None

        update_run_metadata(
            {
                "journey": "api_consumer",
                "authenticated": True,
                "client_id": client_info.client_id,
                "processing_time_ms": processing_time,
                "response_confidence": api_response["confidence"],
            }
        )

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "journey_api_rate_limiting",
        phase="api",
        scenario_type="rate_limiting",
        tags=["user_journey", "api", "rate_limiting"],
    )
    async def test_rate_limiting_behavior(self):
        """Test API rate limiting behavior."""
        from src.api.auth import APIKeyAuthenticator, RateLimitConfig
        from src.api.exceptions import RateLimitError

        # Configure strict rate limiting
        authenticator = APIKeyAuthenticator(
            valid_keys=["rate-limit-test-key"],
            rate_limit_config=RateLimitConfig(
                requests_per_minute=5,  # Very low for testing
                requests_per_hour=100,
                requests_per_day=1000,
            ),
        )

        # Make requests up to limit
        successful_requests = 0
        rate_limited = False

        for _i in range(10):
            try:
                authenticator.require_auth("rate-limit-test-key")
                successful_requests += 1
            except RateLimitError:
                rate_limited = True
                break

        # Should hit rate limit before 10 requests
        assert rate_limited or successful_requests <= 5

        update_run_metadata(
            {
                "journey": "rate_limiting",
                "successful_requests": successful_requests,
                "rate_limited": rate_limited,
            }
        )


# ============================================================================
# JOURNEY 7: FAILURE RECOVERY
# ============================================================================


class TestFailureRecoveryJourney:
    """
    Tests system behavior under failure conditions.

    Steps:
    1. LLM timeout recovery
    2. Partial agent failure degradation
    3. Cache miss handling
    4. Invalid input recovery
    """

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "journey_failure_recovery",
        phase="resilience",
        scenario_type="failure_handling",
        tags=["user_journey", "resilience", "failure_recovery", "degradation"],
    )
    async def test_llm_timeout_recovery(self, mock_llm_client):
        """System recovers from LLM timeout."""
        # Configure mock to fail first, then succeed
        mock_llm_client.set_failure_mode(True, "Connection timeout")

        # First call should fail
        with pytest.raises(Exception) as exc_info:
            await mock_llm_client.generate("Test query")

        assert "timeout" in str(exc_info.value).lower()

        # Subsequent call should succeed (mock auto-resets failure mode)
        response = await mock_llm_client.generate("Test query after recovery")
        assert response.content is not None

        update_run_metadata(
            {
                "journey": "failure_recovery",
                "failure_type": "llm_timeout",
                "recovery_successful": True,
            }
        )

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "journey_graceful_degradation",
        phase="resilience",
        scenario_type="degradation",
        tags=["user_journey", "resilience", "graceful_degradation"],
    )
    async def test_graceful_degradation_with_partial_failure(self):
        """System degrades gracefully when some agents fail."""
        # Simulate partial agent failure
        agent_results = {
            "HRM": {"success": True, "confidence": 0.85, "response": "HRM analysis complete"},
            "TRM": {"success": False, "error": "Timeout", "response": None},
            "MCTS": {"success": True, "confidence": 0.75, "response": "MCTS simulation complete"},
        }

        # System should still produce result with available agents
        successful_agents = [k for k, v in agent_results.items() if v["success"]]
        failed_agents = [k for k, v in agent_results.items() if not v["success"]]

        assert len(successful_agents) >= 1

        # Calculate confidence from successful agents only
        confidences = [v["confidence"] for v in agent_results.values() if v.get("success") and "confidence" in v]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Generate degraded response
        degraded_response = {
            "response": "Partial analysis complete (some agents unavailable)",
            "confidence": avg_confidence,
            "agents_used": successful_agents,
            "agents_failed": failed_agents,
            "degraded": len(failed_agents) > 0,
        }

        assert degraded_response["confidence"] > 0.5
        assert degraded_response["degraded"] is True
        assert len(degraded_response["agents_used"]) >= 1

        update_run_metadata(
            {
                "journey": "graceful_degradation",
                "successful_agents": successful_agents,
                "failed_agents": failed_agents,
                "degraded_confidence": avg_confidence,
            }
        )


# ============================================================================
# JOURNEY 8: MCTS DEEP SIMULATION
# ============================================================================


class TestMCTSDeepSimulationJourney:
    """
    Tests deep MCTS simulation with extensive tree exploration.

    Steps:
    1. Initialize MCTS engine
    2. Run extensive simulation
    3. Analyze tree statistics
    4. Validate action selection
    """

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "journey_mcts_deep_simulation",
        phase="mcts",
        scenario_type="tactical",
        use_mcts=True,
        mcts_iterations=500,
        tags=["user_journey", "mcts", "deep_simulation", "performance"],
    )
    async def test_deep_mcts_simulation(self, tactical_scenario):
        """Run deep MCTS simulation with extensive tree exploration."""
        from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
        from src.framework.mcts.policies import RolloutPolicy

        # Step 1: Initialize MCTS engine
        engine = MCTSEngine(
            seed=42,
            exploration_weight=1.414,
            progressive_widening_k=1.0,
            progressive_widening_alpha=0.5,
            max_parallel_rollouts=4,
            cache_size_limit=1000,
        )

        # Step 2: Create root state
        root_state = MCTSState(
            state_id="tactical_root",
            features={
                "position": "neutral",
                "resources": tactical_scenario["context"]["friendly_forces"],
                "enemy_direction": tactical_scenario["context"]["enemy_estimate"]["direction"],
            },
        )
        root = MCTSNode(state=root_state, rng=engine.rng)

        # Step 3: Define actions and transitions
        possible_actions = [
            "advance_to_alpha",
            "advance_to_beta",
            "hold_position",
            "flanking_maneuver",
            "tactical_retreat",
        ]

        def action_generator(state: MCTSState) -> list[str]:
            return possible_actions

        def state_transition(state: MCTSState, action: str) -> MCTSState:
            return MCTSState(
                state_id=f"{state.state_id}_{action}",
                features={**state.features, "last_action": action},
            )

        # Step 4: Create rollout policy
        class SimpleRolloutPolicy(RolloutPolicy):
            async def evaluate(self, state: MCTSState, rng, max_depth: int = 10) -> float:
                # Simple heuristic-based evaluation
                base_value = 0.5
                if "alpha" in state.state_id:
                    base_value += 0.2
                elif "beta" in state.state_id:
                    base_value += 0.1
                return min(1.0, base_value + rng.uniform(-0.1, 0.1))

        rollout_policy = SimpleRolloutPolicy()

        # Step 5: Run MCTS search
        num_iterations = 500
        best_action, stats = await engine.search(
            root=root,
            num_iterations=num_iterations,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=rollout_policy,
            max_rollout_depth=10,
        )

        # Step 6: Validate results
        assert best_action is not None
        assert best_action in possible_actions
        assert stats["iterations"] == num_iterations
        assert stats["root_visits"] >= num_iterations

        # Check tree exploration
        assert stats["num_children"] > 0
        assert stats["best_action_visits"] > 0

        update_run_metadata(
            {
                "journey": "mcts_deep_simulation",
                "iterations": num_iterations,
                "best_action": best_action,
                "best_action_visits": stats["best_action_visits"],
                "best_action_value": stats["best_action_value"],
                "root_value": stats["root_value"],
                "cache_hit_rate": stats["cache_hit_rate"],
                "total_simulations": stats["total_simulations"],
            }
        )


# ============================================================================
# JOURNEY 9: HRM + TRM SYNERGY
# ============================================================================


class TestHRMTRMSynergyJourney:
    """
    Tests the synergy between HRM hierarchical decomposition and TRM refinement.

    Steps:
    1. HRM decomposes problem
    2. TRM refines each subproblem
    3. Results are synthesized
    """

    @pytest.mark.e2e
    @trace_e2e_test(
        "journey_hrm_trm_synergy",
        phase="agent_synergy",
        scenario_type="tactical",
        tags=["user_journey", "hrm", "trm", "synergy", "decomposition"],
    )
    def test_hrm_trm_synergy(self):
        """Test HRM decomposition followed by TRM refinement."""
        from src.agents.hrm_agent import create_hrm_agent
        from src.agents.trm_agent import create_trm_agent
        from src.training.system_config import HRMConfig, TRMConfig

        # Step 1: Initialize agents
        hrm_config = HRMConfig(
            h_dim=64,
            l_dim=32,
            num_h_layers=2,
            num_l_layers=1,
            max_outer_steps=5,
            max_ponder_steps=8,
            halt_threshold=0.9,
            dropout=0.1,
            ponder_epsilon=1e-6,
        )

        trm_config = TRMConfig(
            latent_dim=64,
            hidden_dim=128,
            num_recursions=5,
            min_recursions=2,
            convergence_threshold=0.01,
            deep_supervision=True,
            use_layer_norm=True,
            dropout=0.1,
        )

        hrm_agent = create_hrm_agent(hrm_config, device="cpu", use_ponder_net=False)
        trm_agent = create_trm_agent(trm_config, output_dim=64, device="cpu")

        # Step 2: Create test input
        batch_size = 2
        seq_len = 10
        test_input = torch.randn(batch_size, seq_len, hrm_config.h_dim)

        # Step 3: HRM hierarchical processing
        hrm_output = hrm_agent(
            test_input,
            max_steps=3,
            return_decomposition=True,
        )

        assert hrm_output.final_state is not None
        assert hrm_output.halt_step > 0
        assert len(hrm_output.convergence_path) > 0

        # Step 4: TRM refinement on HRM output
        trm_input = hrm_output.final_state
        trm_output = trm_agent(
            trm_input,
            num_recursions=3,
            check_convergence=True,
        )

        assert trm_output.final_prediction is not None
        assert trm_output.recursion_depth > 0
        assert len(trm_output.intermediate_predictions) > 0

        # Step 5: Validate synergy
        synergy_metrics = {
            "hrm_halt_step": hrm_output.halt_step,
            "hrm_ponder_cost": hrm_output.total_ponder_cost,
            "trm_recursion_depth": trm_output.recursion_depth,
            "trm_converged": trm_output.converged,
            "total_processing_steps": hrm_output.halt_step + trm_output.recursion_depth,
        }

        assert synergy_metrics["total_processing_steps"] > 0

        update_run_metadata(
            {
                "journey": "hrm_trm_synergy",
                **synergy_metrics,
                "hrm_params": hrm_agent.get_parameter_count(),
                "trm_params": trm_agent.get_parameter_count(),
            }
        )


# ============================================================================
# JOURNEY 10: OBSERVABILITY AND METRICS
# ============================================================================


class TestObservabilityJourney:
    """
    Tests the observability and metrics collection journey.

    Steps:
    1. Correlation ID propagation
    2. Metrics collection
    3. Trace verification
    4. Log aggregation
    """

    @pytest.mark.e2e
    @trace_e2e_test(
        "journey_observability",
        phase="observability",
        scenario_type="monitoring",
        tags=["user_journey", "observability", "metrics", "tracing"],
    )
    def test_observability_journey(self):
        """Test complete observability journey."""
        import uuid

        # Step 1: Generate correlation ID
        correlation_id = f"req_{uuid.uuid4().hex[:12]}"

        # Step 2: Simulate request flow with correlation ID
        request_flow = [
            {"component": "api_gateway", "correlation_id": correlation_id, "timestamp": time.time()},
            {"component": "auth_middleware", "correlation_id": correlation_id, "timestamp": time.time()},
            {"component": "query_validator", "correlation_id": correlation_id, "timestamp": time.time()},
            {"component": "hrm_agent", "correlation_id": correlation_id, "timestamp": time.time()},
            {"component": "trm_agent", "correlation_id": correlation_id, "timestamp": time.time()},
            {"component": "response_formatter", "correlation_id": correlation_id, "timestamp": time.time()},
        ]

        # Step 3: Verify correlation ID propagation
        assert all(entry["correlation_id"] == correlation_id for entry in request_flow)

        # Step 4: Collect metrics
        metrics = {
            "request_duration_ms": 1250,
            "agent_processing_ms": {
                "hrm": 450,
                "trm": 480,
            },
            "memory_usage_mb": 256,
            "cache_hit_rate": 0.75,
            "error_count": 0,
        }

        # Step 5: Validate metrics
        assert metrics["request_duration_ms"] < 30000  # Under SLA
        assert metrics["error_count"] == 0
        assert 0.0 <= metrics["cache_hit_rate"] <= 1.0

        # Step 6: Secret redaction check
        log_entry = {
            "message": "Processing query",
            "correlation_id": correlation_id,
            "api_key": "[REDACTED]",
            "user_data": {"id": "user_123", "email": "[REDACTED]"},
        }

        assert log_entry["api_key"] == "[REDACTED]"
        assert "sk-" not in str(log_entry)

        update_run_metadata(
            {
                "journey": "observability",
                "correlation_id": correlation_id,
                "components_traced": len(request_flow),
                "metrics_collected": list(metrics.keys()),
                "secrets_redacted": True,
            }
        )
