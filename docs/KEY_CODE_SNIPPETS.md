# Key Code Snippets - Architecture Patterns

## 1. AsyncAgentBase - Core Agent Pattern

```python
# From: src/framework/agents/base.py
class AsyncAgentBase(ABC):
    """Base class for all agents with async-first design."""
    
    def __init__(
        self,
        model_adapter: LLMClient,
        logger: Any = None,
        name: str | None = None,
        metrics_collector: MetricsCollector | None = None,
        **config: Any,
    ):
        self.model_adapter = model_adapter
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.name = name or self.__class__.__name__
        self.metrics = metrics_collector or NoOpMetricsCollector()
        self.config = config

    @abstractmethod
    async def _process_impl(self, context: AgentContext) -> AgentResult:
        """Core processing logic implemented by subclasses."""
        pass

    async def process(
        self,
        query: str | None = None,
        context: AgentContext | None = None,
        *,
        rag_context: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Main entry point with timing and metrics collection."""
        # Build context if not provided
        if context is None:
            context = AgentContext(
                query=query,
                rag_context=rag_context,
                additional_context=kwargs,
            )
        
        # Run processing with error handling
        try:
            context = await self.pre_process(context)
            result = await self._process_impl(context)
            result = await self.post_process(context, result)
        except Exception as e:
            result = await self.on_error(context, e)
        
        # Return backward-compatible format
        return {
            "response": result.response,
            "metadata": {
                "agent_name": result.agent_name,
                "confidence": result.confidence,
                "processing_time_ms": result.processing_time_ms,
                "success": result.success,
            },
        }
```

## 2. MCTS Core - Deterministic Search

```python
# From: src/framework/mcts/core.py
class MCTSEngine:
    """Deterministic MCTS with seeded RNG and caching."""
    
    async def search(
        self,
        root: MCTSNode,
        num_iterations: int,
        action_generator: Callable,
        state_transition: Callable,
        rollout_policy: RolloutPolicy,
    ) -> Tuple[str, Dict]:
        """Run MCTS iterations with select-expand-simulate-backprop phases."""
        
        for _ in range(num_iterations):
            # 1. Selection & Expansion
            node = self._select_node(root)
            
            # 2. Simulation (rollout)
            rollout_value = await rollout_policy.rollout(
                node.state,
                action_generator,
                state_transition,
            )
            
            # 3. Backpropagation
            self._backpropagate(node, rollout_value)
        
        # 4. Return best action
        best_action = self._select_action(root, policy=self.selection_policy)
        return best_action, self._get_stats(root)

    def _select_node(self, root: MCTSNode) -> MCTSNode:
        """Navigate tree using UCB1 until reaching unexpanded node."""
        node = root
        while not node.is_terminal and node.is_fully_expanded:
            node = node.select_child(self.exploration_weight)
        return node if node.is_terminal else self._expand(node)
```

## 3. Provider-Agnostic LLM Pattern

```python
# From: src/adapters/llm/base.py & __init__.py
@runtime_checkable
class LLMClient(Protocol):
    """Any implementation of this protocol works."""
    
    async def generate(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text from LLM."""
        ...

# Factory function
def create_client(provider: str, **kwargs) -> LLMClient:
    """Create provider-specific client implementing LLMClient protocol."""
    if provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "anthropic":
        return AnthropicClient(**kwargs)
    elif provider == "lmstudio":
        return LMStudioClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# Usage (works with any provider)
async def main():
    client = create_client("openai", model="gpt-4")
    response = await client.generate(
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
    )
    print(response.text)
```

## 4. Neural Meta-Controller - Agent Routing

```python
# From: src/agents/meta_controller/base.py & rnn_controller.py
@dataclass
class MetaControllerFeatures:
    """8-dimensional feature vector for agent selection."""
    hrm_confidence: float
    trm_confidence: float
    mcts_value: float
    consensus_score: float
    last_agent: str
    iteration: int
    query_length: int
    has_rag_context: bool

class RNNMetaController(AbstractMetaController):
    """GRU-based neural controller (99.78% accuracy)."""
    
    def __init__(self, hidden_dim=64, num_layers=1):
        self.model = RNNMetaControllerModel(
            input_dim=8,           # Features above
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_agents=3,          # HRM, TRM, MCTS
        )
    
    def predict(self, features: MetaControllerFeatures) -> MetaControllerPrediction:
        """Select best agent based on features."""
        tensor = features_to_tensor([features])
        logits = self.model(tensor)
        probabilities = F.softmax(logits, dim=-1)[0].detach()
        agent_idx = torch.argmax(probabilities).item()
        agent_name = self.INDEX_TO_LABEL[agent_idx]
        
        return MetaControllerPrediction(
            agent=agent_name,
            confidence=float(probabilities[agent_idx]),
            probabilities={
                "hrm": float(probabilities[0]),
                "trm": float(probabilities[1]),
                "mcts": float(probabilities[2]),
            },
        )
```

## 5. LangGraph State Machine Integration

```python
# From: src/framework/graph/ (LangGraph orchestration package)
class AgentState(TypedDict):
    """Shared state across all graph nodes."""
    query: str
    use_mcts: bool
    use_rag: bool
    rag_context: NotRequired[str]
    hrm_results: NotRequired[Dict]
    trm_results: NotRequired[Dict]
    agent_outputs: Annotated[List[Dict], operator.add]
    mcts_stats: NotRequired[Dict]
    confidence_scores: NotRequired[Dict[str, float]]

async def initialize_state(state: AgentState) -> AgentState:
    """Entry point - initialize shared state."""
    return {
        **state,
        "agent_outputs": [],
        "confidence_scores": {},
    }

async def execute_mcts_node(state: AgentState) -> AgentState:
    """Execute MCTS simulation if requested."""
    if not state.get("use_mcts"):
        return state
    
    engine = MCTSEngine(seed=42, exploration_weight=1.414)
    root = MCTSNode(state=MCTSState(state_id="root", features={}), rng=engine.rng)
    
    best_action, stats = await engine.search(
        root=root,
        num_iterations=100,
        action_generator=lambda s: ["action_a", "action_b", "action_c"],
        state_transition=lambda s, a: MCTSState(f"{s.state_id}_{a}"),
        rollout_policy=HybridRolloutPolicy(),
    )
    
    return {
        **state,
        "mcts_best_action": best_action,
        "mcts_stats": stats,
        "agent_outputs": state["agent_outputs"] + [{"agent": "mcts", "action": best_action}],
    }

# Build graph
graph = StateGraph(AgentState)
graph.add_node("init", initialize_state)
graph.add_node("rag", retrieve_context)
graph.add_node("mcts", execute_mcts_node)
graph.add_edge("init", "rag")
graph.add_edge("rag", "mcts")

# Compile and run
app = graph.compile(checkpointer=MemorySaver())
result = await app.ainvoke({"query": "Test", "use_mcts": True, "use_rag": True})
```

## 6. Training Pipeline Pattern

```python
# From: src/training/train_rnn.py
class RNNTrainer:
    """Complete training pipeline with early stopping & metrics."""
    
    def __init__(self, hidden_dim=64, epochs=20):
        self.model = RNNMetaControllerModel(
            input_dim=8,
            hidden_dim=hidden_dim,
            num_layers=1,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = epochs
    
    def train(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
    ) -> Dict:
        """Train with validation and early stopping."""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs, y_val)
                val_acc = (val_outputs.argmax(1) == y_val).float().mean()
            
            history["val_loss"].append(val_loss.item())
            history["val_accuracy"].append(val_acc.item())
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    break
        
        return history
```

## 7. Configuration with Pydantic Settings

```python
# From: src/config/settings.py
class Settings(BaseSettings):
    """Secure configuration with environment variable validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )
    
    # LLM Provider
    LLM_PROVIDER: LLMProvider = Field(default=LLMProvider.OPENAI)
    OPENAI_API_KEY: Optional[SecretStr] = None
    ANTHROPIC_API_KEY: Optional[SecretStr] = None
    
    # MCTS Parameters
    MCTS_ITERATIONS: int = Field(default=100, ge=1, le=100000)
    MCTS_C: float = Field(default=1.414, ge=0, le=10)
    SEED: int = Field(default=42)
    
    # Storage
    PINECONE_API_KEY: Optional[SecretStr] = None
    
    # Logging
    LOG_LEVEL: LogLevel = LogLevel.INFO

# Usage
def main():
    settings = get_settings()  # Loads from .env automatically
    
    client = create_client(
        settings.LLM_PROVIDER,
        api_key=settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else None,
    )
```

## 8. Composite & Parallel Agents

```python
# From: src/framework/agents/base.py
class ParallelAgent(CompositeAgent):
    """Run multiple agents concurrently, return best result."""
    
    async def _process_impl(self, context: AgentContext) -> AgentResult:
        # Run all sub-agents in parallel
        tasks = [agent.process(context=context) for agent in self.sub_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate: highest confidence wins
        successful_results = [r for r in results if isinstance(r, dict)]
        if not successful_results:
            return AgentResult(response="All agents failed", confidence=0.0)
        
        best = max(successful_results, key=lambda r: r["metadata"]["confidence"])
        return AgentResult(
            response=best["response"],
            confidence=best["metadata"]["confidence"],
            metadata={"aggregation": "highest_confidence", "sub_results": successful_results},
        )

class SequentialAgent(CompositeAgent):
    """Chain agents: output of one becomes context for next."""
    
    async def _process_impl(self, context: AgentContext) -> AgentResult:
        current_context = context
        
        for agent in self.sub_agents:
            result = await agent.process(context=current_context)
            
            # Update context with this agent's output
            current_context = AgentContext(
                query=current_context.query,
                rag_context=result["response"],  # Previous output -> context
                additional_context={
                    **current_context.additional_context,
                    f"{agent.name}_result": result["response"],
                },
            )
        
        return AgentResult(response=current_context.rag_context, confidence=0.9)
```

---

## File Locations Quick Reference

| Purpose | File |
|---------|------|
| Agent Base Classes | `src/framework/agents/base.py` |
| MCTS Engine | `src/framework/mcts/core.py` |
| MCTS Config | `src/framework/mcts/config.py` |
| LLM Adapters | `src/adapters/llm/base.py` |
| LLM Factory | `src/adapters/llm/__init__.py` |
| Settings | `src/config/settings.py` |
| Meta-Controller Base | `src/agents/meta_controller/base.py` |
| RNN Controller | `src/agents/meta_controller/rnn_controller.py` |
| Training (RNN) | `src/training/train_rnn.py` |
| Training (BERT) | `src/training/train_bert_lora.py` |
| Data Generator | `src/training/data_generator.py` |
| Graph Integration | `src/framework/graph/` |
