# Troubleshooting Playbook
## LangGraph Multi-Agent MCTS Framework

**Purpose:** Quick reference for common issues and resolutions
**Audience:** Developers, SRE, QA engineers
**Last Updated:** 2025-11-19

---

## Table of Contents

1. [Setup and Configuration Issues](#setup-and-configuration-issues)
2. [LangSmith Tracing Issues](#langsmith-tracing-issues)
3. [Agent Behavior Issues](#agent-behavior-issues)
4. [MCTS Performance Issues](#mcts-performance-issues)
5. [API and Integration Issues](#api-and-integration-issues)
6. [Test Failures](#test-failures)
7. [CI/CD Issues](#cicd-issues)
8. [Production Debugging](#production-debugging)

---

## Setup and Configuration Issues

### Issue: `ModuleNotFoundError: No module named 'langchain'`

**Symptoms:**
```
ImportError: No module named 'langchain'
```

**Root Cause:**
Dependencies not installed or wrong Python environment.

**Resolution:**
```bash
# Ensure you're in the right directory
cd langgraph_multi_agent_mcts

# Activate virtual environment (if using)
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r training/requirements.txt

# Verify installation
python -c "import langchain; print(langchain.__version__)"
```

**Prevention:**
- Always use a virtual environment
- Run `python scripts/verification/verify_setup.py` before starting work

---

### Issue: `LANGSMITH_API_KEY not set`

**Symptoms:**
```
ValueError: LANGSMITH_API_KEY environment variable is not set
```

**Root Cause:**
Missing or incorrect LangSmith configuration.

**Resolution:**

1. **Get API Key:**
   - Go to https://smith.langchain.com/
   - Settings → API Keys → Create Key

2. **Set Environment Variables:**
   ```bash
   # Linux/Mac
   export LANGSMITH_API_KEY="your-key-here"
   export LANGSMITH_PROJECT="your-project-name"
   export LANGSMITH_TRACING_ENABLED="true"

   # Windows PowerShell
   $env:LANGSMITH_API_KEY="your-key-here"
   $env:LANGSMITH_PROJECT="your-project-name"
   $env:LANGSMITH_TRACING_ENABLED="true"
   ```

3. **Or use `.env` file:**
   ```bash
   # Create .env file
   cat > .env << EOF
   LANGSMITH_API_KEY=your-key-here
   LANGSMITH_PROJECT=your-project-name
   LANGSMITH_TRACING_ENABLED=true
   EOF

   # Load in Python
   from dotenv import load_dotenv
   load_dotenv()
   ```

**Verification:**
```bash
python -c "import os; print(os.getenv('LANGSMITH_API_KEY'))"
```

**Prevention:**
- Document environment setup in team onboarding
- Use `.env.example` template file
- See [SECRETS_MANAGEMENT.md](../SECRETS_MANAGEMENT.md)

---

### Issue: `OpenAI API rate limit exceeded`

**Symptoms:**
```
RateLimitError: Rate limit exceeded. Please try again later.
```

**Root Cause:**
Too many API requests in short time period.

**Resolution:**

**Immediate:**
```python
# Add retry logic with exponential backoff
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def call_llm(prompt):
    return openai.ChatCompletion.create(...)
```

**Long-term:**
```python
# Implement rate limiting
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=50, period=60)  # 50 calls per minute
def call_llm(prompt):
    return openai.ChatCompletion.create(...)
```

**Prevention:**
- Use caching for repeated queries
- Implement request batching
- Monitor usage in OpenAI dashboard
- Consider upgrading API tier

---

## LangSmith Tracing Issues

### Issue: Traces not appearing in LangSmith UI

**Symptoms:**
- Tests run successfully
- No traces visible in LangSmith dashboard

**Diagnosis:**
```python
# Check if tracing is enabled
import os
print("Tracing enabled:", os.getenv("LANGSMITH_TRACING_ENABLED"))
print("API key set:", bool(os.getenv("LANGSMITH_API_KEY")))
print("Project:", os.getenv("LANGSMITH_PROJECT"))
```

**Resolution:**

1. **Verify environment variables:**
   ```bash
   python -c "
   import os
   from langsmith import Client
   client = Client()
   print('Connected to:', client.api_url)
   print('Project:', client.get_project())
   "
   ```

2. **Check decorator usage:**
   ```python
   # Ensure decorator is applied
   from tests.utils.langsmith_tracing import trace_e2e_test

   @trace_e2e_test(scenario="test_scenario")  # ← Must have decorator
   def test_my_feature():
       pass
   ```

3. **Verify network connectivity:**
   ```bash
   curl -H "Authorization: Bearer $LANGSMITH_API_KEY" \
        https://api.smith.langchain.com/api/v1/sessions
   ```

**Common Mistakes:**
- Forgot to set `LANGSMITH_TRACING_ENABLED=true`
- Wrong project name (case-sensitive)
- Firewall blocking api.smith.langchain.com
- Running tests too quickly (traces may have 1-2 minute delay)

**Prevention:**
- Add tracing verification to CI setup
- Use `scripts/smoke_test_traced.py` to test tracing

---

### Issue: Trace hierarchy is incorrect

**Symptoms:**
- Traces appear flat instead of nested
- Parent-child relationships missing

**Root Cause:**
Incorrect context propagation in LangSmith tracing.

**Resolution:**

**Check context propagation:**
```python
from langsmith import Client
from langchain.callbacks import tracing_v2_enabled

# Ensure context is preserved
with tracing_v2_enabled(project_name="your-project") as cb:
    # All calls within this block will be traced as children
    hrm_result = hrm_agent.decompose(query)
    trm_result = trm_agent.refine(hrm_result)
```

**For async code:**
```python
import asyncio
from langsmith import Client

async def traced_workflow():
    # Use run_tree to create hierarchy
    client = Client()
    with client.run_tree("workflow", run_type="chain"):
        with client.run_tree("hrm_phase", run_type="chain"):
            await hrm_agent.decompose_async(query)
        with client.run_tree("trm_phase", run_type="chain"):
            await trm_agent.refine_async(result)
```

**Prevention:**
- Use provided tracing decorators from [tests/utils/langsmith_tracing.py](../../tests/utils/langsmith_tracing.py)
- Review [LANGSMITH_E2E.md](../LANGSMITH_E2E.md) for hierarchy patterns

---

## Agent Behavior Issues

### Issue: HRM decomposition is too generic

**Symptoms:**
- HRM returns vague or generic tasks
- Tasks don't match query specifics

**Diagnosis:**
```python
# Check HRM prompt and LLM parameters
from src.agents.hrm_agent import HRMAgent

agent = HRMAgent()
print("Prompt template:", agent.prompt_template)
print("LLM config:", agent.llm_config)
```

**Resolution:**

1. **Improve prompt specificity:**
   ```python
   # Before
   prompt = "Break this query into tasks: {query}"

   # After
   prompt = """
   Analyze this {domain} query and decompose it into specific, actionable tasks.
   Query: {query}

   Guidelines:
   - Each task should be concrete and measurable
   - Focus on {domain}-specific considerations
   - Order tasks by logical dependencies
   """
   ```

2. **Add examples (few-shot):**
   ```python
   prompt = """
   Example 1:
   Query: "Urban warfare tactics"
   Tasks:
   1. Analyze urban terrain characteristics
   2. Identify force composition needs
   3. Develop Rules of Engagement

   Now decompose this query:
   {query}
   """
   ```

3. **Tune LLM parameters:**
   ```python
   # Increase temperature for more creative decomposition
   llm = ChatOpenAI(temperature=0.7)  # Default is 0.0

   # Or try a more capable model
   llm = ChatOpenAI(model="gpt-4")  # Instead of gpt-3.5-turbo
   ```

**Prevention:**
- Maintain prompt versioning in git
- Add unit tests for decomposition quality
- Monitor decomposition quality in LangSmith experiments

---

### Issue: TRM refinement not converging

**Symptoms:**
- TRM runs maximum iterations without convergence
- Quality improves very slowly or not at all

**Diagnosis:**
```python
# Check convergence threshold and improvement metrics
from src.agents.trm_agent import TRMAgent

agent = TRMAgent()
print("Convergence threshold:", agent.convergence_threshold)
print("Max iterations:", agent.max_iterations)

# Run with debug logging
agent.refine(task, debug=True)
```

**Resolution:**

1. **Adjust convergence threshold:**
   ```python
   # If too strict (e.g., 1% improvement required)
   agent.convergence_threshold = 0.05  # Allow 5% improvement

   # If too loose (e.g., 20% improvement required)
   agent.convergence_threshold = 0.10  # Require 10% improvement
   ```

2. **Improve critique prompt:**
   ```python
   critique_prompt = """
   Analyze this solution critically:
   {solution}

   Identify specific weaknesses:
   1. Missing information
   2. Logical gaps
   3. Unclear explanations
   4. Factual errors

   Provide concrete improvement suggestions.
   """
   ```

3. **Use a stronger model for critique:**
   ```python
   # Use GPT-4 for critique, GPT-3.5 for refinement
   critique_llm = ChatOpenAI(model="gpt-4")
   refine_llm = ChatOpenAI(model="gpt-3.5-turbo")
   ```

**Prevention:**
- Monitor TRM convergence rates in LangSmith
- Run experiments with different thresholds
- Add timeout logic to prevent infinite loops

---

### Issue: MCTS selects suboptimal moves

**Symptoms:**
- MCTS chooses clearly worse options
- Win probability estimates are inaccurate

**Diagnosis:**
```python
# Visualize MCTS tree
from src.framework.mcts import MCTSEngine

engine = MCTSEngine()
result = engine.search(initial_state, iterations=100)

# Print tree statistics
print("Root visit count:", result.root.visit_count)
print("Root win rate:", result.root.win_rate)
for child in result.root.children:
    print(f"Child: visits={child.visit_count}, win_rate={child.win_rate}")
```

**Resolution:**

1. **Increase iteration count:**
   ```python
   # If using too few iterations (e.g., 50)
   result = engine.search(state, iterations=200)  # Or more
   ```

2. **Tune exploration constant:**
   ```python
   # If exploiting too much (always picking current best)
   engine.exploration_constant = 2.0  # Increase from default 1.41

   # If exploring too much (trying random moves)
   engine.exploration_constant = 1.0  # Decrease from default 1.41
   ```

3. **Fix simulation rollout bias:**
   ```python
   # Ensure rollout is unbiased or uses learned policy
   def simulate(state):
       while not state.is_terminal():
           # Use uniform random or learned policy
           action = random.choice(state.legal_actions())
           state = state.apply_action(action)
       return state.get_reward()
   ```

4. **Check backpropagation logic:**
   ```python
   def backpropagate(node, reward):
       while node is not None:
           node.visit_count += 1
           node.total_reward += reward
           node.win_rate = node.total_reward / node.visit_count
           node = node.parent  # ← Ensure this traverses correctly
   ```

**Prevention:**
- Unit test UCB1 calculation
- Validate win probability against ground truth
- Run experiments comparing different MCTS configs

---

## MCTS Performance Issues

### Issue: MCTS is too slow

**Symptoms:**
- High latency (> 5 seconds for 100 iterations)
- Timeouts in production

**Diagnosis:**
```python
import time

start = time.time()
result = mcts_engine.search(state, iterations=100)
elapsed = time.time() - start

print(f"Total time: {elapsed:.2f}s")
print(f"Time per iteration: {elapsed/100*1000:.2f}ms")
```

**Resolution:**

1. **Profile to find bottleneck:**
   ```python
   import cProfile
   import pstats

   profiler = cProfile.Profile()
   profiler.enable()
   result = mcts_engine.search(state, iterations=100)
   profiler.disable()

   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(10)  # Top 10 slowest functions
   ```

2. **Common bottlenecks and fixes:**

   **Slow LLM calls:**
   ```python
   # Cache simulation results
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def simulate(state_hash):
       # Simulation logic
       pass
   ```

   **Inefficient state copying:**
   ```python
   # Use copy-on-write or immutable data structures
   from dataclasses import dataclass

   @dataclass(frozen=True)
   class GameState:
       # Immutable state
       pass
   ```

   **Too many iterations:**
   ```python
   # Reduce iterations or use early stopping
   result = mcts_engine.search(
       state,
       iterations=100,  # Reduced from 500
       early_stop_threshold=0.95  # Stop if win rate > 95%
   )
   ```

3. **Parallelize simulations:**
   ```python
   import concurrent.futures

   def parallel_mcts(state, iterations=100, workers=4):
       # Run iterations in parallel
       with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
           futures = [
               executor.submit(mcts_engine.search, state, iterations=iterations//workers)
               for _ in range(workers)
           ]
           results = [f.result() for f in futures]
       # Merge results
       return merge_mcts_results(results)
   ```

**Prevention:**
- Set performance budgets (e.g., max 2s latency)
- Monitor MCTS latency in production
- Run load tests regularly

---

## API and Integration Issues

### Issue: `/analyze` endpoint returns 500 error

**Symptoms:**
```
HTTP 500 Internal Server Error
{"detail": "Unexpected error during analysis"}
```

**Diagnosis:**

1. **Check API logs:**
   ```bash
   # If using Docker
   docker logs langgraph-api

   # If running locally
   tail -f logs/api.log
   ```

2. **Enable debug mode:**
   ```python
   # In src/api/inference_server.py
   app = FastAPI(debug=True)
   ```

3. **Check LangSmith trace:**
   - Find trace for the failing request
   - Look for exceptions in trace metadata

**Resolution:**

**Common causes:**

1. **Missing environment variable:**
   ```python
   # Add validation at startup
   required_vars = ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]
   for var in required_vars:
       if not os.getenv(var):
           raise ValueError(f"{var} is not set")
   ```

2. **Unhandled exception:**
   ```python
   # Add try-except in endpoint
   @app.post("/analyze")
   async def analyze(query: str):
       try:
           result = await analyze_query(query)
           return result
       except Exception as e:
           logger.error(f"Analysis failed: {e}", exc_info=True)
           return JSONResponse(
               status_code=500,
               content={"detail": str(e)}
           )
   ```

3. **Timeout:**
   ```python
   # Add timeout to long-running operations
   import asyncio

   try:
       result = await asyncio.wait_for(
           mcts_engine.search_async(state),
           timeout=30.0  # 30 second timeout
       )
   except asyncio.TimeoutError:
       return {"error": "MCTS search timed out"}
   ```

**Prevention:**
- Add comprehensive error handling
- Use Sentry or similar for error tracking
- Set up health check endpoint

---

### Issue: Pinecone integration failing

**Symptoms:**
```
PineconeConnectionError: Failed to connect to Pinecone index
```

**Diagnosis:**
```python
# Test Pinecone connectivity
import pinecone

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

# List indexes
print("Available indexes:", pinecone.list_indexes())

# Test query
index = pinecone.Index("your-index-name")
results = index.query(vector=[0.1]*1536, top_k=1)
print("Query successful:", results)
```

**Resolution:**

1. **Verify API key and environment:**
   ```bash
   # Check Pinecone dashboard
   echo $PINECONE_API_KEY
   echo $PINECONE_ENVIRONMENT  # Should be like "us-west1-gcp"
   ```

2. **Check index exists:**
   ```python
   # Create index if missing
   if "your-index-name" not in pinecone.list_indexes():
       pinecone.create_index(
           "your-index-name",
           dimension=1536,
           metric="cosine"
       )
   ```

3. **Handle connection errors gracefully:**
   ```python
   from tenacity import retry, stop_after_attempt

   @retry(stop=stop_after_attempt(3))
   def query_pinecone(vector):
       index = pinecone.Index("your-index-name")
       return index.query(vector=vector, top_k=10)
   ```

**Prevention:**
- Add Pinecone health check to startup
- Fall back to local FAISS if Pinecone unavailable
- See [PINECONE_INTEGRATION.md](../PINECONE_INTEGRATION.md)

---

## Test Failures

### Issue: E2E tests failing intermittently

**Symptoms:**
- Tests pass locally, fail in CI
- "Flaky" tests that pass/fail randomly

**Common Causes:**

1. **Non-deterministic LLM outputs:**
   ```python
   # Fix: Use mocks for E2E tests
   from unittest.mock import patch

   @patch('src.agents.hrm_agent.call_llm')
   def test_e2e_flow(mock_llm):
       mock_llm.return_value = {"tasks": ["Task 1", "Task 2"]}
       result = analyze_query("test query")
       assert len(result["tasks"]) == 2
   ```

2. **Race conditions in async code:**
   ```python
   # Fix: Use proper async coordination
   import asyncio

   async def test_concurrent_agents():
       # Ensure proper awaiting
       results = await asyncio.gather(
           hrm_agent.decompose_async(query),
           trm_agent.refine_async(task)
       )
   ```

3. **Timing-dependent assertions:**
   ```python
   # Bad: Depends on exact timing
   assert latency < 1.0

   # Good: Allow reasonable margin
   assert latency < 2.0  # Or use ranges
   ```

**Prevention:**
- Use deterministic fixtures
- Add retries for genuinely flaky external calls
- Mark flaky tests with `@pytest.mark.flaky(reruns=3)`

---

### Issue: Coverage threshold not met

**Symptoms:**
```
FAIL Required test coverage of 50% not met. Got 45%
```

**Resolution:**

1. **Identify uncovered code:**
   ```bash
   pytest --cov=src --cov-report=html
   open htmlcov/index.html  # View coverage report
   ```

2. **Add missing tests:**
   - Focus on uncovered branches
   - Add tests for error cases
   - Test edge cases

3. **Exclude non-critical files:**
   ```ini
   # In pyproject.toml
   [tool.coverage.run]
   omit = [
       "*/tests/*",
       "*/migrations/*",
       "*/scripts/*"
   ]
   ```

**Prevention:**
- Enforce coverage in PR reviews
- Add coverage diff reporting in CI
- Set realistic coverage targets (50% is reasonable)

---

## CI/CD Issues

### Issue: GitHub Actions failing on `pip install`

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement package-name
```

**Resolution:**

1. **Check dependency pinning:**
   ```bash
   # Regenerate requirements with pinned versions
   pip freeze > requirements.txt
   ```

2. **Use Python version matrix:**
   ```yaml
   # In .github/workflows/ci.yml
   strategy:
     matrix:
       python-version: ["3.11", "3.12"]
   ```

3. **Add dependency caching:**
   ```yaml
   - name: Cache pip dependencies
     uses: actions/cache@v3
     with:
       path: ~/.cache/pip
       key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
   ```

**Prevention:**
- Test locally in clean environment before pushing
- Use `pip-compile` for deterministic requirements
- Pin major versions for stability

---

## Production Debugging

### Issue: High latency in production

**Symptoms:**
- P95 latency > 5 seconds
- User complaints about slow responses

**Diagnosis:**

1. **Check LangSmith traces:**
   - Filter by: `latency > 5000ms`
   - Identify: Which phase is slow (HRM, TRM, MCTS)?

2. **Check infrastructure metrics:**
   ```bash
   # CPU, memory, network
   kubectl top pods  # If using Kubernetes
   docker stats      # If using Docker
   ```

3. **Profile in production:**
   ```python
   # Add performance logging
   import time
   import logging

   def analyze_query(query):
       start = time.time()

       hrm_start = time.time()
       hrm_result = hrm_agent.decompose(query)
       logging.info(f"HRM latency: {time.time() - hrm_start:.2f}s")

       trm_start = time.time()
       trm_result = trm_agent.refine(hrm_result)
       logging.info(f"TRM latency: {time.time() - trm_start:.2f}s")

       logging.info(f"Total latency: {time.time() - start:.2f}s")
   ```

**Resolution:**

1. **Scale horizontally:**
   ```bash
   # Add more replicas
   kubectl scale deployment langgraph-api --replicas=5
   ```

2. **Add caching:**
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def analyze_cached(query):
       return analyze_query(query)
   ```

3. **Optimize slow agents:**
   - Reduce MCTS iterations
   - Use faster LLM (gpt-3.5-turbo instead of gpt-4)
   - Parallelize independent operations

**Prevention:**
- Set SLAs and monitor them (see [SLA.md](../SLA.md))
- Add performance tests to CI
- Use auto-scaling based on latency metrics

---

## Getting Help

### Where to Look First

1. **Documentation:**
   - [architecture.md](../architecture.md)
   - [LANGSMITH_E2E.md](../LANGSMITH_E2E.md)
   - [API_CONFIGURATION_GUIDE.md](../API_CONFIGURATION_GUIDE.md)

2. **LangSmith Traces:**
   - Most issues are visible in traces
   - Look for exceptions, high latency, unexpected outputs

3. **Logs:**
   - API logs: `logs/api.log`
   - Agent logs: `logs/agents.log`
   - MCTS logs: `logs/mcts.log`

4. **Tests:**
   - Run relevant tests to isolate issue
   - Check if tests are passing in CI

### Escalation Path

1. **Self-Service:** Check this playbook and documentation
2. **Peer Help:** Ask in team chat or training forum
3. **Office Hours:** Attend weekly training office hours
4. **Instructor:** Email training lead for complex issues
5. **Incident:** Page on-call if production is down

### Reporting Bugs

When reporting a bug, include:
- **Description:** What went wrong?
- **Expected Behavior:** What should have happened?
- **Steps to Reproduce:** Exact commands to recreate
- **Logs:** Relevant error messages
- **LangSmith Trace:** Link to trace (if applicable)
- **Environment:** OS, Python version, package versions

**Template:**
```markdown
## Bug Report

**Description:**
E2E test failing with "MCTS timeout"

**Expected:**
MCTS should complete within 5 seconds

**Steps to Reproduce:**
1. Run: pytest tests/e2e/test_mcts_simulation_flow.py::test_mcts_100_iterations
2. Wait for timeout error

**Logs:**
```
ERROR: MCTS search timed out after 5.0s
```

**Trace:** https://smith.langchain.com/trace/abc123

**Environment:**
- OS: Ubuntu 22.04
- Python: 3.11.5
- langgraph: 0.2.0
```

---

**Last Updated:** 2025-11-19

For questions or suggestions, contact: [Training Program Lead]
