# Module 7: CI/CD & Observability Integration

**Duration:** 10 hours (2 days)
**Format:** Workshop + Production Readiness Lab
**Difficulty:** Advanced
**Prerequisites:** Completed Modules 1-6, basic Docker and GitHub Actions knowledge

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Configure GitHub Actions** for automated CI/CD pipelines
2. **Implement observability** with OpenTelemetry, Prometheus, and Grafana
3. **Deploy containerized applications** using Docker and Kubernetes
4. **Validate production readiness** using automated checks
5. **Handle incidents** with comprehensive monitoring and alerting

---

## Session 1: CI/CD Pipeline (3 hours)

### Pre-Reading (30 minutes)

Before the session, review:
- [.github/workflows/ci.yml](../../.github/workflows/ci.yml) - CI pipeline configuration
- [DEPLOYMENT_REPORT.md](../reports/DEPLOYMENT_REPORT.md) - Deployment best practices
- GitHub Actions documentation: https://docs.github.com/en/actions

### Lecture: GitHub Actions Fundamentals (60 minutes)

#### CI/CD Pipeline Architecture

**Pipeline Stages:**
```
┌──────────────┐
│  Code Push   │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│         Continuous Integration       │
├──────────────────────────────────────┤
│ 1. Lint (Ruff)                       │
│ 2. Type Check (MyPy)                 │
│ 3. Security Scan (Bandit)            │
│ 4. Dependency Audit (pip-audit)      │
│ 5. Unit Tests (pytest)               │
│ 6. Integration Tests                 │
│ 7. Coverage Report                   │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│      Continuous Deployment           │
├──────────────────────────────────────┤
│ 1. Build Docker Image                │
│ 2. Smoke Tests                       │
│ 3. Push to Registry                  │
│ 4. Deploy to Staging                 │
│ 5. E2E Tests                         │
│ 6. Deploy to Production              │
└──────────────────────────────────────┘
```

#### GitHub Actions Workflow Structure

**Basic workflow (.github/workflows/ci.yml):**
```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:  # Manual trigger

env:
  PYTHON_VERSION: "3.11"

jobs:
  lint:
    name: Lint with Ruff
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install ruff
        run: pip install ruff

      - name: Run ruff linter
        run: ruff check . --output-format=github

      - name: Run ruff formatter check
        run: ruff format --check .

  type-check:
    name: Type Check with MyPy
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install mypy types-psutil types-requests
          pip install -r requirements.txt

      - name: Run mypy
        run: mypy src/ --ignore-missing-imports

  test:
    name: Run Tests
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio

      - name: Run tests with coverage
        run: pytest --cov=src --cov-report=xml --cov-report=html

      - name: Upload coverage reports
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
```

#### Key GitHub Actions Concepts

**1. Triggers (on):**
```yaml
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight
  workflow_dispatch:  # Manual trigger
```

**2. Jobs and Dependencies:**
```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    # ...

  test:
    runs-on: ubuntu-latest
    needs: lint  # Wait for lint to complete
    # ...

  deploy:
    runs-on: ubuntu-latest
    needs: [lint, test]  # Wait for both
    # ...
```

**3. Secrets:**
```yaml
- name: Run with secrets
  env:
    LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: python scripts/run_tests.py
```

**4. Artifacts:**
```yaml
- name: Upload test results
  uses: actions/upload-artifact@v4
  with:
    name: test-results
    path: test-results/

- name: Download test results
  uses: actions/download-artifact@v4
  with:
    name: test-results
```

### Lecture: Security Scanning (45 minutes)

#### Bandit Security Scan

**Configuration:**
```yaml
security-scan:
  name: Security Scan with Bandit
  runs-on: ubuntu-latest

  steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install bandit
      run: pip install bandit[toml]

    - name: Run bandit security scan
      run: bandit -r src/ -f json -o bandit-report.json || true

    - name: Upload bandit report
      uses: actions/upload-artifact@v4
      with:
        name: bandit-security-report
        path: bandit-report.json

    - name: Check for high severity issues
      run: |
        python -c "
        import json, sys
        with open('bandit-report.json') as f:
            data = json.load(f)
        high_severity = [r for r in data.get('results', [])
                         if r.get('issue_severity') == 'HIGH']
        if high_severity:
            print('HIGH SEVERITY ISSUES FOUND:')
            for issue in high_severity:
                print(f\"  {issue['filename']}:{issue['line_number']}: {issue['issue_text']}\")
            sys.exit(1)
        else:
            print('No high severity issues found')
        "
```

#### Dependency Vulnerability Scan

**Configuration:**
```yaml
dependency-audit:
  name: Dependency Vulnerability Scan
  runs-on: ubuntu-latest

  steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install pip-audit
      run: pip install pip-audit

    - name: Run pip-audit
      run: pip-audit --requirement requirements.txt --format json --output audit-report.json || true

    - name: Upload audit report
      uses: actions/upload-artifact@v4
      with:
        name: dependency-audit-report
        path: audit-report.json
```

### Hands-On Exercise: Create CI Pipeline (45 minutes)

**Exercise 1: Complete CI Workflow**

**Objective:** Create a comprehensive CI workflow for your module.

**Requirements:**
1. Lint check (Ruff)
2. Type check (MyPy)
3. Security scan (Bandit)
4. Unit tests with coverage
5. Integration tests
6. Generate and upload reports

**Template:**
```yaml
name: Complete CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  lint:
    # TODO: Add lint job

  type-check:
    # TODO: Add type check job

  security:
    # TODO: Add security scan job

  test:
    name: Run Tests
    runs-on: ubuntu-latest
    needs: [lint, type-check]  # Run after quality checks

    steps:
      # TODO: Add test steps
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: pytest --cov=src --cov-report=xml

      - name: Check coverage threshold
        run: |
          coverage report --fail-under=50
```

**Deliverable:** Working CI workflow file

---

## Session 2: Observability Stack (3 hours)

### Pre-Reading (30 minutes)

- [DEPLOYMENT_REPORT.md](../reports/DEPLOYMENT_REPORT.md) - Observability setup
- OpenTelemetry documentation: https://opentelemetry.io/docs/
- Prometheus documentation: https://prometheus.io/docs/

### Lecture: OpenTelemetry Integration (60 minutes)

#### Observability Pillars

**Three Pillars:**
1. **Traces:** Request flow through system (LangSmith)
2. **Metrics:** Numerical measurements (Prometheus)
3. **Logs:** Event records (structured logging)

#### OpenTelemetry Setup

**Installation:**
```bash
pip install opentelemetry-api opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-grpc \
    opentelemetry-instrumentation-httpx
```

**Basic configuration:**
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Set up tracer provider
trace.set_tracer_provider(TracerProvider())

# Configure OTLP exporter
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",
    insecure=True,
)

# Add span processor
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Get tracer
tracer = trace.get_tracer(__name__)
```

**Instrumenting code:**
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("process_query")
def process_query(query: str) -> dict:
    """Process query with tracing."""
    span = trace.get_current_span()

    # Add attributes
    span.set_attribute("query.length", len(query))
    span.set_attribute("query.type", detect_type(query))

    # Process
    result = do_processing(query)

    # Add result attributes
    span.set_attribute("result.confidence", result["confidence"])

    return result

# Nested spans
@tracer.start_as_current_span("hrm_decomposition")
def hrm_decompose(query: str) -> dict:
    """HRM with nested tracing."""
    with tracer.start_as_current_span("llm_call") as llm_span:
        response = llm.invoke(query)
        llm_span.set_attribute("tokens", len(response))

    with tracer.start_as_current_span("parse_response") as parse_span:
        tasks = parse(response)
        parse_span.set_attribute("task_count", len(tasks))

    return {"tasks": tasks}
```

### Lecture: Prometheus Metrics (60 minutes)

#### Metrics Types

**1. Counter:** Monotonically increasing value
```python
from prometheus_client import Counter

request_counter = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# Increment
request_counter.labels(method='POST', endpoint='/analyze', status='200').inc()
```

**2. Gauge:** Value that can go up or down
```python
from prometheus_client import Gauge

active_requests = Gauge(
    'active_requests',
    'Number of active requests'
)

# Set value
active_requests.set(5)

# Increment/decrement
active_requests.inc()
active_requests.dec()
```

**3. Histogram:** Distribution of values
```python
from prometheus_client import Histogram

request_latency = Histogram(
    'request_duration_seconds',
    'Request latency distribution',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Observe value
with request_latency.time():
    process_request()

# Or manually
request_latency.observe(1.234)
```

**4. Summary:** Similar to histogram, calculates quantiles
```python
from prometheus_client import Summary

response_size = Summary(
    'response_size_bytes',
    'Response size distribution'
)

response_size.observe(1024)
```

#### Application Metrics

**Define custom metrics:**
```python
from prometheus_client import Counter, Histogram, Gauge

# Agent metrics
hrm_requests = Counter(
    'hrm_requests_total',
    'Total HRM agent requests',
    ['status']
)

trm_iterations = Histogram(
    'trm_refinement_iterations',
    'TRM refinement iteration count',
    buckets=[1, 2, 3, 5, 10]
)

mcts_win_probability = Histogram(
    'mcts_win_probability',
    'MCTS win probability distribution',
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)

# System metrics
active_llm_calls = Gauge(
    'active_llm_calls',
    'Number of active LLM calls'
)
```

**Instrumenting agents:**
```python
from prometheus_client import Counter, Histogram
import time

hrm_requests = Counter('hrm_requests_total', 'HRM requests', ['status'])
hrm_latency = Histogram('hrm_latency_seconds', 'HRM latency')

class HRMAgent:
    def decompose(self, query: str) -> dict:
        """Decompose with metrics."""
        start = time.time()

        try:
            result = self._decompose_impl(query)
            hrm_requests.labels(status='success').inc()
            return result
        except Exception as e:
            hrm_requests.labels(status='error').inc()
            raise
        finally:
            hrm_latency.observe(time.time() - start)
```

**Exposing metrics endpoint:**
```python
from fastapi import FastAPI
from prometheus_client import make_asgi_app

app = FastAPI()

# Add prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Now accessible at http://localhost:8000/metrics
```

### Hands-On Exercise: Add Observability (60 minutes)

**Exercise 2: Instrument Application**

**Objective:** Add comprehensive observability to an agent.

**Requirements:**
1. OpenTelemetry tracing for all methods
2. Prometheus metrics for key operations
3. Structured logging
4. Metrics endpoint
5. Test and verify metrics

**Template:**
```python
from opentelemetry import trace
from prometheus_client import Counter, Histogram
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Set up tracing
tracer = trace.get_tracer(__name__)

# Define metrics
requests = Counter('agent_requests_total', 'Total requests', ['agent', 'status'])
latency = Histogram('agent_latency_seconds', 'Latency', ['agent'])
confidence = Histogram('agent_confidence', 'Confidence scores', ['agent'])

class ObservableAgent:
    """Agent with full observability."""

    def __init__(self, name: str):
        self.name = name

    @tracer.start_as_current_span("agent_process")
    def process(self, query: str) -> dict:
        """Process with observability."""
        span = trace.get_current_span()
        span.set_attribute("agent.name", self.name)
        span.set_attribute("query.length", len(query))

        logger.info(f"Processing query: {query[:50]}...", extra={
            "agent": self.name,
            "query_length": len(query),
        })

        # TODO: Add metrics collection
        # TODO: Implement processing
        # TODO: Record results

        return result

# TODO: Create FastAPI app with metrics endpoint
# TODO: Test metrics collection
```

**Deliverable:** Fully instrumented agent with working metrics endpoint

---

## Session 3: Production Readiness (2.5 hours)

### Pre-Reading (30 minutes)

- [scripts/production_readiness_check.py](../../scripts/production_readiness_check.py) - Readiness checks
- [DEPLOYMENT_REPORT.md](../reports/DEPLOYMENT_REPORT.md) - Production requirements

### Lecture: Production Checklist (60 minutes)

#### Production Readiness Criteria

**1. Code Quality:**
- [ ] All tests passing (80%+ coverage)
- [ ] No linting errors
- [ ] Type checking passing
- [ ] Security scan clean

**2. Performance:**
- [ ] Latency targets met (<5s p99)
- [ ] Throughput targets met (>10 qps)
- [ ] Resource limits defined
- [ ] Load testing completed

**3. Reliability:**
- [ ] Error handling comprehensive
- [ ] Retry logic implemented
- [ ] Circuit breakers configured
- [ ] Graceful degradation

**4. Observability:**
- [ ] Distributed tracing enabled
- [ ] Metrics exposed
- [ ] Structured logging
- [ ] Dashboards created
- [ ] Alerts configured

**5. Security:**
- [ ] Authentication enabled
- [ ] Authorization implemented
- [ ] Secrets management
- [ ] TLS/HTTPS enforced
- [ ] Input validation

**6. Operations:**
- [ ] Health checks
- [ ] Readiness checks
- [ ] Deployment automation
- [ ] Rollback procedures
- [ ] Runbooks documented

#### Automated Readiness Checks

**production_readiness_check.py structure:**
```python
import sys
from typing import List, Tuple

def check_tests() -> Tuple[bool, str]:
    """Verify test suite passes."""
    import subprocess

    result = subprocess.run(
        ["pytest", "--tb=short", "-q"],
        capture_output=True,
        text=True,
    )

    passed = result.returncode == 0
    message = "All tests passing" if passed else f"Tests failing:\n{result.stdout}"
    return passed, message

def check_coverage() -> Tuple[bool, str]:
    """Verify coverage threshold."""
    import subprocess

    result = subprocess.run(
        ["pytest", "--cov=src", "--cov-report=term-missing", "--cov-fail-under=50"],
        capture_output=True,
        text=True,
    )

    passed = result.returncode == 0
    message = "Coverage threshold met (50%+)" if passed else "Coverage below threshold"
    return passed, message

def check_linting() -> Tuple[bool, str]:
    """Verify linting passes."""
    import subprocess

    result = subprocess.run(
        ["ruff", "check", "."],
        capture_output=True,
        text=True,
    )

    passed = result.returncode == 0
    message = "No linting errors" if passed else f"Linting errors:\n{result.stdout}"
    return passed, message

def check_type_checking() -> Tuple[bool, str]:
    """Verify type checking passes."""
    import subprocess

    result = subprocess.run(
        ["mypy", "src/", "--ignore-missing-imports"],
        capture_output=True,
        text=True,
    )

    passed = result.returncode == 0
    message = "Type checking passed" if passed else f"Type errors:\n{result.stdout}"
    return passed, message

def check_security() -> Tuple[bool, str]:
    """Verify security scan."""
    import subprocess
    import json

    result = subprocess.run(
        ["bandit", "-r", "src/", "-f", "json"],
        capture_output=True,
        text=True,
    )

    if result.stdout:
        data = json.loads(result.stdout)
        high_severity = [r for r in data.get("results", [])
                         if r.get("issue_severity") == "HIGH"]

        if high_severity:
            return False, f"High severity security issues: {len(high_severity)}"

    return True, "No high severity security issues"

def check_environment() -> Tuple[bool, str]:
    """Verify required environment variables."""
    import os

    required_vars = [
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "OPENAI_API_KEY",
    ]

    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        return False, f"Missing environment variables: {', '.join(missing)}"

    return True, "All required environment variables set"

def main():
    """Run all production readiness checks."""
    checks = [
        ("Tests", check_tests),
        ("Coverage", check_coverage),
        ("Linting", check_linting),
        ("Type Checking", check_type_checking),
        ("Security", check_security),
        ("Environment", check_environment),
    ]

    results = []
    all_passed = True

    print("=" * 60)
    print("PRODUCTION READINESS CHECK")
    print("=" * 60)

    for name, check_func in checks:
        passed, message = check_func()
        results.append((name, passed, message))
        all_passed = all_passed and passed

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n{status} - {name}")
        print(f"  {message}")

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - READY FOR PRODUCTION")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ SOME CHECKS FAILED - NOT READY FOR PRODUCTION")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Lecture: Docker Deployment (45 minutes)

#### Multi-Stage Dockerfile

**Optimized Dockerfile:**
```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Set environment
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose ports
EXPOSE 8000 9090

# Run application
CMD ["uvicorn", "src.api.inference_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose for Development

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"
    environment:
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./src:/app/src
    depends_on:
      - redis
      - prometheus

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
```

### Hands-On Exercise: Production Deployment (45 minutes)

**Exercise 3: Deploy to Docker**

**Objective:** Create production-ready Docker deployment.

**Requirements:**
1. Multi-stage Dockerfile
2. docker-compose.yml with all services
3. Health check endpoint
4. Prometheus metrics endpoint
5. Smoke tests

**Deliverable:** Working Docker deployment

---

## Session 4: Incident Response (1.5 hours)

### Lecture: Monitoring and Alerting (45 minutes)

#### Alert Rules

**Prometheus alert rules:**
```yaml
# monitoring/alerts.yml
groups:
  - name: application_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          rate(http_requests_total{status=~"5.."}[5m])
          / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.99,
            rate(request_duration_seconds_bucket[5m])
          ) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "P99 latency is {{ $value }}s (threshold: 5s)"

      # Low confidence scores
      - alert: LowConfidenceScores
        expr: |
          avg(agent_confidence) < 0.7
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Agent confidence scores low"
          description: "Average confidence is {{ $value }} (threshold: 0.7)"

      # MCTS convergence issues
      - alert: MCTSConvergenceIssues
        expr: |
          rate(mcts_convergence_failures_total[10m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "MCTS convergence issues"
          description: "Convergence failure rate: {{ $value }}/s"
```

#### Incident Runbook

**Example runbook:**
```markdown
# Incident Runbook: High Error Rate

## Alert
**Alert Name:** HighErrorRate
**Severity:** Critical
**Threshold:** >5% error rate for 5 minutes

## Symptoms
- 5xx HTTP responses
- Failed agent executions
- LLM API errors

## Investigation Steps

1. **Check dashboards:**
   - Error rate by endpoint
   - Error distribution by type
   - Recent deployments

2. **Review logs:**
   ```bash
   kubectl logs -l app=langgraph-mcts --tail=100 | grep ERROR
   ```

3. **Check LangSmith traces:**
   - Filter: `success: false`
   - Look for common error patterns
   - Check LLM provider status

4. **Verify dependencies:**
   - LLM provider API status
   - Database connectivity
   - Redis availability

## Resolution Steps

### If LLM Provider Issue:
1. Switch to fallback model:
   ```bash
   kubectl set env deployment/langgraph-mcts FALLBACK_MODEL=gpt-3.5-turbo
   ```

2. Monitor error rate
3. Notify LLM provider support

### If Database Issue:
1. Check connection pool:
   ```bash
   kubectl exec -it langgraph-mcts-xxx -- python -c "from src.storage import check_db; check_db()"
   ```

2. Restart if necessary:
   ```bash
   kubectl rollout restart deployment/langgraph-mcts
   ```

### If Code Issue:
1. Identify problematic deployment:
   ```bash
   kubectl rollout history deployment/langgraph-mcts
   ```

2. Rollback to previous version:
   ```bash
   kubectl rollout undo deployment/langgraph-mcts
   ```

3. Create hotfix branch
4. Deploy fix through CI/CD

## Communication
- Update status page
- Notify #incidents channel
- Post-mortem after resolution

## Prevention
- Add retry logic for transient errors
- Improve error handling
- Add more comprehensive testing
```

### Hands-On Exercise: Incident Simulation (45 minutes)

**Exercise 4: Handle Simulated Incident**

**Objective:** Respond to a simulated production incident.

**Scenario:** High latency alert triggered (P99 > 5s)

**Tasks:**
1. Review monitoring dashboards
2. Identify root cause using traces
3. Implement fix
4. Verify resolution
5. Write post-mortem

**Deliverable:** Incident report with resolution steps

---

## Module 7 Assessment

### Practical Assessment

**Task:** Build complete CI/CD pipeline with production deployment

**Requirements:**
1. GitHub Actions workflow (20 points)
   - Lint, type check, security scan
   - Unit and integration tests
   - Docker build and push

2. Observability (25 points)
   - OpenTelemetry tracing
   - Prometheus metrics
   - Grafana dashboard

3. Docker deployment (20 points)
   - Multi-stage Dockerfile
   - docker-compose.yml
   - Health checks

4. Production readiness (20 points)
   - Automated checks script
   - All checks passing
   - Documentation

5. Incident response (15 points)
   - Alert rules
   - Runbooks
   - Monitoring setup

**Total:** 100 points (passing: 70+)

**Submission:** GitHub repository with complete CI/CD setup

---

## Assessment Rubric

| Criteria | Weight | Description |
|----------|--------|-------------|
| **CI/CD Pipeline** | 20% | Complete automated pipeline |
| **Observability** | 25% | Comprehensive monitoring and tracing |
| **Deployment** | 20% | Production-ready Docker deployment |
| **Readiness** | 20% | All production checks passing |
| **Operations** | 15% | Alerts, runbooks, incident handling |

**Minimum Passing:** 70% overall

---

## Additional Resources

### Reading
- [.github/workflows/ci.yml](../../.github/workflows/ci.yml) - CI configuration
- [DEPLOYMENT_REPORT.md](../reports/DEPLOYMENT_REPORT.md) - Deployment guide
- [scripts/production_readiness_check.py](../../scripts/production_readiness_check.py) - Readiness checks

### Tools
- GitHub Actions: https://docs.github.com/en/actions
- Docker: https://docs.docker.com/
- Prometheus: https://prometheus.io/docs/
- Grafana: https://grafana.com/docs/

### Office Hours
- When: [Schedule TBD]
- Topics: CI/CD troubleshooting, deployment strategies, incident handling

---

## Congratulations!

You have completed the LangGraph Multi-Agent MCTS Framework training program!

### Next Steps

1. **Apply your knowledge** to real projects
2. **Contribute** to the framework
3. **Share learnings** with the community
4. **Continue learning** with advanced topics

### Advanced Topics (Future Modules)

- **Module 8:** Advanced MCTS Techniques
- **Module 9:** Neural Network Integration
- **Module 10:** Multi-Model Orchestration
- **Module 11:** Production Scaling Strategies
- **Module 12:** Cost Optimization

---

**Training Program Complete!** 🎉
