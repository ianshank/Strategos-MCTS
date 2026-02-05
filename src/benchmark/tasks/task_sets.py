"""
Default benchmark task set definitions.

Tasks are defined as data (not code) for extensibility.
Add new tasks by appending to the appropriate task set list
or by loading from external JSON/YAML configuration.
"""

from __future__ import annotations

from src.benchmark.tasks.models import BenchmarkTask, TaskCategory, TaskComplexity

# ─── Task Set A: Software Quality Engineering ───

TASK_A1_CODE_REVIEW = BenchmarkTask(
    task_id="A1",
    category=TaskCategory.QE,
    description="Code Review: Identify bugs in PR diff",
    input_data=(
        "Review this Python code diff for a new MCTS node selection function:\n"
        "```python\n"
        "def select_node(self, node):\n"
        "    best_score = -float('inf')\n"
        "    best_child = None\n"
        "    for child in node.children:\n"
        "        exploitation = child.value / child.visits\n"
        "        exploration = math.sqrt(math.log(node.visits) / child.visits)\n"
        "        score = exploitation + self.c * exploration\n"
        "        if score > best_score:\n"
        "            best_score = score\n"
        "            best_child = child\n"
        "    return best_child\n"
        "```\n\n"
        "Identify all bugs, edge cases, and improvement opportunities."
    ),
    expected_outputs=(
        "division by zero when child.visits is 0",
        "no handling for leaf nodes with no children",
        "UCB1 formula correctness",
        "thread safety concerns",
    ),
    complexity=TaskComplexity.MEDIUM,
    metadata={"domain": "mcts", "language": "python"},
)

TASK_A2_SECURITY_ANALYSIS = BenchmarkTask(
    task_id="A2",
    category=TaskCategory.QE,
    description="Security Vulnerability Analysis: Analyze codebase for vulnerabilities",
    input_data=(
        "Analyze the following agent orchestration API endpoint for security vulnerabilities:\n"
        "```python\n"
        "from fastapi import FastAPI, Request\n"
        "import subprocess\n"
        "import sqlite3\n\n"
        "app = FastAPI()\n"
        "db = sqlite3.connect('agents.db')\n\n"
        "@app.post('/agents/execute')\n"
        "async def execute_agent(request: Request):\n"
        "    body = await request.json()\n"
        "    agent_name = body['agent_name']\n"
        "    query = body['query']\n"
        "    \n"
        "    # Log the request\n"
        "    db.execute(f\"INSERT INTO logs (agent, query) VALUES ('{agent_name}', '{query}')\")\n"
        "    db.commit()\n"
        "    \n"
        "    # Execute agent\n"
        "    result = subprocess.run(\n"
        "        f'python agents/{agent_name}.py --query \"{query}\"',\n"
        "        shell=True, capture_output=True, text=True\n"
        "    )\n"
        "    \n"
        "    return {'output': result.stdout, 'agent': agent_name}\n"
        "```\n\n"
        "Map vulnerabilities to OWASP Top 10, prioritize by severity, "
        "and suggest specific remediations."
    ),
    expected_outputs=(
        "SQL injection vulnerability",
        "command injection via subprocess",
        "no input validation or sanitization",
        "no authentication or authorization",
        "OWASP mapping and severity rating",
    ),
    complexity=TaskComplexity.HIGH,
    metadata={"domain": "security", "framework": "fastapi"},
)

TASK_A3_TEST_PLAN = BenchmarkTask(
    task_id="A3",
    category=TaskCategory.QE,
    description="Test Plan: Generate comprehensive test plan from requirements",
    input_data=(
        "Generate a test plan for the MangoMAS agent orchestration system:\n\n"
        "Requirements:\n"
        "1. Agents must be registered with unique IDs and capability manifests\n"
        "2. The MCTS planner selects optimal agent delegation paths\n"
        "3. Policy gates enforce safety constraints before agent actions execute\n"
        "4. Graduated rollout supports canary deployments of new agent versions\n"
        "5. Cell-based processing ensures agent isolation and fault containment\n"
        "6. Real-time monitoring tracks agent performance and SLA compliance\n\n"
        "Create a prioritized test plan covering all test types."
    ),
    expected_outputs=(
        "unit tests for each component",
        "integration tests for agent communication",
        "load/performance tests",
        "chaos/fault injection tests",
        "security tests for policy gates",
    ),
    complexity=TaskComplexity.HIGH,
    metadata={"domain": "testing", "system": "mangomas"},
)

TASK_A4_ARCHITECTURE_REVIEW = BenchmarkTask(
    task_id="A4",
    category=TaskCategory.QE,
    description="Architecture Decision Record Review: Evaluate ADR for risks and alternatives",
    input_data=(
        "Review this Architecture Decision Record (ADR) for a multi-agent system:\n\n"
        "# ADR-007: Use MCTS for Agent Orchestration\n\n"
        "## Context\n"
        "We need to decide how to coordinate multiple AI agents for complex tasks. "
        "Options considered: round-robin, priority queue, auction-based, and MCTS.\n\n"
        "## Decision\n"
        "We will use Monte Carlo Tree Search (MCTS) with UCB1 selection policy "
        "for agent orchestration. The MCTS planner will explore agent delegation "
        "paths and select optimal sequences based on simulated rollouts.\n\n"
        "## Consequences\n"
        "- Higher computational overhead than simpler approaches\n"
        "- Better exploration of multi-step agent coordination paths\n"
        "- Requires reward function design per domain\n"
        "- Can leverage neural network guidance for efficiency\n\n"
        "Evaluate this ADR: identify risks, suggest alternatives, "
        "analyze tradeoffs, and recommend improvements."
    ),
    expected_outputs=(
        "risk identification",
        "alternative evaluation",
        "tradeoff analysis",
        "recommendation for improvements",
    ),
    complexity=TaskComplexity.VERY_HIGH,
    metadata={"domain": "architecture", "pattern": "adr"},
)

# ─── Task Set B: Regulatory Compliance ───

TASK_B1_REQUIREMENT_EXTRACTION = BenchmarkTask(
    task_id="B1",
    category=TaskCategory.COMPLIANCE,
    description="Compliance Requirement Extraction: Extract requirements from regulation",
    input_data=(
        "Extract compliance requirements from this regulation excerpt:\n\n"
        "EU AI Act - Title III, Chapter 2, Article 9 (Risk Management System)\n\n"
        "1. A risk management system shall be established, implemented, documented "
        "and maintained in relation to high-risk AI systems.\n"
        "2. The risk management system shall consist of a continuous iterative process "
        "planned and run throughout the entire lifecycle of a high-risk AI system, "
        "requiring regular systematic review and updating.\n"
        "3. The risk management system shall include:\n"
        "   (a) identification and analysis of known and foreseeable risks\n"
        "   (b) estimation and evaluation of risks that may emerge\n"
        "   (c) evaluation of risks based on post-market monitoring data\n"
        "   (d) adoption of suitable risk management measures\n"
        "4. Risk management measures shall give due consideration to the effects "
        "and possible interactions resulting from the combined application of requirements.\n"
        "5. Testing shall be carried out at appropriate points during development "
        "and in any event prior to placing on the market or putting into service.\n\n"
        "Extract structured compliance requirements with references."
    ),
    expected_outputs=(
        "structured requirements list",
        "regulation article references",
        "risk management system components",
        "testing requirements",
    ),
    complexity=TaskComplexity.MEDIUM,
    metadata={"regulation": "eu_ai_act", "article": "9"},
)

TASK_B2_CONTROL_GAP_ANALYSIS = BenchmarkTask(
    task_id="B2",
    category=TaskCategory.COMPLIANCE,
    description="Control Gap Analysis: Map requirements to controls and find gaps",
    input_data=(
        "Given the following compliance requirements and existing controls, "
        "identify gaps:\n\n"
        "REQUIREMENTS (from EU AI Act Article 9):\n"
        "R1: Continuous risk management lifecycle process\n"
        "R2: Regular systematic review of risk assessments\n"
        "R3: Post-market monitoring data integration\n"
        "R4: Combined effect analysis of risk measures\n"
        "R5: Pre-deployment testing at appropriate points\n\n"
        "EXISTING CONTROLS:\n"
        "C1: Annual risk assessment review (manual process)\n"
        "C2: Unit test suite with 80% coverage\n"
        "C3: Incident response procedure (documented, untested)\n"
        "C4: Model versioning with Git-based tracking\n\n"
        "Map controls to requirements, calculate coverage, and identify gaps "
        "with severity ratings."
    ),
    expected_outputs=(
        "control-to-requirement mapping",
        "gap identification with severity",
        "coverage percentage calculation",
        "remediation priority ranking",
    ),
    complexity=TaskComplexity.HIGH,
    metadata={"regulation": "eu_ai_act", "analysis_type": "gap"},
)

TASK_B3_REMEDIATION_PLAN = BenchmarkTask(
    task_id="B3",
    category=TaskCategory.COMPLIANCE,
    description="Remediation Plan: Generate prioritized remediation with constraints",
    input_data=(
        "Generate a remediation plan for these compliance gaps:\n\n"
        "GAP-1 (Critical): No continuous risk management process\n"
        "  - Requirement: EU AI Act Art. 9(1) - continuous iterative process\n"
        "  - Current state: Annual manual review only\n\n"
        "GAP-2 (High): No post-market monitoring integration\n"
        "  - Requirement: EU AI Act Art. 9(3)(c)\n"
        "  - Current state: No production monitoring feedback loop\n\n"
        "GAP-3 (High): No combined effect analysis\n"
        "  - Requirement: EU AI Act Art. 9(4)\n"
        "  - Current state: Risks assessed in isolation\n\n"
        "GAP-4 (Medium): Incomplete pre-deployment testing\n"
        "  - Requirement: EU AI Act Art. 9(5)\n"
        "  - Current state: Unit tests only, no integration or adversarial testing\n\n"
        "CONSTRAINTS:\n"
        "- Budget: $150,000 for remediation\n"
        "- Timeline: Must achieve compliance within 6 months\n"
        "- Team: 2 engineers, 1 compliance officer\n"
        "- Cannot disrupt current production operations\n\n"
        "Create a prioritized remediation plan with timeline, budget allocation, "
        "risk mitigation, and success criteria."
    ),
    expected_outputs=(
        "prioritized action items",
        "timeline with milestones",
        "budget allocation breakdown",
        "risk mitigation strategies",
        "success criteria per gap",
    ),
    complexity=TaskComplexity.VERY_HIGH,
    metadata={"regulation": "eu_ai_act", "plan_type": "remediation"},
)

# ─── Task Set C: Strategic Decision Making ───

TASK_C1_INVESTMENT_STRATEGY = BenchmarkTask(
    task_id="C1",
    category=TaskCategory.STRATEGIC,
    description="Investment Strategy Evaluation: Evaluate multiple strategies",
    input_data=(
        "Given the following market data, evaluate 3 investment strategies "
        "for an AI infrastructure portfolio:\n\n"
        "MARKET CONTEXT:\n"
        "- AI compute demand growing 40% YoY\n"
        "- GPU supply constrained for next 18 months\n"
        "- Cloud provider revenue growth: AWS 17%, Azure 29%, GCP 28%\n"
        "- Emerging inference-optimized chip companies raising $2B+ in funding\n"
        "- Open-source model quality approaching proprietary levels\n\n"
        "STRATEGIES TO EVALUATE:\n"
        "Strategy A: Concentrated bet on NVIDIA + major cloud providers\n"
        "Strategy B: Diversified across compute, inference chips, and AI tooling\n"
        "Strategy C: Focus on AI application layer (SaaS companies using AI)\n\n"
        "For each strategy: analyze risk/return profile, identify catalysts "
        "and risks, recommend allocation, and provide confidence-weighted ranking."
    ),
    expected_outputs=(
        "analysis of all three strategies",
        "risk/return profiles",
        "catalyst and risk identification",
        "confidence-weighted ranking",
        "allocation recommendations",
    ),
    complexity=TaskComplexity.HIGH,
    metadata={"domain": "investment", "sector": "ai_infrastructure"},
)

TASK_C2_PROJECT_PLANNING = BenchmarkTask(
    task_id="C2",
    category=TaskCategory.STRATEGIC,
    description="Resource-Constrained Project Planning: Commercialize MangoMAS",
    input_data=(
        "Plan the commercialization of MangoMAS as an enterprise product.\n\n"
        "CONSTRAINTS:\n"
        "- Solo developer (Ian) with 20 hours/week available\n"
        "- $5,000 budget for infrastructure and tools\n"
        "- Target: first paying customer within 6 months\n"
        "- Competing with CrewAI, AutoGen, LangGraph (open source)\n\n"
        "TECHNICAL ASSETS:\n"
        "- MCTS-based agent orchestration (unique differentiator)\n"
        "- Policy gates for safety constraints\n"
        "- Cell-based processing for isolation\n"
        "- Neural meta-controller for routing\n"
        "- Enterprise use cases (M&A, Clinical Trial, Compliance)\n\n"
        "Consider: product positioning, MVP scope, go-to-market, pricing, "
        "technical differentiation.\n\n"
        "Evaluate at least 3 strategic approaches with tradeoffs."
    ),
    expected_outputs=(
        "multiple strategic options (3+)",
        "resource allocation plan",
        "risk assessment per strategy",
        "timeline with milestones",
        "competitive differentiation strategy",
    ),
    complexity=TaskComplexity.VERY_HIGH,
    metadata={"domain": "commercialization", "product": "mangomas"},
)

TASK_C3_COMPETITIVE_ANALYSIS = BenchmarkTask(
    task_id="C3",
    category=TaskCategory.STRATEGIC,
    description="Competitive Analysis: Multi-agent framework landscape with scenarios",
    input_data=(
        "Conduct a competitive analysis of the multi-agent AI framework landscape:\n\n"
        "FRAMEWORKS TO ANALYZE:\n"
        "1. LangGraph (LangChain) - graph-based orchestration\n"
        "2. CrewAI - role-based agent teams\n"
        "3. AutoGen (Microsoft) - conversational agents\n"
        "4. Google ADK - Agent Development Kit\n"
        "5. MangoMAS - MCTS-guided multi-agent system (our system)\n\n"
        "FOR EACH FRAMEWORK:\n"
        "- Architecture approach and key differentiators\n"
        "- Target market and use cases\n"
        "- Strengths and weaknesses\n"
        "- Community and ecosystem maturity\n\n"
        "MODEL SCENARIOS:\n"
        "Scenario A: Enterprise buyer prioritizing reliability and compliance\n"
        "Scenario B: Startup prioritizing speed and cost-efficiency\n"
        "Scenario C: Research team prioritizing flexibility and novel architectures\n\n"
        "For each scenario, rank frameworks and explain positioning strategy for MangoMAS."
    ),
    expected_outputs=(
        "per-framework analysis",
        "scenario-based rankings",
        "MangoMAS positioning strategy per scenario",
        "market opportunity assessment",
        "contingency recommendations",
    ),
    complexity=TaskComplexity.VERY_HIGH,
    metadata={"domain": "competitive_analysis", "frameworks": 5},
)


# ─── Task Set Collections ───

TASK_SET_A: tuple[BenchmarkTask, ...] = (
    TASK_A1_CODE_REVIEW,
    TASK_A2_SECURITY_ANALYSIS,
    TASK_A3_TEST_PLAN,
    TASK_A4_ARCHITECTURE_REVIEW,
)

TASK_SET_B: tuple[BenchmarkTask, ...] = (
    TASK_B1_REQUIREMENT_EXTRACTION,
    TASK_B2_CONTROL_GAP_ANALYSIS,
    TASK_B3_REMEDIATION_PLAN,
)

TASK_SET_C: tuple[BenchmarkTask, ...] = (
    TASK_C1_INVESTMENT_STRATEGY,
    TASK_C2_PROJECT_PLANNING,
    TASK_C3_COMPETITIVE_ANALYSIS,
)

ALL_TASKS: tuple[BenchmarkTask, ...] = TASK_SET_A + TASK_SET_B + TASK_SET_C
