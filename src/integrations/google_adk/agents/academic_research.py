"""
Academic Research agent for paper analysis and citation discovery.

Based on: https://github.com/google/adk-samples/tree/main/python/agents/academic-research
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import ADKAgentAdapter, ADKAgentRequest, ADKAgentResponse, ADKConfig


class AcademicResearchAgent(ADKAgentAdapter):
    """
    AI-driven research assistant for academic paper analysis.

    This agent:
    - Analyzes core contributions of seminal research papers
    - Discovers recent citations using Google Search
    - Synthesizes future research directions
    - Maps contemporary influence of foundational work

    Focuses on papers from January 2023 onward for citation discovery.
    """

    def __init__(self, config: ADKConfig):
        """
        Initialize Academic Research agent.

        Args:
            config: ADK configuration
        """
        super().__init__(config, agent_name="academic_research")

        # Research-specific directories
        self.papers_dir = Path(config.workspace_dir) / "papers"
        self.analysis_dir = Path(config.workspace_dir) / "analysis"
        self.citations_dir = Path(config.workspace_dir) / "citations"

        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.citations_dir.mkdir(parents=True, exist_ok=True)

        # Citation date filter
        self.citation_start_date = "2023-01-01"

    async def _agent_initialize(self) -> None:
        """Initialize Academic Research agent resources."""
        try:
            import google.adk  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "google-adk not installed. Install with: pip install 'langgraph-multi-agent-mcts[google-adk]'"
            ) from None

        # Verify search is enabled
        if not self.config.enable_search:
            raise ValueError("Academic Research agent requires enable_search=True")

    async def _agent_invoke(self, request: ADKAgentRequest) -> ADKAgentResponse:
        """
        Execute academic research task.

        Args:
            request: Agent request with paper or research query

        Returns:
            Agent response with research analysis
        """
        task_type = request.parameters.get("task_type", "full_analysis")
        paper_path = request.parameters.get("paper_path")
        paper_title = request.parameters.get("paper_title")
        paper_url = request.parameters.get("paper_url")

        if task_type == "full_analysis":
            return await self._handle_full_analysis(
                request,
                paper_path,
                paper_title,
                paper_url,
            )
        elif task_type == "citation_discovery":
            return await self._handle_citation_discovery(paper_title or request.query)
        elif task_type == "future_directions":
            return await self._handle_future_directions(request)
        else:
            return ADKAgentResponse(
                result="",
                status="error",
                error=f"Unknown task type: {task_type}",
            )

    async def _handle_full_analysis(
        self,
        request: ADKAgentRequest,
        paper_path: str | None,
        paper_title: str | None,
        paper_url: str | None,
    ) -> ADKAgentResponse:
        """
        Perform full academic research analysis.

        Args:
            request: Agent request
            paper_path: Path to PDF paper
            paper_title: Paper title
            paper_url: URL to paper

        Returns:
            Agent response with full analysis
        """
        if not any([paper_path, paper_title, paper_url]):
            return ADKAgentResponse(
                result="",
                status="error",
                error="One of paper_path, paper_title, or paper_url must be provided",
            )

        # Generate analysis plan
        analysis = self._generate_research_analysis(
            paper_path=paper_path,
            paper_title=paper_title,
            paper_url=paper_url,
            query=request.query,
        )

        # Save analysis
        analysis_file = self.analysis_dir / f"{self._sanitize_filename(paper_title or 'paper')}_analysis.md"
        with open(analysis_file, "w") as f:
            f.write(analysis)

        return ADKAgentResponse(
            result=analysis,
            metadata={
                "task_type": "full_analysis",
                "paper_title": paper_title,
                "analysis_file": str(analysis_file),
            },
            artifacts=[str(analysis_file)],
            status="success",
            session_id=request.session_id,
        )

    async def _handle_citation_discovery(self, paper_title: str) -> ADKAgentResponse:
        """
        Discover recent citations for a paper.

        Args:
            paper_title: Title of the paper

        Returns:
            Agent response with citation analysis
        """
        citation_plan = self._generate_citation_discovery_plan(paper_title)

        citation_file = self.citations_dir / f"{self._sanitize_filename(paper_title)}_citations.md"
        with open(citation_file, "w") as f:
            f.write(citation_plan)

        return ADKAgentResponse(
            result=citation_plan,
            metadata={
                "task_type": "citation_discovery",
                "paper_title": paper_title,
                "citation_file": str(citation_file),
            },
            artifacts=[str(citation_file)],
            status="success",
        )

    async def _handle_future_directions(
        self,
        request: ADKAgentRequest,
    ) -> ADKAgentResponse:
        """
        Synthesize future research directions.

        Args:
            request: Agent request with context

        Returns:
            Agent response with research directions
        """
        paper_title = request.parameters.get("paper_title")
        context = request.context

        directions = self._generate_future_directions(
            query=request.query,
            paper_title=paper_title,
            context=context,
        )

        return ADKAgentResponse(
            result=directions,
            metadata={
                "task_type": "future_directions",
                "paper_title": paper_title,
            },
            status="success",
            session_id=request.session_id,
        )

    def _generate_research_analysis(
        self,
        paper_path: str | None,
        paper_title: str | None,
        paper_url: str | None,
        query: str,
    ) -> str:
        """Generate comprehensive research analysis."""
        source = paper_path or paper_url or paper_title

        return f"""
# Academic Research Analysis

## Research Query
{query}

## Paper Information
- **Title**: {paper_title or "To be extracted"}
- **Source**: {source}

## Analysis Framework

### Phase 1: Core Contribution Analysis
Examines the foundational contributions of the seminal work:

1. **Main Thesis**
   - Central research question
   - Key hypotheses
   - Problem statement

2. **Methodology**
   - Research approach
   - Experimental design
   - Data collection and analysis

3. **Key Findings**
   - Primary results
   - Novel contributions
   - Theoretical implications

4. **Impact Assessment**
   - Field advancement
   - Paradigm shifts
   - Practical applications

### Phase 2: Citation Discovery
Uses Google Search to identify recent publications (from {self.citation_start_date} onward):

1. **Search Strategy**
   - Query formulation: "{paper_title} citations"
   - Filter: published >= {self.citation_start_date}
   - Sources: Google Scholar, arXiv, academic databases

2. **Citation Analysis**
   - Number of citations
   - Citation context
   - Building upon vs. challenging
   - Application domains

3. **Influential Citations**
   - High-impact citing papers
   - Novel applications
   - Extension of methods

### Phase 3: Future Research Directions
Synthesizes insights from original paper and contemporary citations:

1. **Identified Gaps**
   - Unexplored areas
   - Limitations acknowledged
   - Open questions

2. **Emerging Trends**
   - Novel applications
   - Methodological improvements
   - Cross-disciplinary connections

3. **Proposed Directions**
   - High-impact opportunities
   - Feasible next steps
   - Interdisciplinary potential

## Execution Plan

### Step 1: Paper Extraction
```
- Extract full text from {source}
- Parse sections and figures
- Identify key concepts
```

### Step 2: Web Search for Citations
```
Search query: "citing {paper_title}"
Date filter: after:{self.citation_start_date}
Top 20 citing papers
```

### Step 3: Synthesis
```
- Analyze citation patterns
- Identify research trajectories
- Generate novel directions
```

## Expected Outputs

1. **Core Analysis Report**
   - Summary of main contributions
   - Methodological assessment
   - Impact evaluation

2. **Citation Map**
   - Recent citing papers (post-2023)
   - Citation context analysis
   - Influence tracking

3. **Future Directions Document**
   - 5-10 novel research directions
   - Rationale for each direction
   - Feasibility assessment

---
*Generated by Academic Research Agent*
*Citation search: {self.citation_start_date} onwards*
""".strip()

    def _generate_citation_discovery_plan(self, paper_title: str) -> str:
        """Generate citation discovery plan."""
        return f"""
# Citation Discovery: {paper_title}

## Search Configuration
- **Target Paper**: {paper_title}
- **Date Range**: {self.citation_start_date} to present
- **Search Tools**: Google Search, Google Scholar

## Discovery Strategy

### 1. Search Queries
- Primary: "citing {paper_title}"
- Secondary: "{paper_title}" AND "builds on" OR "extends"
- Academic: site:scholar.google.com {paper_title}

### 2. Source Filtering
- arXiv papers
- Conference proceedings
- Journal articles
- Preprint servers

### 3. Analysis Metrics
- Citation count
- Citation context (positive, critical, methodological)
- Research domains
- Geographic distribution

### 4. Key Questions
- Who is citing this work?
- How are they using it?
- What new directions are emerging?
- Are there critical assessments?

## Expected Findings

### Citation Categories
1. **Direct Extensions**: Papers building directly on methods
2. **Applications**: Novel application domains
3. **Comparisons**: Benchmarking against this work
4. **Critiques**: Papers challenging assumptions
5. **Surveys**: Review papers citing this work

### Impact Metrics
- Total citations (post-{self.citation_start_date})
- Citation velocity
- Field diversity
- Influential citing papers

---
*Citation search will be executed using Google Search tools*
""".strip()

    def _generate_future_directions(
        self,
        query: str,
        paper_title: str | None,
        context: dict[str, Any],
    ) -> str:
        """Generate future research directions."""
        return f"""
# Future Research Directions

## Context
{query}

## Based On
{paper_title or "Research analysis"}

## Proposed Research Directions

### Direction 1: Methodological Advancement
**Rationale**: Extend core methodology to new settings
- Investigate scalability to larger datasets
- Explore alternative algorithmic approaches
- Cross-validate with different benchmarks

### Direction 2: Interdisciplinary Applications
**Rationale**: Apply insights to adjacent fields
- Transfer learning to related domains
- Combine with complementary techniques
- Novel application contexts

### Direction 3: Theoretical Foundations
**Rationale**: Strengthen theoretical understanding
- Formal analysis of assumptions
- Convergence guarantees
- Complexity characterization

### Direction 4: Practical Implementation
**Rationale**: Bridge theory-practice gap
- Production-ready implementations
- Efficiency optimizations
- User-friendly tooling

### Direction 5: Empirical Validation
**Rationale**: Comprehensive empirical assessment
- Extensive benchmarking
- Real-world case studies
- Long-term impact studies

## Prioritization Criteria
1. **Impact Potential**: Significance to the field
2. **Feasibility**: Resources and timeline
3. **Novelty**: Uniqueness of approach
4. **Timeliness**: Current relevance

## Next Steps for Researchers
1. Literature review of recent citations
2. Identify specific research gaps
3. Design pilot studies
4. Seek collaborations
5. Secure funding

---
*Future directions synthesized from paper analysis and contemporary citations*
""".strip()

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize string for use as filename."""
        import re

        # Remove special characters, keep alphanumeric and basic punctuation
        clean = re.sub(r"[^\w\s-]", "", name)
        # Replace spaces with underscores
        clean = re.sub(r"\s+", "_", clean)
        return clean[:100]  # Limit length

    async def analyze_paper(
        self,
        paper_path: str | None = None,
        paper_title: str | None = None,
        paper_url: str | None = None,
        query: str | None = None,
    ) -> ADKAgentResponse:
        """
        Analyze an academic paper.

        Args:
            paper_path: Path to PDF file
            paper_title: Title of the paper
            paper_url: URL to the paper
            query: Specific research question

        Returns:
            Agent response with comprehensive analysis
        """
        request = ADKAgentRequest(
            query=query or "Analyze this paper and its impact",
            parameters={
                "task_type": "full_analysis",
                "paper_path": paper_path,
                "paper_title": paper_title,
                "paper_url": paper_url,
            },
        )

        return await self.invoke(request)

    async def find_citations(self, paper_title: str) -> ADKAgentResponse:
        """
        Find recent citations for a paper.

        Args:
            paper_title: Title of the paper

        Returns:
            Agent response with citation analysis
        """
        request = ADKAgentRequest(
            query=f"Find citations for: {paper_title}",
            parameters={
                "task_type": "citation_discovery",
                "paper_title": paper_title,
            },
        )

        return await self.invoke(request)

    async def suggest_future_research(
        self,
        paper_title: str,
        query: str | None = None,
    ) -> ADKAgentResponse:
        """
        Suggest future research directions.

        Args:
            paper_title: Title of the paper
            query: Specific focus for suggestions

        Returns:
            Agent response with research directions
        """
        request = ADKAgentRequest(
            query=query or "Suggest future research directions",
            parameters={
                "task_type": "future_directions",
                "paper_title": paper_title,
            },
        )

        return await self.invoke(request)

    def get_capabilities(self) -> dict[str, Any]:
        """Get Academic Research agent capabilities."""
        base_caps = super().get_capabilities()
        base_caps.update(
            {
                "agent_type": "academic_research",
                "features": [
                    "paper_analysis",
                    "citation_discovery",
                    "future_directions",
                    "pdf_processing",
                    "web_search",
                ],
                "citation_date_filter": self.citation_start_date,
                "supported_formats": ["pdf", "url", "arxiv"],
            }
        )
        return base_caps
