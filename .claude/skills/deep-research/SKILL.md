---
name: deep-research
description: Defines the standard operating procedure and output format for the multi-agent /deep-research command.
---

# Deep Research Workflow

This skill defines the execution environment and artifact standards for the `/deep-research` command, orchestrated by the `research-*` subagents.

## Core Directives

1. **Multi-Agent Execution:** The research must follow a strict pipeline: Planner -> Fetcher -> Critic -> Synthesizer. A single agent should NOT attempt to do the entire research task in one pass.
2. **Statefulness:** If the Critic rejects the Fetcher's data, the Planner must formulate new queries and retry. The system should loop until the Critic is satisfied.
3. **Traceability:** All claims in the final report MUST be cited. Use brackets for citations: `[Author, Year - Source]`.
4. **Feasibility:** Research is only useful if it applies to the current architecture. The report must specifically address how the researched topic integrates with the `Strategos-MCTS` codebase (e.g., PyTorch models, LangGraph orchestration).

## Artifact Formatting

The final output MUST be written to `docs/reports/YYYY-MM-DD_<topic>_deep_research.md`.

It must follow this markdown structure:

```markdown
# Deep Research: [Topic]

## Executive Summary
[High-level overview of the findings and their relevance to Strategos-MCTS.]

## Background & Literature
[Detailed analysis of the topic, heavily cited from the Fetcher's data.]

## Architectural Analysis & Feasibility
[How this topic maps to our current codebase. What would need to change? What are the risks?]

## Spec Brief
[A concise summary of the proposed implementation path, designed to be passed directly as the <goal> parameter in the `/spec-new` command.]
```

## Bridging to SDD

Upon completing the report, the `research-synthesizer` should instruct the user to run the following command to begin formal specification:

```bash
/spec-new <new-id> docs/reports/YYYY-MM-DD_<topic>_deep_research.md
```
