# Documentation Index

The map of Strategos-MCTS documentation, organized by purpose
([Diátaxis](https://diataxis.fr/): tutorials/how-to, reference, explanation, plus status and archives).
Start with the [project README](../README.md) for installation and a feature overview.

> **Authority is scoped by axis, not by rank.** [`STATUS.md`](STATUS.md) governs **measured status** —
> point-in-time reports elsewhere are historical, and where they disagree with `STATUS.md`, `STATUS.md`
> governs. [`../CHARTER.md`](../CHARTER.md) governs **durable intent**: scope, non-goals, and invariants.
> The two never overlap: the charter restates no measured value, and `STATUS.md` sets no boundary.

## Charter, status & roadmap

- [`../CHARTER.md`](../CHARTER.md) — vision, scope, non-goals, invariants, amendment protocol
  (**source of truth for intent**).
- [`STATUS.md`](STATUS.md) — reproducible test/coverage baseline (**source of truth for status**).
- [`NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md`](NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md) — active roadmap.

## Explanation (architecture & concepts)

- [`architecture.md`](architecture.md) — system architecture overview.
- [`C4_ARCHITECTURE.md`](C4_ARCHITECTURE.md) — C4 diagrams of system components.
- [`C4_MERMAID_ARCHITECTURE.md`](C4_MERMAID_ARCHITECTURE.md) — C4 diagrams (Mermaid source narrative).
- [`langgraph_mcts_architecture.md`](langgraph_mcts_architecture.md) — MCTS + LangGraph architecture details.
- [`DEEPMIND_IMPLEMENTATION.md`](DEEPMIND_IMPLEMENTATION.md) — AlphaZero/DeepMind-style implementation notes.
- [`ASSEMBLY_THEORY.md`](ASSEMBLY_THEORY.md) — assembly-theory background.
- [`feature_engineering_enhancement.md`](feature_engineering_enhancement.md) — feature-engineering design.
- [`E2E_USER_JOURNEYS.md`](E2E_USER_JOURNEYS.md) — end-to-end user journeys.
- [`BUSINESS_VALUE_ROI.md`](BUSINESS_VALUE_ROI.md) — business value and ROI framing.
- [`related-work.md`](related-work.md) — related research and prior art.

## How-to guides

- [`RUN_ALL_LOCALLY.md`](RUN_ALL_LOCALLY.md) — run the entire framework locally.
- [`LOCAL_TRAINING_GUIDE.md`](LOCAL_TRAINING_GUIDE.md) — train models locally or in the cloud.
- [`GPU_TRAINING_GUIDE.md`](GPU_TRAINING_GUIDE.md) — GPU hardware setup and memory management.
- [`META_CONTROLLER_TRAINING.md`](META_CONTROLLER_TRAINING.md) — meta-controller routing learning loop.
- [`DATASET_SETUP.md`](DATASET_SETUP.md) — dataset preparation.
- [`DOCKER_DEPLOYMENT.md`](DOCKER_DEPLOYMENT.md) — containerized deployment.
- [`SECRETS_MANAGEMENT.md`](SECRETS_MANAGEMENT.md) — External Secrets Operator setup and rotation.
- [`API_CONFIGURATION_GUIDE.md`](API_CONFIGURATION_GUIDE.md) — API configuration.
- [`MCP_SERVER_GUIDE.md`](MCP_SERVER_GUIDE.md) — MCP server setup.
- [`PINECONE_INTEGRATION.md`](PINECONE_INTEGRATION.md) — Pinecone vector-DB integration.
- [`RAG_EVALUATION.md`](RAG_EVALUATION.md) — RAG evaluation.
- [`RAG_EVAL_DATASET_QUICKSTART.md`](RAG_EVAL_DATASET_QUICKSTART.md) — RAG evaluation dataset quickstart.
- [`AGENT_TRACING_GUIDE.md`](AGENT_TRACING_GUIDE.md) — agent tracing.
- [`EXPERIMENT_TRACKING_GUIDE.md`](EXPERIMENT_TRACKING_GUIDE.md) — experiment tracking (W&B/Braintrust).
- [`LANGSMITH_E2E.md`](LANGSMITH_E2E.md) — LangSmith end-to-end tracing.
- [`LANGSMITH_EXPERIMENTS.md`](LANGSMITH_EXPERIMENTS.md) — LangSmith experiments.
- [`LINTING_SETUP.md`](LINTING_SETUP.md) — linting/formatting setup.
- [`quickstart/`](quickstart/) — Docker and synthetic-generation quickstarts.
- [`runbooks/`](runbooks/) — operational runbooks (high error rate, high latency, service down, incident response).

## Reference

- [`API_QUICK_REFERENCE.md`](API_QUICK_REFERENCE.md) — API quick reference.
- [`KEY_CODE_SNIPPETS.md`](KEY_CODE_SNIPPETS.md) — key code snippets.
- [`GAME_DOMAINS.md`](GAME_DOMAINS.md) — supported gameplay domains.
- [`SLA.md`](SLA.md) — service-level agreement.
- [`MIGRATION_NOTES.md`](MIGRATION_NOTES.md) — migration notes.
- [`PHASE2_MCTS_CODE_GENERATION_TEMPLATE.md`](PHASE2_MCTS_CODE_GENERATION_TEMPLATE.md) — MCTS code-generation template.
- [`diagrams/`](diagrams/) — rendered architecture SVGs ([gallery](diagrams/README.md)).
- [`mermaid/`](mermaid/) — Mermaid diagram sources.

## Reports & archive

Historical, point-in-time material — read for context, not as current status.

- [`reports/`](reports/) — live output sink for `/deep-research` reports.
- [`archive/`](archive/) — frozen point-in-time documents: [`archive/reports/`](archive/reports/)
  (completion reports, gap analyses, setup records) and [`archive/summaries/`](archive/summaries/)
  (feature/module implementation summaries). Each carries a historical-snapshot banner.
- [`plans/`](plans/) — roadmaps, implementation plans, and PR descriptions.
- [`templates/`](templates/) — implementation/subagent reference templates
  (`MULTI_AGENT_MCTS_TEMPLATE.md`, `CLAUDE_CODE_SUBAGENT_TEMPLATE.md`).
- [`reviews/`](reviews/) — peer/design reviews.
- [`testing/`](testing/) — test plan documentation.
- [`training/`](training/) — the developer training curriculum (modules, labs, assessments).

## Contributing & policies

- [Contributing guide](../.github/CONTRIBUTING.md) · [Code of Conduct](../.github/CODE_OF_CONDUCT.md)
- [Security policy](../.github/SECURITY.md) · [Support](../.github/SUPPORT.md)
- [Changelog](../CHANGELOG.md) · [Project structure](../PROJECT_STRUCTURE.md)
