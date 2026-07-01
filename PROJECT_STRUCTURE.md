# Project Structure

```
langgraph-multi-agent-mcts/
├── README.md                    # Project overview and quick start
├── CHANGELOG.md                 # Version history
├── ATTRIBUTION.md               # Third-party attributions
├── pyproject.toml               # Package configuration and dependencies
├── requirements.txt             # Direct dependencies for Docker/pip
├── Dockerfile                   # Multi-stage production Docker build
├── docker-compose.yml           # Full stack orchestration (8 services)
│
├── src/                         # Core application source code
│   ├── adapters/                # External service adapters
│   │   └── llm/                 # LLM provider clients (OpenAI, Anthropic, LMStudio)
│   ├── agents/                  # AI agent implementations
│   │   └── meta_controller/     # Neural meta-controller (RNN/BERT)
│   ├── api/                     # REST API server (FastAPI)
│   ├── config/                  # Application configuration
│   ├── data/                    # Data loading and preprocessing
│   ├── framework/               # Core MCTS framework
│   │   ├── agents/              # Framework agent base classes
│   │   └── mcts/                # Monte Carlo Tree Search implementation
│   ├── models/                  # Data models and validation
│   ├── observability/           # Logging, metrics, tracing
│   ├── storage/                 # Persistent storage (S3, Pinecone)
│   └── training/                # Model training utilities
│
├── config/                      # Configuration files
│   ├── README.md                # Configuration guide
│   ├── mcp_config.json          # Active MCP server config
│   ├── mcp_config.example.json  # Example configuration
│   └── mcp_config_template.json # Configuration template
│
├── docs/                        # Documentation
│   ├── architecture.md          # System architecture
│   ├── langgraph_mcts_architecture.md  # MCTS architecture details
│   ├── API_CONFIGURATION_GUIDE.md      # API configuration
│   ├── API_QUICK_REFERENCE.md          # API quick reference
│   ├── MCP_SERVER_GUIDE.md             # MCP server setup
│   ├── DEPLOYMENT_REPORT.md            # Deployment status report
│   ├── NEURAL_TRAINING_SUMMARY.md      # Neural network training
│   ├── INTEGRATION_STATUS.md           # Integration status
│   ├── STATUS.md                # Reproducible test/coverage baseline (source of truth)
│   ├── reports/                 # Analyses & completion reports (e.g. SCALABILITY_ANALYSIS.md)
│   ├── summaries/               # Feature/module implementation summaries
│   ├── plans/                   # Roadmaps, implementation plans, PR descriptions
│   ├── quickstart/              # Quickstart guides (Docker, synthetic generation)
│   ├── img/                     # Documentation images
│   ├── mermaid/                 # Architecture diagrams
│   ├── runbooks/                # Operational runbooks
│   └── testing/                 # Test documentation
│
├── examples/                    # Example scripts and demos
│   ├── langgraph_multi_agent_mcts.py   # Main framework demo
│   ├── lmstudio_mcp_demo.py            # LM Studio MCP integration
│   ├── mcp_usage_example.py            # MCP usage patterns
│   ├── llm_provider_usage.py           # LLM provider examples
│   └── mcts_determinism_demo.py        # MCTS determinism tests
│
├── demos/                       # Interactive demonstrations
│   └── neural_meta_controller_demo.py  # Neural controller demo
│
├── scripts/                     # Automation and utility scripts
│   ├── smoke_test.sh                   # Docker deployment smoke tests
│   ├── verify_setup.py                 # Setup verification
│   ├── verify_all_integrations.py      # Full integration check
│   ├── verify_pinecone_integration.py  # Pinecone connectivity
│   ├── verify_braintrust_wandb_integration.py  # Experiment tracking
│   ├── test_api_integrations.py        # API integration tests
│   ├── test_lmstudio_connection.py     # LM Studio connection
│   ├── export_architecture_diagrams.py # Export Mermaid diagrams
│   ├── production_readiness_check.py   # Pre-production validation
│   └── security_audit.py               # Security scanning
│
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── e2e/                     # End-to-end tests
│   ├── api/                     # API endpoint tests
│   ├── chaos/                   # Chaos engineering tests
│   ├── performance/             # Load and performance tests
│   ├── training/                # Training pipeline tests
│   ├── fixtures/                # Test fixtures
│   └── mocks/                   # Mock implementations
│
├── tools/                       # Development tools
│   ├── cli/                     # Command-line tools
│   └── mcp/                     # MCP server implementation
│
├── huggingface_space/           # HuggingFace Spaces deployment
│   ├── app.py                   # Gradio demo application
│   ├── requirements.txt         # Space dependencies
│   ├── README.md                # Space metadata
│   ├── DEPLOYMENT_GUIDE.md      # Deployment instructions
│   └── demo_src/                # Demo source modules
│       ├── agents_demo.py       # Agent implementations
│       ├── mcts_demo.py         # MCTS implementation
│       ├── llm_mock.py          # Mock LLM client
│       └── wandb_tracker.py     # W&B integration
│
├── kubernetes/                  # Kubernetes deployment
│   └── deployment.yaml          # K8s manifests (HPA, PDB, Ingress)
│
├── monitoring/                  # Observability infrastructure
│   ├── prometheus.yml           # Prometheus configuration
│   ├── alerts.yml               # Alert rules (15 rules)
│   ├── alertmanager.yml         # Alert routing
│   ├── otel-collector-config.yaml  # OpenTelemetry collector
│   └── grafana/                 # Grafana dashboards
│
├── training/                    # Advanced training pipeline
│   ├── README.md                # Training documentation
│   ├── config.yaml              # Training configuration
│   ├── requirements.txt         # Training dependencies
│   ├── agent_trainer.py         # Agent training logic
│   ├── data_pipeline.py         # Data preprocessing
│   ├── evaluation.py            # Model evaluation
│   ├── orchestrator.py          # Training orchestration
│   └── tests/                   # Training tests
│
├── .github/                     # GitHub configuration
│   └── workflows/
│       └── ci.yml               # CI/CD pipeline (9 jobs)
│
└── artifacts/                   # Generated artifacts (gitignored)
    ├── models/                  # Trained model weights
    └── logs/                    # Execution logs
```

## Directory Purposes

| Directory | Purpose |
|-----------|---------|
| `src/` | Core application code - the main package |
| `config/` | Configuration files for deployment |
| `docs/` | All documentation and guides |
| `examples/` | Working examples and demos |
| `scripts/` | Automation, verification, and utility scripts |
| `tests/` | Comprehensive test suite |
| `tools/` | Development and debugging tools |
| `huggingface_space/` | HuggingFace Spaces POC deployment |
| `kubernetes/` | Container orchestration manifests |
| `monitoring/` | Observability stack configuration |
| `training/` | Advanced ML training pipeline |
| `artifacts/` | Generated files (models, logs) - not in git |

## Quick Navigation

- **Getting Started**: See `README.md`
- **Architecture**: See `docs/architecture.md`
- **API Documentation**: Run server and visit `http://localhost:8000/docs`
- **Examples**: Browse `examples/` directory
- **Configuration**: See `config/README.md`
- **Deployment**: See `docs/DEPLOYMENT_REPORT.md`
