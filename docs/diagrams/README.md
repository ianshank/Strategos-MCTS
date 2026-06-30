# Architecture Diagrams (rendered)

Rendered SVGs of the Mermaid diagrams embedded in the C4 architecture docs. Regenerate with:

```bash
npm install -g @mermaid-js/mermaid-cli   # or use npx
python scripts/render_mermaid_diagrams.py
```

## From `docs/C4_ARCHITECTURE.md`

### 1. Level 1: System Context Diagram

![Level 1: System Context Diagram](c4_architecture-01.svg)

### 2. Level 2: Container Diagram

![Level 2: Container Diagram](c4_architecture-02.svg)

### 3. 3.1 Training Orchestrator Components

![3.1 Training Orchestrator Components](c4_architecture-03.svg)

### 4. 3.2 Neural Network Components (DeepMind Implementation)

![3.2 Neural Network Components (DeepMind Implementation)](c4_architecture-04.svg)

### 5. Docker & Production Deployment

![Docker & Production Deployment](c4_architecture-05.svg)

### 6. Neural self-play & multi-domain learning (M5 / Phase 5)

![Neural self-play & multi-domain learning (M5 / Phase 5)](c4_architecture-06.svg)

## From `docs/C4_MERMAID_ARCHITECTURE.md`

### 1. Primary System Context

![Primary System Context](c4_mermaid_architecture-01.svg)

### 2. Extended Context with Data Flows

![Extended Context with Data Flows](c4_mermaid_architecture-02.svg)

### 3. Main Container Architecture

![Main Container Architecture](c4_mermaid_architecture-03.svg)

### 4. Container Communication Matrix

![Container Communication Matrix](c4_mermaid_architecture-04.svg)

### 5. 3.1 MCTS Engine Components

![3.1 MCTS Engine Components](c4_mermaid_architecture-05.svg)

### 6. 3.2 Agent Layer Components

![3.2 Agent Layer Components](c4_mermaid_architecture-06.svg)

### 7. 3.3 Meta-Controller Components

![3.3 Meta-Controller Components](c4_mermaid_architecture-07.svg)

### 8. 3.4 LangGraph Orchestration Components

![3.4 LangGraph Orchestration Components](c4_mermaid_architecture-08.svg)

### 9. 4.1 MCTS Core Classes

![4.1 MCTS Core Classes](c4_mermaid_architecture-09.svg)

### 10. 4.2 Agent Interfaces

![4.2 Agent Interfaces](c4_mermaid_architecture-10.svg)

### 11. 4.3 Configuration Classes

![4.3 Configuration Classes](c4_mermaid_architecture-11.svg)

### 12. Query Processing Sequence

![Query Processing Sequence](c4_mermaid_architecture-12.svg)

### 13. MCTS Search Sequence

![MCTS Search Sequence](c4_mermaid_architecture-13.svg)

### 14. Training Pipeline Sequence

![Training Pipeline Sequence](c4_mermaid_architecture-14.svg)

### 15. LangGraph State Machine

![LangGraph State Machine](c4_mermaid_architecture-15.svg)

### 16. Meta-Controller State Machine

![Meta-Controller State Machine](c4_mermaid_architecture-16.svg)

### 17. Request Data Flow

![Request Data Flow](c4_mermaid_architecture-17.svg)

### 18. Training Data Flow

![Training Data Flow](c4_mermaid_architecture-18.svg)

### 19. Observability Data Flow

![Observability Data Flow](c4_mermaid_architecture-19.svg)

### 20. Kubernetes Deployment

![Kubernetes Deployment](c4_mermaid_architecture-20.svg)

