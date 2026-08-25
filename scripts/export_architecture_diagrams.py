#!/usr/bin/env python3
"""
Export architecture diagrams:
- C4 views (C1..C4) via Mermaid rendered through Kroki (PNG)
- Neural network diagrams (RNN and BERT) via Matplotlib (PNG)
Outputs to: docs/img/
"""

import os
import sys

import requests

try:
    from matplotlib.patches import FancyArrow, Rectangle
    import matplotlib.pyplot as plt
except Exception:
    plt = None


IMG_DIR = os.path.join("docs", "img")
MMD_DIR = os.path.join("docs", "mermaid")


def ensure_dirs() -> None:
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(MMD_DIR, exist_ok=True)


def get_mermaid_texts() -> dict[str, str]:
    c1 = """graph TB
  user[Developer / Operator]:::person
  system[LangGraph Multi‑Agent MCTS Framework]:::system

  subgraph External Systems
    openai[OpenAI API]
    anthropic[Anthropic API]
    lmstudio[LM Studio (Local LLM)]
    pinecone[(Pinecone Vector DB)]
    braintrust[Braintrust]
    wandb[Weights & Biases]
    s3[(S3 Object Storage)]
  end

  user -- runs demos/tests/training --> system
  system -- prompts/completions --> openai
  system -- prompts/completions --> anthropic
  system -- prompts/completions --> lmstudio
  system -- upsert/query 10D vectors --> pinecone
  system -- experiments/metrics --> braintrust
  system -- runs/metrics/artifacts --> wandb
  system -- model checkpoints/artifacts --> s3

  classDef person fill:#ffd,stroke:#333
  classDef system fill:#bdf,stroke:#333
"""
    c2 = """graph TB
  cli[CLI/Demos/Tests\\n- demos/neural_meta_controller_demo.py\\n- tests/*]:::app
  training[Training Pipelines\\n- src/training/train_rnn.py\\n- src/training/train_bert_lora.py]:::app
  framework[Core Framework\\n- src/framework/*]:::core
  agents[Meta‑Controllers\\n- src/agents/meta_controller/*]:::core
  adapters[LLM Adapters\\n- src/adapters/llm/*]:::core
  storage[Storage\\n- src/storage/s3_client.py\\n- src/storage/pinecone_store.py]:::infra
  observ[Observability\\n- src/observability/*]:::infra
  config[Configuration\\n- src/config/settings.py]:::infra
  tools[MCP Server\\n- tools/mcp/server.py]:::app

  cli --> agents
  cli --> training
  training --> agents
  agents --> adapters
  agents --> storage
  agents --> observ
  framework --- agents
  framework --- adapters
  framework --- storage
  framework --- observ
  training --> observ
  config --> agents
  config --> adapters
  config --> storage
  config --> observ
  tools --> framework

  classDef app fill:#e7f7ff,stroke:#333
  classDef core fill:#d7ffd7,stroke:#333
  classDef infra fill:#fff1cc,stroke:#333
"""
    c3 = """graph TB
  subgraph Meta‑Controller
    utils[utils.py\\nnormalize_features(10D)\\nfeatures_to_text()]:::code
    rnn[RNNMetaController\\nGRU(hidden_dim, num_layers)\\nDropout -> Linear(3)]:::code
    bert[BERTMetaController\\nHF AutoModelForSeqCls\\nOptional LoRA(r=4, alpha=16)]:::code
  end

  subgraph Integrations
    pstore[PineconeVectorStore\\nVECTOR_DIMENSION=10\\nupsert/query in namespace]:::code
    btracker[BraintrustTracker\\nexperiment->log_*->summarize]:::code
    s3c[S3StorageClient (async, retries)]:::code
    llm[Adapters: OpenAI, Anthropic, LM Studio]:::code
  end

  utils --> rnn
  utils --> bert
  rnn --> pstore
  bert --> pstore
  rnn --> btracker
  bert --> btracker
  training[Training (RNN/BERT)] --> rnn
  training --> btracker
  rnn --> llm
  bert --> llm
  checkpoints((Checkpoints)) --> s3c

  classDef code fill:#f6f6ff,stroke:#333
"""
    c4 = """sequenceDiagram
  participant App as Demo/Framework
  participant MC as Meta‑Controller (RNN/BERT)
  participant LLM as LLM Adapter
  participant PC as Pinecone
  participant BT as Braintrust

  App->>MC: features (state) or text (BERT)
  MC->>LLM: (optional) prompt for feature extraction/aux
  LLM-->>MC: completion
  MC-->>App: MetaControllerPrediction(agent, confidence, probs)

  App->>PC: upsert 10D vector + metadata (agent, probs, len, iter)
  App->>BT: log hyperparams/epoch or prediction
  Note over PC,BT: buffering when not configured
"""
    return {
        "c1_system_context": c1,
        "c2_containers": c2,
        "c3_components": c3,
        "c4_sequence": c4,
        "user_journey_flow": """flowchart LR
    user[User/Developer] -->|CLI/Demos| framework[Framework]
    user -->|REST API| api[FastAPI Server]
    user -->|MCP| mcp[MCP Server]

    api --> framework
    mcp --> framework

    framework -->|LangGraph| graph[StateGraph]
    graph --> route[Route Decision]

    route -->|parallel| parallel[Parallel Agents\\nHRM + TRM]
    route -->|mcts| mcts[MCTS Simulation]
    route -->|aggregate| agg[Aggregate Results]

    parallel --> agg
    mcts --> agg

    agg --> eval[Evaluate Consensus]
    eval -->|consensus| synth[Synthesize Response]
    eval -->|iterate| route

    synth --> output[Final Response]
    output --> obs[Observability\\n(OTel, Prometheus)]
    output --> storage[S3 Artifacts]

    classDef svc fill:#e1d5e7,stroke:#333
    classDef core fill:#d5e8d4,stroke:#333
    classDef flow fill:#fff2cc,stroke:#333

    class api,mcp svc
    class framework core
    class graph,route,parallel,mcts,agg,eval,synth,output flow
""",
        "agent_workflow": """flowchart TB
    A[Entry] --> B[Retrieve Context]
    B --> C{Route Decision}

    C -->|HRM| HRM[HRM Agent Node]
    C -->|TRM| TRM[TRM Agent Node]
    C -->|MCTS| MCTS[MCTS Simulator Node]
    C -->|Parallel HRM+TRM| PAR[Parallel Agents]

    PAR --> AGG[Aggregate Results]
    HRM --> AGG
    TRM --> AGG
    MCTS --> AGG

    AGG --> EVAL[Evaluate Consensus]

    EVAL -->|Consensus| SYN[Synthesize]
    EVAL -->|Iterate| C

    SYN --> END[(END)]

    click HRM "#hrm" "HRM decomposition and scoring"
    click TRM "#trm" "TRM iterative refinement"
    click MCTS "#mcts" "Deterministic MCTS search"
""",
        "training_pipeline": """flowchart LR
    D[Dataset\\n(data/, generated_dataset.json)] --> FE[Feature Extraction\\nnormalize_features/ features_to_text]

    FE --> RNN[RNN Training\\nPyTorch GRU]
    FE --> BERT[BERT LoRA Fine-tune\\nTransformers/Diffusers]

    RNN --> E1[Eval/Validation]
    BERT --> E2[Eval/Validation]

    E1 --> TRK[Experiment Tracking\\nW&B / Braintrust]
    E2 --> TRK

    RNN --> CK1[S3 Checkpoint]
    BERT --> CK2[S3 Checkpoint]

    E1 --> PC[(Pinecone 10D Vectors)]
    E2 --> PC

    TRK --> RPT[Reports & Dashboards]
""",
        "external_systems_overview": """graph TB
    subgraph Framework
      FW[LangGraph Multi-Agent MCTS]
    end

    subgraph LLM Providers
      OAI[OpenAI API]
      ANT[Anthropic API]
      LMS[LM Studio]
    end

    subgraph Storage & Vectors
      S3[(AWS S3)]
      PINE[(Pinecone 10D)]
    end

    subgraph Observability
      OTel[OpenTelemetry]
      PROM[Prometheus]
      WAND[Weights & Biases]
      BT[Braintrust]
    end

    FW --> OAI
    FW --> ANT
    FW --> LMS

    FW --> S3
    FW --> PINE

    FW --> OTel
    FW --> PROM
    FW --> WAND
    FW --> BT

    classDef fw fill:#d5e8d4,stroke:#333
    classDef ext fill:#e7f7ff,stroke:#333
    classDef obs fill:#fff1cc,stroke:#333

    class FW fw
    class OAI,ANT,LMS,S3,PINE ext
    class OTel,PROM,WAND,BT obs
""",
    }


def render_mermaid_via_kroki(name: str, source: str, out_png: str) -> None:
    """Render Mermaid source to PNG using Kroki."""
    try:
        # Write .mmd source for reference
        with open(os.path.join(MMD_DIR, f"{name}.mmd"), "w", encoding="utf-8") as f:
            f.write(source)
        # Render to PNG
        url = "https://kroki.io/mermaid/png"
        headers = {"Content-Type": "text/plain"}
        resp = requests.post(url, data=source.encode("utf-8"), headers=headers, timeout=30)
        resp.raise_for_status()
        with open(out_png, "wb") as f:
            f.write(resp.content)
        print(f"[OK] Mermaid rendered: {out_png}")
    except Exception as e:
        print(f"[WARN] Mermaid render failed for {name}: {e}")
        # Leave the .mmd file for manual rendering


def _add_box(ax, xy, w, h, label, fc="#eef", ec="#333"):
    x, y = xy
    rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)


def _add_arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrow(
            x1,
            y1,
            x2 - x1,
            y2 - y1,
            width=0.05,
            head_width=0.3,
            head_length=0.4,
            length_includes_head=True,
            color="#333",
        )
    )


def draw_rnn_meta_controller(path: str, hidden_dim: int = 64, num_layers: int = 1, num_agents: int = 3) -> None:
    if plt is None:
        print("[WARN] Matplotlib not available; skipping RNN diagram")
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 4)
    ax.axis("off")

    _add_box(ax, (0.5, 1.25), 3.0, 1.5, "Input Features\n10-D normalized")
    gru_label = f"GRU\nhidden_dim={hidden_dim}\nlayers={num_layers}"
    _add_box(ax, (4.5, 1.0), 4.0, 2.0, gru_label, fc="#e9f7ef")
    _add_box(ax, (9.5, 1.25), 2.0, 1.5, "Dropout\np=0.1", fc="#fff3cd")
    _add_box(ax, (12.5, 1.25), 3.0, 1.5, f"Linear\n{hidden_dim} → {num_agents}", fc="#e8eaf6")
    _add_box(ax, (16.0, 1.25), 1.5, 1.5, "Softmax", fc="#fce4ec")

    _add_arrow(ax, 3.5, 2.0, 4.5, 2.0)
    _add_arrow(ax, 8.5, 2.0, 9.5, 2.0)
    _add_arrow(ax, 11.5, 2.0, 12.5, 2.0)
    _add_arrow(ax, 15.5, 2.0, 16.0, 2.0)

    ax.text(17.7, 2.7, "P(hrm)", fontsize=9)
    ax.text(17.7, 2.1, "P(trm)", fontsize=9)
    ax.text(17.7, 1.5, "P(mcts)", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Matplotlib rendered: {path}")


def draw_bert_meta_controller(path: str, use_lora: bool = True, num_labels: int = 3) -> None:
    if plt is None:
        print("[WARN] Matplotlib not available; skipping BERT diagram")
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 4)
    ax.axis("off")

    _add_box(ax, (0.5, 1.25), 3.5, 1.5, "Features → Text\nfeatures_to_text()", fc="#eef")
    _add_box(ax, (4.5, 1.25), 2.5, 1.5, "Tokenizer\nAutoTokenizer", fc="#e8eaf6")
    bert_label = "BERT SeqCls\nAutoModelForSeqCls"
    if use_lora:
        bert_label += "\n+ LoRA(r=4, α=16)"
    _add_box(ax, (7.5, 1.0), 5.0, 2.0, bert_label, fc="#e9f7ef")
    _add_box(ax, (13.5, 1.25), 2.0, 1.5, "Logits → Softmax", fc="#fff3cd")
    _add_box(ax, (16.0, 1.25), 3.5, 1.5, f"Probabilities\n{num_labels} labels (hrm,trm,mcts)", fc="#fce4ec")

    _add_arrow(ax, 4.0, 2.0, 4.5, 2.0)
    _add_arrow(ax, 7.0, 2.0, 7.5, 2.0)
    _add_arrow(ax, 12.5, 2.0, 13.5, 2.0)
    _add_arrow(ax, 15.5, 2.0, 16.0, 2.0)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Matplotlib rendered: {path}")


def main() -> int:
    ensure_dirs()

    # Mermaid (C1..C4)
    mermaid = get_mermaid_texts()
    for name, text in mermaid.items():
        out_png = os.path.join(IMG_DIR, f"{name}.png")
        render_mermaid_via_kroki(name, text, out_png)

    # Matplotlib (NN diagrams)
    draw_rnn_meta_controller(os.path.join(IMG_DIR, "rnn_meta_controller.png"))
    draw_bert_meta_controller(os.path.join(IMG_DIR, "bert_meta_controller.png"))
    print("[DONE] Diagram export complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
