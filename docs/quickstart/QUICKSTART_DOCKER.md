# Docker Training Quick Start

## 🚀 Run Training in 3 Commands

### Prerequisites
- Docker Desktop with GPU support (NVIDIA Docker)
- API keys for Pinecone, W&B, GitHub

---

## Option 1: Automated Script (Recommended)

```powershell
# 1. Ensure .env is configured
# Edit .env file with your API keys

# 2. Run training (builds image + starts container)
.\scripts\docker_train.ps1

# 3. Monitor logs
docker logs -f mcts-training-demo
```

That's it! The script handles everything automatically.

---

## Option 2: Manual Step-by-Step

### Step 1: Configure Environment Variables

Check your `.env` file has these keys:

```bash
PINECONE_API_KEY=your-key-here
WANDB_API_KEY=your-key-here
GITHUB_TOKEN=your-token-here
```

### Step 2: Build Docker Image

```powershell
# Build demo image (16GB GPU)
docker build -f Dockerfile.train --target demo -t langgraph-mcts-train:demo .

# Or build production image (multi-GPU)
docker build -f Dockerfile.train --target production -t langgraph-mcts-train:prod .
```

### Step 3: Run Training

```powershell
# Using docker-compose (recommended)
docker-compose -f docker-compose.train.yml up training-demo

# Or using docker directly
docker run --gpus all `
  --env-file .env `
  -v ${PWD}/checkpoints:/app/checkpoints `
  -v ${PWD}/logs:/app/logs `
  -v ${PWD}/cache:/app/cache `
  --name mcts-training-demo `
  langgraph-mcts-train:demo
```

### Step 4: Monitor Training

```powershell
# View logs
docker logs -f mcts-training-demo

# Check GPU usage
nvidia-smi -l 1

# View container status
docker ps

# Check health
docker inspect --format='{{.State.Health.Status}}' mcts-training-demo
```

---

## Verify Everything is Working

### 1. Check Container is Running

```powershell
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Expected output:
```
NAMES                STATUS              PORTS
mcts-training-demo   Up 2 minutes (healthy)
```

### 2. Verify GPU Access

```powershell
docker exec mcts-training-demo python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

Expected output:
```
CUDA: True
GPUs: 1
```

### 3. Check Training Progress

```powershell
# View last 50 lines of logs
docker logs --tail 50 mcts-training-demo

# Follow logs in real-time
docker logs -f mcts-training-demo
```

Look for:
```
==========================================
Multi-Agent MCTS Training Pipeline
==========================================
PyTorch: 2.x.x
CUDA: True
GPUs: 1
==========================================
Starting training pipeline...
Phase 1/5: Base Pre-training
```

### 4. Check W&B Dashboard

1. Go to https://wandb.ai
2. Find project: `multi-agent-mcts-demo`
3. View your training run
4. Monitor metrics: loss, accuracy, GPU usage

---

## Common Commands

### Container Management

```powershell
# Start container
docker-compose -f docker-compose.train.yml up -d training-demo

# Stop container
docker-compose -f docker-compose.train.yml stop

# Remove container
docker-compose -f docker-compose.train.yml down

# Restart container
docker-compose -f docker-compose.train.yml restart training-demo
```

### Logs and Debugging

```powershell
# View logs
docker logs mcts-training-demo

# Follow logs
docker logs -f --tail 100 mcts-training-demo

# Execute command in container
docker exec mcts-training-demo python --version

# Interactive shell
docker exec -it mcts-training-demo bash
```

### View Generated Artifacts

```powershell
# Checkpoints
ls checkpoints\

# Logs
ls logs\demo\

# Cache
ls cache\

# Reports
ls reports\
```

---

## Troubleshooting

### Issue: "No such image"

**Solution:**
```powershell
# Rebuild image
docker build -f Dockerfile.train --target demo -t langgraph-mcts-train:demo .
```

### Issue: "CUDA not available"

**Solution:**
```powershell
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Ensure GPU is specified in docker run
docker run --gpus all ...
```

### Issue: "Container unhealthy"

**Solution:**
```powershell
# Check health check logs
docker inspect mcts-training-demo | Select-String -Pattern "Health"

# View recent logs
docker logs --tail 100 mcts-training-demo

# Restart container
docker restart mcts-training-demo
```

### Issue: "Port already in use"

**Solution:**
```powershell
# Find and stop conflicting container
docker ps -a
docker stop <container-name>

# Or use different port
# Edit docker-compose.train.yml to change port mappings
```

---

## Advanced Usage

### Run in Background

```powershell
# Start detached
docker-compose -f docker-compose.train.yml up -d training-demo

# View status
docker ps

# Attach to logs
docker logs -f mcts-training-demo
```

### Custom Configuration

```powershell
# Use custom config
docker run --gpus all `
  -e DEMO_MODE=1 `
  -v ${PWD}/training/my_config.yaml:/app/training/config.yaml `
  langgraph-mcts-train:demo
```

### Production Mode

```powershell
# Build production image
docker build -f Dockerfile.train --target production -t langgraph-mcts-train:prod .

# Run production training
docker-compose -f docker-compose.train.yml --profile production up training-prod
```

---

## Next Steps

After training completes:

1. **Review Results**
   - Check W&B dashboard for metrics
   - Review logs in `./logs/demo/`
   - Inspect checkpoints in `./checkpoints/`

2. **Run Evaluation**
   ```powershell
   docker exec mcts-training-demo python -m training.cli evaluate --model checkpoints/demo/hrm_checkpoint_epoch_3.pt
   ```

3. **Scale to Production**
   - Use production config: `training/config.yaml`
   - Deploy with multi-GPU support
   - Enable full dataset processing

---

## Quick Reference

| Task | Command |
|------|---------|
| Build image | `docker build -f Dockerfile.train -t mcts-train:demo --target demo .` |
| Start training | `docker-compose -f docker-compose.train.yml up training-demo` |
| View logs | `docker logs -f mcts-training-demo` |
| Stop training | `docker-compose -f docker-compose.train.yml stop` |
| GPU check | `docker exec mcts-training-demo nvidia-smi` |
| Health check | `docker inspect --format='{{.State.Health.Status}}' mcts-training-demo` |

---

## Support

- **Documentation:** [docs/DOCKER_DEPLOYMENT.md](docs/DOCKER_DEPLOYMENT.md)
- **Issues:** [GitHub Issues](https://github.com/your-org/langgraph_multi_agent_mcts/issues)
- **Training Guide:** [docs/LOCAL_TRAINING_GUIDE.md](docs/LOCAL_TRAINING_GUIDE.md)

---

**Ready to train? Run:** `.\scripts\docker_train.ps1` 🚀
