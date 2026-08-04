# Docker Deployment Guide

Comprehensive guide for deploying the Multi-Agent MCTS Training Pipeline using Docker with GPU support.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Docker Images](#docker-images)
- [Deployment Workflows](#deployment-workflows)
- [Testing](#testing)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Production Deployment](#production-deployment)

---

## Prerequisites

### Required Software

- **Docker**: 20.10+ with BuildKit enabled
- **Docker Compose**: 2.0+
- **NVIDIA Docker Runtime**: For GPU support
- **Python**: 3.11+ (for local testing)

### Hardware Requirements

| Deployment | GPU | VRAM | CPU | RAM |
|------------|-----|------|-----|-----|
| Demo | 1x | 16GB | 4 cores | 16GB |
| Production | 1-4x | 40GB | 16 cores | 64GB |

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required for training
PINECONE_API_KEY=your-pinecone-api-key
WANDB_API_KEY=your-wandb-api-key
GITHUB_TOKEN=your-github-token

# Optional
OPENAI_API_KEY=your-openai-api-key
NEO4J_PASSWORD=your-neo4j-password

# Docker registry (for CI/CD)
DOCKER_REGISTRY=ghcr.io
DOCKER_USERNAME=your-username
```

---

## Quick Start

### 1. Pre-Deployment Sanity Check

```bash
# Run sanity checks
python scripts/deployment_sanity_check.py --verbose
```

### 2. Build Docker Images

```bash
# Build demo image
docker build -f Dockerfile.train -t langgraph-mcts-train:demo --target demo .

# Build production image
docker build -f Dockerfile.train -t langgraph-mcts-train:prod --target production .
```

### 3. Run Demo Training

```bash
# Using docker-compose
docker-compose -f docker-compose.train.yml up training-demo

# Or using docker directly
docker run --gpus all \
  -e PINECONE_API_KEY=$PINECONE_API_KEY \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  langgraph-mcts-train:demo
```

### 4. Run Smoke Tests

```bash
# Install test dependencies
pip install pytest docker requests

# Run smoke tests
pytest tests/deployment/test_docker_smoke.py -v -m smoke
```

---

## Docker Images

### Available Images

| Image | Purpose | Base | Size |
|-------|---------|------|------|
| `langgraph-mcts-train:demo` | 16GB GPU demo | CUDA 12.1 | ~8GB |
| `langgraph-mcts-train:prod` | Full training | CUDA 12.1 | ~8GB |
| `langgraph-mcts:api` | API server | Python 3.11 | ~2GB |

### Build Targets

```bash
# Demo mode (16GB optimized)
docker build -f Dockerfile.train --target demo -t mcts:demo .

# Production mode (full training)
docker build -f Dockerfile.train --target production -t mcts:prod .

# Base image (for development)
docker build -f Dockerfile.train --target base -t mcts:base .
```

### Multi-Platform Build

```bash
# Build for multiple architectures
docker buildx create --use
docker buildx build -f Dockerfile.train \
  --platform linux/amd64,linux/arm64 \
  --target demo \
  -t langgraph-mcts-train:demo \
  --push .
```

---

## Deployment Workflows

### Local Development

```bash
# Start Jupyter for interactive development
docker-compose --profile development up jupyter

# Access at http://localhost:8888
```

### Staging Environment

```bash
# Deploy to staging
docker-compose -f docker-compose.train.yml \
  --profile production \
  up -d training-prod

# Monitor logs
docker-compose -f docker-compose.train.yml logs -f training-prod
```

### Production Environment

```bash
# Pull latest images
docker pull ghcr.io/your-org/langgraph-mcts-train:prod

# Run with monitoring
docker-compose \
  -f docker-compose.yml \
  -f docker-compose.train.yml \
  --profile production \
  --profile monitoring \
  up -d

# Check health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

---

## Testing

### Pre-Deployment Tests

```bash
# Run sanity checks
python scripts/deployment_sanity_check.py

# Expected output:
#   ✓ Configuration Files - PASS
#   ✓ Docker Files - PASS
#   ✓ Python Syntax - PASS
#   ✓ Dependencies - PASS
#   ✓ Required Scripts - PASS
#   ✓ Test Suite - PASS
#   ✓ Documentation - PASS
```

### Post-Deployment Smoke Tests

```bash
# Test running containers
pytest tests/deployment/test_docker_smoke.py -v

# Test specific container
pytest tests/deployment/test_docker_smoke.py::test_cuda_available_in_container -v

# Generate coverage report
pytest tests/deployment/ --cov=. --cov-report=html
```

### GPU Tests

```bash
# Verify GPU access
docker run --gpus all langgraph-mcts-train:demo python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU Count: {torch.cuda.device_count()}')
print(f'GPU Name: {torch.cuda.get_device_name(0)}')
"
```

---

## Monitoring

### Container Health

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' mcts-training-demo

# View health check logs
docker inspect --format='{{json .State.Health}}' mcts-training-demo | jq
```

### Resource Usage

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor container resources
docker stats mcts-training-demo

# Container logs
docker logs -f --tail=100 mcts-training-demo
```

### Metrics and Dashboards

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access dashboards:
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9090
# - Jaeger: http://localhost:16686
```

---

## Troubleshooting

### Common Issues

#### 1. GPU Not Accessible

**Symptom:**
```
RuntimeError: No CUDA GPUs are available
```

**Solution:**
```bash
# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

#### 2. Out of Memory (OOM)

**Symptom:**
```
CUDA out of memory
```

**Solution:**
```bash
# Use demo mode with smaller batch size
docker run --gpus all \
  -e DEMO_MODE=1 \
  langgraph-mcts-train:demo
```

#### 3. Container Fails Health Check

**Symptom:**
```
Container unhealthy
```

**Solution:**
```bash
# Check logs
docker logs mcts-training-demo

# Run health check manually
docker exec mcts-training-demo python /app/healthcheck.py

# Restart container
docker restart mcts-training-demo
```

#### 4. Network Connectivity Issues

**Symptom:**
```
Cannot connect to external services
```

**Solution:**
```bash
# Check DNS
docker exec mcts-training-demo ping -c 3 google.com

# Check environment variables
docker exec mcts-training-demo env | grep API_KEY

# Test service verification
docker exec mcts-training-demo \
  python scripts/verify_external_services.py
```

---

## Production Deployment

### Security Best Practices

```yaml
# docker-compose.prod.yml
services:
  training-prod:
    # Use specific image tags (not :latest)
    image: ghcr.io/org/langgraph-mcts-train:v1.2.3

    # Run as non-root user
    user: "1000:1000"

    # Read-only root filesystem
    read_only: true

    # Drop unnecessary capabilities
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE

    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '16'
          memory: 64G
          reservations:
            devices:
              - capabilities: [gpu]
```

### High Availability

```bash
# Multiple replicas with load balancing
docker-compose up --scale api-server=3

# Health checks with automatic restart
services:
  training-prod:
    restart: unless-stopped
    healthcheck:
      interval: 30s
      timeout: 10s
      retries: 3
```

### Backup and Recovery

```bash
# Backup checkpoints
docker run --rm \
  -v mcts-checkpoints:/data \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/checkpoints-$(date +%Y%m%d).tar.gz -C /data .

# Restore from backup
docker run --rm \
  -v mcts-checkpoints:/data \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/checkpoints-20250120.tar.gz -C /data
```

### CI/CD Integration

The pipeline automatically:

1. **Runs sanity checks** on every push
2. **Builds Docker images** for demo and production
3. **Runs smoke tests** on built images
4. **Scans for vulnerabilities** with Trivy — an advisory CRITICAL+HIGH scan that uploads SARIF to
   GitHub code scanning, plus a **blocking** scan that fails the build on fixable CRITICAL
   findings. Accepted exceptions live in `.trivyignore` (each needs a rationale, an expiry date and
   a tracking link). Note also that `docker-deployment.yml` now filters its `pull_request` trigger
   by `paths`, so docs-only PRs no longer run the image matrix.
5. **Deploys to staging/production** on main branch

```bash
# Trigger deployment manually
gh workflow run docker-deployment.yml \
  -f deploy_environment=production
```

---

## Performance Optimization

### Build Optimization

```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Use build cache
docker build --cache-from langgraph-mcts-train:latest \
  -t langgraph-mcts-train:new .

# Multi-stage builds to reduce image size
# Already implemented in Dockerfile.train
```

### Runtime Optimization

```bash
# Use tmpfs for temporary files
docker run --gpus all \
  --tmpfs /tmp:rw,size=4g \
  langgraph-mcts-train:demo

# Optimize GPU memory
docker run --gpus all \
  --shm-size=8g \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  langgraph-mcts-train:demo
```

---

## Additional Resources

- [Dockerfile.train](../Dockerfile.train) - Training image definition
- [docker-compose.train.yml](../docker-compose.train.yml) - Training services
- [deployment_sanity_check.py](../scripts/deployment_sanity_check.py) - Pre-deployment checks
- [test_docker_smoke.py](../tests/deployment/test_docker_smoke.py) - Post-deployment tests
- [LOCAL_TRAINING_GUIDE.md](LOCAL_TRAINING_GUIDE.md) - Local training setup

---

## Support

For issues or questions:

- [GitHub Issues](https://github.com/your-org/langgraph_multi_agent_mcts/issues)
- [Discussions](https://github.com/your-org/langgraph_multi_agent_mcts/discussions)

---

**Last Updated:** 2025-01-20
**Version:** 1.0.0
