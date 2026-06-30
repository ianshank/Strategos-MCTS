# Secrets Management Guide

This guide outlines best practices for managing API keys and sensitive credentials in the Multi-Agent MCTS system.

## Overview

The system uses multiple external services that require API keys:

| Service | Environment Variable | Sensitivity | Rotation Frequency |
|---------|---------------------|-------------|-------------------|
| OpenAI | `OPENAI_API_KEY` | **HIGH** | Every 90 days |
| Anthropic | `ANTHROPIC_API_KEY` | **HIGH** | Every 90 days |
| Braintrust | `BRAINTRUST_API_KEY` | **MEDIUM** | Every 180 days |
| Pinecone | `PINECONE_API_KEY` | **HIGH** | Every 90 days |
| Weights & Biases | `WANDB_API_KEY` | **MEDIUM** | Every 180 days |
| HuggingFace | `HF_TOKEN` | **LOW** | Annually |

## Environment Setup

### Development Environment

1. Create `.env` file (NEVER commit this file):
```bash
# Copy from template
cp .env.example .env

# Edit with your actual keys
```

2. Use environment variables:
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

3. Verify `.gitignore` includes:
```
.env
.env.*
!.env.example
```

### Production Environment

**NEVER** store API keys in:
- Source code
- Git history
- Docker images
- CI/CD logs

**DO** use:
- Secret management services (AWS Secrets Manager, HashiCorp Vault)
- Kubernetes secrets
- CI/CD secret variables

## Key Rotation Procedures

### OpenAI API Key

1. **Generate new key**:
   - Go to https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Name it with date: `mcts-production-2025-11-16`

2. **Update configuration**:
   ```bash
   # Update .env
   OPENAI_API_KEY=sk-new-key-here

   # If using secrets manager, update there
   ```

3. **Verify new key works**:
   ```bash
   python -c "
   import openai
   client = openai.OpenAI()
   response = client.chat.completions.create(
       model='gpt-4o-mini',
       messages=[{'role':'user','content':'test'}]
   )
   print('Key valid:', response.id)
   "
   ```

4. **Revoke old key**:
   - Wait 24-48 hours to ensure all systems updated
   - Delete old key from OpenAI dashboard

### Anthropic API Key

1. **Generate new key**:
   - Go to https://console.anthropic.com/settings/keys
   - Click "Create Key"
   - Name descriptively

2. **Update configuration**:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-new-key-here
   ```

3. **Verify**:
   ```bash
   python -c "
   import anthropic
   client = anthropic.Anthropic()
   response = client.messages.create(
       model='claude-3-5-sonnet-20241022',
       max_tokens=10,
       messages=[{'role':'user','content':'test'}]
   )
   print('Key valid:', response.id)
   "
   ```

4. **Revoke old key** after verification

### Pinecone API Key

1. **Generate new key**:
   - Go to https://app.pinecone.io/
   - Navigate to API Keys
   - Create new key

2. **Update**:
   ```bash
   PINECONE_API_KEY=pcsk_new_key_here
   ```

3. **Verify**:
   ```python
   from pinecone import Pinecone
   pc = Pinecone()
   indexes = pc.list_indexes()
   print('Key valid:', indexes)
   ```

### Braintrust API Key

1. **Generate**:
   - Go to https://www.braintrust.dev/app/settings
   - Create new API key

2. **Update**:
   ```bash
   BRAINTRUST_API_KEY=sk-new-key
   ```

3. **Verify**:
   ```python
   import braintrust
   braintrust.login(api_key=os.getenv("BRAINTRUST_API_KEY"))
   ```

### Weights & Biases API Key

1. **Generate**:
   - Go to https://wandb.ai/settings
   - Create new API key

2. **Update**:
   ```bash
   WANDB_API_KEY=new_key
   ```

3. **Verify**:
   ```bash
   wandb login --verify
   ```

## Automated Key Rotation

### Using Python Script

```python
#!/usr/bin/env python3
"""Key rotation reminder script."""

import os
from datetime import datetime, timedelta
from pathlib import Path

KEY_ROTATION_SCHEDULE = {
    "OPENAI_API_KEY": 90,  # days
    "ANTHROPIC_API_KEY": 90,
    "PINECONE_API_KEY": 90,
    "BRAINTRUST_API_KEY": 180,
    "WANDB_API_KEY": 180,
}

def check_key_rotation():
    rotation_file = Path(".key_rotation_dates.json")

    if not rotation_file.exists():
        print("WARNING: No rotation tracking file found")
        print("Create .key_rotation_dates.json with last rotation dates")
        return

    import json
    with open(rotation_file) as f:
        dates = json.load(f)

    now = datetime.now()

    for key_name, days in KEY_ROTATION_SCHEDULE.items():
        last_rotated = datetime.fromisoformat(dates.get(key_name, "2000-01-01"))
        next_rotation = last_rotated + timedelta(days=days)

        if now > next_rotation:
            print(f"[CRITICAL] {key_name} needs rotation (overdue)")
        elif (next_rotation - now).days < 14:
            print(f"[WARNING] {key_name} rotation due in {(next_rotation - now).days} days")
        else:
            print(f"[OK] {key_name} - next rotation in {(next_rotation - now).days} days")

if __name__ == "__main__":
    check_key_rotation()
```

## CI/CD Secrets Configuration

### GitHub Actions

Add secrets in repository settings:

```yaml
# .github/workflows/ci.yml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  BRAINTRUST_API_KEY: ${{ secrets.BRAINTRUST_API_KEY }}
  PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
  WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
```

### Kubernetes Secrets (External Secrets Operator)

> **Do NOT commit plaintext `kind: Secret` manifests with `stringData` values.** `kubernetes/deployment.yaml`
> defines an `ExternalSecret` (not an inline Secret) that materializes `llm-secrets` at runtime from an
> external store via the [External Secrets Operator](https://external-secrets.io/). The Deployment's
> `secretKeyRef` references are unchanged.

One-time cluster setup:

```bash
# 1. Install the operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  -n external-secrets --create-namespace

# 2. Create a SecretStore in the mcts-framework namespace (AWS Secrets Manager example;
#    swap the provider block for vault/gcpsm/azurekv as needed).
cat <<'EOF' | kubectl apply -f -
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: llm-secret-store
  namespace: mcts-framework
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2
      auth:
        jwt:
          serviceAccountRef:
            name: mcts-eso-sa   # IRSA-bound SA with secretsmanager:GetSecretValue
EOF
```

Store the actual values in the backend under key `strategos/llm` with properties `openai-api-key`,
`anthropic-api-key`, `api-keys`. The `ExternalSecret` in `kubernetes/deployment.yaml` syncs them into the
`llm-secrets` Secret every `refreshInterval` (1h). Force an immediate sync after rotation:

```bash
kubectl annotate externalsecret llm-secrets -n mcts-framework force-sync=$(date +%s) --overwrite
kubectl rollout restart deployment/langgraph-mcts -n mcts-framework
kubectl rollout status deployment/langgraph-mcts -n mcts-framework
```

### Application auth secrets (JWT)

API authentication is selected by `AUTH_MODE` (`api_key` default, or `jwt`). In `jwt` mode the signing
secret comes from `JWT_SECRET` (env → `src/config/settings.py`, `SecretStr`); `JWT_ALGORITHM` /
`JWT_EXPIRY_HOURS` are settings too. Never hardcode `JWT_SECRET` — supply it via the same external store
(add a `jwt-secret` property and an `ExternalSecret` data entry).

## Security Best Practices

### DO

1. **Use environment variables** - Never hardcode keys
2. **Rotate regularly** - Follow the rotation schedule
3. **Least privilege** - Use read-only keys where possible
4. **Monitor usage** - Check for anomalous API usage
5. **Audit logs** - Track who accessed what keys
6. **Separate environments** - Different keys for dev/staging/prod
7. **Backup keys** - Store securely in password manager

### DON'T

1. **Commit keys** - Ever, even temporarily
2. **Share keys** - Each developer should have their own
3. **Log keys** - Be careful with debug output
4. **Email keys** - Use secure channels only
5. **Store in plaintext** - Use encrypted storage
6. **Reuse keys** - Different services, different keys
7. **Ignore expiration** - Rotate before problems occur

## Emergency Procedures

### If Key is Compromised

1. **Immediately revoke** the compromised key
2. **Generate new key** from service dashboard
3. **Update all environments** (dev, staging, prod)
4. **Check for unauthorized usage** in API logs
5. **Document incident** for audit trail
6. **Review git history** for any committed secrets

### If Key is Lost

1. **Regenerate** from service dashboard
2. **Update** environment configurations
3. **Notify team** of key change
4. **Verify** all systems reconnect

## Monitoring and Alerts

### Set Up Usage Monitoring

1. **OpenAI**: Dashboard at https://platform.openai.com/usage
2. **Anthropic**: Console at https://console.anthropic.com/
3. **Pinecone**: Usage metrics in dashboard
4. **W&B**: Project settings

### Alert Thresholds

Configure alerts for:
- Unexpected spike in API calls
- Failed authentication attempts
- Approaching usage limits
- Key expiration warnings

## Compliance Checklist

- [ ] All API keys stored in environment variables
- [ ] `.env` file is in `.gitignore`
- [ ] No secrets in git history
- [ ] Key rotation schedule documented
- [ ] CI/CD secrets properly configured
- [ ] Emergency procedures documented
- [ ] Team trained on key management
- [ ] Regular security audits scheduled
- [ ] Usage monitoring in place
- [ ] Incident response plan ready

## Regular Security Tasks

### Weekly
- Review API usage dashboards
- Check for failed authentication logs

### Monthly
- Verify .gitignore is comprehensive
- Run security audit script
- Review team access to keys

### Quarterly
- Rotate high-sensitivity keys
- Review and update this documentation
- Test emergency procedures
- Update key rotation tracking

### Annually
- Full secrets audit
- Review all service accounts
- Update security procedures
- Train new team members
