> **⚠️ UNRATIFIED TEMPLATE — not an operative commitment.** Like `docs/SLA.md`, this document
> assumes a staffed "Operations Team" with PagerDuty on-call and a CTO escalation ladder that does
> not exist for this project — see `CHARTER.md` §3 NG-1. Retained as reference for the alert-response
> procedures below (Prometheus queries, remediation steps), which are useful independent of staffing.

# Incident Response Runbook

## MCTS Framework Operations

**Version**: 1.0.0
**Last Updated**: 2025-01-15
**Owner**: Operations Team

---

## 1. On-Call Responsibilities

### 1.1 Initial Response Checklist

When an alert fires or incident is reported:

1. [ ] **Acknowledge** the alert (within 5 minutes for P0/P1)
2. [ ] **Assess** severity using the severity matrix below
3. [ ] **Communicate** status to stakeholders
4. [ ] **Investigate** using diagnostic procedures
5. [ ] **Mitigate** with appropriate remediation steps
6. [ ] **Document** all actions in incident log
7. [ ] **Notify** customers if impact is significant

### 1.2 Severity Assessment Matrix

| Severity | Criteria | Example Symptoms |
|----------|----------|------------------|
| **P0 - Critical** | Complete service outage | All requests failing, health check down |
| **P1 - High** | Major functionality broken | MCTS disabled, auth failures > 50% |
| **P2 - Medium** | Degraded performance | P95 latency > 30s, error rate > 1% |
| **P3 - Low** | Minor issues | Cosmetic bugs, non-critical warnings |

---

## 2. Common Incident Scenarios

### 2.1 Service Completely Down

**Symptoms**:
- `/health` endpoint returns 5xx or timeout
- All requests failing
- Prometheus metrics show 0 requests

**Diagnostic Steps**:

```bash
# 1. Check container status
docker ps -a | grep mcts

# 2. Check container logs
docker logs langgraph-mcts --tail 100

# 3. Check resource usage
docker stats langgraph-mcts

# 4. Verify network connectivity
docker exec langgraph-mcts curl -f http://localhost:8000/health

# 5. Check Kubernetes pods (if using K8s)
kubectl get pods -l app=mcts-framework
kubectl describe pod <pod-name>
kubectl logs <pod-name> --tail 100
```

**Remediation Steps**:

```bash
# Option 1: Restart service
docker-compose restart mcts-framework

# Option 2: Full redeploy
docker-compose down
docker-compose up -d

# Option 3: Roll back to previous version
docker-compose down
git checkout <previous-tag>
docker-compose up -d --build

# Kubernetes rollback
kubectl rollout undo deployment/langgraph-mcts
```

### 2.2 High Error Rate (> 1%)

**Symptoms**:
- Alert: `HighErrorRate` firing
- Dashboard showing spike in 5xx responses
- Customer complaints about failed requests

**Diagnostic Steps**:

```bash
# 1. Check error logs
docker logs langgraph-mcts 2>&1 | grep ERROR | tail 50

# 2. Check error distribution
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(mcts_errors_total[5m])' | jq .

# 3. Check specific error types
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=mcts_errors_total{error_type=~".*"}' | jq .

# 4. Check recent changes
git log --oneline -20

# 5. Verify LLM provider status
curl -I https://api.openai.com
curl -I https://api.anthropic.com
```

**Remediation Steps**:

1. **If LLM provider issue**:
   ```bash
   # Switch to backup provider
   export LLM_PROVIDER=anthropic
   docker-compose restart mcts-framework
   ```

2. **If memory issue**:
   ```bash
   # Increase memory limit
   docker-compose down
   # Edit docker-compose.yml to increase memory
   docker-compose up -d
   ```

3. **If code bug**:
   ```bash
   # Roll back to last known good
   git checkout <stable-tag>
   docker-compose up -d --build
   ```

### 2.3 High Latency (P95 > 30s)

**Symptoms**:
- Alert: `HighLatencyP95` firing
- Users reporting slow responses
- Timeouts in client applications

**Diagnostic Steps**:

```bash
# 1. Check current latency
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, rate(mcts_request_duration_seconds_bucket[5m]))' | jq .

# 2. Check CPU/Memory usage
docker stats langgraph-mcts

# 3. Check MCTS iteration settings
docker exec langgraph-mcts env | grep MCTS

# 4. Check active requests
curl -s http://localhost:9090/api/v1/query \
  --data-urlencode 'query=mcts_active_requests' | jq .

# 5. Check for resource contention
docker exec langgraph-mcts top -bn1
```

**Remediation Steps**:

1. **Reduce MCTS iterations temporarily**:
   ```bash
   export MCTS_DEFAULT_ITERATIONS=50
   docker-compose restart mcts-framework
   ```

2. **Scale up resources**:
   ```bash
   # Increase replicas (Kubernetes)
   kubectl scale deployment/langgraph-mcts --replicas=5
   ```

3. **Enable request queuing**:
   ```bash
   # Add rate limiting at nginx/ingress level
   ```

### 2.4 Memory Leak Detected

**Symptoms**:
- Alert: `HighMemoryUsage` firing
- Memory growing continuously
- OOM kills in container logs

**Diagnostic Steps**:

```bash
# 1. Check memory trend
curl -s http://localhost:9090/api/v1/query_range \
  --data-urlencode 'query=process_resident_memory_bytes' \
  --data-urlencode 'start=now-1h' \
  --data-urlencode 'end=now' \
  --data-urlencode 'step=60' | jq .

# 2. Check for memory profile
docker exec langgraph-mcts python -c "import tracemalloc; tracemalloc.start()"

# 3. Check garbage collection
docker exec langgraph-mcts python -c "import gc; print(gc.get_stats())"

# 4. Check active objects
docker exec langgraph-mcts python -c "
import sys
print(f'Loaded modules: {len(sys.modules)}')
"
```

**Remediation Steps**:

1. **Restart service (immediate mitigation)**:
   ```bash
   docker-compose restart mcts-framework
   ```

2. **Enable more aggressive GC**:
   ```python
   # Add to startup
   import gc
   gc.set_threshold(700, 10, 10)
   ```

3. **Deploy fix for memory leak** (long-term)

### 2.5 Authentication Failures

**Symptoms**:
- Alert: `AuthenticationErrors` firing
- Users reporting 401 errors
- Rate limit bypass attempts

**Diagnostic Steps**:

```bash
# 1. Check auth error logs
docker logs langgraph-mcts 2>&1 | grep "AUTH_ERROR" | tail 20

# 2. Check for brute force attempts
docker logs langgraph-mcts 2>&1 | grep "Invalid API key" | wc -l

# 3. Verify API key configuration
docker exec langgraph-mcts env | grep API_KEY

# 4. Check rate limit status
curl -s http://localhost:8000/stats -H "X-API-Key: demo-key"
```

**Remediation Steps**:

1. **Block suspicious IPs**:
   ```bash
   # Add to nginx/firewall
   iptables -A INPUT -s <suspicious-ip> -j DROP
   ```

2. **Rotate API keys**:
   ```bash
   # Generate new keys for affected clients
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

3. **Increase rate limits temporarily**:
   ```bash
   # If legitimate traffic spike
   export RATE_LIMIT_PER_MINUTE=120
   docker-compose restart mcts-framework
   ```

---

## 3. Escalation Procedures

### 3.1 Escalation Matrix

| Time Elapsed | Action Required |
|--------------|-----------------|
| 0 min | On-call acknowledges alert |
| 15 min | If P0/P1 unresolved, page backup on-call |
| 30 min | If P0 unresolved, page engineering lead |
| 1 hour | If P0 unresolved, page CTO |
| 2 hours | If P0 unresolved, executive briefing |

### 3.2 Contact List

| Role | Name | Phone | Email |
|------|------|-------|-------|
| Primary On-Call | See PagerDuty | Via PagerDuty | oncall@mcts.io |
| Backup On-Call | See PagerDuty | Via PagerDuty | oncall@mcts.io |
| Engineering Lead | [Name] | [Phone] | [Email] |
| Operations Lead | [Name] | [Phone] | [Email] |
| CTO | [Name] | [Phone] | [Email] |

---

## 4. Communication Templates

### 4.1 Initial Incident Notification

```
Subject: [P0/P1/P2] MCTS Framework Incident - [Brief Description]

Status: Investigating

What's happening:
[Brief description of symptoms]

Impact:
[Number of affected customers/requests]

Current actions:
[What's being done]

Next update: [Time]

Incident Commander: [Name]
```

### 4.2 Status Update

```
Subject: UPDATE - [Incident ID] MCTS Framework

Status: [Investigating/Identified/Monitoring/Resolved]

Current situation:
[What we know now]

Actions taken:
[List of remediation steps]

Remaining work:
[What's still needed]

Next update: [Time]
```

### 4.3 Resolution Notification

```
Subject: RESOLVED - [Incident ID] MCTS Framework

Duration: [Start time] to [End time] ([Duration])

Root cause:
[Brief description]

Impact:
[Summary of impact]

Resolution:
[What fixed the issue]

Preventive measures:
[Actions to prevent recurrence]

Post-mortem: [Link to post-mortem document]
```

---

## 5. Post-Incident Procedures

### 5.1 Post-Mortem Template

```markdown
# Post-Mortem: [Incident ID]

## Summary
- **Date**: [Date]
- **Duration**: [Duration]
- **Severity**: [P0/P1/P2/P3]
- **Incident Commander**: [Name]

## Timeline
- **[Time]**: [Event]
- **[Time]**: [Event]
- ...

## Root Cause
[Detailed explanation]

## Impact
- Requests affected: [Number]
- Customers affected: [Number]
- Financial impact: [Estimate]

## What Went Well
- [Point 1]
- [Point 2]

## What Went Wrong
- [Point 1]
- [Point 2]

## Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| [Action] | [Name] | [Date] | [Status] |

## Lessons Learned
[Key takeaways]
```

### 5.2 Metrics to Capture

- Time to detect (TTD)
- Time to acknowledge (TTA)
- Time to resolve (TTR)
- Customer impact duration
- Error budget consumed
- Number of escalations

---

## 6. Preventive Maintenance

### 6.1 Daily Checks

- [ ] Review alert dashboard for warnings
- [ ] Check error rate trends
- [ ] Verify backup health checks
- [ ] Monitor disk space usage

### 6.2 Weekly Tasks

- [ ] Review performance metrics
- [ ] Check for dependency updates
- [ ] Verify backup restoration
- [ ] Review security alerts

### 6.3 Monthly Tasks

- [ ] Conduct chaos engineering tests
- [ ] Review SLA compliance
- [ ] Update runbooks if needed
- [ ] Perform disaster recovery drill

---

## 7. Useful Commands Quick Reference

```bash
# Service health
curl http://localhost:8000/health

# Current metrics
curl http://localhost:8000/metrics

# Container logs
docker logs langgraph-mcts -f

# Restart service
docker-compose restart mcts-framework

# Check Prometheus alerts
curl http://localhost:9090/api/v1/alerts

# View Grafana dashboards
open http://localhost:3000

# Kubernetes pod status
kubectl get pods -n mcts

# Force pod restart
kubectl delete pod <pod-name> -n mcts
```

---

**Remember**: Document everything. Every action you take during an incident should be recorded. This helps with post-mortem analysis and improves future incident response.

**Stay calm, communicate clearly, and follow the procedures.**

---

*This runbook is maintained by the Operations team. Report any issues or suggestions to ops@mcts-framework.io*
