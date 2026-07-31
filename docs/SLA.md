> **⚠️ UNRATIFIED TEMPLATE — not an operative commitment.** This document has never been ratified:
> its contacts, jurisdiction, and approval block are unfilled placeholders, its dates are from
> January 2025, and it names the project by a superseded title. Strategos-MCTS makes **no** uptime,
> latency, or support-response commitment to anyone — see `CHARTER.md` §3 NG-1. Retained for
> reference as a starting point should the project ever be operated as a service.

# Service Level Agreement (SLA)

## LangGraph Multi-Agent MCTS Framework

**Version**: 1.0.0
**Effective Date**: 2025-01-01
**Last Updated**: 2025-01-15

---

## 1. Service Overview

The LangGraph Multi-Agent MCTS Framework provides intelligent multi-agent reasoning with Monte Carlo Tree Search tactical simulation. This SLA defines the performance commitments and operational standards for the production service.

---

## 2. Availability Commitment

### 2.1 Uptime Guarantee

| Tier | Uptime Target | Max Downtime/Year | Max Downtime/Month |
|------|--------------|-------------------|-------------------|
| **Standard** | 99.5% | 43.8 hours | 3.65 hours |
| **Production** | 99.9% | 8.76 hours | 43.8 minutes |
| **Enterprise** | 99.95% | 4.38 hours | 21.9 minutes |

**Measurement**: Availability = (Total Minutes - Downtime Minutes) / Total Minutes × 100

### 2.2 Scheduled Maintenance

- **Maintenance Window**: Sundays 02:00-06:00 UTC
- **Advance Notice**: 72 hours minimum
- **Emergency Patches**: 24 hours notice (security-critical only)
- **Maintenance Impact**: Not counted against uptime SLA if announced

### 2.3 Excluded Downtime

The following are **not** counted against uptime commitments:
- Scheduled maintenance windows
- Customer-caused outages (invalid API keys, rate limit violations)
- Force majeure events (natural disasters, major infrastructure failures)
- Third-party service outages (LLM providers, cloud infrastructure)

---

## 3. Performance Objectives

### 3.1 Latency Targets

| Metric | Standard Tier | Production Tier | Enterprise Tier |
|--------|--------------|-----------------|-----------------|
| **P50 Latency** | < 10 seconds | < 5 seconds | < 3 seconds |
| **P95 Latency** | < 30 seconds | < 15 seconds | < 10 seconds |
| **P99 Latency** | < 60 seconds | < 30 seconds | < 20 seconds |
| **Max Latency** | < 120 seconds | < 60 seconds | < 45 seconds |

**Notes**:
- Latency measured from request receipt to response sent
- MCTS-enabled queries may have higher latency (scales with iterations)
- RAG retrieval adds 1-3 seconds to baseline

### 3.2 Throughput Guarantees

| Tier | Requests/Minute | Requests/Hour | Burst Capacity |
|------|----------------|---------------|----------------|
| **Standard** | 60 | 1,000 | 100/second |
| **Production** | 300 | 10,000 | 500/second |
| **Enterprise** | 1,000 | 50,000 | 2,000/second |

### 3.3 Error Rate Limits

| Error Type | Target | Critical Threshold |
|-----------|--------|-------------------|
| **Total Error Rate** | < 0.1% | > 1% |
| **Server Errors (5xx)** | < 0.05% | > 0.5% |
| **Timeout Errors** | < 0.1% | > 1% |
| **LLM Provider Errors** | < 0.5% | > 2% |

---

## 4. Response Quality

### 4.1 Agent Confidence Scores

- **Minimum Average Confidence**: 0.65 (65%)
- **Target Average Confidence**: 0.80 (80%)
- **Consensus Threshold**: Configurable (default: 0.75)

### 4.2 MCTS Simulation Quality

- **Minimum Iterations**: 10 (guaranteed exploration)
- **Default Iterations**: 100 (balanced performance)
- **Maximum Iterations**: 10,000 (deep analysis)
- **Exploration Completeness**: > 90% of action space explored

### 4.3 RAG Retrieval Quality

- **Retrieval Relevance**: > 0.7 cosine similarity
- **Context Coverage**: Top-k documents (k = 3-10)
- **Freshness**: Vector store updates within 1 hour of source changes

---

## 5. Support Response Times

### 5.1 Incident Severity Levels

| Severity | Definition | Examples |
|----------|-----------|----------|
| **P0 - Critical** | Service completely unavailable | Total outage, data corruption |
| **P1 - High** | Major feature unavailable | MCTS disabled, auth failures |
| **P2 - Medium** | Degraded performance | High latency, partial errors |
| **P3 - Low** | Minor issues | UI bugs, documentation errors |

### 5.2 Response Time Commitments

| Severity | Initial Response | Status Update | Resolution Target |
|----------|-----------------|---------------|-------------------|
| **P0 - Critical** | < 15 minutes | Every 30 minutes | < 4 hours |
| **P1 - High** | < 1 hour | Every 2 hours | < 8 hours |
| **P2 - Medium** | < 4 hours | Daily | < 48 hours |
| **P3 - Low** | < 24 hours | As needed | < 7 days |

### 5.3 Communication Channels

- **Status Page**: https://status.mcts-framework.io
- **Email**: support@mcts-framework.io
- **Slack**: #mcts-support (Enterprise tier)
- **Phone**: +1-xxx-xxx-xxxx (P0/P1 only)

---

## 6. Data & Security

### 6.1 Data Retention

- **Request Logs**: 90 days
- **Response Data**: Not stored (stateless)
- **Metrics Data**: 30 days (Prometheus)
- **Traces**: 7 days (Jaeger)

### 6.2 Security Commitments

- **Encryption at Rest**: AES-256
- **Encryption in Transit**: TLS 1.3
- **Authentication**: API key with SHA-256 hashing
- **Rate Limiting**: Per-client enforcement
- **Security Patching**: < 24 hours for critical CVEs

### 6.3 Compliance

- **SOC 2 Type II**: Planned
- **GDPR**: Compliant
- **Data Residency**: Configurable per deployment

---

## 7. Service Credits

### 7.1 Credit Calculation

| Monthly Uptime | Credit Percentage |
|----------------|-------------------|
| 99.0% - 99.9% | 10% of monthly fee |
| 95.0% - 99.0% | 25% of monthly fee |
| < 95.0% | 50% of monthly fee |

### 7.2 Credit Request Process

1. Submit request within 30 days of incident
2. Include incident ID and impact description
3. Credits applied to next billing cycle
4. Maximum credit: 50% of monthly fee

### 7.3 Exclusions

Credits are **not** available for:
- Customer-caused issues
- Third-party service failures outside our control
- Scheduled maintenance
- Beta/preview features

---

## 8. Monitoring & Reporting

### 8.1 Available Metrics

- **Real-time Dashboard**: Grafana (https://metrics.mcts-framework.io)
- **API Metrics**: `/metrics` endpoint (Prometheus format)
- **Health Checks**: `/health`, `/ready` endpoints

### 8.2 Monthly Reports

- Uptime percentage
- Latency percentiles
- Error rate summary
- Incident post-mortems
- Capacity planning recommendations

### 8.3 Alerting

Customers are notified of:
- P0/P1 incidents (within 15 minutes)
- Scheduled maintenance (72+ hours advance)
- Security advisories (as needed)
- Service degradation (within 30 minutes)

---

## 9. Capacity Planning

### 9.1 Resource Allocation

| Component | Minimum | Recommended | Maximum |
|-----------|---------|-------------|---------|
| **CPU** | 2 cores | 4 cores | 8 cores |
| **Memory** | 4 GB | 8 GB | 16 GB |
| **Storage** | 10 GB | 50 GB | 200 GB |
| **Network** | 100 Mbps | 1 Gbps | 10 Gbps |

### 9.2 Scaling Guarantees

- **Horizontal Scaling**: Auto-scale based on load
- **Scale-up Time**: < 5 minutes
- **Scale-down Time**: < 15 minutes (graceful)
- **Max Replicas**: Configurable (default: 10)

---

## 10. Terms & Conditions

### 10.1 SLA Modifications

- 30 days advance notice for changes
- Customer consent required for downgrades
- Grandfathering for existing commitments

### 10.2 Termination

Either party may terminate with:
- 30 days written notice
- Immediate termination for material breach
- Pro-rated refunds for prepaid services

### 10.3 Governing Law

- Jurisdiction: [Your Jurisdiction]
- Arbitration: [Arbitration Provider]
- Language: English

---

## 11. Contact Information

**Technical Support**
Email: support@mcts-framework.io
Phone: +1-xxx-xxx-xxxx

**Account Management**
Email: accounts@mcts-framework.io

**Security Issues**
Email: security@mcts-framework.io

**Status Page**
https://status.mcts-framework.io

---

## Appendix A: Service Level Indicators (SLIs)

### A.1 Availability SLI
```
SLI = (Successful Health Checks) / (Total Health Checks) × 100
Measurement: Every 30 seconds via /health endpoint
```

### A.2 Latency SLI
```
SLI = P95 latency of /query endpoint
Measurement: Prometheus histogram quantile
```

### A.3 Error Rate SLI
```
SLI = (5xx Responses) / (Total Responses) × 100
Measurement: HTTP status code counter
```

### A.4 Throughput SLI
```
SLI = Requests per second
Measurement: Rate of successful requests
```

---

## Appendix B: Service Level Objectives (SLOs)

| SLO Name | Target | Error Budget (30 days) |
|----------|--------|------------------------|
| Availability | 99.9% | 43.2 minutes downtime |
| P95 Latency | < 15s | 5% of requests > 15s |
| Error Rate | < 0.1% | 0.1% of requests fail |
| Throughput | > 100 req/min | Never below minimum |

---

**Document Approval**

- Technical Lead: _________________ Date: _________
- Operations Lead: _________________ Date: _________
- Product Owner: _________________ Date: _________
- Customer Success: _________________ Date: _________

---

*This SLA is a living document and will be updated as the service evolves. All changes will be communicated to stakeholders with appropriate notice.*
