# Log Monitoring Patterns for Security Operations

## Overview
Effective detection requires knowing what "normal" looks like and alerting on deviations. These patterns cover web application attacks, authentication anomalies, and host-based indicators.

## Web Application Attack Patterns

### SQL Injection Indicators
```
# Regex for SIEM (Splunk/Elastic)
UNION.*SELECT|SELECT.*FROM.*WHERE|1=1|OR.*1.*=.*1|--\s*$|xp_cmdshell
```
Alert threshold: >10 matches/minute from same IP.

### Command Injection
```
;.*whoami|&&.*id|\$\(.*\)|`.*`|%0a.*cat.*passwd
```

### Path Traversal
```
\.\./|\.\.\%2f|%2e%2e%2f
```

## Authentication Anomalies
- **Brute force**: >10 failed logins in 5 minutes from single IP.
- **Credential stuffing**: failed logins across >50 distinct accounts from single IP/ASN.
- **Impossible travel**: same user authenticating from two geos >500km apart within 1 hour.
- **Off-hours admin access**: privileged login between 22:00–06:00 local time.

## Host-Based Indicators
- New SUID files created outside package manager.
- Crontab modifications by non-root users.
- Outbound connections from database servers (should be zero).
- `/etc/passwd` or `/etc/shadow` read by non-system processes.
- Large outbound data transfers (>100MB) from internal hosts.

## SIEM Alert Tuning
1. Start with high-fidelity, low-volume rules to avoid alert fatigue.
2. Baseline normal traffic per endpoint for 2 weeks before enabling anomaly rules.
3. Suppress known-good scanner IPs and monitoring agents.

## Log Retention Policy
| Log Type | Retention |
|----------|-----------|
| Auth logs | 1 year |
| Web access logs | 90 days |
| Database audit | 1 year |
| Network flow | 90 days |
| Security alerts | 2 years |
