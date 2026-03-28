# Network Segmentation Guide

## Overview
Flat networks let attackers move laterally without friction. Segmentation limits blast radius by ensuring a compromised host in one zone cannot directly reach assets in another.

## Zone Architecture

### Recommended Zones
| Zone | Assets | Trust Level |
|------|--------|-------------|
| DMZ | Public-facing web, load balancers | Untrusted |
| App | Application servers, APIs | Semi-trusted |
| Data | Databases, S3, secrets stores | Trusted |
| Mgmt | Jump hosts, monitoring, CI/CD | High-trust |
| Corp | Employee workstations | Untrusted |

DMZ hosts must never initiate connections to Data zone without explicit firewall rules.

## Firewall Rules
Default-deny between zones. Allow only documented, business-required flows:
```
# Allow: DMZ web → App API on port 8443
# Allow: App → Data DB on port 5432 (specific src IPs only)
# Deny: DMZ → Data (all)
# Deny: Corp → Data (all, use App tier as proxy)
```

## Micro-Segmentation
In container/cloud environments use:
- **Kubernetes NetworkPolicy** to restrict pod-to-pod traffic
- **AWS Security Groups** with source-group references (not CIDR)
- **Calico/Cilium** for eBPF-enforced policies with identity-based rules

## Zero Trust Principles
1. No implicit trust based on network location.
2. Verify identity (mTLS) and device posture before every request.
3. Log all access decisions.
4. Re-verify on privilege escalation.

## Detection
Alert on: traffic crossing zone boundaries outside approved rules, new outbound connections from Data zone, lateral movement patterns (same-subnet scanning, SMB to multiple hosts).
