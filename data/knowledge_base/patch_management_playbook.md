# Patch Management Playbook

## Overview
Unpatched vulnerabilities are the #1 initial access vector. A structured patch program reduces mean time to patch (MTTP) and ensures critical fixes reach production before exploitation.

## Prioritization Framework

### SLA by Severity
| CVSS Score | Classification | Patch SLA |
|------------|---------------|-----------|
| 9.0–10.0 | Critical | 24–72 hours |
| 7.0–8.9 | High | 7 days |
| 4.0–6.9 | Medium | 30 days |
| 0.1–3.9 | Low | 90 days |

Adjust SLA downward if: CVE is in CISA KEV list, public PoC exists, or asset is internet-facing.

## Testing Procedure
1. Deploy patch to isolated staging environment matching production.
2. Run automated regression suite and smoke tests.
3. Check vendor release notes for known incompatibilities.
4. Conduct manual verification for critical services (auth, payments).
5. Document test results with sign-off before production deployment.

## Rollback Plan
- Capture system snapshot/AMI before applying patch.
- Keep previous package version pinned in artifact repository.
- Define rollback trigger criteria: >5% error rate increase, service health check failures.
- Target rollback completion within 30 minutes.

## Emergency Patching (0-day / Active Exploitation)
1. Convene Incident Command within 1 hour of disclosure.
2. Assess exploitability and exposure (internet-facing? auth required?).
3. Apply compensating controls immediately: WAF rule, network block, feature flag.
4. Expedite patch through accelerated test window (4–8 hours).
5. Deploy with change freeze exception; notify stakeholders.

## Tooling
Automate patch scanning with Tenable, Qualys, or `apt-get --simulate`. Track MTTP per team in your vulnerability management dashboard.
