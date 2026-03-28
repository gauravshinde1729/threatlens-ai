# Incident Response Checklist

## Phase 1 — Detection & Triage (0–1 hour)
- [ ] Confirm alert is a true positive (not scanner noise or misconfiguration).
- [ ] Assign Incident Commander and Scribe.
- [ ] Open incident channel (Slack/Teams) and war room bridge.
- [ ] Classify severity: P1 (active breach/data loss), P2 (contained threat), P3 (potential risk).
- [ ] Notify legal and privacy team if PII may be involved (GDPR clock starts).
- [ ] Document timeline in incident log from this point forward.

## Phase 2 — Containment (1–4 hours)
- [ ] Isolate affected systems: revoke credentials, block IPs, quarantine hosts.
- [ ] Preserve evidence: snapshot disk images before remediation.
- [ ] Block attacker's C2 IPs/domains at perimeter firewall and DNS.
- [ ] Rotate all secrets/tokens that may have been exposed.
- [ ] Determine lateral movement scope — check authentication logs across all systems.

## Phase 3 — Eradication (4–24 hours)
- [ ] Identify and remove all attacker artifacts (web shells, cron jobs, new accounts).
- [ ] Patch or mitigate the initial access vulnerability.
- [ ] Rebuild compromised systems from clean image if persistence is suspected.
- [ ] Verify no secondary backdoors remain via integrity checks.

## Phase 4 — Recovery (24–72 hours)
- [ ] Restore services from clean backups after confirming eradication.
- [ ] Enhanced monitoring for 14 days post-recovery.
- [ ] Gradual traffic restoration with monitoring at each step.

## Phase 5 — Lessons Learned (1–2 weeks post)
- [ ] Post-mortem with timeline, root cause, contributing factors.
- [ ] Action items with owners and deadlines (no blame, system focus).
- [ ] Update playbooks, detection rules, and training based on findings.
- [ ] Share sanitized summary with stakeholders.
