"""Prompt templates for the RAG-based remediation playbook generator."""

REMEDIATION_PROMPT_TEMPLATE = """You are a senior security engineer generating a remediation playbook for a vulnerability.

## CVE Details
{cve_details}

## ML Risk Assessment
{ml_prediction}

## Relevant Security Documentation
{retrieved_context}

Generate a structured remediation playbook with these sections:
1. Executive Summary (2-3 sentences)
2. Risk Assessment (severity, exploitability, business impact)
3. Immediate Actions (numbered steps, specific and actionable)
4. Detection Rules (log patterns, SIEM queries to detect exploitation)
5. Monitoring Recommendations (what to watch post-remediation)

Be specific and actionable. Reference the security documentation where applicable. Do not be generic."""
