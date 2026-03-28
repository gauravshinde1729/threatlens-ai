---
description: Analyze a CVE through the full ThreatLens pipeline (read-only)
context: fork
allowed-tools: Read, Bash, Grep
argument-hint: CVE ID (e.g., CVE-2024-3400)
---

Analyze CVE $ARGUMENTS through the ThreatLens pipeline:

1. Look up the CVE details using src/data/nvd_client.py
2. Extract features using src/data/preprocessor.py
3. Run severity prediction using src/models/severity_predictor.py
4. Generate SHAP explanation for the prediction
5. Output a structured threat assessment

Do NOT modify any source files. This is a read-only analysis.