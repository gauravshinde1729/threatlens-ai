---
description: Check and fix linting and formatting issues
context: fork
allowed-tools: Bash, Read, Write
---

Check code quality for both src/ and tests/ in 3 steps:

Step 1 — LINT CHECK
Run: ruff check src/ tests/
Show me the results. If there are errors, list them and 
ask for my approval before fixing.

Step 2 — LINT FIX (only after my approval)
Run: ruff check --fix src/ tests/
Show what was fixed. If any errors remain that --fix can't 
handle, show them and stop.

Step 3 — FORMAT CHECK AND FIX
Run: ruff format --check src/ tests/
If any files need formatting, show which files, then run:
ruff format src/ tests/
Show what was reformatted.

At the end, run ruff check src/ tests/ one final time to 
confirm everything is clean. Report: total issues found, 
auto-fixed, remaining.