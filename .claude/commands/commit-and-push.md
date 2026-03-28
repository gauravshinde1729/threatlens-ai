---
description: Lint, fix, commit, and push to main in one go
context: fork
allowed-tools: Bash, Read, Write
---

Ship the current changes to main:

1. Run ruff check on the entire src/ directory
2. If there are lint errors, fix them automatically with ruff check --fix src/
3. Run ruff format src/ to ensure consistent formatting
4. Run git add -A
5. Look at the staged diff and generate a conventional commit message 
   (feat:, fix:, docs:, test:, ci:) summarizing what changed
6. Show me the commit message and wait for approval
7. Run git commit with that message
8. Run git push origin main

If ruff --fix can't auto-fix something, show me the remaining 
issues and stop — don't commit broken code.
```