---
description: Commit and push clean code to main
context: fork
allowed-tools: Bash, Read
---

Ship the current changes to main:

1. Run ruff check src/ — if errors exist, STOP and tell me 
   to run /lint first
2. Run git add -A
3. Look at the staged diff and generate a conventional commit 
   message (feat:, fix:, docs:, test:, ci:)
4. Show me the commit message and wait for approval
5. Run git commit with that message
6. Run git push origin main
```
