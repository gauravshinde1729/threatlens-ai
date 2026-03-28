Run the test suite with coverage:
```bash
pytest tests/ --cov=src --cov-report=term-missing -v
```

After running:
1. Report pass/fail summary
2. If coverage < 80%, identify under-covered files
3. Suggest specific tests to add for uncovered lines