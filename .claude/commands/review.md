Review the current staged git changes against project conventions:

1. Type hints on all public functions
2. Docstrings on all public classes and methods
3. No hardcoded hyperparameters — must use configs/
4. SHAP explainability for any model changes
5. Tests for any new functionality
6. Pydantic schemas for any API changes
7. No real API calls in tests — must use mocks

Group issues by: critical, warning, suggestion.