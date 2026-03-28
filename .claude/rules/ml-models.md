---
globs: src/models/**/*.py
---

When working on ML model files:
- All models must implement fit(), predict(), and explain() methods
- Use scikit-learn Pipeline for preprocessing chains
- Save models via model_registry.py with version tags (joblib)
- Every model change requires updated evaluation metrics
- Use SHAP TreeExplainer for tree-based models (RF, XGBoost)
- Hyperparameters must come from configs/model_config.yaml
- Include cross-validation (StratifiedKFold, 5 folds) in fit()