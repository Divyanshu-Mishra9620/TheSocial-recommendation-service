"""Offline XGBoost training pipeline for the Recommendation Service.

OFFLINE ONLY. Nothing here is imported by the serving app (`app/`). Building or
training a model does NOT activate it: the heuristic prior (`score_features`)
remains the production scorer. Heavy ML deps (pandas/numpy/scikit-learn/xgboost)
are imported lazily inside functions so the pure helpers are testable without
them. Install the ML stack with: pip install -r requirements-train.txt
"""
