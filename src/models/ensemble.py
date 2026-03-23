"""
AILS Ensemble Methods Module
Bagging, Boosting, Stacking, and Voting ensemble strategies.
Artificial Intelligence Learning System — Created by Cherry Computer Ltd.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional


class AILSEnsemble:
    """
    AILS Ensemble Methods — combining multiple models for improved robustness.

    Supports:
    - Voting (hard/soft) for classification
    - Simple averaging for regression
    - Stacking with a meta-learner
    - Bagging (train multiple models on bootstrap samples)

    Example:
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC

        base_models = [LogisticRegression(), DecisionTreeClassifier(), SVC(probability=True)]
        ensemble = AILSEnsemble(base_models=base_models, method="soft_vote")
        ensemble.fit(X_train, y_train)
        preds = ensemble.predict(X_test)
    """

    METHODS = {"hard_vote", "soft_vote", "average", "stacking", "bagging"}

    def __init__(self, base_models: List,
                 method: str = "soft_vote",
                 meta_learner=None,
                 n_bags: int = 10):
        """
        Args:
            base_models: List of scikit-learn compatible estimators.
            method: Ensemble strategy: 'hard_vote', 'soft_vote',
                    'average', 'stacking', 'bagging'.
            meta_learner: For 'stacking' — the meta-model (default: LogisticRegression).
            n_bags: For 'bagging' — number of bootstrap models.
        """
        if method not in self.METHODS:
            raise ValueError(
                f"method must be one of {self.METHODS}, got '{method}'"
            )
        self.base_models = base_models
        self.method = method
        self.n_bags = n_bags
        self.logger = logging.getLogger("AILS.Models.Ensemble")

        if method == "stacking":
            if meta_learner is None:
                from sklearn.linear_model import LogisticRegression
                meta_learner = LogisticRegression(max_iter=1000)
            self.meta_learner = meta_learner
        else:
            self.meta_learner = None

        self._bag_models: List = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AILSEnsemble":
        """
        Fit all base models (and meta-learner if stacking).

        Args:
            X: Training feature matrix.
            y: Training labels.

        Returns:
            self
        """
        if self.method == "bagging":
            self._fit_bagging(X, y)
        elif self.method == "stacking":
            self._fit_stacking(X, y)
        else:
            for i, model in enumerate(self.base_models):
                model.fit(X, y)
                self.logger.info(
                    f"  ✅ Fitted model {i+1}/{len(self.base_models)}: "
                    f"{model.__class__.__name__}"
                )
        self.logger.info(
            f"✅ Ensemble ({self.method}) training complete."
        )
        return self

    def _fit_bagging(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train n_bags models on bootstrap samples."""
        import copy
        self._bag_models = []
        n = len(X)
        for i in range(self.n_bags):
            idx = np.random.choice(n, n, replace=True)
            X_bag, y_bag = X[idx], y[idx]
            model = copy.deepcopy(self.base_models[0])
            model.fit(X_bag, y_bag)
            self._bag_models.append(model)
            self.logger.info(f"  Bag {i+1}/{self.n_bags} fitted.")

    def _fit_stacking(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train base models and use their predictions to train meta-learner."""
        from sklearn.model_selection import cross_val_predict
        meta_features = []
        for model in self.base_models:
            preds = cross_val_predict(
                model, X, y, cv=5, method="predict_proba"
            )[:, 1]
            meta_features.append(preds)
            model.fit(X, y)
        meta_X = np.column_stack(meta_features)
        self.meta_learner.fit(meta_X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions."""
        if self.method == "hard_vote":
            votes = np.array([m.predict(X) for m in self.base_models])
            from scipy import stats
            return stats.mode(votes, axis=0)[0].flatten()

        elif self.method == "soft_vote":
            probs = np.mean(
                [m.predict_proba(X) for m in self.base_models], axis=0
            )
            return np.argmax(probs, axis=1)

        elif self.method == "average":
            preds = np.array([m.predict(X) for m in self.base_models])
            return np.mean(preds, axis=0)

        elif self.method == "stacking":
            meta_X = np.column_stack([
                m.predict_proba(X)[:, 1] for m in self.base_models
            ])
            return self.meta_learner.predict(meta_X)

        elif self.method == "bagging":
            votes = np.array([m.predict(X) for m in self._bag_models])
            from scipy import stats
            return stats.mode(votes, axis=0)[0].flatten()

        raise ValueError(f"Unknown method: {self.method}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return averaged probability estimates (soft methods only)."""
        if self.method in {"soft_vote", "stacking", "bagging"}:
            models = self._bag_models if self.method == "bagging" else self.base_models
            return np.mean(
                [m.predict_proba(X) for m in models], axis=0
            )
        raise NotImplementedError(
            f"predict_proba not supported for method='{self.method}'"
        )
