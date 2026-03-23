"""
AILS Model Trainer — Unified Training & Evaluation Pipeline
Handles cross-validation, hyperparameter logging, and model persistence.
Artificial Intelligence Learning System — Created by Cherry Computer Ltd.
"""

import numpy as np
import logging
import time
import os
from typing import Dict, Optional, List, Tuple


class AILSTrainer:
    """
    AILS Unified Model Trainer.

    Wraps any AILS model with a consistent fit/evaluate/save interface,
    plus cross-validation, learning-curve logging, and checkpoint support.

    Example:
        from src.models.neural_network import AILSNeuralNetwork

        nn = AILSNeuralNetwork(input_dim=50, hidden_units=[128, 64])
        trainer = AILSTrainer(model=nn, model_name="sentiment_nn")
        trainer.fit(X_train, y_train)
        report = trainer.evaluate(X_test, y_test)
        trainer.save("checkpoints/")
    """

    def __init__(self, model, model_name: str = "ails_model",
                 checkpoint_dir: str = "checkpoints"):
        """
        Args:
            model: Any AILS model instance with .train() / .evaluate() / .save().
            model_name: Identifier for checkpoint and log files.
            checkpoint_dir: Directory to save model checkpoints.
        """
        self.model = model
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.train_history: List[Dict] = []
        self.logger = logging.getLogger("AILS.Trainer")
        os.makedirs(checkpoint_dir, exist_ok=True)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
             epochs: int = 50, batch_size: int = 32,
             validation_split: float = 0.2) -> "AILSTrainer":
        """
        Train the underlying model and record metrics.

        Args:
            X_train: Training features.
            y_train: Training labels.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            validation_split: Fraction for validation.

        Returns:
            self (for chaining)
        """
        self.logger.info(
            f"🚀 Starting training — model: {self.model_name}, "
            f"samples: {len(X_train)}, epochs: {epochs}"
        )
        start = time.time()

        history = self.model.train(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
        )

        elapsed = round(time.time() - start, 2)
        self.logger.info(f"✅ Training complete in {elapsed}s")

        # Store run summary
        if hasattr(history, "history"):
            last_epoch = {
                k: round(float(v[-1]), 4)
                for k, v in history.history.items()
            }
            self.train_history.append({
                "model": self.model_name,
                "epochs_run": len(history.history.get("loss", [])),
                "elapsed_sec": elapsed,
                **last_epoch
            })
        return self

    def evaluate(self, X_test: np.ndarray,
                  y_test: np.ndarray) -> Dict:
        """
        Evaluate on test data and log results.

        Returns:
            Dict of metric names → values.
        """
        metrics = self.model.evaluate(X_test, y_test)
        self.logger.info(f"📊 Test metrics: {metrics}")
        return metrics

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                        k: int = 5) -> Dict:
        """
        K-fold cross-validation using sklearn.

        Args:
            X: Full feature matrix.
            y: Full labels array.
            k: Number of folds.

        Returns:
            Dict with mean and std of cross-validated accuracy.
        """
        from sklearn.model_selection import KFold
        from sklearn.base import clone

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = []
        self.logger.info(f"Starting {k}-fold cross-validation...")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            self.model.train(X_tr, y_tr, epochs=10, validation_split=0.0)
            metrics = self.model.evaluate(X_val, y_val)
            acc = metrics.get("accuracy", 0.0)
            scores.append(acc)
            self.logger.info(f"  Fold {fold+1}/{k} — accuracy: {acc:.4f}")

        result = {
            "k_folds": k,
            "scores": scores,
            "mean_accuracy": round(float(np.mean(scores)), 4),
            "std_accuracy": round(float(np.std(scores)), 4),
        }
        self.logger.info(
            f"✅ CV complete — mean acc: {result['mean_accuracy']:.4f} "
            f"± {result['std_accuracy']:.4f}"
        )
        return result

    def save(self, directory: Optional[str] = None) -> str:
        """
        Save model to disk.

        Returns:
            Path where model was saved.
        """
        save_dir = directory or self.checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{self.model_name}.h5")
        self.model.save(path)
        self.logger.info(f"💾 Model checkpoint saved: {path}")
        return path

    def get_training_summary(self) -> List[Dict]:
        """Return list of all recorded training run summaries."""
        return self.train_history
