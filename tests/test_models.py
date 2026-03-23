"""
AILS Unit Tests — Neural Network Module
Tests for AILSNeuralNetwork build, compile, train, evaluate, predict.
Created by Cherry Computer Ltd.
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestAILSNeuralNetwork:
    """Unit tests for AILSNeuralNetwork."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.models.neural_network import AILSNeuralNetwork
        self.nn = AILSNeuralNetwork(
            input_dim=20,
            hidden_units=[64, 32],
            output_dim=1,
            task="binary_classification"
        )

    def test_build_creates_model(self):
        self.nn.build()
        assert self.nn.model is not None

    def test_compile_model(self):
        self.nn.compile_model()
        assert self.nn.model is not None

    def test_predict_output_shape(self):
        self.nn.compile_model()
        X = np.random.rand(10, 20).astype(np.float32)
        preds = self.nn.predict(X)
        assert preds.shape == (10, 1), f"Expected (10,1), got {preds.shape}"

    def test_predict_classes_binary(self):
        self.nn.compile_model()
        X = np.random.rand(5, 20).astype(np.float32)
        classes = self.nn.predict_classes(X)
        assert set(classes).issubset({0, 1}), "Binary classes must be 0 or 1"

    def test_train_returns_history(self):
        self.nn.compile_model()
        X = np.random.rand(100, 20).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.float32)
        history = self.nn.train(X, y, epochs=2, validation_split=0.2)
        assert "accuracy" in history.history

    def test_evaluate_returns_metrics_dict(self):
        self.nn.compile_model()
        X = np.random.rand(50, 20).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.float32)
        self.nn.train(X, y, epochs=2, validation_split=0.0)
        metrics = self.nn.evaluate(X, y)
        assert "accuracy" in metrics
        assert "loss" in metrics

    def test_multiclass_model(self):
        from src.models.neural_network import AILSNeuralNetwork
        nn = AILSNeuralNetwork(
            input_dim=20,
            hidden_units=[64],
            output_dim=4,
            task="multiclass_classification"
        )
        nn.compile_model()
        X = np.random.rand(10, 20).astype(np.float32)
        preds = nn.predict(X)
        assert preds.shape == (10, 4)

    def test_regression_model(self):
        from src.models.neural_network import AILSNeuralNetwork
        nn = AILSNeuralNetwork(
            input_dim=10,
            hidden_units=[32],
            output_dim=1,
            task="regression"
        )
        nn.compile_model()
        X = np.random.rand(8, 10).astype(np.float32)
        preds = nn.predict(X)
        assert preds.shape == (8, 1)
