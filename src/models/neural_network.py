"""
AILS Core Neural Network Module
TensorFlow Sequential and Functional models for classification and regression.
Created by Cherry Computer Ltd.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple


class AILSNeuralNetwork:
    """
    AILS Core Neural Network — TensorFlow Sequential Model.
    Supports binary and multi-class classification, and regression.

    Example:
        nn = AILSNeuralNetwork(input_dim=100, hidden_units=[128, 64], output_dim=1)
        nn.compile_model()
        nn.train(X_train, y_train, epochs=30)
        metrics = nn.evaluate(X_test, y_test)
    """

    def __init__(self, input_dim: int, hidden_units: List[int],
                 output_dim: int = 1, dropout_rate: float = 0.3,
                 task: str = "binary_classification"):
        """
        Args:
            input_dim: Number of input features.
            hidden_units: List of units per hidden layer.
            output_dim: Number of output units (1 for binary, N for multi-class).
            dropout_rate: Dropout probability for regularization.
            task: 'binary_classification', 'multiclass_classification', 'regression'.
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.task = task
        self.model = None
        self.history = None
        self.logger = logging.getLogger("AILS.NeuralNetwork")

    def build(self):
        """Build the Sequential neural network architecture."""
        import tensorflow as tf

        layers = [
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.Dense(
                self.hidden_units[0], activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.dropout_rate),
        ]
        for units in self.hidden_units[1:]:
            layers += [
                tf.keras.layers.Dense(
                    units, activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.001)
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(self.dropout_rate),
            ]

        # Output layer
        if self.task == "binary_classification":
            layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
        elif self.task == "multiclass_classification":
            layers.append(tf.keras.layers.Dense(self.output_dim, activation="softmax"))
        else:  # regression
            layers.append(tf.keras.layers.Dense(self.output_dim, activation="linear"))

        self.model = tf.keras.Sequential(layers)
        return self.model

    def compile_model(self, optimizer: str = "adam",
                       learning_rate: float = 0.001) -> None:
        """Compile the model with optimizer, loss function, and metrics."""
        import tensorflow as tf
        if self.model is None:
            self.build()

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if self.task == "binary_classification":
            loss = "binary_crossentropy"
            metrics = [
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc")
            ]
        elif self.task == "multiclass_classification":
            loss = "sparse_categorical_crossentropy"
            metrics = ["accuracy"]
        else:
            loss = "mse"
            metrics = ["mae"]

        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
        self.logger.info("✅ AILS Neural Network compiled.")

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
               epochs: int = 50, batch_size: int = 32,
               validation_split: float = 0.2,
               callbacks: Optional[List] = None):
        """
        Train the neural network with early stopping and LR scheduling.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            epochs: Max training epochs.
            batch_size: Mini-batch size.
            validation_split: Fraction of training data for validation.
            callbacks: Optional custom Keras callbacks.

        Returns:
            Keras History object.
        """
        import tensorflow as tf
        if self.model is None:
            self.compile_model()

        default_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
        ]
        all_callbacks = (callbacks or []) + default_callbacks

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=all_callbacks,
            verbose=1
        )
        self.logger.info("✅ Training complete.")
        return self.history

    def evaluate(self, X_test: np.ndarray,
                  y_test: np.ndarray) -> Dict:
        """
        Evaluate the model on test data.

        Returns:
            Dictionary of metric names to values.
        """
        if self.model is None:
            raise RuntimeError("Model not compiled yet.")
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))
        self.logger.info(f"📊 Evaluation: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate raw predictions (probabilities or values)."""
        return self.model.predict(X, verbose=0)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Generate binary or multi-class predictions."""
        preds = self.predict(X)
        if self.task == "binary_classification":
            return (preds >= 0.5).astype(int).flatten()
        return np.argmax(preds, axis=1)

    def save(self, path: str) -> None:
        """Save the full model to disk."""
        self.model.save(path)
        self.logger.info(f"💾 Model saved to '{path}'")

    @classmethod
    def load(cls, path: str) -> "AILSNeuralNetwork":
        """Load a saved Keras model from disk."""
        import tensorflow as tf
        instance = cls.__new__(cls)
        instance.model = tf.keras.models.load_model(path)
        instance.logger = logging.getLogger("AILS.NeuralNetwork")
        return instance

    def summary(self) -> None:
        """Print the model architecture summary."""
        if self.model:
            self.model.summary()
        else:
            print("Model not yet built. Call compile_model() first.")
