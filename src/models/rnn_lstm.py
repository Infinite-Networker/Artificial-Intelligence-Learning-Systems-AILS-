"""
AILS LSTM / GRU / RNN Models
Sequential and time-series deep learning architectures.
Artificial Intelligence Learning System — Created by Cherry Computer Ltd.
"""

import logging
import numpy as np
from typing import List, Optional, Tuple


class AILSLSTMModel:
    """
    AILS Recurrent Neural Network Model.
    Supports LSTM, GRU, and SimpleRNN architectures with
    optional bidirectional layers.

    Example:
        model = AILSLSTMModel(vocab_size=10000, embedding_dim=128,
                              lstm_units=64, output_dim=1, model_type="lstm")
        model.train(X_train, y_train, epochs=10)
    """

    MODEL_TYPES = {"lstm", "gru", "rnn"}

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 lstm_units: int = 64, output_dim: int = 1,
                 model_type: str = "lstm", bidirectional: bool = True,
                 dropout_rate: float = 0.3):
        """
        Args:
            vocab_size: Size of vocabulary (embedding input dimension).
            embedding_dim: Dimensionality of token embeddings.
            lstm_units: Number of units in recurrent layer.
            output_dim: 1 for binary, N for multi-class classification.
            model_type: 'lstm', 'gru', or 'rnn'.
            bidirectional: Wrap recurrent layers with Bidirectional.
            dropout_rate: Dropout rate for regularization.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.output_dim = output_dim
        self.model_type = model_type.lower()
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        self.logger = logging.getLogger(f"AILS.Models.{model_type.upper()}")

        if self.model_type not in self.MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {self.MODEL_TYPES}, "
                f"got '{model_type}'"
            )
        self._build()

    def _get_recurrent_layer(self, units: int,
                              return_sequences: bool = False):
        """Return the correct recurrent layer class."""
        import tensorflow as tf
        cell_map = {
            "lstm": tf.keras.layers.LSTM,
            "gru":  tf.keras.layers.GRU,
            "rnn":  tf.keras.layers.SimpleRNN,
        }
        layer = cell_map[self.model_type](
            units,
            return_sequences=return_sequences,
            dropout=self.dropout_rate,
            recurrent_dropout=0.1,
        )
        if self.bidirectional:
            return tf.keras.layers.Bidirectional(layer)
        return layer

    def _build(self):
        """Build the recurrent model architecture."""
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                self.vocab_size, self.embedding_dim, mask_zero=True
            ),
            self._get_recurrent_layer(self.lstm_units, return_sequences=True),
            tf.keras.layers.Dropout(self.dropout_rate),
            self._get_recurrent_layer(self.lstm_units // 2, return_sequences=False),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(
                self.output_dim,
                activation="sigmoid" if self.output_dim == 1 else "softmax"
            ),
        ])
        loss = (
            "binary_crossentropy" if self.output_dim == 1
            else "sparse_categorical_crossentropy"
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=["accuracy"]
        )
        self.model = model
        self.logger.info(
            f"✅ {self.model_type.upper()} model built "
            f"({'Bidirectional' if self.bidirectional else 'Unidirectional'})"
        )
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
               epochs: int = 20, batch_size: int = 32,
               validation_split: float = 0.2):
        """Train the recurrent model."""
        import tensorflow as tf

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            ),
        ]
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test: np.ndarray,
                  y_test: np.ndarray) -> dict:
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return dict(zip(self.model.metrics_names, results))

    def save(self, path: str) -> None:
        self.model.save(path)
        self.logger.info(f"💾 Model saved to '{path}'")
