"""
AILS Computer Vision Module — CNN Image Classifier
Convolutional Neural Network for image classification and object recognition.
Artificial Intelligence Learning System — Created by Cherry Computer Ltd.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict


class AILSCNNModel:
    """
    AILS Convolutional Neural Network for Image Classification.

    Builds a customizable CNN with convolutional blocks, pooling,
    batch normalization, and fully-connected classification head.

    Example:
        cnn = AILSCNNModel(input_shape=(128, 128, 3), num_classes=10)
        cnn.compile_model()
        cnn.train(X_train, y_train, epochs=20)
        metrics = cnn.evaluate(X_test, y_test)
    """

    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 10, dropout_rate: float = 0.4):
        """
        Args:
            input_shape: (height, width, channels).
            num_classes: Number of output classes.
            dropout_rate: Dropout probability in the classifier head.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        self.logger = logging.getLogger("AILS.Vision.CNN")

    def build(self):
        """Build the CNN architecture."""
        import tensorflow as tf

        inputs = tf.keras.Input(shape=self.input_shape)

        # ── Block 1 ─────────────────────────────────
        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        # ── Block 2 ─────────────────────────────────
        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # ── Block 3 ─────────────────────────────────
        x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # ── Classifier Head ─────────────────────────
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)

        activation = "sigmoid" if self.num_classes == 2 else "softmax"
        out_units = 1 if self.num_classes == 2 else self.num_classes
        outputs = tf.keras.layers.Dense(out_units, activation=activation)(x)

        self.model = tf.keras.Model(inputs, outputs, name="AILS_CNN")
        self.logger.info("✅ CNN model built.")
        return self.model

    def compile_model(self, learning_rate: float = 0.001) -> None:
        """Compile the CNN with Adam optimizer and appropriate loss."""
        import tensorflow as tf
        if self.model is None:
            self.build()

        loss = (
            "binary_crossentropy" if self.num_classes == 2
            else "sparse_categorical_crossentropy"
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=loss,
            metrics=["accuracy"]
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
               epochs: int = 30, batch_size: int = 32,
               validation_split: float = 0.2,
               augment: bool = True):
        """
        Train the CNN with optional data augmentation.

        Args:
            X_train: Images array, shape (N, H, W, C), values in [0, 1].
            y_train: Labels array.
            epochs: Max training epochs.
            batch_size: Batch size.
            validation_split: Validation fraction.
            augment: Apply random flips/rotations.
        """
        import tensorflow as tf
        if self.model is None:
            self.compile_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
            ),
        ]

        if augment:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,
                horizontal_flip=True,
                zoom_range=0.1,
                width_shift_range=0.1,
                height_shift_range=0.1,
                validation_split=validation_split,
            )
            train_gen = datagen.flow(
                X_train, y_train, batch_size=batch_size, subset="training"
            )
            val_gen = datagen.flow(
                X_train, y_train, batch_size=batch_size, subset="validation"
            )
            self.history = self.model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1,
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
            )
        return self.history

    def evaluate(self, X_test: np.ndarray,
                  y_test: np.ndarray) -> Dict:
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return dict(zip(self.model.metrics_names, results))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0)

    def save(self, path: str) -> None:
        self.model.save(path)
        self.logger.info(f"💾 CNN saved to '{path}'")


class AILSImageProcessor:
    """
    AILS Image Preprocessing Utility.
    Handles loading, resizing, normalization, and augmentation.
    """

    @staticmethod
    def normalize(images: np.ndarray) -> np.ndarray:
        """Normalize pixel values to [0, 1]."""
        return images.astype(np.float32) / 255.0

    @staticmethod
    def resize(images: np.ndarray,
               target_size: Tuple[int, int]) -> np.ndarray:
        """Resize a batch of images to target_size (H, W)."""
        try:
            import cv2
            return np.array([
                cv2.resize(img, (target_size[1], target_size[0]))
                for img in images
            ])
        except ImportError:
            from PIL import Image
            return np.array([
                np.array(Image.fromarray(img).resize(
                    (target_size[1], target_size[0])
                ))
                for img in images
            ])

    @staticmethod
    def to_grayscale(images: np.ndarray) -> np.ndarray:
        """Convert RGB images to grayscale."""
        if images.ndim == 4 and images.shape[-1] == 3:
            return np.mean(images, axis=-1, keepdims=True)
        return images
