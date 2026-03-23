"""
AILS Data Preprocessor Module
Data cleaning, normalization, and feature engineering pipelines.
Created by Cherry Computer Ltd.
"""

import re
import string
import logging
from typing import List, Dict, Optional, Tuple, Union
import numpy as np


class AILSPreprocessor:
    """
    AILS Data Preprocessing Module.
    Handles text cleaning, normalization, tokenization, and feature extraction.

    Example:
        prep = AILSPreprocessor()
        clean_texts = prep.clean_text_batch(["Hello  World!", "  AI is great! "])
    """

    def __init__(self, language: str = "english"):
        self.language = language
        self.logger = logging.getLogger("AILS.Preprocessor")
        self._stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set:
        """Load NLTK stopwords (fallback to basic set if unavailable)."""
        try:
            import nltk
            from nltk.corpus import stopwords
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words(self.language))
        except Exception:
            return {
                "i", "me", "my", "myself", "we", "our", "you", "your",
                "he", "she", "it", "they", "what", "which", "who", "is",
                "are", "was", "were", "be", "been", "being", "have", "has",
                "had", "do", "does", "did", "will", "would", "could",
                "should", "may", "might", "must", "shall", "can", "not",
                "the", "a", "an", "and", "or", "but", "in", "on", "at",
                "to", "for", "of", "with", "by", "from", "as", "into",
                "through", "during", "before", "after", "about", "above",
            }

    def clean_text(self, text: str, remove_stopwords: bool = True,
                   lowercase: bool = True, remove_numbers: bool = False) -> str:
        """
        Clean a single text string.

        Args:
            text: Input text.
            remove_stopwords: Whether to remove stopwords.
            lowercase: Convert to lowercase.
            remove_numbers: Remove numeric characters.

        Returns:
            Cleaned text string.
        """
        if not isinstance(text, str):
            return ""
        if lowercase:
            text = text.lower()
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove special characters and punctuation
        text = re.sub(r"[^\w\s]", " ", text)
        if remove_numbers:
            text = re.sub(r"\d+", " ", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if remove_stopwords:
            words = text.split()
            text = " ".join(w for w in words if w not in self._stopwords)
        return text

    def clean_text_batch(self, texts: List[str],
                          **kwargs) -> List[str]:
        """Clean a list of text strings."""
        cleaned = [self.clean_text(t, **kwargs) for t in texts]
        self.logger.info(f"✅ Cleaned {len(cleaned)} texts.")
        return cleaned

    def normalize_numerical(self, data: np.ndarray,
                             method: str = "minmax") -> np.ndarray:
        """
        Normalize numerical features.

        Args:
            data: Input numpy array.
            method: 'minmax' (0-1 scaling) or 'zscore' (standardization).

        Returns:
            Normalized array.
        """
        if method == "minmax":
            min_val = data.min(axis=0)
            max_val = data.max(axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1  # Avoid division by zero
            return (data - min_val) / range_val
        elif method == "zscore":
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            std[std == 0] = 1  # Avoid division by zero
            return (data - mean) / std
        else:
            raise ValueError(f"Unknown normalization method: {method}. "
                             "Use 'minmax' or 'zscore'.")

    def handle_missing_values(self, data: np.ndarray,
                               strategy: str = "mean") -> np.ndarray:
        """
        Handle missing (NaN) values in a numerical array.

        Args:
            data: Input array with potential NaN values.
            strategy: 'mean', 'median', or 'zero'.

        Returns:
            Array with NaN values replaced.
        """
        result = data.copy().astype(float)
        for col in range(result.shape[1] if result.ndim > 1 else 1):
            col_data = result[:, col] if result.ndim > 1 else result
            nan_mask = np.isnan(col_data)
            if nan_mask.any():
                if strategy == "mean":
                    fill_value = np.nanmean(col_data)
                elif strategy == "median":
                    fill_value = np.nanmedian(col_data)
                else:
                    fill_value = 0
                col_data[nan_mask] = fill_value
        return result

    def tokenize(self, text: str) -> List[str]:
        """Basic whitespace tokenizer."""
        return text.split()

    def remove_duplicates(self, texts: List[str]) -> List[str]:
        """Remove duplicate texts while preserving order."""
        seen = set()
        result = []
        for t in texts:
            if t not in seen:
                seen.add(t)
                result.append(t)
        original = len(texts)
        self.logger.info(
            f"Removed {original - len(result)} duplicates "
            f"({len(result)} unique)."
        )
        return result

    def encode_labels(self, labels: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Encode string labels to integers.

        Returns:
            Tuple of (encoded array, label-to-index mapping).
        """
        unique_labels = sorted(set(labels))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        encoded = np.array([label_map[l] for l in labels])
        return encoded, label_map

    def train_test_split(self, X: np.ndarray, y: np.ndarray,
                          test_size: float = 0.2,
                          random_state: int = 42) -> Tuple:
        """
        Split data into training and test sets.

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state)
