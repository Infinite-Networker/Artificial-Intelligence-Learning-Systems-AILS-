"""
AILS NLP Sentiment Analysis Module
Tokenization, stemming, TF-IDF vectorization, and sentiment classification.
Created by Cherry Computer Ltd.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional


class SentimentAnalyzer:
    """
    AILS Sentiment Analysis Engine.
    Combines rule-based and ML-based sentiment classification.

    Example:
        analyzer = SentimentAnalyzer()
        results = analyzer.analyze(["Great product!", "Terrible service."])
        # Returns: ['positive', 'negative']
    """

    POSITIVE_WORDS = frozenset({
        "good", "great", "excellent", "amazing", "love", "best",
        "wonderful", "fantastic", "superb", "happy", "brilliant",
        "outstanding", "perfect", "awesome", "beautiful", "positive",
        "pleasant", "delightful", "impressive", "exceptional", "splendid",
        "remarkable", "terrific", "marvelous", "enjoy", "liked", "loved",
        "recommend", "satisfied", "quality", "fast", "easy", "helpful",
    })

    NEGATIVE_WORDS = frozenset({
        "bad", "terrible", "awful", "poor", "hate", "worst", "horrible",
        "disappointing", "broken", "sad", "dreadful", "disgusting",
        "unacceptable", "useless", "failure", "defective", "misleading",
        "fraud", "scam", "waste", "ugly", "cheap", "slow", "difficult",
        "confusing", "unreliable", "faulty", "inferior", "damaged", "toxic",
    })

    def __init__(self):
        self.logger = logging.getLogger("AILS.NLP.SentimentAnalyzer")
        self.vectorizer = None
        self.model = None

    def _init_vectorizer(self):
        """Lazy-load TF-IDF vectorizer."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    def preprocess(self, text: str) -> str:
        """
        Tokenize and stem text.

        Args:
            text: Raw input string.

        Returns:
            Preprocessed string.
        """
        try:
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.stem import PorterStemmer
            nltk.download("punkt", quiet=True)
            stemmer = PorterStemmer()
            tokens = word_tokenize(text.lower())
            return " ".join(stemmer.stem(t) for t in tokens if t.isalpha())
        except ImportError:
            # Fallback: basic whitespace tokenization
            return " ".join(
                w.lower().strip(".,!?;:'\"") for w in text.split()
                if w.isalpha()
            )

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and transform texts.

        Args:
            texts: List of text strings.

        Returns:
            Dense feature matrix.
        """
        if self.vectorizer is None:
            self._init_vectorizer()
        preprocessed = [self.preprocess(t) for t in texts]
        return self.vectorizer.fit_transform(preprocessed).toarray()

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using already-fitted vectorizer."""
        if self.vectorizer is None:
            raise RuntimeError("Call fit_transform first.")
        preprocessed = [self.preprocess(t) for t in texts]
        return self.vectorizer.transform(preprocessed).toarray()

    def analyze(self, texts: List[str]) -> List[str]:
        """
        Perform rule-based sentiment analysis.

        Args:
            texts: List of text strings.

        Returns:
            List of 'positive', 'negative', or 'neutral'.
        """
        results = []
        for text in texts:
            tokens = set(text.lower().split())
            pos = len(tokens & self.POSITIVE_WORDS)
            neg = len(tokens & self.NEGATIVE_WORDS)
            if pos > neg:
                results.append("positive")
            elif neg > pos:
                results.append("negative")
            else:
                results.append("neutral")
        return results

    def analyze_with_scores(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiments and return detailed score dictionaries.

        Returns:
            List of dicts with 'sentiment', 'positive_score', 'negative_score'.
        """
        results = []
        for text in texts:
            tokens = text.lower().split()
            token_set = set(tokens)
            pos = len(token_set & self.POSITIVE_WORDS)
            neg = len(token_set & self.NEGATIVE_WORDS)
            total = pos + neg if (pos + neg) > 0 else 1
            results.append({
                "text": text[:100],
                "sentiment": "positive" if pos > neg
                             else "negative" if neg > pos else "neutral",
                "positive_score": round(pos / total, 4),
                "negative_score": round(neg / total, 4),
                "confidence": round(abs(pos - neg) / total, 4),
            })
        return results

    def train_ml_model(self, texts: List[str],
                        labels: List[int]) -> None:
        """
        Train a TF-IDF + Neural Network sentiment model.

        Args:
            texts: Training texts.
            labels: Binary labels (1=positive, 0=negative).
        """
        import tensorflow as tf

        X = self.fit_transform(texts)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu",
                                  input_shape=(X.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy",
                           metrics=["accuracy"])
        y = np.array(labels)
        self.model.fit(X, y, epochs=15, batch_size=32,
                       validation_split=0.2, verbose=0)
        self.logger.info("✅ ML sentiment model trained.")

    def predict_ml(self, texts: List[str]) -> List[str]:
        """Predict sentiment using trained ML model."""
        if self.model is None:
            raise RuntimeError("Train a model first with train_ml_model().")
        X = self.transform(texts)
        probs = self.model.predict(X).flatten()
        return ["positive" if p >= 0.5 else "negative" for p in probs]
