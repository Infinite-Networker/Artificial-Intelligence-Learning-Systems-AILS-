"""
AILS Unit Tests — Ethics & NLP Modules
Tests for bias detection, fairness metrics, privacy, and sentiment analysis.
Created by Cherry Computer Ltd.
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Bias Detector Tests ───────────────────────────────────────────────────────

class TestAILSBiasDetector:
    """Unit tests for AILSBiasDetector."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.ethics.bias_detector import AILSBiasDetector
        self.detector = AILSBiasDetector(
            disparity_threshold=0.1,
            impact_threshold=0.8
        )

    def test_demographic_parity_fair(self):
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        result = self.detector.demographic_parity(y_pred, sensitive)
        assert "disparity" in result
        assert "passed" in result
        assert isinstance(result["passed"], bool)

    def test_demographic_parity_biased(self):
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        result = self.detector.demographic_parity(y_pred, sensitive)
        assert result["passed"] is False

    def test_equalized_odds_returns_groups(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
        sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        result = self.detector.equalized_odds(y_true, y_pred, sensitive)
        assert "tpr_gap" in result
        assert "groups" in result

    def test_disparate_impact_ratio_range(self):
        y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 1])
        sensitive = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        ratio_result = self.detector.disparate_impact(y_pred, sensitive, privileged_group=1)
        assert 0.0 <= ratio_result["ratio"] <= 2.0

    def test_full_fairness_report_structure(self):
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        sensitive = np.random.randint(0, 2, 100)
        report = self.detector.generate_fairness_report(y_true, y_pred, sensitive)
        for key in ["representation", "demographic_parity",
                    "equalized_odds", "disparate_impact", "overall_verdict"]:
            assert key in report

    def test_representation_analysis(self):
        sensitive = np.array([0, 0, 1, 1, 1, 1])
        result = self.detector.representation_analysis(sensitive)
        assert result["total_samples"] == 6
        assert "group_0" in result["groups"]
        assert result["groups"]["group_0"]["percentage"] == pytest.approx(33.33, abs=0.01)


# ── Privacy Preserver Tests ───────────────────────────────────────────────────

class TestPrivacyPreserver:
    """Unit tests for PrivacyPreserver."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.ethics.privacy import PrivacyPreserver
        self.pp = PrivacyPreserver()

    def test_differential_privacy_laplace_shape(self):
        from src.ethics.privacy import PrivacyPreserver
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        noisy = PrivacyPreserver.add_differential_privacy(data, epsilon=1.0)
        assert noisy.shape == data.shape

    def test_differential_privacy_modifies_data(self):
        from src.ethics.privacy import PrivacyPreserver
        data = np.ones(50)
        noisy = PrivacyPreserver.add_differential_privacy(data, epsilon=0.1)
        assert not np.allclose(data, noisy), "Noise should change the data"

    def test_gaussian_mechanism(self):
        from src.ethics.privacy import PrivacyPreserver
        data = np.ones(20)
        noisy = PrivacyPreserver.add_differential_privacy(
            data, epsilon=1.0, mechanism="gaussian"
        )
        assert noisy.shape == data.shape

    def test_data_minimization_keeps_essential(self):
        from src.ethics.privacy import PrivacyPreserver
        record = {"name": "Alice", "age": 30, "email": "a@b.com", "score": 99}
        minimized = PrivacyPreserver.data_minimization(record, ["age", "score"])
        assert set(minimized.keys()) == {"age", "score"}

    def test_anonymize_returns_same_length(self):
        from src.ethics.privacy import PrivacyPreserver
        ids = ["alice@example.com", "bob@example.com"]
        anon = PrivacyPreserver.anonymize(ids)
        assert len(anon) == len(ids)
        assert all(a.startswith("ANON_") for a in anon)

    def test_pseudonymize_reversible(self):
        from src.ethics.privacy import PrivacyPreserver
        ids = ["Alice", "Bob"]
        pseudos, mapping = PrivacyPreserver.pseudonymize(ids)
        assert mapping["Alice"] == pseudos[0]
        assert all(p.startswith("USER_") for p in pseudos)

    def test_privacy_budget_within_limit(self):
        result = self.pp.check_privacy_budget([0.2, 0.3, 0.1], total_budget=1.0)
        assert result["within_budget"] is True
        assert result["budget_used"] == pytest.approx(0.6, abs=1e-9)

    def test_privacy_budget_exceeded(self):
        result = self.pp.check_privacy_budget([0.5, 0.6], total_budget=1.0)
        assert result["within_budget"] is False


# ── Sentiment Analyzer Tests ──────────────────────────────────────────────────

class TestSentimentAnalyzer:
    """Unit tests for SentimentAnalyzer."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.nlp.sentiment import SentimentAnalyzer
        self.analyzer = SentimentAnalyzer()

    def test_positive_sentiment(self):
        results = self.analyzer.analyze(["This product is amazing and excellent!"])
        assert results[0] == "positive"

    def test_negative_sentiment(self):
        results = self.analyzer.analyze(["This is terrible and horrible waste."])
        assert results[0] == "negative"

    def test_neutral_sentiment(self):
        results = self.analyzer.analyze(["I received the item today."])
        assert results[0] == "neutral"

    def test_batch_analysis_length(self):
        texts = ["Great!", "Awful!", "OK I guess."]
        results = self.analyzer.analyze(texts)
        assert len(results) == 3

    def test_analyze_with_scores_structure(self):
        texts = ["Absolutely fantastic product!"]
        results = self.analyzer.analyze_with_scores(texts)
        assert "sentiment" in results[0]
        assert "positive_score" in results[0]
        assert "confidence" in results[0]

    def test_preprocess_returns_string(self):
        result = self.analyzer.preprocess("Hello World! 123")
        assert isinstance(result, str)

    def test_fit_transform_shape(self):
        texts = ["Good product", "Bad product", "Great service", "Poor quality"] * 10
        X = self.analyzer.fit_transform(texts)
        assert X.ndim == 2
        assert X.shape[0] == len(texts)


# ── Preprocessor Tests ────────────────────────────────────────────────────────

class TestAILSPreprocessor:
    """Unit tests for AILSPreprocessor."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.data.preprocessor import AILSPreprocessor
        self.prep = AILSPreprocessor()

    def test_clean_text_lowercase(self):
        result = self.prep.clean_text("HELLO WORLD", remove_stopwords=False)
        assert result == result.lower()

    def test_clean_text_removes_urls(self):
        result = self.prep.clean_text("Visit https://example.com for more info.")
        assert "http" not in result

    def test_clean_text_removes_html(self):
        result = self.prep.clean_text("<b>Bold text</b>", remove_stopwords=False)
        assert "<" not in result and ">" not in result

    def test_clean_text_batch_length(self):
        texts = ["Hello", "World", "AI is great"]
        result = self.prep.clean_text_batch(texts)
        assert len(result) == len(texts)

    def test_normalize_minmax(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        norm = self.prep.normalize_numerical(data, method="minmax")
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

    def test_normalize_zscore(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        norm = self.prep.normalize_numerical(data, method="zscore")
        assert abs(norm.mean()) < 1e-10

    def test_remove_duplicates(self):
        texts = ["hello", "world", "hello", "AI"]
        result = self.prep.remove_duplicates(texts)
        assert len(result) == 3
        assert result.count("hello") == 1

    def test_encode_labels(self):
        labels = ["positive", "negative", "neutral", "positive"]
        encoded, mapping = self.prep.encode_labels(labels)
        assert len(encoded) == 4
        assert len(mapping) == 3
