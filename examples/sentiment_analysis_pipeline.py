"""
AILS Full Sentiment Analysis Pipeline
Demonstrates the complete AILS workflow:
Scrape → Store → Preprocess → Train → Evaluate → Predict

Created by Cherry Computer Ltd.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocessor import AILSPreprocessor
from src.nlp.sentiment import SentimentAnalyzer
from src.models.neural_network import AILSNeuralNetwork
from src.ethics.bias_detector import AILSBiasDetector
from src.utils.metrics import evaluate_model
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)

# ── Sample Dataset ────────────────────────────────────────────────────────────
SAMPLE_REVIEWS = [
    ("This product is absolutely amazing! Best purchase ever.", 1),
    ("Terrible quality. Broke after two days. Complete waste of money.", 0),
    ("Great value for money. Works exactly as described.", 1),
    ("Worst customer service I have ever experienced. Never buying again.", 0),
    ("Excellent build quality and very fast delivery. Highly recommend!", 1),
    ("Disappointed with the product. Not as advertised at all.", 0),
    ("Outstanding performance! Exceeded all my expectations.", 1),
    ("Faulty product. Returned immediately. Poor quality control.", 0),
    ("Wonderful experience from start to finish. Will buy again!", 1),
    ("Horrible. Doesn't work at all. Requested a refund.", 0),
    ("Superb product, very happy with my purchase.", 1),
    ("Cheap materials, broke within a week. Very disappointing.", 0),
    ("Love this! It's exactly what I needed. Fast shipping too.", 1),
    ("Not worth the price. Feels flimsy and poorly made.", 0),
    ("Fantastic product! My whole family loves it.", 1),
    ("Arrived damaged and support was unhelpful. Avoid.", 0),
    ("Very pleased with this purchase. Good quality at a fair price.", 1),
    ("Confusing instructions and poor build quality.", 0),
    ("Brilliant! Works perfectly and looks great.", 1),
    ("Scam product. Nothing like the pictures. Do not buy.", 0),
] * 50  # Scale up for training


def main():
    print("=" * 65)
    print("  AILS — Sentiment Analysis Pipeline")
    print("  Created by Cherry Computer Ltd.")
    print("=" * 65)

    texts = [r[0] for r in SAMPLE_REVIEWS]
    labels = np.array([r[1] for r in SAMPLE_REVIEWS])

    # ── Step 1: Preprocess ──────────────────────────────────────────────────
    print("\n[1/5] Preprocessing data...")
    preprocessor = AILSPreprocessor()
    clean_texts = preprocessor.clean_text_batch(texts, remove_stopwords=False)
    print(f"  ✅ Cleaned {len(clean_texts)} text samples.")

    # ── Step 2: Rule-Based Sentiment Analysis ──────────────────────────────
    print("\n[2/5] Running rule-based sentiment analysis (sample)...")
    analyzer = SentimentAnalyzer()
    sample_results = analyzer.analyze_with_scores(texts[:5])
    for r in sample_results:
        print(f"  ➜ [{r['sentiment'].upper():8}] {r['text'][:60]}...")

    # ── Step 3: Vectorize ──────────────────────────────────────────────────
    print("\n[3/5] Vectorizing with TF-IDF...")
    X = analyzer.fit_transform(clean_texts)
    print(f"  ✅ Feature matrix shape: {X.shape}")

    # ── Step 4: Train Neural Network ──────────────────────────────────────
    print("\n[4/5] Training AILS Neural Network...")
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, labels)

    nn = AILSNeuralNetwork(
        input_dim=X.shape[1],
        hidden_units=[256, 128, 64],
        output_dim=1,
        task="binary_classification"
    )
    nn.compile_model()
    nn.train(X_train, y_train, epochs=15, batch_size=32)

    # ── Step 5: Evaluate ──────────────────────────────────────────────────
    print("\n[5/5] Evaluating model performance...")
    metrics = nn.evaluate(X_test, y_test)
    print(f"\n  📊 Results:")
    for k, v in metrics.items():
        print(f"     {k}: {v:.4f}")

    # ── Bias Check ────────────────────────────────────────────────────────
    print("\n[BONUS] Running fairness audit...")
    y_pred = nn.predict_classes(X_test)
    sensitive_attr = np.random.randint(0, 2, size=len(y_test))  # Synthetic
    detector = AILSBiasDetector()
    report = detector.generate_fairness_report(y_test, y_pred, sensitive_attr)
    print(f"  Fairness Verdict: {report['overall_verdict']}")

    print("\n" + "=" * 65)
    print("  ✅ AILS Pipeline Complete!")
    print("  Created by Cherry Computer Ltd.")
    print("=" * 65)


if __name__ == "__main__":
    main()
