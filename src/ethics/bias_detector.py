"""
AILS Ethics Module — Bias Detection & Fairness Analysis
Implements demographic parity, equalized odds, and disparate impact metrics.
Created by Cherry Computer Ltd.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple


class AILSBiasDetector:
    """
    AILS Bias Detection and Fairness Auditing Module.

    Implements industry-standard fairness metrics to ensure responsible AI:
    - Demographic Parity
    - Equalized Odds
    - Disparate Impact (80% Rule)
    - Individual Fairness
    - Data Representation Analysis

    Example:
        detector = AILSBiasDetector()
        report = detector.generate_fairness_report(
            y_true, y_pred, sensitive_attr, privileged_group=1
        )
    """

    def __init__(self, disparity_threshold: float = 0.1,
                 impact_threshold: float = 0.8):
        """
        Args:
            disparity_threshold: Max acceptable demographic disparity (default 0.1 = 10%).
            impact_threshold: Min acceptable disparate impact ratio (default 0.8 = 80% rule).
        """
        self.disparity_threshold = disparity_threshold
        self.impact_threshold = impact_threshold
        self.logger = logging.getLogger("AILS.Ethics.BiasDetector")

    def demographic_parity(self, y_pred: np.ndarray,
                            sensitive_attr: np.ndarray) -> Dict:
        """
        Demographic Parity: Equal positive prediction rates across groups.
        A disparity > threshold indicates potential bias.

        Returns:
            Dict with per-group positive rates, disparity score, and bias flag.
        """
        groups = np.unique(sensitive_attr)
        parity = {}
        for g in groups:
            mask = sensitive_attr == g
            parity[f"group_{g}"] = {
                "positive_rate": float(np.mean(y_pred[mask])),
                "count": int(np.sum(mask))
            }
        rates = [v["positive_rate"] for v in parity.values()]
        disparity = max(rates) - min(rates)
        result = {
            "groups": parity,
            "disparity": round(disparity, 4),
            "threshold": self.disparity_threshold,
            "passed": disparity <= self.disparity_threshold,
            "verdict": "✅ FAIR" if disparity <= self.disparity_threshold
                       else "⚠️ POTENTIAL BIAS DETECTED",
        }
        self.logger.info(
            f"Demographic Parity — disparity: {disparity:.4f} — "
            f"{result['verdict']}"
        )
        return result

    def equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray,
                        sensitive_attr: np.ndarray) -> Dict:
        """
        Equalized Odds: Equal TPR and FPR across groups.

        Returns:
            Dict with TPR/FPR per group and equality verdict.
        """
        from sklearn.metrics import confusion_matrix

        groups = np.unique(sensitive_attr)
        odds = {}
        for g in groups:
            mask = sensitive_attr == g
            yt, yp = y_true[mask], y_pred[mask]
            try:
                tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            except ValueError:
                tpr, fpr = 0.0, 0.0
            odds[f"group_{g}"] = {
                "TPR": round(float(tpr), 4),
                "FPR": round(float(fpr), 4),
                "count": int(np.sum(mask))
            }
        tpr_values = [v["TPR"] for v in odds.values()]
        fpr_values = [v["FPR"] for v in odds.values()]
        tpr_gap = max(tpr_values) - min(tpr_values)
        fpr_gap = max(fpr_values) - min(fpr_values)
        passed = tpr_gap <= 0.1 and fpr_gap <= 0.1
        result = {
            "groups": odds,
            "tpr_gap": round(tpr_gap, 4),
            "fpr_gap": round(fpr_gap, 4),
            "passed": passed,
            "verdict": "✅ FAIR" if passed else "⚠️ UNEQUAL ODDS",
        }
        self.logger.info(
            f"Equalized Odds — TPR gap: {tpr_gap:.4f}, FPR gap: {fpr_gap:.4f} "
            f"— {result['verdict']}"
        )
        return result

    def disparate_impact(self, y_pred: np.ndarray,
                          sensitive_attr: np.ndarray,
                          privileged_group: int = 1) -> Dict:
        """
        Disparate Impact (80% Rule):
        P(Ŷ=1 | unprivileged) / P(Ŷ=1 | privileged) ≥ 0.8

        Returns:
            Dict with ratio and 80% rule verdict.
        """
        priv_mask = sensitive_attr == privileged_group
        unpriv_mask = ~priv_mask
        priv_rate = np.mean(y_pred[priv_mask]) if priv_mask.any() else 0
        unpriv_rate = np.mean(y_pred[unpriv_mask]) if unpriv_mask.any() else 0
        ratio = float(unpriv_rate / priv_rate) if priv_rate > 0 else 0.0
        passed = ratio >= self.impact_threshold
        result = {
            "privileged_rate": round(float(priv_rate), 4),
            "unprivileged_rate": round(float(unpriv_rate), 4),
            "ratio": round(ratio, 4),
            "threshold": self.impact_threshold,
            "passed": passed,
            "verdict": "✅ FAIR (≥ 80% rule)" if passed
                       else "⚠️ DISCRIMINATORY (< 80% rule)",
        }
        self.logger.info(
            f"Disparate Impact Ratio: {ratio:.4f} — {result['verdict']}"
        )
        return result

    def representation_analysis(self, sensitive_attr: np.ndarray) -> Dict:
        """
        Analyze dataset representation across sensitive groups.

        Returns:
            Dict with group counts and percentages.
        """
        total = len(sensitive_attr)
        groups = np.unique(sensitive_attr)
        rep = {}
        for g in groups:
            count = int(np.sum(sensitive_attr == g))
            rep[f"group_{g}"] = {
                "count": count,
                "percentage": round(count / total * 100, 2)
            }
        self.logger.info(f"Representation: {rep}")
        return {"total_samples": total, "groups": rep}

    def generate_fairness_report(self, y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   sensitive_attr: np.ndarray,
                                   privileged_group: int = 1) -> Dict:
        """
        Generate a comprehensive bias and fairness audit report.

        Args:
            y_true: Ground truth labels.
            y_pred: Model predictions (binary).
            sensitive_attr: Sensitive attribute array (e.g., gender, race).
            privileged_group: Value representing the privileged group.

        Returns:
            Full fairness report dict.
        """
        self.logger.info("=" * 60)
        self.logger.info("AILS FAIRNESS AUDIT REPORT — Cherry Computer Ltd.")
        self.logger.info("=" * 60)

        report = {
            "representation": self.representation_analysis(sensitive_attr),
            "demographic_parity": self.demographic_parity(y_pred, sensitive_attr),
            "equalized_odds": self.equalized_odds(y_true, y_pred, sensitive_attr),
            "disparate_impact": self.disparate_impact(
                y_pred, sensitive_attr, privileged_group
            ),
        }

        # Overall verdict
        all_passed = all([
            report["demographic_parity"]["passed"],
            report["equalized_odds"]["passed"],
            report["disparate_impact"]["passed"],
        ])
        report["overall_verdict"] = "✅ MODEL IS FAIR" if all_passed \
            else "⚠️ BIAS DETECTED — REVIEW REQUIRED"
        self.logger.info(f"OVERALL: {report['overall_verdict']}")
        return report
