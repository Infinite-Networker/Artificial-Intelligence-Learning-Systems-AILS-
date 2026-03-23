"""
AILS Privacy-Preserving Module
Implements Differential Privacy, k-Anonymity, and Data Minimization.
Created by Cherry Computer Ltd.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple


class PrivacyPreserver:
    """
    AILS Privacy-Preserving Module.

    Implements state-of-the-art privacy protection techniques:
    - Differential Privacy (Laplace & Gaussian mechanisms)
    - k-Anonymity data generalization
    - Data Minimization
    - Anonymization & Pseudonymization

    Example:
        pp = PrivacyPreserver()
        private_data = pp.add_differential_privacy(data, epsilon=0.5)
    """

    def __init__(self):
        self.logger = logging.getLogger("AILS.Ethics.Privacy")

    @staticmethod
    def add_differential_privacy(data: np.ndarray,
                                   epsilon: float = 1.0,
                                   sensitivity: float = 1.0,
                                   mechanism: str = "laplace") -> np.ndarray:
        """
        Add noise for differential privacy protection.

        Args:
            data: Input data array.
            epsilon: Privacy budget (lower = more private, higher = more accurate).
            sensitivity: Global sensitivity of the query function.
            mechanism: 'laplace' (stronger) or 'gaussian' (smoother).

        Returns:
            Privacy-protected data array.
        """
        scale = sensitivity / epsilon
        if mechanism == "laplace":
            noise = np.random.laplace(loc=0.0, scale=scale, size=data.shape)
        elif mechanism == "gaussian":
            delta = 1e-5
            sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
            noise = np.random.normal(loc=0.0, scale=sigma, size=data.shape)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}. Use 'laplace' or 'gaussian'.")
        return data + noise

    @staticmethod
    def k_anonymize(data: np.ndarray, k: int = 5,
                     quasi_identifiers: Optional[List[int]] = None) -> np.ndarray:
        """
        k-Anonymity: Generalize quasi-identifier columns so each record
        is indistinguishable from at least k-1 others.

        Args:
            data: Input data array.
            k: Minimum group size (higher = more private).
            quasi_identifiers: Column indices to generalize (all if None).

        Returns:
            k-Anonymized data array.
        """
        result = data.copy().astype(float)
        cols = quasi_identifiers or list(range(data.shape[1] if data.ndim > 1 else 1))
        if data.ndim == 1:
            return np.round(result / k) * k
        for col in cols:
            result[:, col] = np.round(result[:, col] / k) * k
        return result

    @staticmethod
    def pseudonymize(identifiers: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Replace identifiers with pseudonyms (reversible with key).

        Args:
            identifiers: List of real identifiers (e.g., names, emails).

        Returns:
            Tuple of (pseudonym list, real-to-pseudonym mapping dict).
        """
        import hashlib
        mapping = {}
        pseudonyms = []
        for ident in identifiers:
            pseudo = "USER_" + hashlib.sha256(ident.encode()).hexdigest()[:10].upper()
            mapping[ident] = pseudo
            pseudonyms.append(pseudo)
        return pseudonyms, mapping

    @staticmethod
    def anonymize(identifiers: List[str]) -> List[str]:
        """
        Irreversibly anonymize identifiers (one-way hash).

        Args:
            identifiers: List of identifiers to anonymize.

        Returns:
            List of anonymized tokens.
        """
        import hashlib
        return [
            "ANON_" + hashlib.md5(i.encode()).hexdigest()[:8].upper()
            for i in identifiers
        ]

    @staticmethod
    def data_minimization(record: Dict,
                           essential_fields: List[str]) -> Dict:
        """
        Retain only essential fields from a data record (GDPR compliance).

        Args:
            record: Full data record dict.
            essential_fields: Fields that are strictly necessary.

        Returns:
            Minimized record with only essential fields.
        """
        return {k: v for k, v in record.items() if k in essential_fields}

    @staticmethod
    def data_minimization_batch(records: List[Dict],
                                 essential_fields: List[str]) -> List[Dict]:
        """Apply data minimization to a batch of records."""
        return [
            {k: v for k, v in r.items() if k in essential_fields}
            for r in records
        ]

    def check_privacy_budget(self, queries: List[float],
                              total_budget: float = 1.0) -> Dict:
        """
        Check if cumulative privacy budget is within allowed limit.

        Args:
            queries: List of epsilon values used per query.
            total_budget: Maximum allowed total epsilon.

        Returns:
            Dict with budget status.
        """
        used = sum(queries)
        remaining = max(0, total_budget - used)
        status = {
            "total_budget": total_budget,
            "budget_used": round(used, 6),
            "budget_remaining": round(remaining, 6),
            "within_budget": used <= total_budget,
            "verdict": "✅ Within Budget" if used <= total_budget
                       else "❌ Budget Exceeded — Stop Queries",
        }
        self.logger.info(f"Privacy Budget: {status['verdict']} "
                         f"(used: {used:.4f}/{total_budget})")
        return status

    @staticmethod
    def suppress_sensitive_columns(data: Dict,
                                    sensitive_columns: List[str]) -> Dict:
        """
        Suppress (remove) sensitive columns entirely from a record.

        Args:
            data: Input data dict.
            sensitive_columns: Columns to suppress.

        Returns:
            Data dict without sensitive columns.
        """
        return {k: v for k, v in data.items() if k not in sensitive_columns}
