"""
AILS Named Entity Recognition (NER) Module
Token-level classification for extracting entities from text.
Artificial Intelligence Learning System — Created by Cherry Computer Ltd.
"""

import re
import logging
from typing import List, Dict, Tuple


# Basic rule-based entity patterns (no dependency on spaCy required)
_PATTERNS: Dict[str, str] = {
    "EMAIL":    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "URL":      r"https?://[^\s]+",
    "PHONE":    r"\b(?:\+?\d[\d\s\-().]{6,14}\d)\b",
    "DATE":     r"\b(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{2}[\/\-]\d{2})\b",
    "MONEY":    r"\$[\d,]+(?:\.\d{2})?|\b\d+(?:\.\d+)?\s*(?:USD|GBP|EUR)\b",
    "PERCENT":  r"\b\d+(?:\.\d+)?%",
    "HASHTAG":  r"#\w+",
    "MENTION":  r"@\w+",
}


class AILSEntityRecognizer:
    """
    AILS Named Entity Recognizer.

    Provides two modes:
    1. **Rule-based** — Fast regex extraction of common entity types
       (email, URL, phone, date, money, percent, hashtag, mention).
    2. **spaCy-based** (optional) — Full NER with PERSON, ORG, GPE, etc.
       Requires `pip install spacy && python -m spacy download en_core_web_sm`.

    Example:
        ner = AILSEntityRecognizer()
        entities = ner.extract_entities(
            "Call us at +1 800-555-0100 or visit https://ails.ai"
        )
        # [{'text': '+1 800-555-0100', 'label': 'PHONE', 'start': 11, 'end': 26},
        #  {'text': 'https://ails.ai', 'label': 'URL', 'start': 37, 'end': 52}]
    """

    def __init__(self, use_spacy: bool = False,
                 spacy_model: str = "en_core_web_sm"):
        self.use_spacy = use_spacy
        self.logger = logging.getLogger("AILS.NLP.NER")
        self._nlp = None

        if use_spacy:
            try:
                import spacy
                self._nlp = spacy.load(spacy_model)
                self.logger.info(f"✅ spaCy model '{spacy_model}' loaded.")
            except (ImportError, OSError) as e:
                self.logger.warning(
                    f"spaCy unavailable ({e}). Falling back to rule-based NER."
                )
                self.use_spacy = False

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text.

        Args:
            text: Input string.

        Returns:
            List of dicts: {text, label, start, end}.
        """
        if self.use_spacy and self._nlp:
            return self._spacy_extract(text)
        return self._regex_extract(text)

    def _regex_extract(self, text: str) -> List[Dict]:
        """Rule-based regex extraction."""
        entities = []
        for label, pattern in _PATTERNS.items():
            for match in re.finditer(pattern, text):
                entities.append({
                    "text":  match.group(),
                    "label": label,
                    "start": match.start(),
                    "end":   match.end(),
                })
        # Sort by start position
        return sorted(entities, key=lambda e: e["start"])

    def _spacy_extract(self, text: str) -> List[Dict]:
        """spaCy-based NER extraction."""
        doc = self._nlp(text)
        return [
            {
                "text":  ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end":   ent.end_char,
            }
            for ent in doc.ents
        ]

    def extract_batch(self, texts: List[str]) -> List[List[Dict]]:
        """Extract entities from a list of texts."""
        return [self.extract_entities(t) for t in texts]

    def anonymize_entities(self, text: str,
                            labels_to_mask: List[str] = None) -> str:
        """
        Replace specified entity types with placeholder tokens.

        Args:
            text: Input text.
            labels_to_mask: Entity labels to anonymize
                            (default: EMAIL, PHONE).

        Returns:
            Text with masked entities.
        """
        if labels_to_mask is None:
            labels_to_mask = ["EMAIL", "PHONE"]

        entities = self.extract_entities(text)
        # Process in reverse to preserve character offsets
        for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
            if ent["label"] in labels_to_mask:
                placeholder = f"[{ent['label']}]"
                text = text[:ent["start"]] + placeholder + text[ent["end"]:]
        return text
