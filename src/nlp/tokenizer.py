"""
AILS NLP Tokenizer Module
Tokenization, stemming, lemmatization, and n-gram generation.
Artificial Intelligence Learning System — Created by Cherry Computer Ltd.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional


class AILSTokenizer:
    """
    AILS Text Tokenizer.

    Provides word tokenization, subword tokenization, n-gram generation,
    stemming, lemmatization, and vocabulary building.

    Example:
        tokenizer = AILSTokenizer()
        tokens = tokenizer.tokenize("The quick brown fox jumps over the lazy dog.")
        # ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']

        vocab = tokenizer.build_vocabulary(corpus_texts, max_vocab=5000)
        encoded = tokenizer.encode("The quick fox", vocab)
    """

    def __init__(self, lowercase: bool = True,
                 remove_punctuation: bool = True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.logger = logging.getLogger("AILS.NLP.Tokenizer")
        self._vocab: Dict[str, int] = {}
        self._reverse_vocab: Dict[int, str] = {}

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a list of word tokens.

        Args:
            text: Raw input string.

        Returns:
            List of token strings.
        """
        if self.lowercase:
            text = text.lower()
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        return [t for t in tokens if t]

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Tokenize a list of texts."""
        return [self.tokenize(t) for t in texts]

    def stem(self, tokens: List[str]) -> List[str]:
        """
        Apply Porter stemming to a list of tokens.
        Falls back to basic suffix stripping if NLTK unavailable.
        """
        try:
            from nltk.stem import PorterStemmer
            import nltk
            nltk.download("punkt", quiet=True)
            stemmer = PorterStemmer()
            return [stemmer.stem(t) for t in tokens]
        except ImportError:
            # Basic suffix stripping fallback
            suffixes = ("ing", "tion", "ed", "ly", "er", "est", "ness")
            stemmed = []
            for t in tokens:
                for suffix in suffixes:
                    if t.endswith(suffix) and len(t) > len(suffix) + 2:
                        t = t[:-len(suffix)]
                        break
                stemmed.append(t)
            return stemmed

    def generate_ngrams(self, tokens: List[str],
                         n: int = 2) -> List[str]:
        """
        Generate n-grams from a token list.

        Args:
            tokens: List of word tokens.
            n: n-gram size (2 = bigram, 3 = trigram).

        Returns:
            List of n-gram strings joined by underscore.
        """
        return [
            "_".join(tokens[i:i + n])
            for i in range(len(tokens) - n + 1)
        ]

    def build_vocabulary(self, texts: List[str],
                          max_vocab: int = 10000,
                          min_freq: int = 2) -> Dict[str, int]:
        """
        Build a vocabulary from a corpus of texts.

        Args:
            texts: List of text strings.
            max_vocab: Maximum vocabulary size.
            min_freq: Minimum token frequency to include.

        Returns:
            Dictionary mapping token → integer index.
        """
        from collections import Counter
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))

        counts = Counter(all_tokens)
        # Filter by frequency and take top max_vocab
        filtered = [
            (tok, cnt) for tok, cnt in counts.most_common()
            if cnt >= min_freq
        ][:max_vocab - 2]  # Reserve 0=PAD, 1=UNK

        self._vocab = {"<PAD>": 0, "<UNK>": 1}
        self._vocab.update({tok: idx + 2 for idx, (tok, _) in enumerate(filtered)})
        self._reverse_vocab = {v: k for k, v in self._vocab.items()}

        self.logger.info(
            f"✅ Vocabulary built: {len(self._vocab)} tokens "
            f"(from {len(texts)} texts)"
        )
        return self._vocab

    def encode(self, text: str,
                vocab: Optional[Dict[str, int]] = None,
                max_len: Optional[int] = None,
                padding: bool = True) -> List[int]:
        """
        Encode text to integer sequence.

        Args:
            text: Input text string.
            vocab: Token→index mapping (uses built vocab if None).
            max_len: Pad/truncate to this length.
            padding: Whether to pad to max_len.

        Returns:
            List of integer token IDs.
        """
        v = vocab or self._vocab
        if not v:
            raise RuntimeError("Build vocabulary first with build_vocabulary().")

        tokens = self.tokenize(text)
        ids = [v.get(t, v.get("<UNK>", 1)) for t in tokens]

        if max_len:
            ids = ids[:max_len]
            if padding:
                ids += [v.get("<PAD>", 0)] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int],
                reverse_vocab: Optional[Dict[int, str]] = None) -> str:
        """Decode integer IDs back to text."""
        rv = reverse_vocab or self._reverse_vocab
        return " ".join(rv.get(i, "<UNK>") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)
