import json
import re
from collections import defaultdict
from pathlib import Path


class JDParser:
    def __init__(self, vocab_path="data/skills_vocab.json"):
        self.vocab = self._load_vocab(vocab_path)

    def _load_vocab(self, path):
        with open(path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        # normalize keys to lowercase
        return {k.lower(): v for k, v in raw_vocab.items()}

    def _normalize_text(self, text):
        """
        Lowercase while preserving tech tokens like C++, Node.js.
        """
        text = text.lower()

        # Keep + and . inside words
        text = re.sub(r"[^\w\s\+\.\-]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _generate_ngrams(self, tokens, n):
        return [
            " ".join(tokens[i:i+n])
            for i in range(len(tokens) - n + 1)
        ]

    def parse(self, jd_path="data/JD.txt"):
        jd_path = Path(jd_path)

        with open(jd_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        normalized = self._normalize_text(raw_text)
        tokens = normalized.split()

        candidates = defaultdict(lambda: {
            "surface": None,
            "canonical": None,
            "count": 0,
            "positions": []
        })

        # Check 1-gram, 2-gram, 3-gram
        for n in [3, 2, 1]:
            ngrams = self._generate_ngrams(tokens, n)

            for i, gram in enumerate(ngrams):
                if gram in self.vocab:
                    canonical = self.vocab[gram]

                    entry = candidates[canonical]
                    entry["surface"] = gram
                    entry["canonical"] = canonical
                    entry["count"] += 1
                    entry["positions"].append(i)

        return {
            "raw_text": raw_text,
            "tokens": tokens,
            "candidates": list(candidates.values())
        }
