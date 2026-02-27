import json
import re
from collections import defaultdict
from pathlib import Path


class JDParser:
    # Initialize the parser with vocabulary data
    def __init__(self, vocab_path="data/skills_vocab.json"):
        self.vocab = self._load_vocab(vocab_path)

    # Load and normalize vocabulary from JSON file
    def _load_vocab(self, path):
        with open(path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        # Convert all keys to lowercase for case-insensitive matching
        return {k.lower(): v for k, v in raw_vocab.items()}

    # Normalize text: convert to lowercase and clean special characters
    def _normalize_text(self, text):
        """
        Lowercase while preserving tech tokens like C++, Node.js.
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters except +, ., and - (for tech names)
        text = re.sub(r"[^\w\s\+\.\-]", " ", text)
        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # Generate n-grams from token list
    def _generate_ngrams(self, tokens, n):
        return [
            " ".join(tokens[i:i+n])
            for i in range(len(tokens) - n + 1)
        ]

    # Parse job description and extract skill candidates
    def parse(self, jd_path="data/JD.txt"):
        jd_path = Path(jd_path)

        # Read raw job description text
        with open(jd_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Normalize text and split into tokens
        normalized = self._normalize_text(raw_text)
        tokens = normalized.split()

        # Initialize candidate dictionary to track matched skills
        candidates = defaultdict(lambda: {
            "surface": None,
            "canonical": None,
            "count": 0,
            "positions": []
        })

        # Check n-grams from largest (3) to smallest (1) to prioritize longer matches
        for n in [3, 2, 1]:
            ngrams = self._generate_ngrams(tokens, n)

            # Match each n-gram against vocabulary
            for i, gram in enumerate(ngrams):
                if gram in self.vocab:
                    canonical = self.vocab[gram]

                    # Record matched skill information
                    entry = candidates[canonical]
                    entry["surface"] = gram
                    entry["canonical"] = canonical
                    entry["count"] += 1
                    entry["positions"].append(i)

        # Return parsed results with raw text, tokens, and extracted skills
        return {
            "raw_text": raw_text,
            "tokens": tokens,
            "candidates": list(candidates.values())
        }
