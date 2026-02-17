import json
import re
from collections import defaultdict
from pathlib import Path


class ATSParser:
    def __init__(self, phrase_path="data/ats_phrases.json"):
        self.phrases = self._load_phrases(phrase_path)
        self.action_verbs = {
            "build", "design", "develop", "implement",
            "optimize", "improve", "reduce", "scale",
            "monitor", "maintain", "debug", "analyze",
            "collaborate", "lead", "automate"
        }

    def _load_phrases(self, path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k.lower(): v for k, v in raw.items()}

    def _normalize_text(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s\+\.\-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def parse(self, jd_path="data/JD.txt"):
        jd_path = Path(jd_path)

        with open(jd_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        normalized = self._normalize_text(raw_text)

        results = defaultdict(lambda: {
            "surface": None,
            "canonical": None,
            "count": 0,
            "type": "ats"
        })

        # ---- Exact Phrase Matching ----
        for phrase, canonical in self.phrases.items():
            matches = len(re.findall(rf"\b{re.escape(phrase)}\b", normalized))
            if matches > 0:
                entry = results[canonical]
                entry["surface"] = phrase
                entry["canonical"] = canonical
                entry["count"] += matches

        # ---- Action Verb Detection ----
        tokens = normalized.split()
        for token in tokens:
            if token in self.action_verbs:
                entry = results[token.capitalize()]
                entry["surface"] = token
                entry["canonical"] = token.capitalize()
                entry["count"] += 1

        return list(results.values())