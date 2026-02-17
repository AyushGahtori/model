"""
src/keyword_ranker.py

Deterministic keyword ranking for JD keywords (core + ATS).
Input: outputs from JDParser and ATSParser (lists of candidate dicts)
Output: sorted list of keywords with scores, percentile, and top50 flag.

Usage (quick):
    from src.keyword_ranker import KeywordRanker
    rk = KeywordRanker()
    ranked = rk.rank(jd_text, core_candidates, ats_candidates)
"""

import re
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any


# -------------------------
# Tunable weights / params
# -------------------------
TYPE_BASE = {"core": 1.5, "ats": 1.0}
IN_TITLE_BONUS = 1.0          # big boost if keyword appears very early (title/summary)
FREQ_WEIGHT = 1.2             # weight for normalized frequency
POSITION_WEIGHT = 0.9         # weight for position-based score (earlier -> higher)
REQUIRED_BONUS = 1.4          # big boost if in "must/required" context
VERB_PROXIMITY_BONUS = 0.3    # small boost if near an action verb
TITLE_WINDOW_CHARS = 80       # chars to consider "title/early" region
REQUIRED_PATTERNS = [
    r"\bmust\b",
    r"\brequired\b",
    r"\bessential\b",
    r"\bminimum\b",
    r"\bmust have\b",
    r"\bmust-have\b",
    r"\bmust have experience\b"
]
ACTION_VERBS = {
    "build", "design", "develop", "implement", "optimize", "improve",
    "reduce", "scale", "monitor", "maintain", "debug", "analyze",
    "collaborate", "lead", "automate", "deploy", "manage", "support"
}


class KeywordRanker:
    def __init__(self,
                 type_base=None,
                 in_title_bonus=IN_TITLE_BONUS,
                 freq_weight=FREQ_WEIGHT,
                 position_weight=POSITION_WEIGHT,
                 required_bonus=REQUIRED_BONUS,
                 verb_proximity_bonus=VERB_PROXIMITY_BONUS):
        self.type_base = type_base or TYPE_BASE
        self.in_title_bonus = in_title_bonus
        self.freq_weight = freq_weight
        self.position_weight = position_weight
        self.required_bonus = required_bonus
        self.verb_proximity_bonus = verb_proximity_bonus
        self.required_regex = re.compile("|".join(REQUIRED_PATTERNS), re.I)

    # -------------------------
    # Helpers
    # -------------------------
    def _collect_all(self, core_candidates: List[Dict[str, Any]],
                     ats_candidates: List[Dict[str, Any]]):
        """
        Merge candidates into canonical-key -> aggregated record.
        Expect candidate dicts like:
           {"surface": "node.js", "canonical": "Node.js", "count": 3, "positions": [12,98], "type":"core"}
        But both parsers may omit some fields; tolerate that.
        """
        agg = {}
        # helper to insert/merge
        def add_item(item, typ):
            canonical = item.get("canonical") or item.get("surface") or item
            key = canonical
            if key not in agg:
                agg[key] = {
                    "canonical": canonical,
                    "surface_examples": set(),
                    "count": 0,
                    "positions": [],
                    "types": set(),
                }
            if item.get("surface"):
                agg[key]["surface_examples"].add(item["surface"])
            agg[key]["count"] += int(item.get("count", 1))
            if "positions" in item and item["positions"]:
                agg[key]["positions"].extend(item["positions"])
            agg[key]["types"].add(typ)

        for c in (core_candidates or []):
            add_item(c, "core")
        for a in (ats_candidates or []):
            add_item(a, "ats")

        # convert sets to lists
        for v in agg.values():
            v["surface_examples"] = list(v["surface_examples"])
            v["types"] = list(v["types"])
        return agg

    def _in_title(self, jd_text: str, canonical: str) -> bool:
        window = jd_text[:TITLE_WINDOW_CHARS].lower()
        return canonical.lower() in window

    def _is_required_context(self, jd_text: str, canonical: str) -> bool:
        """
        Return True if the canonical keyword appears in a sentence or bullet that also
        contains a required-like pattern such as 'must', 'required', 'essential', etc.
        """
        lowered = jd_text.lower()
        # find occurrences and check surrounding sentence
        for match in re.finditer(re.escape(canonical.lower()), lowered):
            start, end = match.start(), match.end()
            # compute context window (grab surrounding 100 chars or the sentence)
            left = max(0, lowered.rfind(".", 0, start))
            right = lowered.find(".", end)
            if right == -1:
                right = min(len(lowered), end + 200)
            # context slice
            ctx = lowered[left+1:right]
            if self.required_regex.search(ctx):
                return True
        # fallback: keyword near a required word anywhere (distance)
        if self.required_regex.search(lowered):
            # check distance to nearest required token occurrence
            req_matches = [m for m in re.finditer(self.required_regex, lowered)]
            key_matches = [m for m in re.finditer(re.escape(canonical.lower()), lowered)]
            if req_matches and key_matches:
                # compute minimal char distance
                min_dist = min(abs(k.start()-r.end()) for k in key_matches for r in req_matches)
                if min_dist < 200:
                    return True
        return False

    def _avg_position_score(self, positions: List[int], total_tokens: int) -> float:
        if not positions:
            return 0.0
        avg_pos = sum(positions)/len(positions)
        # convert token index into [0,1], then invert so smaller index -> larger score
        # If total_tokens unknown or 0, just use a weak default
        if total_tokens <= 1:
            return 0.5
        p = avg_pos / float(total_tokens)
        return max(0.0, 1.0 - p)  # earlier -> closer to 1

    def _verb_proximity(self, jd_text: str, canonical: str) -> bool:
        """
        Check whether any action verb occurs within N words of the canonical phrase.
        """
        lowered = jd_text.lower()
        # split words and build index map of word -> positions
        words = re.findall(r"\w+", lowered)
        if not words:
            return False
        # find indices where canonical appears as sequence
        can_tokens = re.findall(r"\w+", canonical.lower())
        if not can_tokens:
            return False

        # naive search: find first index of can_tokens in words
        indices = []
        for i in range(len(words)-len(can_tokens)+1):
            if words[i:i+len(can_tokens)] == can_tokens:
                indices.append(i)
        if not indices:
            return False

        # check for action verbs within window of 6 words
        window = 6
        for idx in indices:
            left = max(0, idx - window)
            right = min(len(words), idx + len(can_tokens) + window)
            window_words = words[left:right]
            if any(v in window_words for v in ACTION_VERBS):
                return True
        return False

    # -------------------------
    # Main API
    # -------------------------
    def rank(self,
             jd_text: str,
             core_candidates: List[Dict[str, Any]],
             ats_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank and return list of dicts:
        {
            "keyword": canonical,
            "types": [...],
            "count": int,
            "score": float,
            "rank": int,
            "percentile": float,
            "top50": bool,
            "details": {...}
        }
        """
        # Merge candidates
        agg = self._collect_all(core_candidates, ats_candidates)

        # total tokens for position normalization
        total_tokens = len(re.findall(r"\w+", jd_text))

        # compute max freq for normalization
        max_freq = max((v["count"] for v in agg.values()), default=1)

        records = []
        for canonical, v in agg.items():
            types = v["types"]
            # pick a primary type if both exist: prefer core
            typ = "core" if "core" in types else "ats"

            base = self.type_base.get(typ, 1.0)

            # in-title
            in_title = self._in_title(jd_text, canonical)

            # freq norm
            freq_norm = v["count"] / float(max_freq) if max_freq > 0 else 0.0

            # position score
            pos_score = self._avg_position_score(v.get("positions", []), total_tokens)

            # required context
            required_ctx = self._is_required_context(jd_text, canonical)

            # verb proximity
            verb_prox = self._verb_proximity(jd_text, canonical)

            # calculate final score
            score = 0.0
            score += base
            if in_title:
                score += self.in_title_bonus
            score += self.freq_weight * freq_norm
            score += self.position_weight * pos_score
            if required_ctx:
                score += self.required_bonus
            if verb_prox:
                score += self.verb_proximity_bonus

            rec = {
                "keyword": canonical,
                "types": types,
                "count": v["count"],
                "surface_examples": v.get("surface_examples", []),
                "positions": v.get("positions", []),
                "score": round(score, 5),
                "details": {
                    "base": base,
                    "in_title": in_title,
                    "freq_norm": round(freq_norm, 5),
                    "position_score": round(pos_score, 5),
                    "required_ctx": required_ctx,
                    "verb_prox": verb_prox
                }
            }
            records.append(rec)

        # sort by score desc
        records.sort(key=lambda r: r["score"], reverse=True)

        # add rank, percentile, top50
        n = max(len(records), 1)
        for i, r in enumerate(records, start=1):
            r["rank"] = i
            # percentile 100 means top
            r["percentile"] = round(100.0 * (1.0 - (i - 1) / n), 2)
            r["top50"] = (i <= (n // 2))  # top 50% integer split

        return records


# -------------------------
# Quick CLI/test harness
# -------------------------
if __name__ == "__main__":
    # Quick test if run directly (will import your parsers if present)
    try:
        from src.jd_parser import JDParser
        from src.ats_parser import ATSParser
    except Exception as e:
        print("Could not import JDParser/ATSParser from src/ — ensure those files exist.")
        raise e

    jdfile = Path("data/JD.txt")
    if not jdfile.exists():
        print("Place a JD at data/JD.txt to test.")
        raise SystemExit(1)

    jd_parser = JDParser()
    core = jd_parser.parse(str(jdfile))["candidates"]

    ats_parser = ATSParser()
    ats = ats_parser.parse(str(jdfile))

    rk = KeywordRanker()
    ranked = rk.rank(jdfile.read_text(encoding="utf-8"), core, ats)

    # print top 20
    print(json.dumps(ranked[:20], indent=2))