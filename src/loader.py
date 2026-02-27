import json
import re
from pathlib import Path
from collections import defaultdict


class ResumeLoader:
    def __init__(self, resume_path="data/resume.json", vocab_path="data/skills_vocab.json"):
        # Initialize file paths and load the skills vocabulary
        self.resume_path = Path(resume_path)
        self.vocab = self._load_vocab(vocab_path)

    # --------------------------
    # Load canonical vocab
    # --------------------------
    def _load_vocab(self, path):
        # Read the skills vocabulary JSON file and normalize keys to lowercase
        with open(path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        return {k.lower(): v for k, v in raw_vocab.items()}

    # --------------------------
    # Normalize skill name
    # --------------------------
    def _canonicalize(self, skill):
        # Convert skill to lowercase and look it up in vocabulary
        # Return canonical form if found, otherwise return trimmed original
        skill_lower = skill.lower()
        return self.vocab.get(skill_lower, skill.strip())

    # --------------------------
    # Normalize text
    # --------------------------
    def _normalize_text(self, text):
        # Strip whitespace and collapse multiple spaces into single space
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    # --------------------------
    # Ensure bullet list format
    # --------------------------
    def _ensure_bullet_list(self, text):
        # Case 1: Already a list - normalize each item
        if isinstance(text, list):
            return [self._normalize_text(t) for t in text if isinstance(t, str)]

        # Case 2: Dictionary of bullets (p1, p2, etc.) - extract values in sorted order
        elif isinstance(text, dict):
            return [
                self._normalize_text(v)
                for k, v in sorted(text.items())
                if isinstance(v, str)
            ]

        # Case 3: Single string - split by newlines or periods and normalize each part
        elif isinstance(text, str):
            parts = re.split(r"\n+|\.\s+", text)
            return [self._normalize_text(p) for p in parts if p.strip()]

        # Fallback - return empty list for unsupported types
        else:
            return []

    # --------------------------
    # Main loader
    # --------------------------
    def load(self):
        # Read and parse the resume JSON file
        with open(self.resume_path, "r", encoding="utf-8") as f:
            resume = json.load(f)

        # Initialize tracking dictionaries for skill-to-experience and skill-to-project mappings
        skill_to_exp_ids = defaultdict(list)
        skill_to_proj_ids = defaultdict(list)
        skill_counts = defaultdict(int)

        # Process each experience entry
        for exp in resume.get("experience", []):
            # Validate that each experience has an id
            if "id" not in exp:
                raise ValueError("Each experience must have an 'id' field")

            # Convert experience text into standardized bullet list format
            exp["text"] = self._ensure_bullet_list(exp.get("text", ""))

            # Canonicalize and de-duplicate skills for this experience
            normalized_skills = []
            for skill in exp.get("skills", []):
                canonical = self._canonicalize(skill)
                normalized_skills.append(canonical)

                # Map skill to this experience and count occurrences
                skill_to_exp_ids[canonical].append(exp["id"])
                skill_counts[canonical] += 1

            exp["skills"] = list(set(normalized_skills))  # Remove duplicates

        # Process each project entry
        for proj in resume.get("projects", []):
            # Validate that each project has an id
            if "id" not in proj:
                raise ValueError("Each project must have an 'id' field")

            # Convert project text into standardized bullet list format
            proj["text"] = self._ensure_bullet_list(proj.get("text", ""))

            # Canonicalize and de-duplicate skills for this project
            normalized_skills = []
            for skill in proj.get("skills", []):
                canonical = self._canonicalize(skill)
                normalized_skills.append(canonical)

                # Map skill to this project and count occurrences
                skill_to_proj_ids[canonical].append(proj["id"])
                skill_counts[canonical] += 1

            proj["skills"] = list(set(normalized_skills))

        # Ensure skills section exists in resume (create empty list if missing)
        if "skills_section" not in resume:
            resume["skills_section"] = []

        # Return normalized resume and skill mapping data
        return {
            "resume": resume,
            "skill_to_exp_ids": dict(skill_to_exp_ids),
            "skill_to_proj_ids": dict(skill_to_proj_ids),
            "skill_counts": dict(skill_counts)
        }