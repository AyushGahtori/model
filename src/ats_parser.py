import json
import re
from collections import defaultdict
from pathlib import Path


class ATSParser:
    # Constructor that initializes the parser with ATS phrases and action verbs
    def __init__(self, phrase_path="data/ats_phrases.json"):
        # Load ATS phrases from the specified JSON file
        self.phrases = self._load_phrases(phrase_path)
        # Define a set of common action verbs used in job descriptions
        self.action_verbs = {
            "build", "design", "develop", "implement",
            "optimize", "improve", "reduce", "scale",
            "monitor", "maintain", "debug", "analyze",
            "collaborate", "lead", "automate"
        }

    # Load and normalize phrases from a JSON file
    def _load_phrases(self, path):
        # Open the JSON file in read mode with UTF-8 encoding
        with open(path, "r", encoding="utf-8") as f:
            # Parse the JSON content into a dictionary
            raw = json.load(f)
        # Convert all keys to lowercase and return the normalized dictionary
        return {k.lower(): v for k, v in raw.items()}

    # Normalize text by converting to lowercase and removing special characters
    def _normalize_text(self, text):
        # Convert all text to lowercase
        text = text.lower()
        # Remove all special characters except word characters, spaces, +, ., and -
        text = re.sub(r"[^\w\s\+\.\-]", " ", text)
        # Replace multiple consecutive spaces with a single space
        text = re.sub(r"\s+", " ", text)
        # Remove leading and trailing whitespace
        return text.strip()

    # Parse a job description file and extract ATS keywords
    def parse(self, jd_path="data/JD.txt"):
        # Convert the file path to a Path object for better file handling
        jd_path = Path(jd_path)

        # Open and read the job description file
        with open(jd_path, "r", encoding="utf-8") as f:
            # Store the entire file content as a string
            raw_text = f.read()

        # Normalize the raw text for consistent matching
        normalized = self._normalize_text(raw_text)

        # Create a defaultdict to store results for each keyword
        results = defaultdict(lambda: {
            "surface": None,           # Original phrase as found in text
            "canonical": None,         # Standardized/normalized form
            "count": 0,                # Number of occurrences
            "type": "ats"              # Type classification
        })

        # ---- Exact Phrase Matching ----
        # Iterate through all loaded ATS phrases
        for phrase, canonical in self.phrases.items():
            # Count how many times the phrase appears as a whole word in the text
            matches = len(re.findall(rf"\b{re.escape(phrase)}\b", normalized))
            # If the phrase was found at least once
            if matches > 0:
                # Get or create the entry for this canonical form
                entry = results[canonical]
                # Store the original phrase as it appeared
                entry["surface"] = phrase
                # Store the standardized form
                entry["canonical"] = canonical
                # Add to the occurrence count
                entry["count"] += matches

        # ---- Action Verb Detection ----
        # Split the normalized text into individual words
        tokens = normalized.split()
        # Check each word to see if it's an action verb
        for token in tokens:
            # If the token matches one of our action verbs
            if token in self.action_verbs:
                # Get or create an entry for this verb (capitalized)
                entry = results[token.capitalize()]
                # Store the lowercase version
                entry["surface"] = token
                # Store the capitalized canonical form
                entry["canonical"] = token.capitalize()
                # Increment the occurrence count
                entry["count"] += 1

        # Convert the defaultdict values to a list and return
        return list(results.values())