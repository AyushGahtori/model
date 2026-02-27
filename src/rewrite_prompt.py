# src/rewrite_prompt.py
import json
import re
from typing import Dict, Any, List, Tuple, Optional


class RewritePromptBuilder:
    """
    Builds a strict prompt for the LLM to rewrite resume bullets.
    Also builds the mapping of where each MUST keyword should appear.
    """

    def __init__(self, max_bullets_per_exp: int = 3, max_projects: int = 2, bullet_word_limit: int = 25):
        self.max_bullets_per_exp = max_bullets_per_exp
        self.max_projects = max_projects
        self.bullet_word_limit = bullet_word_limit

    def _map_must_keywords_to_locations(self, must_keywords: List[str], resume_data: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Build a mapping for each must keyword -> where it should appear.
        Returns dict:
          { keyword: {"experiences": [exp_id,...], "projects": [proj_id,...], "skills_section": bool} }
        Logic:
          - If a keyword exists in resume experience skills => list those exp ids
          - Else if exists in resume project skills => list those proj ids
          - Else mark skills_section True (LLM must add to skills_section)
        """
        out = {}
        skill_to_exp = resume_data.get("skill_to_exp_ids", {})
        skill_to_proj = resume_data.get("skill_to_proj_ids", {})

        for kw in must_keywords:
            exp_ids = skill_to_exp.get(kw, [])
            proj_ids = skill_to_proj.get(kw, [])
            needs_skills_section = False
            if not exp_ids and not proj_ids:
                needs_skills_section = True

            out[kw] = {
                "experiences": list(exp_ids),
                "projects": list(proj_ids),
                "skills_section": needs_skills_section
            }
        return out

    def build_prompt(self, decision_plan: Dict[str, Any], resume_data: Dict[str, Any]) -> str:
        """
        Build the instruction string to send to the LLM.
        decision_plan: produced by matcher.create_decision_plan()
        resume_data: loader.load() output
        """
        must_keywords = decision_plan.get("must_include_keywords", [])
        selected_projects = decision_plan.get("selected_projects", [])[: self.max_projects]
        experience_plan = decision_plan.get("experience_plan", {})
        skills_additions = decision_plan.get("skills_section_additions", [])

        # Build mapping of must keywords -> allowed locations
        must_map = self._map_must_keywords_to_locations(must_keywords, resume_data)

        # Build experience payload: include all experiences but mark editable indices
        experiences_payload = []
        for exp in resume_data["resume"].get("experience", []):
            exp_id = exp["id"]
            original_bullets = exp.get("text", [])
            # selected bullets for rewrite (from experience_plan) - list of bullet strings
            plan = experience_plan.get(exp_id, {})
            selected_bullets = plan.get("selected_bullets", [])
            # Convert selected_bullets to editable indices (match by exact string)
            editable_indices = []
            for i, b in enumerate(original_bullets):
                if b in selected_bullets:
                    editable_indices.append(i)
            experiences_payload.append({
                "id": exp_id,
                "original_bullets": original_bullets,
                "editable_indices": editable_indices,
                "allowed_skills": plan.get("all_skills", exp.get("skills", []))
            })

        # Build projects payload: include only selected projects (matcher picked them)
        projects_payload = []
        # Build an id->project map for quick access
        proj_map = {p["id"]: p for p in resume_data["resume"].get("projects", [])}
        for pid in selected_projects:
            proj = proj_map.get(pid)
            if proj is None:
                # If matcher asked for a proj id that does not exist, skip
                continue
            projects_payload.append({
                "id": pid,
                "original_bullets": proj.get("text", []),
                "allowed_skills": proj.get("skills", [])
            })

        # Compose strict instruction text
        instruction = []
        append = instruction.append

        append("You are a controlled resume rewriting assistant. Follow ALL rules exactly.")
        append("")
        append("GLOBAL RULES (READ CAREFULLY):")
        append("1) Do NOT invent any new projects or experiences. You may only rewrite bullets provided.")
        append("2) Do NOT add any skills anywhere except those listed in allowed_skills for each section.")
        append("3) Do NOT change company names, role titles, dates, durations, or numeric facts. You may shorten/clarify numbers but do NOT invent new numeric achievements.")
        append("4) You MUST include every keyword in the 'must_keywords' list. For each keyword we also provide allowed locations (experience IDs, project IDs).")
        append("   - If a keyword maps to one or more experience IDs, it must appear in at least one of those experiences.")
        append("   - If a keyword maps only to project IDs, it must appear in at least one of those projects.")
        append("   - If a keyword maps to no experience/project, add it ONLY to the skills_section (do NOT invent an experience).")
        append(f"5) You MUST NOT return more than {self.max_projects} projects. Only the project IDs provided may be used.")
        append("6) You MUST return ALL experiences (do not drop any). For experiences marked with editable_indices, you may rewrite only those bullets; for other bullets return text unchanged or minor punctuation fixes only.")
        append("7) Bullets you rewrite MUST be: active voice, achievement-first, one sentence, <= {} words, and avoid weasel terms.".format(self.bullet_word_limit))
        append("8) Output ONLY valid JSON matching the schema below. NO extra text, no commentary. If you cannot comply, return an object with key 'error' and a short explanation.")
        append("")
        append("OUTPUT JSON SCHEMA (exact):")
        append(json.dumps({
            "experience": [
                {"id": "exp_id", "bullets": ["full list of bullets for this experience (all bullets)"]}
            ],
            "projects": [
                {"id": "proj_id", "bullets": ["rewritten bullets for selected project(s)"]}
            ],
            "skills_section": ["list", "of", "skills", "strings"]
        }, indent=2))
        append("")
        append("INPUT DATA (do not change):")
        payload = {
            "must_keywords": must_keywords,
            "must_keyword_locations": must_map,
            "experiences": experiences_payload,
            "projects": projects_payload,
            "skills_section_additions": skills_additions
        }
        append(json.dumps(payload, indent=2))
        append("")
        append("INSTRUCTIONS FOR HOW TO REWRITE (must follow):")
        append("- For each experience: return the full bullets list. For editable_indices: rewrite those bullets. For non-editable bullets: return text unchanged except tiny punctuation/formatting fixes.")
        append("- For projects: return only the selected project entries (up to max_projects). You may rewrite their bullets; do NOT add new project IDs.")
        append("- When including a MUST keyword in an experience or project, prefer to place the keyword in the rewritten bullet(s) that are most relevant. If it cannot be included sensibly, add it to skills_section (only for keywords mapped to skills_section).")
        append("- All returned bullets must be truthful and derived from the original bullet or allowed_skills for that section.")
        append("- If you must abbreviate or shorten a bullet, preserve the factual meaning.")
        append("- If you detect ambiguous numeric claims in a rewritten bullet, do NOT invent new numbers; instead keep the original numeric value or remove quantification.")
        append("")
        append("Return JSON only. No commentary, no markdown, nothing else.")

        return "\n".join(instruction)


class RewriteValidator:
    """
    Validates the JSON produced by the LLM against the decision plan and resume_data rules.
    Returns (True, None) on success or (False, "error message") on failure.
    """

    sentence_split_re = re.compile(r"[.!?]+")
    word_split_re = re.compile(r"\w+")
    def __init__(self, max_projects: int = 2, bullet_word_limit: int = 25):
        self.max_projects = max_projects
        self.bullet_word_limit = bullet_word_limit

    # helper: count words
    def _word_count(self, s: str) -> int:
        return len(self.word_split_re.findall(s))

    # helper: check one-sentence (very conservative)
    def _is_one_sentence(self, s: str) -> bool:
        # consider a single sentence if there is at most one terminal punctuation
        parts = [p for p in self.sentence_split_re.split(s) if p.strip()]
        return len(parts) <= 1

    def validate(self, llm_output_str: str, decision_plan: Dict[str, Any], resume_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        llm_output_str: raw string response from LLM
        Returns: (ok: bool, error_message or None)
        """
        # parse JSON only
        try:
            out = json.loads(llm_output_str)
        except Exception as e:
            return False, f"Invalid JSON: {e}"

        # error-object from LLM allowed
        if isinstance(out, dict) and out.get("error"):
            return False, f"LLM returned error object: {out.get('error')}"

        # top-level keys
        if not isinstance(out, dict) or not all(k in out for k in ("experience", "projects", "skills_section")):
            return False, "Output JSON must contain keys: experience, projects, skills_section"

        # Check experiences: must include all experiences' ids
        expected_exp_ids = [e["id"] for e in resume_data["resume"].get("experience", [])]
        out_exp_map = {e["id"]: e for e in out["experience"] if "id" in e}
        for exp_id in expected_exp_ids:
            if exp_id not in out_exp_map:
                return False, f"Missing experience id in output: {exp_id}"

        # Validate bullets lengths, one-sentence, and editable indices behavior
        exp_plan = decision_plan.get("experience_plan", {})
        for exp in resume_data["resume"].get("experience", []):
            eid = exp["id"]
            original = exp.get("text", [])
            out_bullets = out_exp_map[eid].get("bullets")
            if not isinstance(out_bullets, list):
                return False, f"Experience {eid} bullets must be a list."
            # same number of bullets required
            if len(out_bullets) != len(original):
                return False, f"Experience {eid} must return same number of bullets ({len(original)})"

            editable_indices = exp_plan.get(eid, {}).get("selected_bullets", [])
            # convert selected bullets to indices
            selected = []
            selected_texts = set(exp_plan.get(eid, {}).get("selected_bullets", []))
            for i, b in enumerate(original):
                if b in selected_texts:
                    selected.append(i)

            for i, b in enumerate(out_bullets):
                # if this index is editable -> allow rewrite checks
                if i in selected:
                    # check word limit and one-sentence
                    if self._word_count(b) > self.bullet_word_limit:
                        return False, f"Experience {eid} bullet {i} exceeds word limit {self.bullet_word_limit}."
                    if not self._is_one_sentence(b):
                        return False, f"Experience {eid} bullet {i} must be one sentence."
                    # cannot invent numeric facts: compare numeric tokens
                    orig_nums = re.findall(r"\d+(?:[.,]\d+)?", original[i])
                    new_nums = re.findall(r"\d+(?:[.,]\d+)?", b)
                    # new numbers must be subset of original numbers (or none)
                    for n in new_nums:
                        if n not in orig_nums:
                            return False, f"Experience {eid} bullet {i} contains a numeric value not present in original."
                else:
                    # non-editable: must be equal to original or only whitespace/punctuation changes
                    # simplified check: normalize letters/numbers and compare
                    norm_orig = re.sub(r"\s+", " ", original[i].strip())
                    norm_out = re.sub(r"\s+", " ", b.strip())
                    # allow small punctuation differences but require core tokens present
                    if len(re.findall(r"\w+", norm_orig)) == 0:
                        continue
                    # require that most words from original appear in output (conservative)
                    orig_words = set(re.findall(r"\w+", norm_orig.lower()))
                    out_words = set(re.findall(r"\w+", norm_out.lower()))
                    if len(orig_words & out_words) / max(1, len(orig_words)) < 0.6:
                        return False, f"Experience {eid} bullet {i} appears substantially altered though not marked editable."

        # Projects checks: must be <= max_projects and IDs must be from decision_plan selected list
        selected_proj_ids = decision_plan.get("selected_projects", [])[: self.max_projects]
        out_projects = out.get("projects", [])
        if len(out_projects) > self.max_projects:
            return False, f"Return at most {self.max_projects} projects."

        for p in out_projects:
            pid = p.get("id")
            if pid not in selected_proj_ids:
                return False, f"Project id {pid} is not allowed. Allowed project ids: {selected_proj_ids}"
            # bullets must be list and meet bullet constraints
            for i, b in enumerate(p.get("bullets", [])):
                if self._word_count(b) > self.bullet_word_limit:
                    return False, f"Project {pid} bullet {i} exceeds word limit."
                if not self._is_one_sentence(b):
                    return False, f"Project {pid} bullet {i} must be one sentence."

        # Skills section: must include the skills_section_additions (these are must-add)
        skills_section = out.get("skills_section", [])
        if not isinstance(skills_section, list):
            return False, "skills_section must be a list."

        required_skills = decision_plan.get("skills_section_additions", [])
        for s in required_skills:
            if s not in skills_section:
                return False, f"Required skill {s} missing from skills_section."

        # Ensure must_keywords are present in expected locations
        must_map = {}
        for k, v in decision_plan.get("must_include_keywords", []), []:
            pass  # placeholder to avoid lint warnings (we build below)

        # Build mapping from id->text content for checking
        # Experience content combined
        combined_exp_text = {}
        for e in out["experience"]:
            combined_exp_text[e["id"]] = " ".join(e.get("bullets", []))

        combined_proj_text = {}
        for p in out.get("projects", []):
            combined_proj_text[p["id"]] = " ".join(p.get("bullets", []))

        must_locations = RewritePromptBuilder()._map_must_keywords_to_locations(decision_plan.get("must_include_keywords", []), resume_data)
        for kw, loc in must_locations.items():
            appeared = False
            # check experiences
            for eid in loc.get("experiences", []):
                text = combined_exp_text.get(eid, "")
                if re.search(r"\b" + re.escape(kw) + r"\b", text, flags=re.I):
                    appeared = True
                    break
            # check projects
            if not appeared:
                for pid in loc.get("projects", []):
                    text = combined_proj_text.get(pid, "")
                    if re.search(r"\b" + re.escape(kw) + r"\b", text, flags=re.I):
                        appeared = True
                        break
            # check skills_section
            if not appeared and loc.get("skills_section"):
                if kw in skills_section:
                    appeared = True
            if not appeared:
                return False, f"Must keyword '{kw}' not found in any allowed location."

        # Final pass OK
        return True, None