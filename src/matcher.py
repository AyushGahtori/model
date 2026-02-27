# src/matcher.py
import json
import re
from collections import defaultdict
from typing import List, Dict, Any


class ResumeMatcher:
    def __init__(self):
        pass

    # ----------------------------------------
    # Select Top 50% Keywords
    # ----------------------------------------
    def _split_keywords(self, ranked_keywords: List[Dict[str, Any]]):
        """
        Splits ranked keywords into must-have and optional categories.
        Keywords marked with top50=True are considered critical (must-have),
        while others are optional for resume enhancement.
        """
        must = []
        optional = []

        # Iterate through each keyword and categorize by top50 flag
        for kw in ranked_keywords:
            if kw.get("top50"):
                must.append(kw["keyword"])
            else:
                optional.append(kw["keyword"])

        return must, optional

    # ----------------------------------------
    # Select Top 2 Projects by Overlap
    # ----------------------------------------
    def _select_projects(self, must_keywords: List[str], resume_data: Dict[str, Any]):
        """
        Selects the top 2 projects that best match the must-have keywords.
        Projects are scored by how many must-have keywords they contain.
        Sorting ensures deterministic selection (higher overlap first, then by ID).
        """
        projects = resume_data["resume"].get("projects", [])
        project_scores = []

        # Score each project by counting overlapping must-have keywords
        for proj in projects:
            overlap = 0
            for skill in must_keywords:
                if skill in proj.get("skills", []):
                    overlap += 1
            project_scores.append((proj["id"], overlap))

        # Sort descending by overlap, deterministic tie-break by id for consistency
        project_scores.sort(key=lambda x: (-x[1], x[0]))

        # Return IDs of top 2 projects
        selected = [p[0] for p in project_scores[:2]]
        return selected

    # ----------------------------------------
    # Improved Experience Bullet Selection
    # ----------------------------------------
    def _select_experience_bullets(self, must_keywords: List[str], resume_data: Dict[str, Any],
                                   xp1_min=3, xp1_max=4, xp2_min=2, xp2_max=3,
                                   allow_extra_for_coverage=True):
        """
        Selects optimal experience bullet points that maximize keyword coverage.
        - xp1 (most recent): target 3-4 bullets
        - xp2 (second most recent): target 2-3 bullets
        - Extras added if must-keywords remain uncovered
        - Ensures xp1_count >= xp2_count for resume emphasis
        """
        experiences = resume_data["resume"].get("experience", [])
        must_lc = [k.lower() for k in must_keywords]

        # -------- HELPER: Check which keywords a bullet covers --------
        def covered_by_bullet(bullet_text):
            """Identifies which must-have keywords appear in a bullet using word boundaries."""
            txt = (bullet_text or "").lower()
            covered = set()
            for kw in must_lc:
                if re.search(r"\b" + re.escape(kw) + r"\b", txt):
                    covered.add(kw)
            return covered

        # -------- BUILD BULLET INFO STRUCTURES --------
        # For each experience entry, analyze all bullets and score them
        exp_bullets_info = []
        for exp in experiences:
            bullets = exp.get("text", [])
            infos = []
            # Create detailed info for each bullet (index, text, covered keywords, score)
            for i, b in enumerate(bullets):
                covered = covered_by_bullet(b)
                score = len(covered)  # Score = number of keywords covered
                infos.append({"index": i, "text": b, "covered": covered, "score": score})
            exp_bullets_info.append({"id": exp["id"], "infos": infos, "skills": exp.get("skills", [])})

        # Track globally covered keywords across all selections
        global_covered = set()

        # -------- SELECTION LOGIC FOR ONE EXPERIENCE --------
        def select_for_exp(exp_info, target_min, target_max):
            """
            Greedily selects bullets from one experience entry to meet targets.
            - First reaches target_min by prioritizing new keyword coverage
            - Then adds up to target_max to maximize coverage
            - Uses score and index as tie-breakers for determinism
            """
            infos = exp_info["infos"].copy()
            # Sort by score (descending) then index (ascending) for deterministic ordering
            candidates = sorted(infos, key=lambda x: (-x["score"], x["index"]))
            selected = []
            covered_so_far = set()

            # -------- PHASE 1: Meet minimum target --------
            # Select bullets until reaching target_min, prioritizing new keyword coverage
            while len(selected) < target_min and candidates:
                best = None
                best_new = -1
                for cand in candidates:
                    new_cov = len(cand["covered"] - covered_so_far)
                    if new_cov > best_new:
                        best_new = new_cov
                        best = cand
                    elif new_cov == best_new and best is not None:
                        # Tie-break: prefer higher score, then lower index
                        if cand["score"] > best["score"] or (cand["score"] == best["score"] and cand["index"] < best["index"]):
                            best = cand
                if best is None:
                    break
                selected.append(best)
                covered_so_far |= best["covered"]
                candidates.remove(best)

            # -------- PHASE 2: Add up to target_max --------
            # Add more bullets to improve coverage without exceeding target_max
            while len(selected) < target_max and candidates:
                best = None
                best_new = -1
                for cand in candidates:
                    new_cov = len(cand["covered"] - covered_so_far)
                    if new_cov > best_new:
                        best_new = new_cov
                        best = cand
                    elif best is not None and new_cov == best_new:
                        # Tie-break: prefer higher score, then lower index
                        if cand["score"] > best["score"] or (cand["score"] == best["score"] and cand["index"] < best["index"]):
                            best = cand
                if best is None:
                    break
                selected.append(best)
                covered_so_far |= best["covered"]
                candidates.remove(best)

            return selected, covered_so_far

        # -------- FIRST-PASS SELECTION: XP1 and XP2 --------
        # Apply selection strategy to first two experiences (most recent)
        exp_plan = {}
        if len(exp_bullets_info) >= 1:
            s1, c1 = select_for_exp(exp_bullets_info[0], xp1_min, xp1_max)
            exp_plan[exp_bullets_info[0]["id"]] = {"selected_infos": s1, "all_skills": exp_bullets_info[0]["skills"]}
            global_covered |= c1
        if len(exp_bullets_info) >= 2:
            s2, c2 = select_for_exp(exp_bullets_info[1], xp2_min, xp2_max)
            exp_plan[exp_bullets_info[1]["id"]] = {"selected_infos": s2, "all_skills": exp_bullets_info[1]["skills"]}
            global_covered |= c2

        # -------- ADDITIONAL EXPERIENCES --------
        # For any further experiences (older positions), pick up to 2 top bullets
        for info in exp_bullets_info[2:]:
            candidates = sorted(info["infos"], key=lambda x: (-x["score"], x["index"]))
            selected = candidates[:2]
            exp_plan[info["id"]] = {"selected_infos": selected, "all_skills": info["skills"]}
            for si in selected:
                global_covered |= si["covered"]

        # -------- IDENTIFY UNCOVERED KEYWORDS --------
        # Find keywords that haven't been covered by any selected bullets yet
        uncovered = set(must_lc) - global_covered

        # -------- EXTRA BULLETS FOR COVERAGE --------
        # If must-keywords remain uncovered and extras are allowed, add bullets that cover them
        if allow_extra_for_coverage and uncovered:
            def add_covering_bullets(exp_info, selected_infos):
                """
                Adds extra bullets from this experience that cover remaining uncovered keywords.
                Prioritizes bullets covering the most uncovered keywords.
                """
                nonlocal uncovered, global_covered
                candidates = [c for c in exp_info["infos"] if c not in selected_infos]
                # Sort by coverage of uncovered keywords, then by index
                candidates = sorted(candidates, key=lambda x: (-len(x["covered"] & uncovered), x["index"]))
                added = []
                for cand in candidates:
                    if len(cand["covered"] & uncovered) > 0:
                        added.append(cand)
                        uncovered -= (cand["covered"] & uncovered)
                        global_covered |= cand["covered"]
                return added

            # Try to cover remaining keywords using xp1, then xp2
            if len(exp_bullets_info) >= 1:
                added1 = add_covering_bullets(exp_bullets_info[0], exp_plan[exp_bullets_info[0]["id"]]["selected_infos"])
                exp_plan[exp_bullets_info[0]["id"]]["selected_infos"].extend(added1)
            if len(exp_bullets_info) >= 2 and uncovered:
                added2 = add_covering_bullets(exp_bullets_info[1], exp_plan[exp_bullets_info[1]["id"]]["selected_infos"])
                exp_plan[exp_bullets_info[1]["id"]]["selected_infos"].extend(added2)

        # -------- REBALANCE: ENSURE XP1 >= XP2 --------
        # Move one bullet from xp2 to xp1 if xp1 has fewer bullets (prioritize recent experience)
        if len(exp_bullets_info) >= 2:
            s1 = exp_plan[exp_bullets_info[0]["id"]]["selected_infos"]
            s2 = exp_plan[exp_bullets_info[1]["id"]]["selected_infos"]
            if len(s1) < len(s2):
                # Sort s2 by score and index to move lowest-value bullet to xp1
                s2_sorted = sorted(s2, key=lambda x: (x["score"], -x["index"]))
                for cand in s2_sorted:
                    s2.remove(cand)
                    s1.append(cand)
                    break
                exp_plan[exp_bullets_info[0]["id"]]["selected_infos"] = s1
                exp_plan[exp_bullets_info[1]["id"]]["selected_infos"] = s2

        # -------- CONVERT TO FINAL FORMAT --------
        # Transform selected_infos back to selected_bullets in original order
        final_plan = {}
        for info in exp_bullets_info:
            eid = info["id"]
            selected_infos = exp_plan.get(eid, {}).get("selected_infos", [])
            # Sort indices to preserve original bullet order
            selected_indices = sorted([si["index"] for si in selected_infos])
            bullets = [info["infos"][i]["text"] for i in selected_indices]
            final_plan[eid] = {"selected_bullets": bullets, "all_skills": exp_plan.get(eid, {}).get("all_skills", [])}

        return final_plan

    # ----------------------------------------
    # Skills Section Handling
    # ----------------------------------------
    def _compute_skills_section_additions(self, must_keywords: List[str], resume_data: Dict[str, Any]):
        """
        Identifies must-have keywords that don't appear in existing experience or projects.
        These are candidates to be added to the dedicated Skills section.
        """
        # Collect all skills already mentioned in experiences and projects
        existing_skills = set()
        for exp in resume_data["resume"].get("experience", []):
            existing_skills.update(exp.get("skills", []))
        for proj in resume_data["resume"].get("projects", []):
            existing_skills.update(proj.get("skills", []))

        # Find keywords not yet covered
        additions = []
        for skill in must_keywords:
            if skill not in existing_skills:
                additions.append(skill)
        return additions

    # ----------------------------------------
    # Main Function
    # ----------------------------------------
    def create_decision_plan(self, ranked_keywords: List[Dict[str, Any]], resume_data: Dict[str, Any]):
        """
        Main orchestration function that creates a comprehensive resume modification plan.
        Combines keyword classification, project selection, experience optimization, and skills additions.
        """
        # Step 1: Categorize keywords into must-have and optional
        must_keywords, optional_keywords = self._split_keywords(ranked_keywords)
        
        # Step 2: Select best 2 projects matching must-have keywords
        selected_projects = self._select_projects(must_keywords, resume_data)
        
        # Step 3: Optimize experience bullets to maximize keyword coverage
        experience_plan = self._select_experience_bullets(must_keywords, resume_data)
        
        # Step 4: Determine keywords to add to dedicated Skills section
        skills_section_additions = self._compute_skills_section_additions(must_keywords, resume_data)

        # Compile all decisions into a comprehensive plan
        decision_plan = {
            "must_include_keywords": must_keywords,
            "optional_keywords": optional_keywords,
            "selected_projects": selected_projects,
            "experience_plan": experience_plan,
            "skills_section_additions": skills_section_additions
        }
        return decision_plan


# -------------------------
# Quick CLI/test harness
# -------------------------
if __name__ == "__main__":
    # Try to import all required modules for the pipeline
    try:
        from src.jd_parser import JDParser
        from src.ats_parser import ATSParser
        from src.keyword_ranker import KeywordRanker
        from src.loader import ResumeLoader
    except Exception as e:
        print("Ensure jd_parser, ats_parser, keyword_ranker, and loader exist in src/:", e)
        raise SystemExit(1)

    # Load and parse job description
    jdfile = "data/JD.txt"
    resume_path = "data/resume.json"
    jd_text = open(jdfile, encoding="utf-8").read()

    # Extract keywords using three different parsing strategies
    core = JDParser().parse(jdfile)["candidates"]
    ats = ATSParser().parse(jdfile)
    ranked = KeywordRanker().rank(jd_text, core, ats)

    # Load resume data and create decision plan
    resume_data = ResumeLoader(resume_path).load()
    matcher = ResumeMatcher()
    decision = matcher.create_decision_plan(ranked, resume_data)

    # Output the experience selection plan for verification
    print(json.dumps(decision["experience_plan"], indent=2))