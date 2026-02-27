import os
import json
import time
from pathlib import Path

from src.jd_parser import JDParser
from src.ats_parser import ATSParser
from src.keyword_ranker import KeywordRanker
from src.loader import ResumeLoader
from src.matcher import ResumeMatcher
from src.model_runner import ResumeRewriter


def main():
    # Record start time for performance tracking
    start_time = time.time()

    print("\n=== Resume Optimization Pipeline Started ===\n")

    # Load and validate job description file
    jd_path = Path("data/JD.txt")

    if not jd_path.exists():
        raise FileNotFoundError("data/JD.txt not found")

    jd_text = jd_path.read_text(encoding="utf-8")

    print("JD Loaded")

    # Parse JD to extract core skills/candidates
    jd_parser = JDParser()
    core_candidates = jd_parser.parse(str(jd_path))["candidates"]

    print(f" Core skills extracted: {len(core_candidates)}")

    # Parse JD for ATS-compatible keywords
    ats_parser = ATSParser()
    ats_candidates = ats_parser.parse(str(jd_path))

    print(f" ATS keywords extracted: {len(ats_candidates)}")

    # Rank keywords by relevance and importance
    ranker = KeywordRanker()
    ranked_keywords = ranker.rank(jd_text, core_candidates, ats_candidates)

    print(f" Keywords ranked: {len(ranked_keywords)}")

    # Load resume data from JSON file
    loader = ResumeLoader("data/resume.json")
    resume_data = loader.load()

    print(" Resume loaded and normalized")

    # Create decision plan by matching resume against ranked keywords
    matcher = ResumeMatcher()
    decision_plan = matcher.create_decision_plan(ranked_keywords, resume_data)

    print(" Decision plan created")

    # Save decision plan to output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "decision_plan.json", "w", encoding="utf-8") as f:
        json.dump(decision_plan, f, indent=2)

    print(" decision_plan.json saved")

    # Run LLM-based resume rewriting using GPU
    print("\n Running LLM rewrite (GPU)...")

    model_runner = ResumeRewriter()
    rewritten_resume = model_runner.rewrite_resume(decision_plan, resume_data)

    print(" LLM rewrite complete")

    # Save rewritten resume to output directory
    with open(output_dir / "rewritten_resume.json", "w", encoding="utf-8") as f:
        json.dump(rewritten_resume, f, indent=2)

    print(" rewritten_resume.json saved")

    # Calculate and display total pipeline execution time
    end_time = time.time()
    print(f"\n=== Pipeline Completed in {round(end_time - start_time, 2)} seconds ===\n")


if __name__ == "__main__":
    main()


from src.latex_generator import LatexGenerator

# these line will only run if the above main() function completes successfully and generates rewritten_resume.json

latex = LatexGenerator()
tex_path = latex.generate_tex(rewritten_resume)
pdf_path = latex.compile_pdf(tex_path)

print("✔ resume.tex generated")
print("✔ resume.pdf generated")    