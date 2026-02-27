# AI Resume Optimizer (GPU-Accelerated, Deterministic, ATS-Aware)

An end-to-end intelligent resume optimization system that:

- Parses job descriptions
- Extracts and ranks technical + ATS keywords
- Deterministically selects optimal resume content
- Uses a constrained LLM (Phi-3 Mini) to rewrite bullets
- Validates output to prevent hallucination
- Generates a professional PDF resume using LaTeX

This project was designed to be:

-  Fast (<20 seconds per JD)
-  Deterministic
-  Hallucination-resistant
-  GPU-only optimized
-  One-page ATS-friendly

---

##  Why This Project Is Non-Trivial

This is not just a “rewrite my resume” script.

The system includes:

- Custom keyword extraction (no LLM)
- Weighted importance ranking
- Top-50% enforcement logic
- Resume bullet selection constraints (xp1 vs xp2 logic)
- Project selection by skill overlap
- Strict prompt engineering
- JSON schema enforcement
- Output validation + retry logic
- Quantized 8-bit GPU inference
- LaTeX PDF generation pipeline

The LLM is never allowed to:
- Invent experience
- Invent projects
- Add fake achievements
- Add unsupported skills

All rewriting happens under deterministic constraints.

---

##  System Architecture

JD.txt
↓
Core Skill Extractor
↓
ATS Keyword Extractor
↓
Keyword Ranker
↓
Resume Loader & Normalizer
↓
Decision Matcher (Top 50% enforcement)
↓
Strict Prompt Builder
↓
Phi-3 Mini (GPU, 8-bit quantized)
↓
Validation Layer
↓
LaTeX Generator
↓
resume.pdf


---

##  Hardware Requirements

Tested on:

- Intel i5-12450H
- 16GB RAM
- RTX 4050 6GB (100W)
- CUDA 12.1

Model runs fully on GPU using 8-bit quantization.

---

##  Installation

###  Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt


project/
│
├── data/
│   ├── JD.txt
│   └── resume.json
│
├── output/
│
├── src/
│   ├── jd_parser.py
│   ├── ats_parser.py
│   ├── keyword_ranker.py
│   ├── loader.py
│   ├── matcher.py
│   ├── rewrite_prompt.py
│   ├── model_runner.py
│   ├── latex_generator.py
│   └── cli.py
│
├── requirements.txt
└── README.md

