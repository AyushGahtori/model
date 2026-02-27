import subprocess
from pathlib import Path


class LatexGenerator:

    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def _escape_latex(self, text):
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def generate_tex(self, rewritten_resume, filename="resume.tex"):

        tex_path = self.output_dir / filename

        with open(tex_path, "w", encoding="utf-8") as f:

            f.write(r"""\documentclass[10pt,a4paper]{article}
\usepackage[a4paper, margin=0.7in]{geometry}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{hyperref}
\usepackage{parskip}

\setlist[itemize]{noitemsep, topsep=0pt}
\titleformat{\section}{\large\bfseries}{}{0em}{}

\begin{document}

\begin{center}
    {\Huge \textbf{Your Name}} \\
    \small Email: your@email.com \quad | \quad Phone: +00 000000000
\end{center}

\vspace{0.2cm}

""")

            # EXPERIENCE
            f.write(r"\section*{Experience}" + "\n")

            for exp in rewritten_resume["experience"]:
                f.write(r"\textbf{" + exp["id"] + "}\\" + "\n")
                f.write(r"\begin{itemize}" + "\n")
                for bullet in exp["bullets"]:
                    escaped = self._escape_latex(bullet)
                    f.write(r"\item " + escaped + "\n")
                f.write(r"\end{itemize}" + "\n")

            # PROJECTS
            if rewritten_resume["projects"]:
                f.write(r"\section*{Projects}" + "\n")
                for proj in rewritten_resume["projects"]:
                    f.write(r"\textbf{" + proj["id"] + "}\\" + "\n")
                    f.write(r"\begin{itemize}" + "\n")
                    for bullet in proj["bullets"]:
                        escaped = self._escape_latex(bullet)
                        f.write(r"\item " + escaped + "\n")
                    f.write(r"\end{itemize}" + "\n")

            # SKILLS
            if rewritten_resume["skills_section"]:
                f.write(r"\section*{Skills}" + "\n")
                skills = ", ".join(rewritten_resume["skills_section"])
                skills = self._escape_latex(skills)
                f.write(skills + "\n")

            f.write(r"\end{document}")

        return tex_path

    def compile_pdf(self, tex_path):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=self.output_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        pdf_path = tex_path.with_suffix(".pdf")
        return pdf_path