from __future__ import annotations

from pathlib import Path


SECTION_FILES = [
    Path("02_sql_and_stats_questions/question_bank.md"),
    Path("03_ml_questions/question_bank.md"),
    Path("04_dl_questions/question_bank.md"),
    Path("05_genai_questions/question_bank.md"),
    Path("06_system_design_questions/question_bank.md"),
    Path("07_case_studies/case_studies.md"),
    Path("08_revision_cheatsheets/revision_sheet.md"),
]


def build_pack(root: Path) -> str:
    chunks: list[str] = ["# Interview Prep Pack", ""]
    for relative_path in SECTION_FILES:
        path = root / relative_path
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8").strip())
            chunks.append("")
    return "\n".join(chunks).strip() + "\n"


def main() -> None:
    root = Path(__file__).resolve().parent
    output_path = root / "interview_pack.md"
    output_path.write_text(build_pack(root), encoding="utf-8")
    print(f"Created {output_path.name}")


if __name__ == "__main__":
    main()
