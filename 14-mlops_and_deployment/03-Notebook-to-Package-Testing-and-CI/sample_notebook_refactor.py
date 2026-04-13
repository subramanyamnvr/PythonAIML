from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_package.features import bucket_age, build_feature_row, normalize_text  # noqa: E402


def main() -> None:
    row = build_feature_row(
        name="  Alice Johnson  ",
        age=31,
        monthly_spend=420.0,
        country="in",
    )
    print("Notebook-friendly package output")
    print("- normalized_name =", normalize_text("  Alice Johnson  "))
    print("- age_bucket      =", bucket_age(31))
    print("- feature_row     =", row)


if __name__ == "__main__":
    main()
