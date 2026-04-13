from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_package.features import bucket_age, build_feature_row, normalize_text


class FeatureTests(unittest.TestCase):
    def test_normalize_text_collapses_spacing(self) -> None:
        self.assertEqual(normalize_text("  Hello   World "), "hello world")

    def test_bucket_age(self) -> None:
        self.assertEqual(bucket_age(21), "young")
        self.assertEqual(bucket_age(33), "mid")
        self.assertEqual(bucket_age(55), "senior")

    def test_build_feature_row_marks_high_value(self) -> None:
        row = build_feature_row("Jane Doe", 29, 300.0, "us")
        self.assertEqual(row["country"], "US")
        self.assertEqual(row["is_high_value"], 1)
        self.assertEqual(row["age_bucket"], "mid")


if __name__ == "__main__":
    unittest.main()
