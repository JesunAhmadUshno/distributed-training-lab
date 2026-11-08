"""Portfolio standard checks: the lock holds, required files exist, no public grade."""

import glob
import hashlib
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# SHA-256 of level-3-deep/PREREG.md as committed in f617be6 (tag proposal-v1), before any Deep run result.
LOCKED_SHA256 = "fa16e55a27d0b230958203423f69d68a10531ba2c2e4cb8747b7426b5abe46a3"


class Layout(unittest.TestCase):
    def test_preregistration_is_unchanged_since_the_lock(self):
        with open(os.path.join(ROOT, "level-3-deep", "PREREG.md"), "rb") as f:
            self.assertEqual(
                hashlib.sha256(f.read()).hexdigest(),
                LOCKED_SHA256,
                "PREREG.md changed after its lock; record an amendment in docs/decisions/ instead",
            )

    def test_required_files_exist(self):
        need = [
            "README.md",
            "PROPOSAL.md",
            "REPORT.md",
            "CHANGELOG.md",
            "CITATION.cff",
            "SECURITY.md",
            "LICENSE",
            "level-3-deep/PREREG.md",
            "level-3-deep/predictions.json",
            "level-3-deep/results_deep.json",
            "level-3-deep/runs/main.jsonl",
            "level-3-deep/runs/RUNLOG.md",
            "docs/decisions/0001-deep-on-two-dgx-spark-nodes.md",
            "docs/decisions/0002-scope-of-the-lock.md",
        ]
        missing = [p for p in need if not os.path.exists(os.path.join(ROOT, p))]
        self.assertFalse(missing, "missing: " + ", ".join(missing))

    def test_no_public_grade(self):
        pages = glob.glob(os.path.join(ROOT, "**", "*.md"), recursive=True) + glob.glob(
            os.path.join(ROOT, "site", "src", "*.jsx")
        )
        for path in pages:
            if "node_modules" in path:
                continue
            with open(path, encoding="utf-8") as f:
                text = f.read()
            for mark in ("of 10,000", "/10000", "Grade:", "/1000 |"):
                self.assertNotIn(mark, text, os.path.relpath(path, ROOT) + " carries a grade; grades are internal")


if __name__ == "__main__":
    unittest.main()
