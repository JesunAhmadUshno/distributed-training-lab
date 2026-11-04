"""Scaffold smoke test. Checks that the scaffold files exist. Nothing more."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
need = ["README.md", "SPEC.md", "DOSSIER.md", "NOVELTY.md", "LICENSE",
        "level-1-learn/README.md", "level-2-build/README.md", "level-3-deep/README.md"]
missing = [p for p in need if not os.path.exists(os.path.join(ROOT, p))]
assert not missing, "missing scaffold files: " + ", ".join(missing)
print("scaffold smoke test: PASS,", len(need), "files present")
