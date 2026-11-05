"""Write results_deep_prep.json deterministically from cpu_equivalence."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cpu_equivalence import build_report  # noqa: E402

if __name__ == "__main__":
    report = build_report()
    out = os.path.join(HERE, "results_deep_prep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {out} all_pass={report['all_pass']}")
