"""Review check: re-derive the Deep headline numbers from the raw run lines.

Written for the 2026-11-08 review (REPORT.md, Review). It does not import
level-3-deep/cluster/analysis.py: it recomputes each number from
level-3-deep/runs/main.jsonl on its own, using the cell statistic stated in
PREREG.md (mean of the per-repeat median step times), so a bug in the
analysis code would show up as a disagreement here.
"""

import json
import os
import unittest
from statistics import fmean, median

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEEP = os.path.join(ROOT, "level-3-deep")


def rows():
    with open(os.path.join(DEEP, "runs", "main.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def cells(mode):
    out = {}
    for r in rows():
        if r["exp"] == "E3" and r["mode"] == mode:
            out.setdefault((r["size"], r["accum"]), []).append(median(r["step_seconds"]))
    return {k: fmean(v) for k, v in out.items()}


class DeepHeadline(unittest.TestCase):
    def test_two_nodes_slower_in_9_of_12_cells_down_to_0_069x(self):
        single, ddp = cells("single"), cells("ddp")
        self.assertEqual(len(ddp), 12)
        speed = {k: single[k] / ddp[k] for k in ddp}
        self.assertEqual(sum(s < 1 for s in speed.values()), 9)
        self.assertEqual(round(min(speed.values()), 3), 0.069)
        self.assertEqual(min(speed, key=speed.get), ("L", 1))
        self.assertTrue(all(speed[(s, 64)] > 1 for s in "SML"))

    def test_link_model_predicted_every_cell_within_2_7_percent(self):
        with open(os.path.join(DEEP, "predictions.json")) as f:
            pred = json.load(f)["cells"]
        ddp = cells("ddp")
        worst = max(
            abs(p["pred_t_ddp"] - ddp[(p["size"], p["accum"])]) / ddp[(p["size"], p["accum"])] for p in pred.values()
        )
        self.assertEqual(len(pred), 12)
        self.assertEqual(round(worst * 100, 1), 2.7)

    def test_link_fit_beta_117_mb_per_s(self):
        pts = [(r["bytes"], t) for r in rows() if r["exp"] == "E2" and r["op"] == "all_reduce" for t in r["seconds"]]
        mx, my = fmean(x for x, _ in pts), fmean(y for _, y in pts)
        slope = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
        self.assertEqual(len(pts), 126)
        self.assertEqual(round(1 / slope / 1e6, 1), 117.4)

    def test_the_two_misses(self):
        ddp, bf16 = cells("ddp"), cells("ddp_bf16")
        self.assertEqual(round((1 - bf16[("S", 16)] / ddp[("S", 16)]) * 100, 1), 29.4)  # H5 line is 30%
        peak = {
            m: max(
                max(r["peak_mib_per_rank"])
                for r in rows()
                if r["exp"] == "E3" and r["mode"] == m and r["size"] == "L" and r["accum"] == 1
            )
            for m in ("ddp", "fsdp")
        }
        self.assertEqual(round(peak["fsdp"] / peak["ddp"] * 100, 1), 65.3)  # H6 line is 65%


if __name__ == "__main__":
    unittest.main()
