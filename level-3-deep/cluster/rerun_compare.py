"""Compare the 2026-11-17 single-node re-execution against the committed record.

Uses the repo's own analysis.py (cell_stat, comm_time) unchanged. Inputs:
../results_deep.json and ../predictions.json (committed, read-only) and the
fresh ../runs/rerun-2026-11-17.jsonl. Output: ../runs/rerun-2026-11-17-comparison.json
plus a printed table. See docs/decisions/0003.
"""

import json
import statistics
import sys

sys.path.insert(0, ".")
import analysis

committed = json.load(open("../results_deep.json"))
pred = json.load(open("../predictions.json"))
fit = pred["link"]
fresh_rows = analysis.load("../runs/rerun-2026-11-17.jsonl")
fresh_single = [r for r in fresh_rows if r["exp"] == "E3" and r["mode"] == "single"]

cells, h3_errs, h4_sides = {}, {}, []
for key in sorted({analysis.cell_key(r) for r in fresh_single}):
    runs = [r for r in fresh_single if analysis.cell_key(r) == key]
    v, ci, _ = analysis.cell_stat(runs)
    name = f"{key[0]}-{key[1]}"
    c = committed["cells"][f"single-{name}"]
    p = pred["cells"][name]
    d = committed["cells"][f"ddp-{name}"]
    comm = analysis.comm_time(runs[0]["params"], fit)
    pred_t_ddp = v / 2 + comm
    err = (pred_t_ddp - d["step_s"]) / d["step_s"]
    h3_errs[name] = err
    h4_sides.append((p["pred_speedup"] > 1) == ((v / d["step_s"]) > 1))
    cells[name] = {
        "fresh_t_single": v,
        "fresh_ci": ci,
        "committed_t_single": c["step_s"],
        "rel_diff_vs_committed": (v - c["step_s"]) / c["step_s"],
        "fresh_peak_mib": max(max(r["peak_mib_per_rank"]) for r in runs),
        "committed_peak_mib": c["peak_mib"],
        "recomputed_pred_t_ddp": pred_t_ddp,
        "committed_measured_t_ddp": d["step_s"],
        "h3_rel_error": err,
        "fresh_loss_last_mean": statistics.fmean(r["loss_last"] for r in runs),
        "committed_loss_last_mean": c["loss_last"],
        "env": {k: runs[0][k] for k in ("hosts", "torch", "cuda", "nccl", "gpu", "link")},
    }

out = {
    "link_fit_used": fit,
    "cells": cells,
    "h3_max_abs_rel_error": max(abs(e) for e in h3_errs.values()),
    "h3_within_25pct_all": all(abs(e) <= 0.25 for e in h3_errs.values()),
    "h4_sides_correct": sum(h4_sides),
    "h4_cells": len(h4_sides),
    "committed_h3_max_abs_rel_error": committed["hypotheses"]["H3"]["max_abs_rel_error"],
    "committed_h4_sides": f"{committed['hypotheses']['H4']['side_correct']}/{committed['hypotheses']['H4']['cells']}",
}
with open("../runs/rerun-2026-11-17-comparison.json", "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
    f.write("\n")
for name, c in cells.items():
    print(
        f"{name:6s} fresh={c['fresh_t_single']:.6f} committed={c['committed_t_single']:.6f} "
        f"diff={c['rel_diff_vs_committed']:+.2%} H3err={c['h3_rel_error']:+.2%}"
    )
print(f"H3 max |err| fresh-singles: {out['h3_max_abs_rel_error']:.4f} (committed: {out['committed_h3_max_abs_rel_error']:.4f})")
print(f"H4 sides: {out['h4_sides_correct']}/{out['h4_cells']} (committed: {out['committed_h4_sides']})")
