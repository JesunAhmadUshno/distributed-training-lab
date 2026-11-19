"""Compare the 2026-11-17 two-node re-execution against the committed record (decision 0004).

Uses the repo's own analysis.py (link_fit, cell_stat, predict) unchanged.
Inputs (all read-only): ../results_deep.json, ../predictions.json,
../runs/main.jsonl (committed lines), ../runs/rerun-2026-11-17.jsonl
(fresh single-node arm, decision 0003), ../runs/twonode-2026-11-17.jsonl
(fresh E1, E2, E4), ../runs/probe-2026-11-17.jsonl (the E3 two-node grid,
which decision 0004's probe rule routed to scratch because node 0 is
co-tenanted by the finetune confirmatory run).

Outputs: ../runs/twonode-2026-11-17-comparison.json (per-cell tables,
link fit, E1 and E4 hash tables, prediction checks) and
../runs/twonode-2026-11-17-rescore.json (the hypotheses that re-score
on uncontended evidence: H1, H2, H7; H3 to H6 are named as not
re-scored, with the reason). Printed tables mirror the files.
"""

import json
import statistics
import sys

sys.path.insert(0, ".")
import analysis

committed_res = json.load(open("../results_deep.json"))
locked_pred = json.load(open("../predictions.json"))
main_rows = analysis.load("../runs/main.jsonl")
fresh_single_rows = [r for r in analysis.load("../runs/rerun-2026-11-17.jsonl") if r["exp"] == "E3"]
fresh_rows = analysis.load("../runs/twonode-2026-11-17.jsonl")
scratch_rows = [r for r in analysis.load("../runs/probe-2026-11-17.jsonl") if r["exp"] == "E3"]

# ---- link fit (H2) ----
fresh_fit = analysis.link_fit(fresh_rows)
committed_fit = locked_pred["link"]
h2_pass = fresh_fit["r2"] >= 0.99 and 80 <= fresh_fit["beta_MBps"] <= 125

# ---- E1 (H1) ----
def e1_table(rows):
    out = {}
    for r in rows:
        if r["exp"] != "E1":
            continue
        e = out.setdefault(r["mode"], {"launches": 0, "max_param_diff": 0.0, "hashes": set()})
        e["launches"] += 1
        e["max_param_diff"] = max(e["max_param_diff"], r["max_param_diff_vs_single"])
        e["hashes"].add(r["param_sha256"])
    for e in out.values():
        e["hashes"] = sorted(e["hashes"])
    return out

fresh_e1, committed_e1 = e1_table(fresh_rows), e1_table(main_rows)
h1_pass = (
    all(e["max_param_diff"] < 1e-5 for e in fresh_e1.values())
    and len(fresh_e1.get("ddp", {}).get("hashes", [])) == 1
    and fresh_e1.get("ddp", {}).get("launches", 0) >= 2
)

# ---- E3 cells: fresh single (0003) and scratch two-node vs committed ----
def cell_table(rows, modes):
    stats = {}
    for r in rows:
        if r["exp"] == "E3" and r["mode"] in modes:
            stats.setdefault((r["mode"], r["size"], r["accum"]), []).append(r)
    out = {}
    for (mode, size, accum), runs in sorted(stats.items()):
        v, ci, _ = analysis.cell_stat(runs)
        c = committed_res["cells"][f"{mode}-{size}-{accum}"]
        out[f"{mode}-{size}-{accum}"] = {
            "fresh_step_s": v,
            "fresh_ci": ci,
            "reps": len(runs),
            "committed_step_s": c["step_s"],
            "rel_diff_vs_committed": (v - c["step_s"]) / c["step_s"],
            "fresh_peak_mib": max(max(r["peak_mib_per_rank"]) for r in runs),
            "committed_peak_mib": c["peak_mib"],
            "fresh_loss_last_mean": statistics.fmean(r["loss_last"] for r in runs),
            "committed_loss_last_mean": c["loss_last"],
        }
    return out

two_cells = cell_table(scratch_rows, ("ddp", "ddp_bf16", "fsdp"))

# ---- prediction checks ----
combined = fresh_rows + fresh_single_rows
fresh_pred = analysis.predict(combined)
pred_checks = {}
for name, p in fresh_pred["cells"].items():
    entry = {
        "fresh_pred_t_ddp": p["pred_t_ddp"],
        "locked_pred_t_ddp": locked_pred["cells"][name]["pred_t_ddp"],
        "committed_t_ddp": committed_res["cells"][f"ddp-{name}"]["step_s"],
    }
    entry["fresh_pred_vs_committed_err"] = (p["pred_t_ddp"] - entry["committed_t_ddp"]) / entry["committed_t_ddp"]
    entry["locked_pred_vs_committed_err"] = (entry["locked_pred_t_ddp"] - entry["committed_t_ddp"]) / entry["committed_t_ddp"]
    sc = two_cells.get(f"ddp-{name}")
    if sc:
        entry["scratch_t_ddp"] = sc["fresh_step_s"]
        entry["fresh_pred_vs_scratch_err"] = (p["pred_t_ddp"] - sc["fresh_step_s"]) / sc["fresh_step_s"]
        entry["locked_pred_vs_scratch_err"] = (entry["locked_pred_t_ddp"] - sc["fresh_step_s"]) / sc["fresh_step_s"]
    pred_checks[name] = entry

def maxerr(checks, key):
    vals = [abs(c[key]) for c in checks.values() if key in c]
    return max(vals) if vals else None

# ---- E4 (H7) ----
def e4_table(rows):
    return {
        str(r["fault_step"]): {
            "restart_count": r["restart_count"],
            "resumed_from_step": r["resumed_from_step"],
            "param_sha256": r["param_sha256"],
        }
        for r in rows if r["exp"] == "E4"
    }

fresh_e4, committed_e4 = e4_table(fresh_rows), e4_table(main_rows)
clean_hashes = {v["param_sha256"] for k, v in fresh_e4.items() if k == "-1"}
h7_pass = (
    len([k for k in fresh_e4 if k != "-1"]) >= 3
    and len(clean_hashes) == 1
    and all(v["restart_count"] >= 1 for k, v in fresh_e4.items() if k != "-1")
    and all(v["param_sha256"] in clean_hashes for k, v in fresh_e4.items() if k != "-1")
)

comparison = {
    "decision": "0004",
    "contention_note": (
        "Node 0 (spark-6aa3) was co-tenanted all night by the finetune-lab confirmatory run "
        "(about 92 GB held, 93-96% GPU utilisation in samples). Decision 0004's probe rule "
        "routed the timed E3 two-node grid to the scratch file runs/probe-2026-11-17.jsonl; "
        "its cells are reported here as contention-affected measurements, not as a clean "
        "re-execution. E1, E2 and E4 are exactness or link measurements and stand as a clean "
        "re-execution."
    ),
    "link_fit": {"fresh": fresh_fit, "committed": committed_fit, "h2_pass_fresh": h2_pass},
    "e1": {"fresh": fresh_e1, "committed": committed_e1, "h1_pass_fresh": h1_pass},
    "e3_twonode_scratch_cells": two_cells,
    "prediction_checks": pred_checks,
    "h3_style_max_abs_errors": {
        "fresh_pred_vs_committed_t_ddp": maxerr(pred_checks, "fresh_pred_vs_committed_err"),
        "locked_pred_vs_committed_t_ddp": maxerr(pred_checks, "locked_pred_vs_committed_err"),
        "fresh_pred_vs_scratch_t_ddp_contention_affected": maxerr(pred_checks, "fresh_pred_vs_scratch_err"),
        "locked_pred_vs_scratch_t_ddp_contention_affected": maxerr(pred_checks, "locked_pred_vs_scratch_err"),
        "committed_h3_max_abs_rel_error": committed_res["hypotheses"]["H3"]["max_abs_rel_error"],
    },
    "e4": {"fresh": fresh_e4, "committed": committed_e4, "h7_pass_fresh": h7_pass},
}
with open("../runs/twonode-2026-11-17-comparison.json", "w") as f:
    json.dump(comparison, f, indent=1, sort_keys=True)
    f.write("\n")

rescore = {
    "decision": "0004",
    "H1": {"pass": h1_pass, "detail": fresh_e1},
    "H2": {"pass": h2_pass, "detail": fresh_fit},
    "H3": {"pass": None, "detail": "Not re-scored: the fresh two-node E3 timings are contention-affected (decision 0004 probe rule). Prediction checks against committed and scratch times are in twonode-2026-11-17-comparison.json."},
    "H4": {"pass": None, "detail": "Not re-scored: same reason as H3."},
    "H5": {"pass": None, "detail": "Not re-scored: same reason as H3."},
    "H6": {"pass": None, "detail": "Not re-scored: same reason as H3 (the fsdp L k=1 cell is in the scratch grid)."},
    "H7": {"pass": h7_pass, "detail": fresh_e4},
}
with open("../runs/twonode-2026-11-17-rescore.json", "w") as f:
    json.dump(rescore, f, indent=1, sort_keys=True)
    f.write("\n")

print(f"LINK fresh beta {fresh_fit['beta_MBps']:.4f} (committed {committed_fit['beta_MBps']:.4f}), "
      f"alpha {fresh_fit['alpha_s'] * 1000:.3f} ms (committed {committed_fit['alpha_s'] * 1000:.3f} ms), "
      f"R2 {fresh_fit['r2']:.8f}; H2 fresh pass: {h2_pass}")
print(f"H1 fresh pass: {h1_pass}; E1 tables: fresh {fresh_e1} committed {committed_e1}")
for name, c in two_cells.items():
    print(f"{name:16s} scratch={c['fresh_step_s']:.4f} committed={c['committed_step_s']:.4f} diff={c['rel_diff_vs_committed']:+.2%}")
print("H3-style max |err|:", comparison["h3_style_max_abs_errors"])
print(f"H7 fresh pass: {h7_pass}; fresh E4 {fresh_e4}")
