"""Compare the 2026-11-19 clean two-node re-execution (decision 0005).

Uses the repo's own analysis.py (link_fit, cell_stat, predict, report)
unchanged. Inputs (all read-only): ../results_deep.json,
../predictions.json, ../runs/main.jsonl (committed lines),
../runs/rerun-2026-11-17.jsonl (fresh single-node arm, decision 0003),
../runs/twonode-2026-11-17.jsonl (fresh E1 and E2, decision 0004),
../runs/twonode-2026-11-19.jsonl (fresh E3 two-node grid and E4,
decision 0005, run on a quiet node 0 after the probe landed within 10%
on both cells: ddp-L-1 +0.00%, ddp-S-64 -0.46%).

Outputs: ../runs/twonode-2026-11-19-comparison.json (per-cell tables,
link fit, E1 and E4 hash tables, prediction checks) and
../runs/twonode-2026-11-19-rescore.json (the full fresh H1 to H7
re-score from analysis.report on the combined fresh rows). No frozen
verdict changes whatever the fresh numbers say.
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
rows_1117 = analysis.load("../runs/twonode-2026-11-17.jsonl")
rows_1119 = analysis.load("../runs/twonode-2026-11-19.jsonl")
fresh_rows = rows_1117 + rows_1119
combined = fresh_rows + fresh_single_rows

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

# ---- E3 two-node cells: clean 2026-11-19 grid vs committed ----
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

two_cells = cell_table(rows_1119, ("ddp", "ddp_bf16", "fsdp"))

# ---- prediction checks ----
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
        entry["clean_t_ddp"] = sc["fresh_step_s"]
        entry["fresh_pred_vs_clean_err"] = (p["pred_t_ddp"] - sc["fresh_step_s"]) / sc["fresh_step_s"]
        entry["locked_pred_vs_clean_err"] = (entry["locked_pred_t_ddp"] - sc["fresh_step_s"]) / sc["fresh_step_s"]
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

# ---- full fresh re-score with the repo's own logic ----
fresh_report = analysis.report(combined, fresh_pred)
H = fresh_report["hypotheses"]

comparison = {
    "decision": "0005",
    "clean_note": (
        "Node 0 (spark-6aa3) was quiet for this run: the finetune confirmatory run was "
        "found already dead (since 2026-11-17 21:59 EST) and stayed stopped, load average "
        "0.00, GPU idle on both nodes at start. Decision 0005's probe (same rule as 0004) "
        "landed within 10% on both cells (ddp-L-1 +0.00%, ddp-S-64 -0.46%), so the timed "
        "E3 grid ran into the record file runs/twonode-2026-11-19.jsonl. The contended "
        "2026-11-17 scratch cells in runs/probe-2026-11-17.jsonl stay labelled scratch "
        "and are not scored here. E1 and E2 are the 2026-11-17 fresh records (not re-run)."
    ),
    "link_fit": {"fresh": fresh_fit, "committed": committed_fit, "h2_pass_fresh": h2_pass},
    "e1": {"fresh": fresh_e1, "committed": committed_e1, "h1_pass_fresh": h1_pass},
    "e3_twonode_clean_cells": two_cells,
    "prediction_checks": pred_checks,
    "h3_style_max_abs_errors": {
        "fresh_pred_vs_committed_t_ddp": maxerr(pred_checks, "fresh_pred_vs_committed_err"),
        "locked_pred_vs_committed_t_ddp": maxerr(pred_checks, "locked_pred_vs_committed_err"),
        "fresh_pred_vs_clean_t_ddp": maxerr(pred_checks, "fresh_pred_vs_clean_err"),
        "locked_pred_vs_clean_t_ddp": maxerr(pred_checks, "locked_pred_vs_clean_err"),
        "committed_h3_max_abs_rel_error": committed_res["hypotheses"]["H3"]["max_abs_rel_error"],
    },
    "e4": {"fresh": fresh_e4, "committed": committed_e4, "h7_pass_fresh": h7_pass},
}
with open("../runs/twonode-2026-11-19-comparison.json", "w") as f:
    json.dump(comparison, f, indent=1, sort_keys=True)
    f.write("\n")

rescore = {
    "decision": "0005",
    "source": "analysis.report on combined fresh rows (E1/E2 2026-11-17, single-node E3 2026-11-17, two-node E3 and E4 2026-11-19), predictions from analysis.predict on the same rows",
    "H1": H.get("H1"),
    "H2": H.get("H2"),
    "H3": H.get("H3"),
    "H4": H.get("H4"),
    "H5": H.get("H5"),
    "H6": H.get("H6"),
    "H7": H.get("H7"),
    "cross_checks": {"h1_pass_explicit": h1_pass, "h2_pass_explicit": h2_pass, "h7_pass_explicit": h7_pass},
}
with open("../runs/twonode-2026-11-19-rescore.json", "w") as f:
    json.dump(rescore, f, indent=1, sort_keys=True)
    f.write("\n")

print(f"LINK fresh beta {fresh_fit['beta_MBps']:.4f} (committed {committed_fit['beta_MBps']:.4f}), "
      f"alpha {fresh_fit['alpha_s'] * 1000:.3f} ms (committed {committed_fit['alpha_s'] * 1000:.3f} ms), "
      f"R2 {fresh_fit['r2']:.8f}; H2 fresh pass: {h2_pass}")
print(f"H1 fresh pass: {h1_pass}")
for name, c in two_cells.items():
    print(f"{name:16s} clean={c['fresh_step_s']:.4f} committed={c['committed_step_s']:.4f} diff={c['rel_diff_vs_committed']:+.2%} reps={c['reps']}")
print("H3-style max |err|:", comparison["h3_style_max_abs_errors"])
print("RESCORE:", {k: (v.get('pass') if isinstance(v, dict) else v) for k, v in rescore.items() if k.startswith('H')})
print(f"H7 fresh pass: {h7_pass}; fresh E4 {fresh_e4}")
