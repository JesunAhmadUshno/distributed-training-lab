"""Analysis for the two-node Deep runs, as fixed in ../PREREG.md. Standard library only.

python3 analysis.py predict  ../runs/main.jsonl   -> ../predictions.json  (after E2 + single arm)
python3 analysis.py report   ../runs/main.jsonl   -> ../results_deep.json (after everything)
"""

import json
import math
import random
import statistics
import sys

BUCKET = 25 * 2**20  # DDP default bucket_cap_mb
B = 2000


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def ols(xs, ys):
    """y = a + b x. Returns a, b, r2."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, 1 - ss_res / ss_tot


def pct(vals, q):
    s = sorted(vals)
    i = (len(s) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def link_fit(rows, seed=0):
    pts = [(r["bytes"], t) for r in rows if r["exp"] == "E2" and r["op"] == "all_reduce" for t in r["seconds"]]
    xs, ys = zip(*pts)
    a, b, r2 = ols(xs, ys)
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        sample = [pts[rng.randrange(len(pts))] for _ in pts]
        if len({x for x, _ in sample}) < 2:  # a resample with one message size cannot fit a line
            continue
        sa, sb, _ = ols(*zip(*sample))
        boots.append((sa, 1 / sb / 1e6))
    return {
        "alpha_s": a,
        "beta_MBps": 1 / b / 1e6,
        "r2": r2,
        "n_trials": len(pts),
        "alpha_ci": [pct([x[0] for x in boots], 0.025), pct([x[0] for x in boots], 0.975)],
        "beta_ci": [pct([x[1] for x in boots], 0.025), pct([x[1] for x in boots], 0.975)],
    }


def cell_key(r):
    return (r["size"], r["accum"])


def cell_stat(runs, seed=0):
    """Mean of per-repeat medians, with a bootstrap over steps within repeats."""
    reps = [r["step_seconds"] for r in sorted(runs, key=lambda r: r["rep"])]
    value = statistics.fmean(statistics.median(s) for s in reps)
    rng = random.Random(seed)
    boots = [statistics.fmean(statistics.median([s[rng.randrange(len(s))] for _ in s]) for s in reps) for _ in range(B)]
    return value, [pct(boots, 0.025), pct(boots, 0.975)], reps


def ratio_ci(num_reps, den_reps, seed=0):
    rng = random.Random(seed)

    def draw(reps):
        return statistics.fmean(statistics.median([s[rng.randrange(len(s))] for _ in s]) for s in reps)

    boots = [draw(num_reps) / draw(den_reps) for _ in range(B)]
    return [pct(boots, 0.025), pct(boots, 0.975)]


def comm_time(params, fit, bytes_per=4):
    g = params * bytes_per
    return math.ceil(g / BUCKET) * fit["alpha_s"] + g / (fit["beta_MBps"] * 1e6)


def predict(rows):
    fit = link_fit(rows)
    cells = {}
    single = [r for r in rows if r["exp"] == "E3" and r["mode"] == "single"]
    for key in sorted({cell_key(r) for r in single}):
        runs = [r for r in single if cell_key(r) == key]
        t_single, _, _ = cell_stat(runs)
        p = runs[0]["params"]
        comm = comm_time(p, fit)
        t_ddp = t_single / 2 + comm
        cells[f"{key[0]}-{key[1]}"] = {
            "size": key[0],
            "accum": key[1],
            "params": p,
            "t_single": t_single,
            "pred_comm": comm,
            "pred_t_ddp": t_ddp,
            "pred_speedup": t_single / t_ddp,
            "pred_comm_share": comm / t_ddp,
        }
    return {"link": fit, "bucket_bytes": BUCKET, "cells": cells}


def report(rows, pred):
    out = {"link": link_fit(rows), "cells": {}, "hypotheses": {}}
    e3 = [r for r in rows if r["exp"] == "E3"]
    stats = {}
    for r in e3:
        stats.setdefault((r["mode"], r["size"], r["accum"]), []).append(r)
    for (mode, size, accum), runs in sorted(stats.items()):
        v, ci, reps = cell_stat(runs)
        out["cells"][f"{mode}-{size}-{accum}"] = {
            "mode": mode,
            "size": size,
            "accum": accum,
            "step_s": v,
            "ci": ci,
            "tokens_per_s": runs[0]["tokens_per_step"] / v,
            "reps": len(runs),
            "peak_mib": max(max(r["peak_mib_per_rank"]) for r in runs),
            "loss_last": statistics.fmean(r["loss_last"] for r in runs),
        }
    H = out["hypotheses"]
    # H1
    e1 = [r for r in rows if r["exp"] == "E1"]
    ddp_hashes = {r["param_sha256"] for r in e1 if r["mode"] == "ddp"}
    H["H1"] = {
        "max_param_diff": {
            r["mode"]: max(x["max_param_diff_vs_single"] for x in e1 if x["mode"] == r["mode"]) for r in e1
        },
        "ddp_launches": sum(r["mode"] == "ddp" for r in e1),
        "ddp_distinct_hashes": len(ddp_hashes),
    }
    H["H1"]["pass"] = (
        all(v < 1e-5 for v in H["H1"]["max_param_diff"].values())
        and len(ddp_hashes) == 1
        and H["H1"]["ddp_launches"] >= 2
    )
    # H2
    lf = out["link"]
    H["H2"] = {"r2": lf["r2"], "beta_MBps": lf["beta_MBps"], "pass": lf["r2"] >= 0.99 and 80 <= lf["beta_MBps"] <= 125}
    # H3, H4, H5
    errs, sides, comp = {}, [], {}
    for k, p in pred["cells"].items():
        d = stats.get(("ddp", p["size"], p["accum"]))
        if not d:
            continue
        t_ddp, _, ddp_reps = cell_stat(d)
        _, _, single_reps = cell_stat(stats[("single", p["size"], p["accum"])])
        speed = p["t_single"] / t_ddp
        out["cells"][f"ddp-{k}"]["speedup"] = speed
        out["cells"][f"ddp-{k}"]["speedup_ci"] = ratio_ci(single_reps, ddp_reps)
        errs[k] = (p["pred_t_ddp"] - t_ddp) / t_ddp
        sides.append((p["pred_speedup"] > 1) == (speed > 1))
        b = stats.get(("ddp_bf16", p["size"], p["accum"]))
        if b and p["pred_comm_share"] >= 0.5:
            t_b, _, _ = cell_stat(b)
            lb = statistics.fmean(r["loss_last"] for r in b)
            lf32 = statistics.fmean(r["loss_last"] for r in d)
            comp[k] = {"cut": 1 - t_b / t_ddp, "loss_rel_diff": abs(lb - lf32) / lf32}
    H["H3"] = {
        "rel_error": errs,
        "max_abs_rel_error": max(abs(e) for e in errs.values()) if errs else None,
        "pass": bool(errs) and all(abs(e) <= 0.25 for e in errs.values()),
    }
    big = max(pred["cells"].values(), key=lambda p: (p["params"], -p["accum"]))
    big_cell = out["cells"].get(f"ddp-{big['size']}-1", {})
    H["H4"] = {
        "side_correct": sum(sides),
        "cells": len(sides),
        "largest_model_accum1_speedup": big_cell.get("speedup"),
        "pass": bool(sides) and sum(sides) / len(sides) >= 0.9 and (big_cell.get("speedup") or 9) < 0.25,
    }
    H["H5"] = {
        "cells": comp,
        "pass": bool(comp) and all(v["cut"] >= 0.30 and v["loss_rel_diff"] <= 0.02 for v in comp.values()),
    }
    # H6
    f = out["cells"].get(f"fsdp-{big['size']}-1")
    dd = out["cells"].get(f"ddp-{big['size']}-1")
    if f and dd:
        H["H6"] = {
            "fsdp_peak_mib": f["peak_mib"],
            "ddp_peak_mib": dd["peak_mib"],
            "ratio": f["peak_mib"] / dd["peak_mib"],
        }
        H["H6"]["pass"] = H["H6"]["ratio"] <= 0.65
    # H7
    e4 = [r for r in rows if r["exp"] == "E4"]
    clean = {r["param_sha256"] for r in e4 if r["fault_step"] < 0}
    faults = [r for r in e4 if r["fault_step"] >= 0]
    H["H7"] = {
        "fault_steps": sorted(r["fault_step"] for r in faults),
        "restarted": all(r["restart_count"] >= 1 for r in faults),
        "matches_clean": [r["param_sha256"] in clean for r in faults],
        "clean_hashes": len(clean),
    }
    H["H7"]["pass"] = len(faults) >= 3 and len(clean) == 1 and H["H7"]["restarted"] and all(H["H7"]["matches_clean"])
    out["env"] = {k: rows[-1][k] for k in ("hosts", "torch", "cuda", "nccl", "gpu", "link")}
    return out


def same(a, b, rel=1e-9):
    """Equal structure and values; floats may differ in the last digits (Python versions sum differently)."""
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=rel)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k], rel) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y, rel) for x, y in zip(a, b))
    return a == b


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    rows = load(path)
    if cmd == "check":
        with open("../predictions.json") as f, open("../results_deep.json") as g:
            ok = same(json.loads(json.dumps(report(rows, json.load(f)))), json.load(g))
        print("results_deep.json reproduced" if ok else "results_deep.json DIFFERS from a fresh re-score")
        sys.exit(0 if ok else 1)
    if cmd == "predict":
        res = predict(rows)
        dest = "../predictions.json"
    else:
        with open("../predictions.json") as f:
            res = report(rows, json.load(f))
        dest = "../results_deep.json"
    with open(dest, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(res.get("hypotheses", {"link": res["link"]}), indent=1)[:3000])
