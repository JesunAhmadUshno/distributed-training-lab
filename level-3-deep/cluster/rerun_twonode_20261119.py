"""Re-execution driver for the two-node arms on spark-6aa3 + spark-31c4 (decision 0004).

Mirrors the order of record in cluster/README.md and runs/RUNLOG.md:
E1 (ddp x2, fsdp x1), E2, the E3 two-node grid (run_grid.py phase 'two'
cell list, per-repeat shuffle seeds, warmup/steps), then E4 (clean,
faults at 13, 25, 37). Appends to ../runs/twonode-2026-11-19.jsonl;
anything already present is skipped, so an interrupted night resumes.

Differences from run_grid.py, both registered in decision 0004: a failed
cell is logged and the grid continues (run_grid.py exits), and GPU
utilisation of BOTH nodes is sampled per repeat (node 0 is co-tenanted
tonight). On node 1, launch.sh's REMOTE points inside the clone in
~/Jesun_Gallea_Dev. A 'probe' stage runs two cells into a scratch file
for the contention check decision 0004 fixes in advance.

Usage: python3 rerun_twonode.py [out.jsonl] [stage]
stage: e1 | e2 | probe | grid | e4 | all   (default all)
"""

import json
import os
import random
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else "../runs/twonode-2026-11-19.jsonl"
STAGE = sys.argv[2] if len(sys.argv) > 2 else "all"
SCRATCH = "../runs/probe-2026-11-19.jsonl"
ENV = dict(
    os.environ,
    REMOTE="Jesun_Gallea_Dev/distributed-training-lab/level-3-deep/cluster",
)


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def launch(args, env_extra=None, timeout=1200):
    env = dict(ENV)
    env.update(env_extra or {})
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, (r.stdout or "")[-600:], (r.stderr or "")[-1200:]
    except subprocess.TimeoutExpired:
        subprocess.run(["ssh", "spark-31c4", "pkill -f e3_scale; pkill -f e4_fault; pkill -f e1_correct; pkill -f e2_comm"], capture_output=True)
        return -9, "", f"TIMEOUT after {timeout}s; peer workers pkilled"


def util(node):
    cmd = (
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"]
        if node == 0
        else ["ssh", "spark-31c4", "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"]
    )
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # probe must never kill the grid
        return f"probe failed: {e}"


def log_util(tag):
    line = f"{tag} node0=[{util(0)}] node1=[{util(1)}]"
    with open(OUT.replace(".jsonl", "-gpu-util.log"), "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def stage_e1():
    rows = load(OUT)
    for mode, want in (("ddp", 2), ("fsdp", 1)):
        have = sum(1 for r in rows if r.get("exp") == "E1" and r.get("mode") == mode)
        for i in range(have, want):
            rc, so, se = launch(["./launch.sh", "2", "e1_correct.py", "--mode", mode, "--out", OUT])
            print(f"E1 {mode} launch {i + 1} rc={rc}", flush=True)
            if rc:
                print(so, se, flush=True)
                return False
            rows = load(OUT)
    return True


def stage_e2():
    if any(r.get("exp") == "E2" for r in load(OUT)):
        print("E2 already present, skipped", flush=True)
        return True
    rc, so, se = launch(["./launch.sh", "2", "e2_comm.py", "--out", OUT])
    print(f"E2 rc={rc}", flush=True)
    if rc:
        print(so, se, flush=True)
    return rc == 0


def stage_probe():
    committed = json.load(open("../results_deep.json"))["cells"]
    probes = [("ddp", "L", 1), ("ddp", "S", 64)]
    done = {(r["mode"], r["size"], r["accum"]) for r in load(SCRATCH) if r.get("exp") == "E3"}
    for mode, size, accum in probes:
        if (mode, size, accum) in done:
            continue
        steps = ["--warmup", "3", "--steps", "10"] if size == "S" else ["--warmup", "2", "--steps", "5"]
        rc, so, se = launch(
            ["./launch.sh", "2", "e3_scale.py", "--mode", mode, "--size", size,
             "--accum", str(accum), "--rep", "0", "--out", SCRATCH, *steps],
            timeout=900,
        )
        print(f"probe {mode} {size} k={accum} rc={rc}", flush=True)
        if rc:
            print(so, se, flush=True)
            return False
    rows = [r for r in load(SCRATCH) if r.get("exp") == "E3"]
    ok = True
    for mode, size, accum in probes:
        r = next(r for r in rows if (r["mode"], r["size"], r["accum"]) == (mode, size, accum))
        fresh = statistics.median(r["step_seconds"])
        ref = committed[f"{mode}-{size}-{accum}"]["step_s"]
        rel = (fresh - ref) / ref
        print(f"PROBE {mode}-{size}-{accum}: fresh rep-median {fresh:.4f}s committed cell {ref:.4f}s rel {rel:+.2%}", flush=True)
        ok = ok and abs(rel) <= 0.10
    print("PROBE VERDICT:", "WITHIN 10% - grid to record file" if ok else "OFF BY >10% - grid to scratch per decision 0004", flush=True)
    return ok


def stage_grid(out):
    SIZES, ACCUM, REPS = ["S", "M", "L"], [1, 4, 16, 64], 3
    cells = [(m, s, k) for m in ("ddp", "ddp_bf16") for s in SIZES for k in ACCUM]
    cells.append(("fsdp", "L", 1))
    failed = []
    for rep in range(REPS):
        done = {(r["mode"], r["size"], r["accum"], r["rep"]) for r in load(out) if r.get("exp") == "E3"}
        order = cells[:]
        random.Random(rep).shuffle(order)
        if out == OUT:
            log_util(f"phase=two rep={rep}")
        for mode, size, accum in order:
            if (mode, size, accum, rep) in done:
                continue
            steps = ["--warmup", "3", "--steps", "10"] if size == "S" else ["--warmup", "2", "--steps", "5"]
            rc, so, se = launch(
                ["./launch.sh", "2", "e3_scale.py", "--mode", mode, "--size", size,
                 "--accum", str(accum), "--rep", str(rep), "--out", out, *steps],
                timeout=1200,
            )
            print(f"rep {rep} {mode} {size} k={accum} rc={rc}", flush=True)
            if rc:
                print(so, se, flush=True)
                failed.append((mode, size, accum, rep, rc))
    print("GRID FAILED CELLS:", failed if failed else "none", flush=True)
    return not failed


def stage_e4():
    rows = load(OUT)
    have = {r["fault_step"] for r in rows if r.get("exp") == "E4"}
    for fault in (-1, 13, 25, 37):
        if fault in have:
            continue
        name = "clean" if fault < 0 else f"f{fault}"
        rc, so, se = launch(
            ["./launch.sh", "2", "e4_fault.py", "--ckpt", f"../ckpt/e4-{name}",
             "--fault-step", str(fault), "--out", OUT],
            env_extra={"P19_PG_TIMEOUT": "20", "TR_EXTRA": "--max-restarts=2"},
            timeout=1800,
        )
        print(f"E4 {name} rc={rc}", flush=True)
        if rc:
            print(so, se, flush=True)
            return False
    return True


if STAGE in ("e1", "all"):
    stage_e1()
if STAGE in ("e2", "all"):
    stage_e2()
if STAGE == "probe":
    stage_probe()
if STAGE == "grid-record":
    stage_grid(OUT)
if STAGE == "grid-scratch":
    stage_grid(SCRATCH)
if STAGE in ("e4", "all"):
    stage_e4()
print("STAGE DONE:", STAGE, flush=True)
