"""Re-execution driver for the E3 single-node arm on spark-6aa3 (decision 0003).

Mirrors run_grid.py phase 'single' exactly (same cell list, same per-repeat
shuffle seeds, same warmup/steps, launched via launch.sh with 1 node), minus
run_grid.py's GPU-util probe, which ssh's to spark-31c4 (out of bounds for
this run). Appends to ../runs/rerun-2026-11-17.jsonl; cells already present
are skipped, so an interrupted grid resumes.
"""

import json
import os
import random
import subprocess
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "../runs/rerun-2026-11-17.jsonl"
SIZES, ACCUM, REPS = ["S", "M", "L"], [1, 4, 16, 64], 3
cells = [("single", s, k) for s in SIZES for k in ACCUM]
done = set()
if os.path.exists(out):
    for line in open(out):
        r = json.loads(line)
        if r.get("exp") == "E3":
            done.add((r["mode"], r["size"], r["accum"], r["rep"]))
for rep in range(REPS):
    order = cells[:]
    random.Random(rep).shuffle(order)
    for mode, size, accum in order:
        if (mode, size, accum, rep) in done:
            continue
        steps = ["--warmup", "3", "--steps", "10"] if size == "S" else ["--warmup", "2", "--steps", "5"]
        args = [
            "./launch.sh", "1", "e3_scale.py",
            "--mode", mode, "--size", size, "--accum", str(accum),
            "--rep", str(rep), "--out", out, *steps,
        ]
        r = subprocess.run(args, capture_output=True, text=True, timeout=1800)
        print(f"rep {rep} {mode} {size} k={accum} rc={r.returncode}", flush=True)
        if r.returncode:
            print(r.stdout[-800:], r.stderr[-1500:], flush=True)
            sys.exit(1)
print("RERUN GRID COMPLETE", flush=True)
