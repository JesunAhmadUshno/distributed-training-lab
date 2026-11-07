"""Runs the PREREG E3 grid in shuffled order per repeat, appending to one JSONL.

  python3 run_grid.py single ../runs/main.jsonl   (single-node arm, node 0)
  python3 run_grid.py two    ../runs/main.jsonl   (two-node arms, after predictions.json is committed)
Cells already present in the file are skipped, so an interrupted grid resumes.
"""

import json
import os
import random
import subprocess
import sys

phase, out = sys.argv[1], sys.argv[2]
SIZES, ACCUM, REPS = ["S", "M", "L"], [1, 4, 16, 64], 3
modes = ["single"] if phase == "single" else ["ddp", "ddp_bf16"]
cells = [(m, s, k) for m in modes for s in SIZES for k in ACCUM]
if phase == "two":
    cells.append(("fsdp", "L", 1))
done = set()
if os.path.exists(out):
    for line in open(out):
        r = json.loads(line)
        if r.get("exp") == "E3":
            done.add((r["mode"], r["size"], r["accum"], r["rep"]))
for rep in range(REPS):
    order = cells[:]
    random.Random(rep).shuffle(order)
    util = subprocess.run(
        ["ssh", "spark-31c4", "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    with open(out.replace(".jsonl", "-gpu-util.log"), "a") as f:
        f.write(f"phase={phase} rep={rep} node1_gpu_util={util}\n")
    for mode, size, accum in order:
        if (mode, size, accum, rep) in done:
            continue
        steps = ["--warmup", "3", "--steps", "10"] if size == "S" else ["--warmup", "2", "--steps", "5"]
        args = [
            "./launch.sh",
            "1" if mode == "single" else "2",
            "e3_scale.py",
            "--mode",
            mode,
            "--size",
            size,
            "--accum",
            str(accum),
            "--rep",
            str(rep),
            "--out",
            out,
            *steps,
        ]
        r = subprocess.run(args, capture_output=True, text=True, timeout=900)
        print(f"rep {rep} {mode} {size} k={accum} rc={r.returncode}", flush=True)
        if r.returncode:
            print(r.stdout[-800:], r.stderr[-1500:], flush=True)
            sys.exit(1)
