"""E2: all-reduce and broadcast cost across the 1 GbE link (feeds the H2 model).

For each message size, 3 warmup calls, then timed calls with a barrier
before and a CUDA synchronise after each one. Every trial time is kept;
analysis.py fits t(S) = alpha + S / beta and bootstraps the fit.
Usage: torchrun ... e2_comm.py --out results.jsonl
"""
import argparse
import time

import torch
import torch.distributed as dist

import common as c

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--max-log2", type=int, default=28)  # 256 MiB
a = ap.parse_args()

rank, world = c.init()
assert world == 2, "E2 needs exactly two ranks"
stamp = c.env_stamp()
for op in ("all_reduce", "broadcast"):
    for k in range(12, a.max_log2 + 1, 2):
        nbytes = 1 << k
        x = torch.ones(nbytes // 4, device="cuda")
        call = (lambda: dist.all_reduce(x)) if op == "all_reduce" else (lambda: dist.broadcast(x, 0))
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        trials = 20 if nbytes <= (1 << 20) else (8 if nbytes <= (1 << 24) else 5)
        times = []
        for _ in range(trials):
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            call()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        # Every rank measured; report the slower rank per trial (the collective finishes when both do).
        t = torch.tensor(times, device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        c.emit(a.out, {"exp": "E2", "op": op, "bytes": nbytes, "seconds": t.tolist(), **stamp})
dist.destroy_process_group()
