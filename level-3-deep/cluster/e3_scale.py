"""E3: does a second node pay off, and does the measured link model predict it? (H2-H5)

The global batch is held fixed between arms: 2 * accum micro-batches per
optimizer step. 'single' runs them all on one GPU; the 2-node arms run
accum per rank and synchronise once per step (no_sync on the first
accum - 1 micro-batches). Modes:
  single     one process, one GPU
  ddp        2 nodes, DistributedDataParallel, fp32 gradients
  ddp_bf16   2 nodes, DDP with the bf16 gradient-compression comm hook
  fsdp       2 nodes, FSDP2 (fully_shard per block), for the memory question
Every timed step is kept (CUDA-synchronised wall time). Peak allocated
memory is reported per rank.
Usage: torchrun ... e3_scale.py --mode M --size S --accum K --rep R --out results.jsonl
"""

import argparse
import contextlib
import time

import common as c
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["single", "ddp", "ddp_bf16", "fsdp"], required=True)
ap.add_argument("--size", default="S")
ap.add_argument("--micro", type=int, default=8)
ap.add_argument("--accum", type=int, default=1)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--rep", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rank, world = c.init()
assert world == (1 if a.mode == "single" else 2), f"mode {a.mode} launched with world size {world}"
torch.manual_seed(100 + a.rep)
torch.backends.cuda.matmul.allow_tf32 = True  # throughput run: production numerics, not the bitwise E1 setup
model = c.GPT(a.size).cuda()
params = c.n_params(model)
if a.mode in ("ddp", "ddp_bf16"):
    model = DDP(model)
    if a.mode == "ddp_bf16":
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        model.register_comm_hook(None, default_hooks.bf16_compress_hook)
elif a.mode == "fsdp":
    from torch.distributed.fsdp import fully_shard

    for b in model.blocks:
        fully_shard(b)
    fully_shard(model)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
local_micro = 2 * a.accum if a.mode == "single" else a.accum
torch.cuda.reset_peak_memory_stats()
times, losses = [], []
for step in range(a.warmup + a.steps):
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    for m in range(local_micro):
        shard = m if a.mode == "single" else rank * a.accum + m
        x, y = c.batch(step, shard, a.micro)
        last = m == local_micro - 1
        sync_off = a.mode in ("ddp", "ddp_bf16") and not last
        with model.no_sync() if sync_off else contextlib.nullcontext():
            if a.mode == "fsdp":
                model.set_requires_gradient_sync(last)
            loss = c.loss_fn(model, x, y) / local_micro
            loss.backward()
    opt.step()
    torch.cuda.synchronize()
    if step >= a.warmup:
        times.append(time.perf_counter() - t0)
        losses.append(loss.item() * local_micro)
peak = torch.tensor([torch.cuda.max_memory_allocated() / 2**20], device="cuda")
peaks = [torch.zeros_like(peak) for _ in range(world)]
dist.all_gather(peaks, peak)
t = torch.tensor(times, device="cuda")
dist.all_reduce(t, op=dist.ReduceOp.MAX)
stamp = c.env_stamp()
c.emit(
    a.out,
    {
        "exp": "E3",
        "mode": a.mode,
        "size": a.size,
        "params": params,
        "micro": a.micro,
        "seq": c.SEQ,
        "accum": a.accum,
        "global_micro_batches": 2 * a.accum,
        "rep": a.rep,
        "tokens_per_step": 2 * a.accum * a.micro * c.SEQ,
        "step_seconds": t.tolist(),
        "peak_mib_per_rank": [p.item() for p in peaks],
        "loss_last": losses[-1],
        **stamp,
    },
)
dist.destroy_process_group()
