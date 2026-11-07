"""E1: two-node DDP and FSDP give the same model as one process (H1, H1b).

Each rank trains on its own shard (shard = rank) for STEPS of SGD with
momentum. Rank 0 then replays the identical global batches in a single
process (mean of the two shard losses, which equals DDP's averaged
gradient for equal shards) and reports the largest parameter difference.
A SHA-256 of the final parameters lets two separate launches be compared
for run-to-run determinism.
Usage: torchrun ... e1_correct.py --mode ddp|fsdp --out results.jsonl
"""
import argparse
import hashlib

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import common as c

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["ddp", "fsdp"], required=True)
ap.add_argument("--size", default="S")
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--micro", type=int, default=4)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rank, world = c.init()
assert world == 2, "E1 needs exactly two ranks; refusing to report a 1-rank result as 2-node"
c.deterministic(7)
model = c.GPT(a.size).cuda()
if a.mode == "ddp":
    model = DDP(model)
else:
    from torch.distributed.fsdp import fully_shard
    for b in model.blocks:
        fully_shard(b)
    fully_shard(model)
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
losses = []
for step in range(a.steps):
    x, y = c.batch(step, rank, a.micro)
    loss = c.loss_fn(model, x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    t = loss.detach().clone()
    dist.all_reduce(t)
    losses.append(t.item() / world)

stamp = c.env_stamp()  # collective: every rank must call it
# Full parameters on every rank (FSDP holds DTensor shards).
full = [(p.full_tensor() if hasattr(p, "full_tensor") else p).detach() for p in model.parameters()]

if rank == 0:
    c.deterministic(7)
    ref = c.GPT(a.size).cuda()
    ropt = torch.optim.SGD(ref.parameters(), lr=0.05, momentum=0.9)
    ref_losses = []
    for step in range(a.steps):
        parts = [c.batch(step, s, a.micro) for s in range(world)]
        loss = sum(c.loss_fn(ref, x, y) for x, y in parts) / world
        ropt.zero_grad(set_to_none=True)
        loss.backward()
        ropt.step()
        ref_losses.append(loss.item())
    diff = max((f.float() - r.detach().float()).abs().max().item() for f, r in zip(full, ref.parameters()))
    digest = hashlib.sha256(b"".join(f.float().cpu().numpy().tobytes() for f in full)).hexdigest()
    c.emit(a.out, {"exp": "E1", "mode": a.mode, "size": a.size, "params": c.n_params(ref), "steps": a.steps,
                   "micro_per_rank": a.micro, "max_param_diff_vs_single": diff,
                   "max_loss_diff_vs_single": max(abs(p - q) for p, q in zip(losses, ref_losses)),
                   "loss_first": losses[0], "loss_last": losses[-1], "param_sha256": digest, **stamp})
dist.barrier()
dist.destroy_process_group()
