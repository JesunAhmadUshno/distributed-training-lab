"""E4: a node crash mid-run is recovered exactly (H6).

Two-node DDP, SGD with momentum, deterministic numerics. Every rank saves
its own local checkpoint (model, optimizer, step) every --every steps;
DDP keeps the replicas identical, so no shared filesystem is needed.
With --fault-step N, rank 1 hard-exits at step N on the first attempt
only (a marker file stops it repeating); torchrun's elastic agent
restarts both workers, which resume from the last checkpoint. The final
parameter hash is compared, in analysis, with an uninterrupted launch.
Usage: torchrun --max-restarts=2 --rdzv-backend=c10d ... e4_fault.py --ckpt DIR [--fault-step N] --out results.jsonl
"""
import argparse
import hashlib
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import common as c

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--every", type=int, default=10)
ap.add_argument("--fault-step", type=int, default=-1)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rank, world = c.init()
assert world == 2, "E4 needs exactly two ranks"
c.deterministic(11)
model = DDP(c.GPT("S").cuda())
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
os.makedirs(a.ckpt, exist_ok=True)
path = os.path.join(a.ckpt, f"rank{rank}.pt")
start, resumed_from = 0, None
if os.path.exists(path):
    s = torch.load(path, map_location="cuda", weights_only=True)
    model.module.load_state_dict(s["model"])
    opt.load_state_dict(s["opt"])
    start = resumed_from = s["step"]
marker = os.path.join(a.ckpt, "fault_fired")
attempt = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", 0))
for step in range(start, a.steps):
    if rank == 1 and step == a.fault_step and not os.path.exists(marker):
        open(marker, "w").close()
        os._exit(1)  # hard crash: no cleanup, no barrier
    x, y = c.batch(step, rank, 4)
    loss = c.loss_fn(model, x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if (step + 1) % a.every == 0:
        torch.save({"model": model.module.state_dict(), "opt": opt.state_dict(), "step": step + 1}, path)
stamp = c.env_stamp()
digest = hashlib.sha256(b"".join(p.detach().float().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
c.emit(a.out, {"exp": "E4", "fault_step": a.fault_step, "restart_count": attempt, "resumed_from_step": resumed_from,
               "steps": a.steps, "checkpoint_every": a.every, "loss_last": loss.item(), "param_sha256": digest, **stamp})
dist.destroy_process_group()
