"""Shared helpers for the Kaggle Deep exercises. Kaggle only.

This module is NOT run on the Muse VM. It is syntax-checked there
with py_compile only. On Kaggle (T4 x2) it guards the backend:
nccl when CUDA is available, gloo otherwise, and it refuses to
report a 2-GPU result when fewer than 2 devices are visible.
"""
import os
import time


def setup_distributed():
    import torch
    import torch.distributed as dist
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return torch, dist, rank, world, local_rank, backend


def tiny_model(torch):
    return torch.nn.Sequential(
        torch.nn.Linear(4, 5), torch.nn.Tanh(), torch.nn.Linear(5, 1)
    )


def synthetic_batch(torch, n=16, device="cpu"):
    xs, ys = [], []
    for i in range(n):
        x = torch.tensor([
            ((i % 4) - 1.5) / 1.5,
            ((i // 4) - 1.5) / 1.5,
            (((i * 3) % 5) - 2) / 2.0,
            float((i * 7) % 3 - 1),
        ], dtype=torch.float32)
        xs.append(x)
        ys.append(2.0 * x[0] - 1.0 * x[1] + 0.5 * x[2] + 1.0 * x[3] + 0.5)
    return torch.stack(xs).to(device), torch.tensor(ys, dtype=torch.float32).unsqueeze(1).to(device)


def timed(fn, repeats=5):
    start = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    return out, (time.perf_counter() - start) / repeats
