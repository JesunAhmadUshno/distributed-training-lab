"""Shared core for the two-node Deep runs (P19, route changed 2026-11-07).

Two DGX Spark nodes (one NVIDIA GB10 GPU each) joined by 1 GbE through a
switch, NCCL over TCP sockets (no InfiniBand / RoCE). Everything a result
line needs to be traceable is stamped by env_stamp(): hosts, torch, CUDA,
NCCL, backend, world size, link speed.
"""
import json
import math
import os
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 4096
SEQ = 256

# name -> (d_model, n_layer, n_head). Parameter counts are measured, not assumed.
SIZES = {"S": (256, 4, 4), "M": (512, 8, 8), "L": (768, 12, 12)}


def init(backend="nccl"):
    """Join the process group; falls back to a 1-process group when not launched by torchrun."""
    if "RANK" not in os.environ:
        os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1",
                          MASTER_PORT=os.environ.get("MASTER_PORT", "29511"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    # P19_PG_TIMEOUT (seconds) bounds how long a rank waits on a dead peer; E4 sets it short
    # so a crashed node is detected in seconds, not the 10-minute default.
    timeout = timedelta(seconds=float(os.environ.get("P19_PG_TIMEOUT", 600)))
    dist.init_process_group(backend, timeout=timeout)
    return dist.get_rank(), dist.get_world_size()


def deterministic(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def link_speed():
    ifname = os.environ.get("NCCL_SOCKET_IFNAME", "")
    try:
        with open(f"/sys/class/net/{ifname}/speed") as f:
            return f"{ifname} {f.read().strip()} Mb/s"
    except OSError:
        return "unknown"


def env_stamp():
    hosts = [None] * dist.get_world_size()
    dist.all_gather_object(hosts, socket.gethostname())
    return {
        "hosts": hosts, "world_size": dist.get_world_size(), "backend": dist.get_backend(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "nccl": ".".join(map(str, torch.cuda.nccl.version())), "gpu": torch.cuda.get_device_name(),
        "link": link_speed(), "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def emit(path, record):
    """Append one JSON line (rank 0 only) and echo it."""
    if dist.get_rank() == 0:
        line = json.dumps(record, sort_keys=True)
        with open(path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.h = h

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        # Explicit masked attention (deterministic on every backend, unlike fused SDPA kernels).
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.h)
        att = att.masked_fill(torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1), float("-inf"))
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, size):
        super().__init__()
        d, n, h = SIZES[size]
        self.tok, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(SEQ, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(n))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, VOCAB, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def batch(step, shard, micro, seed=1234, device="cuda"):
    """Deterministic synthetic tokens from a fixed order-1 Markov chain.

    Keyed by (seed, step, shard) so any rank, any restart, and the
    single-process reference all see the same examples. The chain has
    learnable structure, so loss falls; it is synthetic, never text.
    """
    g = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + step * 1009 + shard)
    shift = torch.randint(1, 64, (1,), generator=torch.Generator().manual_seed(seed)).item()
    start = torch.randint(0, VOCAB, (micro, 1), generator=g)
    noise = torch.randint(0, 8, (micro, SEQ), generator=g)
    seq = (start + torch.arange(SEQ + 1)[: SEQ].mul(shift) + noise) % VOCAB
    nxt = (seq + shift) % VOCAB
    return seq.to(device), nxt.to(device)


def loss_fn(model, x, y):
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten())


def max_param_diff(a, b):
    return max((pa.detach().float() - pb.detach().float()).abs().max().item()
               for pa, pb in zip(a.parameters(), b.parameters()))
