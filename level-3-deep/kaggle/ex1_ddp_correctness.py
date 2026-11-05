"""Exercise 1: DDP correctness on Kaggle T4 x2. Kaggle only.

Trains the tiny MLP with DistributedDataParallel for a fixed step
count and compares the final parameters and loss against a
single-process run on rank 0 with the same seed and data. The
exercise passes only if the max parameter difference is below
1e-5. Run with: torchrun --nproc_per_node=2 ex1_ddp_correctness.py
Not run on the Muse VM: syntax-checked with py_compile only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup_distributed, synthetic_batch, tiny_model


def main():
    torch, dist, rank, world, local_rank, backend = setup_distributed()
    assert world == 2, f"Exercise 1 requires world_size 2, got {world}"
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)
    x, y = synthetic_batch(torch, device=device)
    model = tiny_model(torch).to(device)
    ddp = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank] if device.startswith("cuda") else None
    )
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
    loss_fn = torch.nn.MSELoss()
    for _ in range(30):
        opt.zero_grad()
        loss = loss_fn(ddp(x), y)
        loss.backward()
        opt.step()
    if rank == 0:
        torch.manual_seed(1234)
        ref = tiny_model(torch).to(device)
        ref_opt = torch.optim.SGD(ref.parameters(), lr=0.05)
        for _ in range(30):
            ref_opt.zero_grad()
            ref_loss = loss_fn(ref(x), y)
            ref_loss.backward()
            ref_opt.step()
        diffs = [float((a - b).abs().max()) for a, b in zip(ddp.module.parameters(), ref.parameters())]
        print({
            "exercise": 1, "backend": backend, "world": world,
            "max_param_diff_vs_single": max(diffs), "final_loss": float(loss),
        })
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
