"""Exercise 2: 1 vs 2 GPU scaling on Kaggle T4 x2. Kaggle only.

Measures wall time per step and examples per second for the same
fixed workload at world_size 1 and world_size 2, launched as two
separate torchrun invocations (see KAGGLE-RUN.md). Each invocation
writes its own timing line; the scaling table is assembled only
from those two measured lines. No scaling number exists until
both runs complete on Kaggle.
Run: torchrun --nproc_per_node=1 ex2_scaling_1v2.py
     torchrun --nproc_per_node=2 ex2_scaling_1v2.py
Not run on the Muse VM: syntax-checked with py_compile only.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup_distributed, tiny_model, synthetic_batch


def main():
    torch, dist, rank, world, local_rank, backend = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)
    x, y = synthetic_batch(torch, n=256, device=device)
    model = tiny_model(torch).to(device)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.startswith("cuda") else None)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
    loss_fn = torch.nn.MSELoss()
    for _ in range(5):
        opt.zero_grad(); loss_fn(ddp(x), y).backward(); opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    steps = 50
    start = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(); loss_fn(ddp(x), y).backward(); opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if rank == 0:
        print(json.dumps({"exercise": 2, "backend": backend, "world": world, "steps": steps, "seconds_total": elapsed, "seconds_per_step": elapsed / steps, "examples_per_second_global": steps * 256 * world / elapsed}))
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
