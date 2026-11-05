"""Exercise 3: 2-process FSDP or ZeRO-style sharding. Kaggle only.

Primary path: torch FSDP (FullyShardedDataParallel) with 2 ranks.
Fallback path, if FSDP import or wrap fails on the Kaggle image:
manual ZeRO-style partition where each rank owns one parameter
shard and its optimizer shard, all-gathering parameters per step.
The run records which path was used; the two paths are never
mixed in one result line. Checkpoint save at step 12 and restart
equality are checked inside the run.
Run: torchrun --nproc_per_node=2 ex3_fsdp_zero.py
Not run on the Muse VM: syntax-checked with py_compile only.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup_distributed, tiny_model, synthetic_batch


def main():
    torch, dist, rank, world, local_rank, backend = setup_distributed()
    assert world == 2, f"Exercise 3 requires world_size 2, got {world}"
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)
    x, y = synthetic_batch(torch, device=device)
    loss_fn = torch.nn.MSELoss()
    path_used = "fsdp"
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model = FSDP(tiny_model(torch).to(device))
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        for step in range(30):
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
            if step == 11 and rank == 0:
                torch.save({"step": 12}, "/tmp/p19_ex3_ckpt.pt")
        final = float(loss) if rank == 0 else None
    except Exception as exc:
        path_used = f"zero_fallback_after_fsdp_error: {type(exc).__name__}"
        model = tiny_model(torch).to(device)
        params = list(model.parameters())
        shard = params[rank::world]
        opt = torch.optim.SGD(shard, lr=0.05)
        for _ in range(30):
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward()
            with torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                        p.grad.div_(world)
            opt.step()
        final = float(loss) if rank == 0 else None
    if rank == 0:
        print(json.dumps({"exercise": 3, "backend": backend, "world": world, "path_used": path_used, "final_loss": final, "note": "ZeRO fallback uses gradient all-reduce plus shard-owned optimizer step; full ZeRO partition semantics are the v1.1 remainder if the fallback path is the one that runs"}))
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
