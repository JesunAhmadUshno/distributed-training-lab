"""Exercise 4: 2-rank communication timing. Kaggle only.

Times all_reduce and broadcast at several message sizes with
warmup, CUDA synchronisation where available, and a barrier
before each timed block. Reports milliseconds per call and
effective bandwidth in MB per second computed from the measured
time and the known message size. No timing exists until this
runs on Kaggle with 2 ranks.
Run: torchrun --nproc_per_node=2 ex4_comm_timing.py
Not run on the Muse VM: syntax-checked with py_compile only.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup_distributed


def main():
    torch, dist, rank, world, local_rank, backend = setup_distributed()
    assert world == 2, f"Exercise 4 requires world_size 2, got {world}"
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    rows = []
    for numel in (1024, 65536, 1048576, 16777216):
        for op_name in ("all_reduce", "broadcast"):
            buf = torch.ones(numel, dtype=torch.float32, device=device)
            def run_op():
                if op_name == "all_reduce":
                    dist.all_reduce(buf)
                else:
                    dist.broadcast(buf, src=0)
            for _ in range(3):
                run_op()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.barrier()
            reps = 20
            start = time.perf_counter()
            for _ in range(reps):
                run_op()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - start) / reps * 1000.0
            mb = numel * 4 / 1e6
            rows.append({"op": op_name, "numel": numel, "bytes": numel * 4, "ms_per_call": ms, "effective_MB_per_s": mb / (ms / 1000.0) if ms > 0 else None})
    if rank == 0:
        print(json.dumps({"exercise": 4, "backend": backend, "world": world, "rows": rows}, indent=2))
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
