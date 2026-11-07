# DEEP-SPEC: P19 distributed-training-lab, Addendum B scope

Status: SUPERSEDED 2026-11-07. Deep moved to two DGX Spark nodes (PREREG.md, README.md). Kept as the original 2026-11-05 prep spec.
Source: December Compression Protocol Addendum B (Jesun ruling
2026-11-05). P19 Deep is the 2-GPU Kaggle exercise set, honestly
labelled. The 4-GPU scope is the parked v1.1 remainder, recorded
in the repo root ROADMAP.md.

## Target environment

- Kaggle free tier, T4 x2 (2 GPUs in one notebook session).
- Quota: 30 GPU hours per week on the free tier (see KAGGLE-RUN.md).
- PyTorch with torch.distributed. Backend: nccl when CUDA is
  available, gloo otherwise. The backend guard lives in
  kaggle/common.py and refuses a 2-GPU result when world size
  is not 2 for exercises 1, 3, and 4.
- No spend, no new credential inside this package. Kaggle account
  state is handled by a separate workstream and is not claimed here.

## The four exercises (the whole Addendum B Deep scope)

1. DDP correctness (kaggle/ex1_ddp_correctness.py): train the tiny
   MLP under DistributedDataParallel at world size 2 for 30 steps
   and compare against a single-process run with the same seed
   and data. Pass line: max parameter difference below 1e-5.
2. 1 vs 2 GPU scaling (kaggle/ex2_scaling_1v2.py): the same fixed
   workload at world size 1 and world size 2, in two separate
   launches. The scaling table is built only from the two measured
   timing lines (seconds per step, examples per second).
3. 2-process FSDP or ZeRO (kaggle/ex3_fsdp_zero.py): primary path
   is torch FSDP at world size 2. The fallback path is a ZeRO-style
   shard-owned optimizer step with gradient all-reduce, and the
   run records which path was used. Checkpoint marker at step 12.
4. 2-rank communication timing (kaggle/ex4_comm_timing.py):
   all_reduce and broadcast at 4 message sizes, with warmup,
   barriers, and CUDA synchronisation where available. Output:
   milliseconds per call and effective MB per second per size.

## What this prep package proves, and what it does not

Proved on CPU in this cycle (see results_deep_prep.json, produced
by cpu_equivalence.py reusing the Build sharding code by import):
the Deep exercise logic at CPU scale, namely gradient all-reduce
equivalence, shard partition sums, and checkpoint restart
equality, with deterministic output across two runs.

Not proved, and explicitly unmeasured at this writing:

- No GPU number of any kind: no tokens per second, no examples
  per second, no scaling ratio, no communication bandwidth.
- No DDP, FSDP, or ZeRO run on real GPUs.
- No Kaggle session was opened in this cycle.
- The Kaggle scripts were syntax-checked with py_compile only.
  Torch is absent on the Muse VM, so they were not imported
  and not executed there.

## Deep closure criteria (unchanged, not met yet)

Deep closes only when all four exercises have run on Kaggle
T4 x2, their measured outputs are committed back to this folder,
the scaling and communication tables are filled with those
measured values, the failure modes section is updated with what
actually broke, and the grade is awarded on that evidence.
A CPU equivalence pass does not close Deep.

## Failure modes expected at the Kaggle run (to be confirmed)

- World size 1 in a 2-GPU exercise: the guard fails the run by
  design, it does not silently report a 1-GPU result as 2-GPU.
- FSDP wrap failure on the Kaggle torch build: exercise 3 falls
  back to the ZeRO-style path and labels the result line.
- Session quota or restart loss: checkpoints and printed JSON
  lines are the recovery path, per KAGGLE-RUN.md.
- Backend mismatch: nccl requires CUDA devices; a CPU-only
  session falls back to gloo and must be labelled CPU, not GPU.
