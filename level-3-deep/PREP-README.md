# Level 3, Deep PREP (2026-11-05, superseded by README.md)

Status: PREP PACKAGE, 2026-11-05. Deep is NOT closed. No GPU
number was measured in this cycle, and no GPU number is claimed
anywhere in this folder.

## Blocker (named, verified this cycle)

Neither GPU route was open in the prep cycle:

- Kaggle: no Kaggle credentials exist on the Muse VM (no
  ~/.kaggle folder, no kaggle Python module). The Kaggle signup
  is a separate workstream whose state is not claimed here, and
  no signup was attempted in this cycle.
- BlackBox (RTX 5050, the fleet GPU of first resort): the SSH
  probe failed this cycle (connection closed), the same as the
  prior cycle. NEEDS-BLACKBOX-ON stands in the batch ledger.

Per the ledger instruction for this case, this cycle built the
Deep code and runbook so they are Kaggle-ready, ran the CPU
equivalence check, and graded Deep as NOT closed.

## What is in this prep package

- DEEP-SPEC.md: the Addendum B scope in full (DDP correctness,
  1 vs 2 GPU scaling, 2-process FSDP or ZeRO, 2-rank
  communication timing, on Kaggle free T4 x2), with what remains
  unmeasured stated plainly.
- kaggle/: five PyTorch scripts (common.py plus one per
  exercise) using torch.distributed with an nccl or gloo guard.
  They were syntax-checked with py_compile only on this VM.
  Torch is absent on this VM, so the scripts were compile-checked
  as text via py_compile without import: they did not run here,
  and no result in this folder comes from them.
- cpu_equivalence.py: a standard library CPU equivalence check
  that reuses the Build sharding code and the Learn data-parallel
  code by import (no copies) and verifies the Deep exercise logic
  at CPU scale.
- run_deep_prep.py and results_deep_prep.json: the measured CPU
  results, deterministic across two runs (identical file hash).
- KAGGLE-RUN.md: the exact runbook for Kaggle T4 x2, the 30 GPU
  hours per week quota note, and the outputs to commit back.

## CPU equivalence results (measured, CPU only, one process)

| Check | Measured result |
| --- | --- |
| Learn all-reduce max gradient diff vs single-process | 8.881784197001252e-16 |
| Learn replica max diff | 0.0 |
| Learn loss | 11.5 initial, 0.1043492323477317 final |
| Build FSDP-class max diff vs single-process | 0.0 |
| Build ZeRO-class max diff vs single-process | 0.0 |
| Build loss (both strategies) | 3.9524402310764573 initial, 0.037717029906809106 final |
| Shard partition | 31 parameters, shards of 16 and 15, every index exactly once |
| Shard element sums | -0.1 and -0.1, total -0.2, full vector sum -0.2, abs diff 0.0 |
| All-gather reconstruction max diff | 0.0 |
| Checkpoint restart (step 12 of 30) max diff vs uninterrupted | 0.0 for FSDP-class, 0.0 for ZeRO-class |
| GPU numbers measured | None. False in the results file by design. |

## Failure modes

- Claiming GPU results from a CPU simulation: blocked by design.
  The results file carries gpu_numbers_measured as false, and
  this README keeps the PREP title until Kaggle evidence lands.
- Kaggle scripts that have never executed: they may still fail
  on first contact with the Kaggle torch build (API drift, FSDP
  wrap rules, torchrun environment). That is the largest known
  risk in this package and is why Deep stays open.
- Exercise 3 fallback simplification: if FSDP fails, the fallback
  is a ZeRO-style shard-owned optimizer step with gradient
  all-reduce. It labels its result line with the path used, and
  full ZeRO partition semantics stay a named remainder if the
  fallback is the path that runs.
- Exercise 3 checkpoint: the Kaggle script saves a step marker,
  not a full training state. Full state restart equality is
  proved on CPU here (step 12, diff 0.0) and remains to be
  re-proved on GPU.
- World size guard: exercises 1, 3, and 4 fail by design at
  world size other than 2, so a 1-GPU session cannot be
  misreported as a 2-GPU result.


## Next step to close Deep

Run KAGGLE-RUN.md on a verified Kaggle T4 x2 session (or, for a
partial 1-GPU check only, on BlackBox when it is back on), commit
results_deep_kaggle.json with the measured outputs, fill the
scaling and communication tables from those outputs, update the
failure modes with what actually broke, then grade Deep.
