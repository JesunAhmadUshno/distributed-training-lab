# KAGGLE-RUN: runbook for the P19 Deep exercises on Kaggle T4 x2

Status: RUNBOOK ONLY, 2026-11-05. No Kaggle run happened in the
prep cycle. Deep is NOT closed.

## Named blocker (current, verified in the prep cycle)

- No Kaggle account state is verified for this work: no Kaggle
  credentials exist on the Muse VM (no ~/.kaggle folder and no
  kaggle Python module), and the Kaggle signup is a separate
  workstream whose state is not claimed here. No signup was
  attempted in the prep cycle.
- BlackBox (RTX 5050) was unreachable in this cycle: the SSH
  probe failed, so the fleet GPU route was also closed.
  NEEDS-BLACKBOX-ON stands in the batch ledger.
- Either route opening (a verified Kaggle session, or BlackBox
  back on for a local 1-GPU partial run) unblocks the next step.
  The full Addendum B set needs the Kaggle T4 x2 route.

## Quota note

Kaggle free tier: 30 GPU hours per week. The four exercises are
small (tiny MLP, at most 50 timed steps, message sizes up to
64 MB). Budget 1 GPU hour for a first full pass including setup
and reruns, and keep the rest of the weekly quota untouched.
Stop the session when the outputs below are saved: an idle
session burns quota.

## Exact steps on Kaggle

1. Create a new Kaggle Notebook. Set the accelerator to GPU T4 x2
   in the notebook settings. Confirm in a cell: torch.cuda.device_count()
   must print 2. If it prints 1 or 0, stop and record that fact.
2. Clone this repo into the notebook working folder, or upload
   the level-3-deep/kaggle folder. No Kaggle API key is needed
   for the exercises themselves.
3. In a terminal cell, run each exercise with torchrun:
   - torchrun --nproc_per_node=2 kaggle/ex1_ddp_correctness.py
   - torchrun --nproc_per_node=1 kaggle/ex2_scaling_1v2.py
   - torchrun --nproc_per_node=2 kaggle/ex2_scaling_1v2.py
   - torchrun --nproc_per_node=2 kaggle/ex3_fsdp_zero.py
   - torchrun --nproc_per_node=2 kaggle/ex4_comm_timing.py
4. Save every printed JSON line exactly as printed. Do not edit
   numbers. If an exercise fails, save the full error text: the
   failure log is part of the Deep evidence.
5. Note the torch version, the backend printed by each exercise,
   and the session date. A gloo or CPU result must never be
   relabelled as an nccl GPU result.

## Outputs to commit back

- level-3-deep/results_deep_kaggle.json: one section per exercise
  with its printed JSON, torch version, backend, and date.
- The filled scaling table (exercise 2) and communication table
  (exercise 4) in level-3-deep/README.md, replacing the PREP
  status only when every table value traces to the JSON above.
- The failure modes update in level-3-deep/README.md: what broke,
  at which exercise, and the fix or the named remainder.
- Only then may Deep be graded and, if the gates pass, closed.
  No v1.0 release is created from the prep package alone.
