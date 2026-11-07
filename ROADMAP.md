# ROADMAP: distributed-training-lab (P19)

Updated 2026-11-07, v1.0 (Deep closed).

## Done

- Level 1, Learn: DONE (2026-11-05). Simulated data-parallel
  training, all-reduce equivalence proved on CPU.
- Level 2, Build: DONE (2026-11-05). Simulated FSDP-class and
  ZeRO-class sharding, checkpoint restart proved on CPU.
- Level 3, Deep: DONE (2026-11-07). Route changed by Owner ruling
  from Kaggle T4 x2 to two DGX Spark nodes over 1 GbE.
  Pre-registered study (PREREG.md), 5 of 7 hypotheses pass, H5 and
  H6 missed narrowly. Write-up: level-3-deep/README.md.

## Parked v1.1 remainder

- The 4-GPU scope: scaling beyond 2 GPUs (4-GPU table, larger
  message and model sizes, multi-session runs). Parked per
  Addendum B. It is not part of the v1.0 Deep scope and no date
  is promised for it.
- DeepSpeed ZeRO on the same two nodes, for the SPEC's
  three-way comparison (DDP and FSDP2 are measured).
- A faster interconnect (the GB10 nodes' ConnectX ports) to test
  where the link model's crossover moves.
- The original SPEC Deep items outside Addendum B labelling:
  the DDP, FSDP, and DeepSpeed same-model comparison write-up
  and the JAX sharding stretch note, now reconciled: DDP and FSDP2 are
  measured in level-3-deep; DeepSpeed and JAX stay parked.
