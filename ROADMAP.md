# ROADMAP: distributed-training-lab (P19)

Updated 2026-11-05, Deep prep cycle.

## Done

- Level 1, Learn: DONE (2026-11-05). Simulated data-parallel
  training, all-reduce equivalence proved on CPU.
- Level 2, Build: DONE (2026-11-05). Simulated FSDP-class and
  ZeRO-class sharding, checkpoint restart proved on CPU.

## Open

- Level 3, Deep: OPEN, prep package only (2026-11-05).
  Addendum B scope: the 2-GPU Kaggle exercise set (DDP
  correctness, 1 vs 2 GPU scaling, 2-process FSDP or ZeRO,
  2-rank communication timing) on Kaggle free T4 x2.
  Prep state: DEEP-SPEC.md, Kaggle-ready scripts, CPU
  equivalence results, and KAGGLE-RUN.md are in level-3-deep/.
  Deep closes only on measured Kaggle outputs committed back
  per KAGGLE-RUN.md. No GPU number has been measured yet.
  Blockers at prep time: no verified Kaggle account state on
  this VM, and BlackBox unreachable (NEEDS-BLACKBOX-ON).

## Parked v1.1 remainder

- The 4-GPU scope: scaling beyond 2 GPUs (4-GPU table, larger
  message and model sizes, multi-session runs). Parked per
  Addendum B. It is not part of the v1.0 Deep scope and no date
  is promised for it.
- Full ZeRO partition semantics if the exercise 3 fallback path
  is the one that runs on Kaggle.
- Full training-state checkpoint save and restore on GPU for
  exercise 3.
- A notebook (.ipynb) wrapper for the Kaggle scripts.
- The original SPEC Deep items outside Addendum B labelling:
  the DDP, FSDP, and DeepSpeed same-model comparison write-up
  and the JAX sharding stretch note, to be reconciled with the
  Addendum B results when Deep closes.
