# distributed-training-lab

Distributed training: DDP, FSDP, and DeepSpeed on the same model, plus a multi-node DDP track on CPU VMs. The GPU scaling table at Deep level stays parked until a free multi-GPU route exists.

Status: Learn DONE (2026-11-05). Build and Deep not started. Learn is a simulated data-parallel run in one process: no real DDP, no GPU claim. Deep is defined by Addendum B as the 2-GPU Kaggle exercise set, honestly labelled; the 4-GPU scope is the parked v1.1 remainder.
Project: P19. Fleet node: BlackBox (GPU). Wave: Wave 4 (Learn and Build).

## Start here

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.

## Level map

- Level 1, Learn: DONE (2026-11-05). Simulated data-parallel training of a 3-parameter linear model by 2 simulated ranks in one process, with all-reduce gradient averaging proven equal to the single-process gradient, replica consistency verified, shard imbalance measured, and communication volume accounted in code. See level-1-learn/README.md.
- Level 2, Build: a real working system other people can run.
- Level 3, Deep: benchmarks, optimization, evaluation, hard cases.

## Benchmarks

Measured 2026-11-05 on the Muse VM (CPU, Python 3.12, standard library only). Simulated ranks in one process: no GPU, no real DDP, no wall time or throughput claimed. Full method and per-step table: level-1-learn/README.md.

| Test | Result |
| --- | --- |
| Simulated ranks / shard sizes | 2 ranks, shards of 4 and 4 from 8 synthetic examples |
| Model | 3-parameter linear model, true parameters [2.0, -1.0, 1.0] |
| Training | 20 synchronous steps, learning rate 0.05 |
| Combined loss | 11.5 initial, 0.104349 final |
| Final parameters | [1.995697, -0.545749, 0.537617] |
| All-reduce gradient vs single-process | max difference 8.88e-16 over 20 steps |
| Replica consistency | max difference 0.0 over 20 steps |
| Communication accounting | 12 bytes per rank per step, 480 bytes total over 20 steps (accounting only, no transfer) |
| Shard imbalance check (6 plus 2) | naive average differs from full gradient by 2.833333; size-weighted average matches (difference 0.0) |
| Tests | 20 unit tests passing plus the scaffold smoke test |

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
