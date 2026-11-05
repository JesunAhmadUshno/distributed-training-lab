---
title: distributed-training-lab
---

# distributed-training-lab

Distributed training: DDP, FSDP, and DeepSpeed on the same model, plus a multi-node DDP track on CPU VMs. Built in levels with measured results at each level. The simulated levels are stated plainly as simulations in every level README: no GPU number is claimed from them.

- Learn: a simulated data-parallel run in one process, 2 simulated ranks on a 3-parameter linear model. Combined loss fell from 11.5 to 0.104349 over 20 steps, and the all-reduce gradient matched the single-process gradient with a max difference of 8.88e-16.
- Build: simulated parameter-sharded training of a 31-parameter 2-layer MLP under FSDP-class and ZeRO-class strategies. Both reduced the loss from 3.952440 to 0.037717, matched the single-process reference with max difference 0.0, and restarted from a step 12 checkpoint with max difference 0.0 against the uninterrupted run.
- Deep: NOT closed. Deep is defined as the 2-GPU Kaggle exercise set (Addendum B), honestly labelled; the prep package is Kaggle-ready with a CPU equivalence check (all checks passed), and the 4-GPU scope is the parked v1.1 remainder in ROADMAP.md.

Grades on record (of 10,000): Learn 8,400, Build 8,300, Deep prep package 7,800 (Deep itself not awarded, not closed).

Every number above is already published in this repo (README.md and the level write-ups) and reproduced by the runners in this repo, checked in CI. The full write-ups are in the level folders: level-1-learn, level-2-build, level-3-deep.

Releases: https://github.com/JesunAhmadUshno/distributed-training-lab/releases
