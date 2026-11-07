# PREREG: P19 Deep on two DGX Spark nodes over 1 GbE

Status: PRE-REGISTERED 2026-11-07, committed before any main-grid run.
The git timestamp of this file is the registration time. Smoke and
sizing runs made before it (harness bring-up, one 4-step timing probe
per model size on one node) are listed in the run log and excluded
from every result.

## Route change (Owner ruling, 2026-11-07)

Deep moves from the Kaggle T4 x2 exercise set (DEEP-SPEC.md, Addendum B)
to two DGX Spark nodes, one NVIDIA GB10 GPU each, joined by 1 GbE through
an office switch, NCCL over TCP sockets (no InfiniBand, no RoCE). This is
harder than two GPUs in one box: the link is about 100 times slower than
NVLink-class links, so communication, not compute, sets the result. The
Kaggle runbook stays in the repo as the superseded route.

## Question

On two GPU nodes joined by commodity 1 GbE, when does data-parallel
training beat one node, and can a two-parameter link model measured in
two minutes predict the answer before the training runs?

## Hypotheses and pass lines

- H1, correctness. After 30 steps (SGD with momentum, fp32, deterministic
  kernels), two-node DDP and two-node FSDP2 parameters each differ from a
  single-process replay of the same global batches by less than 1e-5
  (max absolute). Two separate DDP launches give the same SHA-256.
- H2, link model. All-reduce time fits t(S) = alpha + S / beta over
  4 KiB to 256 MiB with R^2 >= 0.99, and the effective beta lies within
  80 to 125 MB/s (1 GbE line rate is 125 MB/s).
- H3, prediction. For every (model size, accumulation) cell, the
  no-overlap model T_ddp = T_single / 2 + n_b * alpha + G / beta
  predicts the measured two-node DDP step time within +/-25%. G is the
  gradient size in bytes (4 x parameters), n_b = ceil(G / 25 MiB) is
  DDP's bucket count, alpha and beta come from E2, T_single from the
  single-node arm. The per-cell predictions are computed and committed
  (predictions.json) after E2 and the single-node arm, before any
  two-node E3 run.
- H4, crossover. Speedup = T_single / T_ddp. The model's predicted side
  of 1.0 (faster or slower than one node) is right in at least 90% of
  cells, and the largest model at accumulation 1 has speedup below 0.25.
- H5, compression. The bf16 gradient-compression hook cuts two-node step
  time by at least 30% in every cell where the model attributes at least
  half the step to communication, and its last-step loss is within 2% of
  fp32 DDP in the same cell.
- H6, memory. FSDP2 peak allocated memory per rank on the largest model
  with AdamW is at most 65% of DDP's.
- H7, recovery. A hard crash of node 1 (process exit, no cleanup) at
  step 13, 25 or 37 of a 40-step run with checkpoints every 10 steps is
  restarted by torchrun and ends with parameters bit-identical (SHA-256)
  to an uninterrupted run, in all three cases.

A hypothesis that misses its pass line is reported as missed, with the
measured value. No pass line moves after this commit.

## Design

- Models: GPT-style decoder, vocabulary 4096, context 256, sizes
  S 5,322,240 / M 29,545,472 / L 91,544,064 parameters (counted by
  common.n_params).
- Data: synthetic order-1 Markov tokens keyed by (seed, step, shard), so
  every arm, rank and restart sees identical examples. Synthetic, not
  text: this study measures systems behaviour, not language modelling.
- E3 grid: sizes {S, M, L} x accumulation {1, 4, 16, 64} x modes
  {single, ddp, ddp_bf16}, micro-batch 8, global batch held fixed at
  2 x accumulation micro-batches per optimizer step, plus FSDP2 on L at
  accumulation 1. Three repeats. Within each repeat the cell order is
  shuffled with a fixed seed (repeat index), so drift on the shared
  switch cannot line up with one arm.
- Timing: CUDA-synchronised wall time per optimizer step after warmup
  (S: 3 warmup, 10 timed; M and L: 2 warmup, 5 timed). Two-node times are
  the slower rank per step.
- E2: all-reduce and broadcast at 4 KiB to 256 MiB in x4 steps, 3 warmup
  calls, 20 / 8 / 5 timed calls by size band, slower rank per call.

## Analysis plan

- Cell statistic: median step time per repeat; the cell value is the
  mean of the three repeat medians, with a 95% percentile bootstrap
  interval (2000 resamples of steps within repeats, seed 0).
- Speedup intervals: the same bootstrap applied to the ratio.
- Link fit: ordinary least squares of all-reduce trial times on bytes;
  alpha, beta and R^2 reported with 95% bootstrap intervals (resampling
  trials).
- All analysis is in analysis.py, standard library only, with unit tests
  on known answers.

## Threats named in advance

- The 1 GbE switch is a shared office network; other traffic can slow a
  run. Mitigation: repeats, shuffled order, slower-rank timing, medians.
- Node 1 also hosts an idle vLLM server holding about 83 GB of unified
  memory. GPU utilisation is sampled before each repeat and reported;
  memory headroom (26 GB free) is far above the largest run (about 3 GB).
- Throughput only: runs are 5 to 10 steps, so no convergence claim.
- Two nodes only; nothing here says how the curve bends at 4 or 8.
- E3 uses TF32 matmuls (production numerics); E1 and E4 use strict fp32
  and deterministic kernels, because they test exactness.
