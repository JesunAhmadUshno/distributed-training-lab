# REPORT: distributed-training-lab (P19)

The study of record is the pre-registered two-node Deep study
(level-3-deep/PREREG.md, tag proposal-v1). This report scores it against
that lock, misses included. The full Deep write-up stays in
level-3-deep/README.md; the sections below carry its substance.

## Background

Learn and Build (2026-11-05) are CPU simulations of data-parallel and
sharded training in one Python process: no GPU, no real DDP, FSDP or
DeepSpeed libraries, no wall time or throughput claimed. They came
before the lock and are not pre-registered. They prove the bookkeeping
(the averaged gradient equals one process's gradient up to float
reduction order; restart from a checkpoint is exact) that Deep then
tests on real hardware.

Status: v1.0. Learn DONE (2026-11-05). Build DONE (2026-11-05). Deep DONE (2026-11-07): measured on two DGX Spark nodes, 5 of 7 pre-registered hypotheses pass, 2 missed narrowly and reported as missed. Learn is a simulated data-parallel run in one process: no real DDP, no GPU claim. Build is a simulated parameter-sharding run in one process (FSDP-class and ZeRO-class strategies with checkpoint save and restart): no real FSDP or DeepSpeed libraries, no GPU claim. Deep moved from the Kaggle T4 x2 route (never run, kept as superseded) to two DGX Spark nodes; scaling beyond 2 nodes is the parked v1.1 remainder.
Project: P19. Hardware: two NVIDIA DGX Spark (GB10) nodes for Deep.

### Level map (moved from the README)

- Level 1, Learn: DONE (2026-11-05). Simulated data-parallel training of a 3-parameter linear model by 2 simulated ranks in one process, with all-reduce gradient averaging proven equal to the single-process gradient, replica consistency verified, shard imbalance measured, and communication volume accounted in code. See level-1-learn/README.md.
- Level 2, Build: DONE (2026-11-05). Simulated parameter-sharded training of a 31-parameter 2-layer MLP by 2 simulated ranks in one process, under FSDP-class and ZeRO-class strategies on the same model and data, with checkpoint save at step 12 of 30 and restart proven identical to the uninterrupted run, plus communication and memory accounting in code. See level-2-build/README.md.
- Level 3, Deep: DONE (2026-11-07). Two-node DDP, bf16-compressed DDP and FSDP2 on two GB10 GPUs over 1 GbE, pre-registered in level-3-deep/PREREG.md: exact DDP and FSDP2 equivalence (0.0 difference), a link fit of R^2 0.9999998 at 117 MB/s, per-cell predictions within 2.7%, bit-identical recovery from crashes at steps 13, 25 and 37. Missed: bf16 compression cut 29.4% in one cell (line 30%), FSDP2 used 65.3% of DDP memory (line 65%). See level-3-deep/README.md.

### CPU levels, measured 2026-11-05

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

Build measured 2026-11-05 on the Muse VM (CPU, Python 3.12, standard library only). Simulated ranks in one process: no GPU, no real FSDP or DeepSpeed libraries, communication and memory are logical accounting only. Full method and per-strategy table: level-2-build/README.md.

| Test | Result |
| --- | --- |
| Simulated ranks / parameter shards | 2 ranks, shards of 16 and 15 from 31 parameters |
| Model | 2-layer MLP, input 4, hidden 5 (tanh), output 1, 31 parameters |
| Training | 30 steps, learning rate 0.05, momentum 0.5, 16 synthetic examples |
| Loss (both strategies) | 3.952440 initial, 0.037717 final |
| Strategies | FSDP-class and ZeRO-class, same model, data, and steps |
| Max diff vs single-process reference | 0.0 for both strategies |
| Checkpoint save and restart | checkpoint at step 12, restarted final max diff 0.0 vs uninterrupted, both strategies |
| Communication accounting | 248 bytes per step all ranks, 7,440 bytes total over 30 steps, both strategies (accounting only, no transfer) |
| Memory accounting | peak 79 elements on the largest rank vs 93-element full-model baseline (accounting only) |
| Tests | 43 unit tests passing (23 new Build tests) plus the scaffold smoke test at Build close; the suite has since grown, see below |

Note on the two strategies: FSDP-class and ZeRO-class share the same update arithmetic in this emulation and differ only in phase order (when the all-gather happens). Their 0.0 difference from each other and from the reference holds by construction; it shows the bookkeeping is right, not that the strategies behave the same on real hardware.

### Added CPU studies (2026-11-07)

- Rank sweep, level-1-learn/results_sweep.json: the Learn run at 1 to 8 simulated ranks. With equal shards (1, 2, 4, 8 ranks) the all-reduce gradient differs from the single process only by float reduction order, at most 8.88e-16, and it does not grow with rank count. With unequal contiguous shards (3, 5, 6, 7 ranks) the naive mean is biased, by 0.94 to 1.70.
- Shard-split sweep, same file: every 2-rank split from 1|7 to 7|1. The size-weighted average matches the full-batch gradient at every split; the naive mean is exact only at 4|4.
- Ring all-reduce, level-2-build/results_scaling.json: a chunked ring (reduce-scatter, then all-gather) over the 31 MLP gradient values at 1, 2, 4, 8 and 16 ranks, with the per-rank volume against 2(N-1)/N x P and the float difference against a plain sum.
- Stage-aware memory, same file: per-rank model-state counts for DDP and ZeRO stages 1 to 3 (logical accounting, no measurement).

Tests now: 65 unit tests plus the scaffold smoke test (`python3 -m unittest discover -s tests`). The CPU review of these levels is in "Earlier review" below.

## Method

Two NVIDIA DGX Spark nodes (one GB10 GPU each), NCCL 2.29.7 over TCP on
a 1000 Mb/s office Ethernet link, torch 2.12.0+cu130, CUDA 13.0. Every
result line in level-3-deep/runs/main.jsonl carries its hosts, versions
and link. The design and analysis plan are as locked in PREREG.md:
E1 correctness, E2 link timing (126 all-reduce trials), E3 grid
(S / M / L x accumulation 1, 4, 16, 64 x single, DDP, bf16-compressed DDP,
plus FSDP2 on L at accumulation 1, three shuffled repeats), E4 crash
recovery. Cell value is the mean of per-repeat median step times with a
95% percentile bootstrap (2000 resamples, seed 0).

Field profile (ML and AI research) against this study:
- Baselines: the single-node arm at the same global batch, run in the
  same session on the same nodes.
- Ablations: one part changed at a time: one node vs two; fp32 vs bf16
  gradient communication; DDP vs FSDP2; accumulation 1 to 64.
- Uncertainty: bootstrap intervals on every cell, speedup and link
  parameter (results_deep.json).
- Compute, data and versions: stamped on every result line; data is
  synthetic order-1 Markov tokens keyed by (seed, step, shard).

## Results

### Answer

Almost never, and yes. Two nodes are slower than one in 9 of 12 cells,
down to 0.07x for the 92M-parameter model at accumulation 1. They win
only at accumulation 64 (1.31x to 1.47x). The link model
t(S) = alpha + S / beta, fitted on all-reduce timings alone, predicted
every two-node step time within 2.7% before any two-node training ran.

### Hypotheses, as scored by analysis.py

| # | Pass line (PREREG.md) | Measured | Result |
| --- | --- | --- | --- |
| H1 | DDP and FSDP2 within 1e-5 of a one-process replay; two DDP launches, one SHA-256 | max diff 0.0 (DDP), 0.0 (FSDP2); 1 hash over 2 launches | pass |
| H2 | R^2 >= 0.99, beta 80 to 125 MB/s | R^2 0.9999998, beta 117.4 MB/s [117.43, 117.44], alpha 0.40 ms [0.36, 0.44] | pass |
| H3 | Prediction within +/-25% in every cell | worst cell 2.7% (L, k=4) | pass |
| H4 | Side of 1.0 right in >= 90% of cells; L at k=1 below 0.25x | 12 of 12; L at k=1 0.069x | pass |
| H5 | bf16 hook cuts >= 30% where comm share >= 0.5, loss within 2% | cuts 29.4% to 48.5%; S at k=16 cut 29.4%; loss within 2e-6 | missed |
| H6 | FSDP2 peak memory <= 65% of DDP on L | 65.3% (2,196 vs 3,362 MiB) | missed |
| H7 | Crash at step 13, 25, 37 resumes bit-identical | 3 of 3 match the clean SHA-256, each restarted once | pass |

H1 is exact, not just close: with two ranks, NCCL's averaged gradient
(a + b) / 2 equals the replay's 0.5a + 0.5b bit for bit in fp32.

### Speedup of two nodes over one (DDP, fp32 gradients)

Speedup = one-node step time / two-node step time at the same global
batch. Brackets are 95% bootstrap intervals.

| Model | k=1 | k=4 | k=16 | k=64 |
| --- | --- | --- | --- | --- |
| S, 5.3M params | 0.10 [0.10, 0.10] | 0.33 [0.33, 0.33] | 0.85 [0.81, 0.90] | 1.47 [1.44, 1.54] |
| M, 29.5M | 0.08 [0.08, 0.08] | 0.25 [0.25, 0.25] | 0.72 [0.72, 0.72] | 1.37 [1.31, 1.40] |
| L, 91.5M | 0.07 [0.07, 0.08] | 0.25 [0.24, 0.26] | 0.65 [0.64, 0.69] | 1.31 [1.30, 1.32] |

Predicted (predictions.json, committed before the two-node runs):
S 0.10 / 0.32 / 0.87 / 1.51, M 0.08 / 0.25 / 0.71 / 1.37,
L 0.07 / 0.24 / 0.64 / 1.31.

Why: the L gradient is 366 MB in fp32. At 117 MB/s one sync costs
3.1 s, while one node computes a k=1 step in 0.22 s. Only when each
node does 64 micro-batches between syncs does the halved compute
outweigh the transfer. The bf16 hook halves the bytes and nearly halves
the step where communication dominates (1.64 s vs 3.19 s on L at k=1).

### What the misses mean

- H5: the 30% line was set for every cell the model marks as at least
  half communication. S at k=16 sits nearest that boundary (predicted
  comm share 0.56), so halving the bytes cut 29.4%, not 30%. The loss
  side of H5 held everywhere (relative difference under 2e-6).
- H6: FSDP2 saved 34.7% of peak memory on L, just short of the 35%
  needed. Activations and the per-block all-gather buffer are not
  sharded, and at 92M parameters they are a large share of the peak.
  FSDP2 was also the slowest mode (4.60 s per step vs 3.19 s for DDP),
  because it moves parameters as well as gradients over the same link.

### Design (summary; full text in PREREG.md)

- E1 correctness: 30 SGD-momentum steps, strict fp32, deterministic
  kernels, against a one-process replay of the same global batches.
- E2 link: all-reduce and broadcast, 4 KiB to 256 MiB in x4 steps, 126
  all-reduce trials, slower rank per call.
- E3 grid: S / M / L x accumulation 1, 4, 16, 64 x single, DDP, DDP with
  bf16 compression hook, plus FSDP2 on L at k=1. Micro-batch 8, sequence
  256, AdamW, TF32. Three repeats, cell order shuffled per repeat. Cell
  value = mean of per-repeat median step times.
- E4 recovery: 40 steps, checkpoints every 10, rank 1 killed with
  os._exit at step 13, 25 or 37; torchrun restarts and resumes.
- Data: synthetic order-1 Markov tokens keyed by (seed, step, shard).

### Threats and limits

- Two nodes only. Nothing here says how the curve bends at 4 or 8.
- Shared office switch. Mitigated by shuffled order, repeats, medians,
  slower-rank timing; intervals are tight (most under 5%).
- Node 1 hosts an idle vLLM server (about 83 GB of unified memory). Its
  GPU utilisation was 0% at every repeat (runs/main-gpu-util.log).
- Throughput only: 5 to 10 timed steps per run, no convergence claim.
- DeepSpeed was not run; the comparison is DDP, compressed DDP, FSDP2.
- Harness bring-up runs (runs/bringup/) and one relaunch of E1 FSDP2
  after a harness bug are logged in runs/RUNLOG.md and excluded.

## Review (2026-11-08)

A second reader re-derived the headline numbers from the raw evidence,
without using the study's analysis code: tests/test_review.py reads
level-3-deep/runs/main.jsonl and recomputes the cell statistic as stated
in PREREG.md. It agrees with results_deep.json on every number checked:
two nodes slower in 9 of 12 cells, the lowest speedup 0.069x at L k=1,
two nodes faster only at k=64, worst prediction error 2.7% over the 12
cells of predictions.json, link beta 117.4 MB/s over 126 trials, and the
two misses (H5 cut 29.4% at S k=16, H6 ratio 65.3%). The analysis code's
own re-score (`analysis.py check`) reproduces results_deep.json, and
`analysis.py predict` reproduces predictions.json byte for byte from the
same evidence.

Gates checked: PREREG.md is byte-identical to its lock commit f617be6
(SHA-256 pinned in tests/test_layout.py); the first result lines arrive
after it (a181dd2); predictions.json arrives before any two-node grid
line; bring-up runs and the one E1 FSDP2 relaunch are logged in
runs/RUNLOG.md and excluded. No pass line moved.

## Earlier review (2026-11-07)

The author's own review of the CPU levels, kept for the record. Its
hypotheses H1 to H5 are CPU checks and are separate from PREREG.md's
H1 to H7. Each criterion lists the evidence in the repo, what that
change fixed, and, for any gap left open, the exact compute or data it
needed. Rule applied: every gap that can be closed on a CPU, without
paid APIs and without long runs, is closed in that change.

### Update, later on 2026-11-07: Deep measured on GPUs

The blocker named below (no multi-GPU run) is closed. Deep ran as a
pre-registered study on two NVIDIA GB10 nodes over 1 GbE
(level-3-deep/PREREG.md, README.md): hypotheses and pass lines were
committed before any main-grid run, per-cell predictions were committed
before any two-node run, and every result line carries its hosts,
versions and link speed. 5 of 7 hypotheses pass; H5 (bf16 compression,
29.4% against a 30% line in one cell) and H6 (FSDP2 memory, 65.3%
against 65%) are reported as missed. The blockers below for the research
question, hypotheses and real-hardware evidence are closed for the
two-node claims. Still open: DeepSpeed, more than two nodes, and a
faster interconnect.

### Summary (CPU review, earlier the same day)

The repo is an honest teaching lab, not yet a research result. Its simulations are correct, deterministic and well tested, and every level says plainly that no GPU was used. The research question the SPEC poses (how DDP, FSDP and DeepSpeed compare on the same model at real scale) cannot be answered until a multi-GPU run happens. This change adds the CPU studies that were missing (rank sweep, all-splits imbalance, ring all-reduce, stage-aware memory), fixes documentation drift, adds related work, and publishes a dashboard whose live trainers are checked against the recorded results at build time.

| Criterion | Remaining blocker (CPU review) |
| --- | --- |
| Research question | Real-hardware question needs GPUs |
| Hypotheses | None for the CPU claims |
| Method | None on CPU |
| Baselines | Real DDP baseline needs 2 GPUs |
| Statistics and uncertainty | Timing variance needs GPU runs |
| Reproducibility | None |
| Threats to validity | None |
| Related work | Novelty claim still draft |
| Limitations | None |

### 1. Research question

- Before: the SPEC asks for a same-model comparison of DDP, FSDP and DeepSpeed, a scaling table and an NCCL bandwidth test. The levels actually answer a narrower question that was never written down: does sharding the data or the parameters across simulated ranks change the arithmetic?
- After: the narrower question is stated on the dashboard and in the README. "Does splitting change the answer, and if so why?" is answered for this toy setting: equal shards change it only by float reduction order; unequal shards with a naive mean bias it; checkpoint restart with momentum state is exact.
- Open: the SPEC question about throughput, cost and failure modes at scale. Needs: 2 x T4 GPUs (Kaggle free tier, about 1 GPU hour per KAGGLE-RUN.md) for exercises 1 to 4; 4 GPUs for the parked v1.1 scaling table.

### 2. Hypotheses

- Before: none were stated in falsifiable form.
- After (each tested in tests/ and recorded):
  - H1. With equal shard sizes the all-reduce mean equals the full-batch gradient up to float rounding (below 1e-12). Holds at 1, 2, 4, 8 ranks (tests/test_learn_sweeps.py).
  - H2. With unequal shards the naive mean is biased and the size-weighted mean is not. Holds at every split 1|7 to 7|1 and at 3, 5, 6, 7 ranks.
  - H3. The rounding gap does not grow steadily with rank count, because it depends on addition order. Supported: 8.88e-16 at 2 ranks, 2.22e-16 at 4, 0 at 8 (results_sweep.json).
  - H4. A ring all-reduce leaves every rank with an identical copy and sends 2(N-1) chunks per rank. Holds at 1, 2, 4, 8, 16 ranks (tests/test_build_collectives.py).
  - H5. Restart from a checkpoint that includes velocity shards reproduces the uninterrupted run exactly. Holds at step 12 in Python and at every step 1 to 29 in the browser port.

### 3. Method

- Strengths: stdlib-only, deterministic, no wall clock or randomness, shard and partition logic validated with named errors, results regenerated byte for byte.
- Weakness found: FSDP-class and ZeRO-class call the same `_update_shards` on the same full-batch gradient, and the ZeRO post-update gather result is discarded. Their 0.0 difference holds by construction.
- Fixed: the README and the dashboard now say this in plain words, and the new `stage_memory_accounting` separates DDP, ZeRO-1, ZeRO-2 and ZeRO-3/FSDP so the strategies differ where they really do differ, in what each rank stores.
- Open: none that a CPU can close. A behavioural difference between FSDP and ZeRO needs real libraries with overlapping communication, so the 2-GPU exercise 3.

### 4. Baselines

- Present: a single-process full-batch trainer in both levels, used as the reference for every equivalence claim.
- Added: a plain left-to-right sum as the baseline for the ring all-reduce, so its rounding difference is measured against something.
- Open: a real `torch.nn.parallel.DistributedDataParallel` run as the baseline for the real FSDP and DeepSpeed runs. Needs 2 GPUs (exercise 1).

### 5. Statistics and uncertainty

- The simulations are deterministic, so there is no sampling noise to put intervals on. The right uncertainty here is numerical: how large is float error and where does it come from.
- Before: the 8.88e-16 gap was reported but not analysed.
- Fixed: the rank sweep and the ring study record the gap at several rank counts and two summation orders, which shows it is rounding (non-associativity of floating-point addition, see Goldberg 1991) and that it does not trend with N.
- Open: timing and bandwidth numbers will need repeated trials with warm-up and confidence intervals. Exercise 4 already includes warm-up and barriers; it needs 2 GPUs to run.

### 6. Reproducibility

- Three runner scripts regenerate every results file; CI reruns them on Python 3.11 and 3.12. The new results files (results_sweep.json, results_scaling.json) are written by the existing runners, so the same drift check covers them.
- Added: the dashboard build runs `site/src/sim.check.mjs`, which replays every browser computation and asserts it matches the committed JSON (Learn log within 1e-12 per step, both MLP strategies within 1e-12, all splits, all rank counts, all accounting formulas). A changed results file without a matching code change fails the Pages build.
- Small gap noted, not changed here: ci.yml lists test modules by name, so the two new test files run under `python3 -m unittest discover -s tests` but not in CI. Adding two steps to ci.yml is a one-line follow-up; it was left alone because ci.yml is outside this change's scope.

### 7. Threats to validity

- Internal: toy models (3 and 31 parameters), full batch only, contiguous sharding only, no randomness. Results say nothing about minibatch sampling or stochastic effects.
- Construct: "communication" and "memory" are logical element counts at 4 bytes, not measurements, and leave out activations, latency, bucketing and allocator overhead. The dashboard labels every such number as accounting.
- External: CPU, one process. No claim transfers to NCCL, GPUs or multi-node.
- Browser port: `Math.tanh` may differ from CPython `math.tanh` in the last bit, so the MLP check uses a 1e-12 tolerance and the site does not claim bit-identical replay across languages (it happens to be identical on the build machine).

### 8. Related work

Before: NOVELTY.md is a draft with "NO MEASUREMENT YET" and no citations. Added here:

- Li et al., "PyTorch Distributed: Experiences on Accelerating Data Parallel Training", VLDB 2020. The DDP design (gradient bucketing, overlap of all-reduce with backward) that the Learn level simplifies to one synchronous average.
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", SC 2020. Source of the stage 1/2/3 partitioning used in `stage_memory_accounting`.
- Zhao et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel", VLDB 2023. The FSDP phase order (all-gather, compute, reduce-scatter, sharded update) the Build level emulates.
- Patarasuk and Yuan, "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations", Journal of Parallel and Distributed Computing, 2009. The ring algorithm and its 2(N-1)/N volume, simulated in `ring_all_reduce`.
- Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic", ACM Computing Surveys, 1991. Why summation order changes the last bits.

Open: the novelty claim itself. On current evidence the honest value is a clean, tested, browser-reproducible teaching implementation, not a new method. A real novelty claim, if any, needs the measured same-model comparison (2 GPUs minimum).

### 9. Limitations

Stated on the dashboard (Method and limits window), in every level README and in the README:

- No real distributed run at any level; the Kaggle scripts are syntax-checked only.
- Strategy comparison is by construction, not by measurement.
- Learn has not converged at 20 steps (final parameters about [1.996, -0.546, 0.538] against true [2, -1, 1]); the run is about equivalence, not accuracy. Running more steps in the dashboard shows it keeps converging.

### Gaps closed in this change

| Gap | Fix | Evidence |
| --- | --- | --- |
| No scaling dimension | Rank sweep 1 to 8 | level-1-learn/sweeps.py, results_sweep.json, 5 tests |
| Single imbalance split | All splits 1|7 to 7|1 | same files, 4 tests |
| Float gap not analysed | Rank sweep plus ring vs plain sum | results_sweep.json, results_scaling.json |
| No ring all-reduce model | Chunked ring simulation | level-2-build/collectives.py, 4 tests |
| One memory formula for all strategies | DDP and ZeRO-1/2/3 counts | `stage_memory_accounting`, 2 tests |
| Strategy equivalence by construction unstated | Plain-language note | README, dashboard |
| README drift (test count, Deep status, jargon) | Updated, glossary added | README.md |
| No citations | Related work above | this file |
| Browser numbers unchecked | Build-time known-answer check | site/src/sim.check.mjs |

### Gaps that need compute

| Gap | Exact need |
| --- | --- |
| Real DDP correctness and 1 vs 2 GPU scaling | Kaggle notebook with GPU T4 x2, exercises 1 and 2, about 1 GPU hour total (KAGGLE-RUN.md) |
| Real FSDP or ZeRO run with checkpoint restart | Same session, exercise 3 |
| Communication timing and bandwidth | Same session, exercise 4 (all_reduce and broadcast up to 16,777,216 floats) |
| Tokens per second and cost table by GPU count, failure log | 4 GPUs and the P01 GPT checkpoint (the parked v1.1 scope, Future work below) |
| NCCL bandwidth test before trusting a scaling plan | nccl-tests on 2 or more GPUs on one node |

## Future work (parked v1.1 remainder, moved from ROADMAP.md)

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

## Glossary of internal terms

- Muse VM: the CPU machine the simulated levels were run on. BlackBox: a local GPU machine in the owner's fleet, offline during these levels. NEEDS-BLACKBOX-ON: a blocker tag meaning that machine must be online first.
- Addendum B: the roadmap addendum that defines Deep as the two-GPU Kaggle exercise set.
- claim_check gate: the review step a novelty claim must pass before it is published.
- D-V3-11: the decision record that set the licences.
