# Level 2, Build: distributed-training-lab

Status: DONE (2026-11-05). Two parameter-sharding strategies on
the same model, data, and step count, with checkpoint save and
restart, built with the Python standard library only. Early pull
from Sprint 3 under the December compression protocol.

## Main claim

Build ran two parameter-sharding strategies on the same 2-layer MLP with 31 parameters and the same 16 synthetic examples for 30 steps at learning rate 0.05 with momentum 0.5 across 2 simulated ranks with parameter shards of 16 and 15: the FSDP-class strategy and the DeepSpeed ZeRO-class strategy each reduced the full-batch loss from 3.952440 to 0.037717, each matched the single-process reference final parameters with max difference 0.0, each saved a checkpoint at step 12 and restarted from it with a final max difference of 0.0 against its uninterrupted run, each accounted 7,440 bytes of logical communication over 30 steps (248 bytes per step), and each held peak logical storage of 79 elements on the largest rank against a 93-element full-model baseline, with 23 new Build tests passing as part of 43 total tests.

## Honest scope line

The two ranks in this level are simulated in one process on a
CPU. This is not real PyTorch FSDP and not real DeepSpeed: no
FSDP or DeepSpeed library is used, no GPU is used, no processes
are started, and no bytes move over any network. Communication
and memory figures are logical accounting computed in code from
the parameter count, never measured transfers, measured memory,
wall time, or throughput, and no GPU number is claimed anywhere.
NEEDS-BLACKBOX-ON: the assigned fleet node BlackBox (RTX 5050)
was checked once by SSH at the start of this cycle and was
unreachable (connection closed, exit 255), so this Build ran as
the CPU sharding emulation on the Muse VM, a limit stated in
the scope notes below. The NCCL interconnect bandwidth
measurement named in the SPEC (register item 542) needs real GPU
hardware and is not done here; it stays Deep and v1.1 scope. The
TensorFlow MirroredStrategy bridge run named in the SPEC also
stays deferred. Deep for this project is defined by Addendum B
of the December compression protocol as the 2-GPU Kaggle
exercise set, honestly labelled; the 4-GPU scope is the parked
v1.1 remainder in this repo ROADMAP scope.

## BUILD-SPEC

1. Model and data. A 2-layer MLP: input 4, hidden 5 with tanh,
   output 1. The flat parameter vector has 31 parameters (W1
   4 by 5 = 20, b1 = 5, W2 = 5, b2 = 1). The dataset is 16
   synthetic examples from deterministic formulas, targets
   y = 2 * x1 - 1 * x2 + 0.5 * x3 + 1 * x4 + 0.5. Loss is mean
   squared error, gradients by hand-written backpropagation,
   verified against numeric gradients in the tests. Initial
   parameters come from a fixed formula, no randomness anywhere.
2. Partition. The 31 parameters are partitioned contiguously
   across 2 simulated ranks: shard sizes 16 and 15, every index
   in exactly one shard, in order. Each rank's trainer state is
   held only as its parameter shard plus its optimizer velocity
   shard; a full vector exists only as a transient gathered copy.
3. FSDP-class strategy. Per step: all-gather the parameter
   shards into a full copy, compute the full-batch gradient,
   reduce-scatter the gradient so each rank holds only its
   gradient shard, then each rank updates only its own shard
   with its own velocity shard (momentum SGD, v = 0.5 * v
   - 0.05 * g, p = p + v).
4. DeepSpeed ZeRO-class strategy. Per step: the gradient is
   partitioned into per-rank gradient shards, optimizer state
   (the velocity vector) is partitioned per rank, each rank
   updates only its own parameter shard, and the updated shards
   are all-gathered after the update.
5. Checkpoint save and restart. At step 12 of 30, each strategy
   saves a checkpoint JSON holding the step, the per-rank
   parameter shards, the per-rank velocity shards, the config,
   and a SHA-256 config hash. A fresh trainer is built from the
   file only (hash recomputed and checked, shard shapes checked,
   step checked), then continues to step 30. The restarted final
   parameters are compared to the uninterrupted run.
6. Reference comparison. A single-process trainer runs the same
   model, data, steps, learning rate, and momentum with full
   vectors. Both strategies do the same arithmetic per element,
   so the final difference against the reference is expected to
   be, and is measured as, exactly 0.0.
7. Accounting. Communication and memory are computed in code
   (see the hand checks below), not measured. Bytes use the
   float32 accounting size of 4 bytes per parameter.

## Run it

    python level-2-build/run_build.py
    python -m unittest tests.test_build_sharding -v
    python -m unittest discover -s tests
    python tests/test_smoke.py

The runner writes level-2-build/results_build.json
deterministically: run it twice and the file is identical
(verified 2026-11-05:
sha256 c1094c9dc1f0537471af2660817503b5e06b28bfc7a549fa2916c47f8049540f
on repeated runs). The checkpoint files are generated inside a
temp directory during the run and are not committed.

## Results (measured 2026-11-05, Muse VM, Python 3.12, CPU)

Both strategies share the same model, data, seed values, and
step count, and both do the same per-element arithmetic, so
their training curves are identical. Step losses (loss before
the update at that step, same for both strategies):

| Step | Full-batch loss before update |
| --- | --- |
| 1 | 3.952440 |
| 5 | 3.518244 |
| 10 | 2.242819 |
| 15 | 0.322613 |
| 20 | 0.046378 |
| 25 | 0.040956 |
| 30 | 0.038199 |

Per strategy:

| Measure | FSDP-class | ZeRO-class |
| --- | --- | --- |
| Parameter shards (2 ranks) | 16 and 15 | 16 and 15 |
| Steps / checkpoint step | 30 / 12 | 30 / 12 |
| Initial loss | 3.952440 | 3.952440 |
| Final loss (after step 30) | 0.037717 | 0.037717 |
| Max diff vs single-process reference | 0.0 | 0.0 |
| Checkpoint restart max diff vs uninterrupted | 0.0 | 0.0 |
| All-gather bytes per step (all ranks) | 124 | 124 |
| Reduce-scatter bytes per step (all ranks) | 124 | 124 |
| Total logical bytes over 30 steps | 7,440 | 7,440 |
| Persistent elements per rank | 32 and 30 | 32 and 30 |
| Peak logical elements per rank | 79 and 76 | 79 and 76 |
| Full-model baseline elements per rank | 93 | 93 |

Final parameters (same for both strategies, first 6 of 31):
[-0.825122, 0.615005, 0.235960, -0.587703, 0.475554, 0.351112].
The full 31 values are in results_build.json.

Hand checks: all-gather received volume per step is
(31 - 16) * 4 + (31 - 15) * 4 = 60 + 64 = 124 bytes.
Reduce-scatter received volume per step is
16 * 4 + 15 * 4 = 64 + 60 = 124 bytes. Total per step is
248 bytes, times 30 steps is 7,440 bytes. Persistent storage
per rank is parameter shard plus velocity shard: 2 * 16 = 32
and 2 * 15 = 30 elements. Peak logical storage adds the
transient gathered full copy (31) and the gradient shard:
31 + 3 * 16 = 79 and 31 + 3 * 15 = 76. The single-process
baseline holds parameters, gradient, and velocity in full:
3 * 31 = 93 elements. The checkpoint restart difference is
exactly 0.0, not merely under 1e-12, because the checkpoint
stores the full shard state as JSON numbers that round-trip
exactly and the restarted run repeats the same arithmetic.

## Failure modes and limits

- Simulated ranks only: one process, no real FSDP or DeepSpeed
  libraries, no GPU, no processes, no network. Nothing here
  measures real distributed speed, scaling, interconnect
  behaviour, or failure at scale, and no speedup, throughput,
  or GPU claim is made. NEEDS-BLACKBOX-ON stands for the GPU
  route (see the honest scope line).
- Identical results by construction: both strategies perform
  the same per-element update arithmetic, so they agree with
  each other and with the single-process reference exactly. In
  real systems the strategies differ in memory timing, overlap,
  and performance, not in the math of a full-batch step; this
  emulation teaches the sharding and checkpoint mechanics and
  the accounting, not a performance ranking.
- Tiny synthetic data and a small model: 16 examples and 31
  parameters. At this size the sharding overhead lesson is
  arithmetic, not operational; nothing is claimed about large
  models, real datasets, or minibatch noise.
- Accounting only: communication bytes and stored elements are
  computed from the parameter count with fixed formulas. Ring
  or tree collective patterns, latency, bandwidth, and real
  memory use are not modelled or measured. The NCCL bandwidth
  item from the SPEC stays open for real hardware.
- Full-batch synchronous steps only: every step waits for the
  gather and scatter by construction. Stragglers, asynchronous
  updates, gradient compression, communication overlap, and
  rank failure mid-run are out of scope at this level; a rank
  failure here would lose its shard, because the checkpoint is
  the only recovery path and it is written once at step 12.
- One checkpoint, one format: the checkpoint is plain JSON for
  teaching and inspection. It is not a production format (no
  binary tensor storage, no sharded checkpoint files per rank,
  no versioning beyond the config hash check).
- Optimizer scope: momentum SGD only, with the velocity vector
  as the whole optimizer state. Adam-class state (two moment
  vectors per parameter) would triple the partitioned optimizer
  storage and is named here as unbuilt, not approximated.


## What Deep adds next

Deep per Addendum B of the December compression protocol: the
2-GPU Kaggle exercise set, honestly labelled, with a scaling
table (tokens per second and cost per run against GPU count),
a failure log at each scale, and the DDP, FSDP, and DeepSpeed
comparison write-up, plus the JAX sharding stretch note. The
4-GPU scope is the parked v1.1 remainder. See the root README
level map.
