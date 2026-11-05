# Level 1, Learn: distributed-training-lab

Status: DONE (2026-11-05). A simulated data-parallel training run
of a tiny linear model, built with the Python standard library
only. Early pull from Sprint 3 under the December compression
protocol.

## Honest scope line

The two ranks in this level are simulated in one process. This is
not real PyTorch DDP, it uses no GPU, it starts no processes, and
it transfers no bytes over any network. Communication volume is
accounting only: a byte count computed from the parameter count,
never a measured transfer, and no wall time or throughput number
is claimed anywhere. The SPEC Learn target of real DDP on 2 GPUs
needs real GPU hardware; the TensorFlow MirroredStrategy bridge
run named in the SPEC (register item 229) is Build and Deep
scope and is deferred, not done here. Deep for this project is
defined by Addendum B of the December compression protocol as
the 2-GPU Kaggle exercise set; this Learn level makes no GPU
claim of any kind.

## Method

1. Model and data. The model is one linear rule, prediction =
   w1 * x1 + w2 * x2 + b, with 3 parameters [w1, w2, b]. The
   dataset is 8 synthetic examples whose targets follow
   y = 2 * x1 - 1 * x2 + 1 exactly, so the true parameters are
   [2.0, -1.0, 1.0] and the data carries no noise. Loss is mean
   squared error.
2. Sharding. The dataset is split into contiguous shards, one
   per simulated rank. The training run uses 2 ranks with
   balanced shards of 4 and 4. Shard sizes from the split
   function differ by at most one.
3. Per-rank gradients. Each rank computes the gradient of the
   loss on its own shard only, at its own replica parameters.
   For one example with error e = prediction - y, the gradient
   contribution is 2 * e * [x1, x2, 1], averaged over the shard.
4. All-reduce. The rank gradients are averaged element by
   element in code. Every rank then holds the same averaged
   gradient. With equal shard sizes this average equals the
   single-process gradient on the full batch; the run records
   the difference at every step and it stays at rounding level.
5. Synchronous step. The same update, parameters minus learning
   rate times the averaged gradient, is applied to every
   replica. Replicas start identical and receive identical
   updates, so they stay identical; the run verifies the
   replica difference at every step instead of assuming it.
6. Logging. Each step logs the per-rank losses, the per-rank
   gradients, the averaged gradient, the full-batch gradient,
   their maximum difference, the combined full-batch loss
   before the update, and the parameters after the update.
7. Communication accounting. Each rank contributes one gradient
   vector per step: 3 parameters at 4 bytes (float32) is a
   12-byte payload per rank per step, 24 bytes per step across
   2 ranks, 480 bytes over 20 steps. This is a computed count,
   not a transfer measurement.
8. Shard imbalance. A separate check splits the data 6 and 2.
   The naive all-reduce average weights both shards equally,
   so it differs from the full-batch gradient. The size-weighted
   average matches the full-batch gradient. Balanced shards, or
   size weighting, are required for the averaging rule to be
   correct.

## The scenario

One training run: 2 simulated ranks, 8 examples in shards of 4
and 4, 20 synchronous steps, learning rate 0.05, initial
parameters [0.0, 0.0, 0.0]. One imbalance check at the initial
parameters with shards of 6 and 2, reported separately below.

## Run it

    python level-1-learn/run_learn.py
    python -m unittest tests.test_learn_data_parallel -v
    python tests/test_smoke.py

The runner writes level-1-learn/results_learn.json
deterministically: run it twice and the file is identical
(verified 2026-11-05:
sha256 1e0e07285fb577b972ceeeaf1c194a3e39f58e02789bd18b3ddb4ab0607ab9d6
on both runs).

## Results (measured 2026-11-05, Muse VM, Python 3.12)

Step 1, hand-checkable: rank 0 loss 9.5 with gradient
[-7.5, -1.0, -5.0], rank 1 loss 13.5 with gradient
[-13.5, -5.0, -5.0]. Their average is [-10.5, -3.0, -5.0],
equal to the single-process full-batch gradient, and the first
update gives parameters [0.525, 0.15, 0.25].

| Step | Rank 0 loss | Rank 1 loss | Combined loss before update |
| --- | --- | --- | --- |
| 1 | 9.500000 | 13.500000 | 11.500000 |
| 2 | 4.989063 | 6.277813 | 5.633437 |
| 3 | 2.956100 | 3.548955 | 3.252528 |
| 4 | 1.951084 | 2.412671 | 2.181878 |
| 5 | 1.399193 | 1.847305 | 1.623249 |
| 6 | 1.063171 | 1.497345 | 1.280258 |
| 7 | 0.839677 | 1.240357 | 1.040017 |
| 8 | 0.680570 | 1.033895 | 0.857233 |
| 9 | 0.561630 | 0.862222 | 0.711926 |
| 10 | 0.469640 | 0.718228 | 0.593934 |
| 11 | 0.396800 | 0.597538 | 0.497169 |
| 12 | 0.338167 | 0.496717 | 0.417442 |
| 13 | 0.290416 | 0.412790 | 0.351603 |
| 14 | 0.251192 | 0.343130 | 0.297161 |
| 15 | 0.218764 | 0.285441 | 0.252102 |
| 16 | 0.191815 | 0.237745 | 0.214780 |
| 17 | 0.169325 | 0.198358 | 0.183841 |
| 18 | 0.150483 | 0.165857 | 0.158170 |
| 19 | 0.134642 | 0.139053 | 0.136847 |
| 20 | 0.121277 | 0.116954 | 0.119116 |

After the step 20 update the final parameters are
[1.995697, -0.545749, 0.537617] and the final combined loss is
0.104349. The w1 estimate is close to the true 2.0 after 20
steps; w2 and b are still moving toward -1.0 and 1.0, which is
expected at this step count and learning rate and is stated
plainly rather than rounded into a convergence claim.

Summary: Learn ran one deterministic simulated data-parallel training run in a single process on the Muse VM: 2 simulated ranks trained a 3-parameter linear model on 8 synthetic examples in balanced shards of 4 for 20 synchronous steps at learning rate 0.05, the all-reduce averaged gradient matched the single-process full-batch gradient at every step with max difference 8.88e-16, replicas stayed identical with max difference 0.0, combined loss fell from 11.5 to 0.104349, logical all-reduce traffic totalled 480 bytes by accounting only, and in a separate 6 plus 2 imbalance check the naive average differed from the full gradient by 2.833333 while the size-weighted average matched it, with 20 unit tests passing plus the scaffold smoke test.

Imbalance check (separate, at the initial parameters, shards
6 and 2): rank gradients are [-7.666667, -1.333333, -4.333333]
and [-19.0, -8.0, -7.0]. The naive average is
[-13.333333, -4.666667, -5.666667], a maximum difference of
2.833333 from the full-batch gradient [-10.5, -3.0, -5.0]. The
size-weighted average is [-10.5, -3.0, -5.0], difference 0.0.

Hand checks: the initial full-batch gradient is
(2/8) * (-42, -12, -20) = (-10.5, -3.0, -5.0), from error sums
-42 for x1, -12 for x2, and -20 for the bias over the 8
examples at zero parameters. The initial loss is
(9 + 0 + 4 + 25 + 1 + 16 + 1 + 36) / 8 = 92 / 8 = 11.5. The
shard gradients are (2/4) * (-15, -2, -10) = (-7.5, -1.0, -5.0)
and (2/4) * (-27, -10, -10) = (-13.5, -5.0, -5.0), and their
mean is the full-batch gradient. Communication accounting is
3 params * 4 bytes = 12 bytes per rank per step, times 2 ranks
times 20 steps = 480 bytes.

## Failure modes and limits

- Simulated ranks only: one process, no real DDP, no GPU, no
  processes, no network. Nothing here measures real distributed
  speed, scaling, or interconnect behaviour, and no speedup or
  throughput claim is made.
- Tiny synthetic data: 8 examples, 2 features, no noise, one
  fixed dataset. The gradient equality is an arithmetic fact
  about averaging, shown on this data; no claim is made about
  real datasets, large models, or stochastic minibatch noise.
- Synchronous updates only: every rank waits for the average
  each step by construction. Stragglers, asynchronous updates,
  gradient compression, and communication overlap are out of
  scope at this level.
- The naive average is wrong under shard imbalance: the 6 and 2
  check differs from the full gradient by 2.833333. A real
  system must balance shards or weight by shard size; this
  level names and measures the effect, it does not fix a real
  sampler.
- Communication bytes are accounting only, from the parameter
  count. Ring or tree all-reduce traffic patterns, latency, and
  bandwidth are Build and Deep scope and are not modelled here.
- One model shape: a 3-parameter linear model with full-batch
  gradients. The P01 GPT, minibatching, and optimizers beyond
  plain gradient descent are later level scope.
- This level was run on the Muse VM (CPU), not on the fleet
  node BlackBox (GPU). That costs a grade deduction, named in
  the grade. The TensorFlow MirroredStrategy bridge run from
  the SPEC is deferred to Build and Deep, not done here.

## Grade

Grade: 8,400 of 10,000 for this Learn level.

Deductions (they sum to 1,600):
- Simulated ranks in one process, not real DDP and no GPU:
  minus 700.
- Tiny synthetic dataset (8 examples) and a 3-parameter linear
  model only: minus 400.
- Single-process sequential execution: no real communication,
  no processes, and no scaling behaviour measured: minus 300.
- Run on the Muse VM, not the assigned BlackBox GPU fleet
  node: minus 200.

## What Build adds next

The same training run under FSDP and under DeepSpeed with
checkpoint save and restart mid-run, plus the deferred
TensorFlow MirroredStrategy bridge run, on real hardware. Deep
then follows Addendum B: the 2-GPU Kaggle exercise set with a
scaling table and failure log, honestly labelled. See the root
README level map.
