"""Sharded training emulation, Level 2 Build.

Two parameter-sharding strategies on the SAME model, data, seed
values, and step count, in one process with the Python standard
library only:

- FSDP-class: parameters are sharded per simulated rank. Each step
  all-gathers the shards into a full parameter copy, computes the
  full-batch gradient, reduce-scatters the gradient so each rank
  holds only its gradient shard, and each rank updates only its
  own parameter shard with its own optimizer (velocity) shard.
- DeepSpeed ZeRO-class: optimizer state and gradients are
  partitioned per simulated rank. Each rank updates only its own
  parameter shard with its own gradient shard and velocity shard,
  and the updated shards are all-gathered after the update.

The ranks are simulated in one process. This is NOT real PyTorch
FSDP and NOT real DeepSpeed, it uses NO GPU, it starts no
processes, and it transfers no bytes over any network.
Communication and memory figures are logical accounting computed
in code from the parameter count, never measured transfers or
measured memory. Both strategies perform the same arithmetic per
element as a single-process reference trainer, so their final
parameters match the reference exactly in this emulation; the
level teaches the sharding, checkpoint, and accounting mechanics,
not a performance difference.

Model: a 2-layer MLP, input 4, hidden 5 with tanh, output 1.
Parameter vector layout (31 parameters, flat list):
  indices 0 to 19:  W1, row-major 4 by 5 (W1[i * 5 + j])
  indices 20 to 24: b1, 5 values
  indices 25 to 29: W2, 5 values
  index 30:         b2, 1 value
Loss: mean squared error over 16 synthetic examples.
Optimizer: SGD with momentum, velocity v = momentum * v - lr * g,
then p = p + v. The velocity vector is sharded with the parameters.
"""

import hashlib
import json
import math


class ShardingError(Exception):
    """Named error for invalid sharding, training, and checkpoint inputs."""


BYTES_PER_PARAM = 4  # float32 accounting only, nothing is transferred
INPUT_DIM = 4
HIDDEN_DIM = 5
NUM_PARAMS = INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM + 1  # 31
STRATEGIES = ("fsdp", "zero")


def build_dataset():
    """16 synthetic examples, deterministic formulas, no randomness.

    x components are fixed arithmetic patterns in [-1, 1]. Targets
    follow y = 2 * x1 - 1 * x2 + 0.5 * x3 + 1 * x4 + 0.5 exactly.
    """
    examples = []
    for i in range(16):
        x = [
            ((i % 4) - 1.5) / 1.5,
            ((i // 4) - 1.5) / 1.5,
            (((i * 3) % 5) - 2) / 2.0,
            float((i * 7) % 3 - 1),
        ]
        y = 2.0 * x[0] - 1.0 * x[1] + 0.5 * x[2] + 1.0 * x[3] + 0.5
        examples.append({"x": x, "y": y})
    return examples


def initial_params():
    """Deterministic small initial parameters, formula only.

    p_k = 0.05 * (((k * 11) mod 7) - 3), so values lie in
    [-0.15, 0.15] and the sequence is fully reproducible.
    """
    return [0.05 * (((k * 11) % 7) - 3) for k in range(NUM_PARAMS)]


def validate_vector(vec, length, name):
    if (not isinstance(vec, list) or len(vec) != length
            or not all(isinstance(v, (int, float)) for v in vec)):
        raise ShardingError(f"{name} must be a list of {length} numbers")


def validate_examples(examples):
    if not isinstance(examples, list) or not examples:
        raise ShardingError("examples must be a non-empty list")
    for ex in examples:
        if (not isinstance(ex.get("x"), list) or len(ex["x"]) != INPUT_DIM
                or not all(isinstance(v, (int, float)) for v in ex["x"])
                or not isinstance(ex.get("y"), (int, float))):
            raise ShardingError("each example needs x of 4 numbers and a numeric y")


def forward(params, x):
    """Return (prediction, hidden activations) for one example."""
    validate_vector(params, NUM_PARAMS, "params")
    hidden = []
    for j in range(HIDDEN_DIM):
        pre = params[20 + j]
        for i in range(INPUT_DIM):
            pre += x[i] * params[i * HIDDEN_DIM + j]
        hidden.append(math.tanh(pre))
    out = params[30]
    for j in range(HIDDEN_DIM):
        out += hidden[j] * params[25 + j]
    return out, hidden


def predict(params, x):
    return forward(params, x)[0]


def compute_loss(params, examples):
    validate_examples(examples)
    total = 0.0
    for ex in examples:
        err = predict(params, ex["x"]) - ex["y"]
        total += err * err
    return total / len(examples)


def compute_gradient(params, examples):
    """Full-batch mean gradient of the MSE loss, flat 31-vector.

    Backpropagation per example: with e = prediction - y,
    dL/dprediction = 2 * e / N. W2 gradient is that times the
    hidden activation, b2 gradient is that value, and the hidden
    layer receives it times W2 times (1 - tanh^2), which gives the
    W1 and b1 gradients through the inputs.
    """
    validate_vector(params, NUM_PARAMS, "params")
    validate_examples(examples)
    grad = [0.0] * NUM_PARAMS
    n = float(len(examples))
    for ex in examples:
        pred, hidden = forward(params, ex["x"])
        d_out = 2.0 * (pred - ex["y"]) / n
        for j in range(HIDDEN_DIM):
            grad[25 + j] += d_out * hidden[j]
        grad[30] += d_out
        for j in range(HIDDEN_DIM):
            d_pre = d_out * params[25 + j] * (1.0 - hidden[j] * hidden[j])
            grad[20 + j] += d_pre
            for i in range(INPUT_DIM):
                grad[i * HIDDEN_DIM + j] += d_pre * ex["x"][i]
    return grad


def partition_indices(num_params, num_ranks):
    """Contiguous partition of range(num_params) across ranks.

    Shard sizes differ by at most one: the first shards take one
    extra index when the count does not divide evenly. Every index
    appears in exactly one shard, in order.
    """
    if not isinstance(num_params, int) or num_params < 1:
        raise ShardingError("num_params must be a positive integer")
    if not isinstance(num_ranks, int) or num_ranks < 1:
        raise ShardingError("num_ranks must be a positive integer")
    if num_ranks > num_params:
        raise ShardingError("num_ranks cannot exceed num_params")
    base = num_params // num_ranks
    extra = num_params % num_ranks
    parts = []
    start = 0
    for rank in range(num_ranks):
        size = base + (1 if rank < extra else 0)
        parts.append(list(range(start, start + size)))
        start += size
    return parts


def shard_vector(vec, parts):
    """Split a flat vector into per-rank shards by index partition."""
    shards = []
    for idxs in parts:
        shard = []
        for i in idxs:
            if i < 0 or i >= len(vec):
                raise ShardingError("partition index out of range for vector")
            shard.append(vec[i])
        shards.append(shard)
    return shards


def all_gather(shards, parts, num_params):
    """Reconstruct the full vector from per-rank shards.

    This is the simulated all-gather: the caller ends with the full
    parameter vector assembled in partition order. Shard lengths
    must match the partition exactly.
    """
    if len(shards) != len(parts):
        raise ShardingError("shards and partition must have the same rank count")
    out = [None] * num_params
    for shard, idxs in zip(shards, parts):
        if len(shard) != len(idxs):
            raise ShardingError("shard length does not match its partition size")
        for value, i in zip(shard, idxs):
            if out[i] is not None:
                raise ShardingError("partition assigns an index twice")
            out[i] = value
    if any(v is None for v in out):
        raise ShardingError("partition does not cover every parameter")
    return out


def max_abs_diff(a, b):
    if len(a) != len(b):
        raise ShardingError("vectors must have the same length")
    return max(abs(x - y) for x, y in zip(a, b))


def config_for(strategy, steps, lr, momentum, checkpoint_step, num_ranks):
    if strategy not in STRATEGIES:
        raise ShardingError(f"strategy must be one of {STRATEGIES}")
    return {
        "strategy": strategy,
        "num_params": NUM_PARAMS,
        "num_ranks": num_ranks,
        "steps": steps,
        "learning_rate": lr,
        "momentum": momentum,
        "checkpoint_step": checkpoint_step,
        "num_examples": len(build_dataset()),
    }


def config_hash(config):
    blob = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def communication_accounting(num_params, shard_sizes, steps):
    """Logical communication accounting per strategy. No transfer.

    Per step, both strategies move the same logical volume in this
    emulation:
    - all-gather of parameters: each rank receives the other ranks
      shards, (num_params - own shard) elements at 4 bytes. Summed
      over 2 ranks with shards s and (num_params - s), that is
      num_params * 4 bytes in total received volume.
    - reduce-scatter of gradients: each rank receives its own
      gradient shard, its shard size at 4 bytes. Summed over ranks
      that is num_params * 4 bytes.
    Per step total across ranks is therefore 2 * num_params * 4
    = 248 bytes for 31 parameters, and the run total is that times
    the step count. These are computed counts, not measurements.
    """
    if num_params < 1 or steps < 1 or not shard_sizes:
        raise ShardingError("accounting inputs must be positive")
    b = BYTES_PER_PARAM
    all_gather = sum((num_params - s) * b for s in shard_sizes)
    reduce_scatter = sum(s * b for s in shard_sizes)
    per_step = all_gather + reduce_scatter
    return {
        "bytes_per_param": b,
        "all_gather_bytes_per_step_all_ranks": all_gather,
        "reduce_scatter_bytes_per_step_all_ranks": reduce_scatter,
        "bytes_per_step_all_ranks": per_step,
        "steps": steps,
        "total_bytes_all_ranks": per_step * steps,
        "total_bytes_per_rank_average": per_step * steps // len(shard_sizes),
    }


def memory_accounting(num_params, shard_sizes):
    """Logical per-rank stored elements. Accounting only, no RSS.

    Persistent storage per rank is the parameter shard plus the
    velocity shard: 2 * shard size. Peak logical storage adds the
    transient gathered full parameter copy and the gradient shard
    held at the update: shard + shard + num_params + shard
    = num_params + 3 * shard size. The single-process baseline
    holds parameters, gradient, and velocity in full:
    3 * num_params = 93 elements.
    """
    if num_params < 1 or not shard_sizes:
        raise ShardingError("accounting inputs must be positive")
    return {
        "baseline_full_elements_per_rank": 3 * num_params,
        "persistent_elements_per_rank": [2 * s for s in shard_sizes],
        "peak_elements_per_rank": [num_params + 3 * s for s in shard_sizes],
        "peak_elements_max_rank": max(num_params + 3 * s for s in shard_sizes),
    }


class ShardedTrainer:
    """One sharded training run for one strategy.

    State is held ONLY as per-rank shards: param_shards and
    velocity_shards. A full parameter vector exists only as a
    transient gathered copy inside a step (FSDP-class: gathered
    before the gradient; ZeRO-class: gathered after the update).
    """

    def __init__(self, strategy, examples, steps=30, lr=0.05, momentum=0.5,
                 num_ranks=2, checkpoint_step=12):
        validate_examples(examples)
        if not isinstance(steps, int) or steps < 1:
            raise ShardingError("steps must be a positive integer")
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ShardingError("lr must be a positive number")
        if not isinstance(momentum, (int, float)) or not (0.0 <= momentum < 1.0):
            raise ShardingError("momentum must be in [0, 1)")
        if (not isinstance(checkpoint_step, int)
                or not (1 <= checkpoint_step < steps)):
            raise ShardingError("checkpoint_step must be between 1 and steps - 1")
        self.config = config_for(strategy, steps, lr, momentum,
                                 checkpoint_step, num_ranks)
        self.examples = examples
        self.parts = partition_indices(NUM_PARAMS, num_ranks)
        self.param_shards = shard_vector(initial_params(), self.parts)
        self.velocity_shards = [[0.0] * len(p) for p in self.parts]
        self.step = 0
        self.log = []

    def full_params(self):
        return all_gather(self.param_shards, self.parts, NUM_PARAMS)

    def _update_shards(self, grad_shards):
        for r in range(len(self.parts)):
            for k in range(len(self.param_shards[r])):
                v = (self.config["momentum"] * self.velocity_shards[r][k]
                     - self.config["learning_rate"] * grad_shards[r][k])
                self.velocity_shards[r][k] = v
                self.param_shards[r][k] += v

    def run_step(self):
        """Advance one step using the strategy phase order."""
        strategy = self.config["strategy"]
        lr_loss_params = self.full_params()
        loss_before = compute_loss(lr_loss_params, self.examples)
        grad = compute_gradient(lr_loss_params, self.examples)
        grad_shards = shard_vector(grad, self.parts)
        if strategy == "fsdp":
            # Phase order: all-gather params (done above for the
            # forward), compute gradient, reduce-scatter gradient
            # into shards, update each shard in place.
            self._update_shards(grad_shards)
        elif strategy == "zero":
            # Phase order: gradient partitioned first, each rank
            # updates only its own shard, all-gather after update.
            self._update_shards(grad_shards)
            _ = self.full_params()  # the post-update all-gather
        else:  # guarded by config_for, kept for direct construction
            raise ShardingError(f"unknown strategy {strategy}")
        self.step += 1
        entry = {
            "step": self.step,
            "loss_before_update": loss_before,
            "params_after_update": self.full_params(),
        }
        self.log.append(entry)
        return entry

    def run_to(self, target_step):
        if target_step < self.step or target_step > self.config["steps"]:
            raise ShardingError("target_step out of range for this trainer")
        while self.step < target_step:
            self.run_step()
        return self

    def save_checkpoint(self, path):
        """Write checkpoint JSON: step, shards, velocity, config hash."""
        ckpt = {
            "strategy": self.config["strategy"],
            "step": self.step,
            "param_shards": self.param_shards,
            "velocity_shards": self.velocity_shards,
            "config": self.config,
            "config_hash": config_hash(self.config),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, indent=2, sort_keys=True)
            f.write("\n")
        return ckpt

    @classmethod
    def from_checkpoint(cls, path, examples):
        """Restart a fresh trainer from a checkpoint file.

        The config hash is recomputed and must match, shard shapes
        must match the partition for the config, and the step must
        be the recorded checkpoint step.
        """
        with open(path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        config = ckpt.get("config")
        if not isinstance(config, dict) or ckpt.get("config_hash") != config_hash(config):
            raise ShardingError("checkpoint config hash mismatch")
        trainer = cls(
            config["strategy"], examples, steps=config["steps"],
            lr=config["learning_rate"], momentum=config["momentum"],
            num_ranks=config["num_ranks"],
            checkpoint_step=config["checkpoint_step"],
        )
        if ckpt.get("step") != config["checkpoint_step"]:
            raise ShardingError("checkpoint step is not the config checkpoint step")
        parts = trainer.parts
        for name in ("param_shards", "velocity_shards"):
            shards = ckpt.get(name)
            if (not isinstance(shards, list) or len(shards) != len(parts)
                    or any(len(s) != len(p) for s, p in zip(shards, parts))):
                raise ShardingError(f"checkpoint {name} shapes do not match the partition")
            for s in shards:
                validate_vector(s, len(s), name)
        trainer.param_shards = [list(s) for s in ckpt["param_shards"]]
        trainer.velocity_shards = [list(s) for s in ckpt["velocity_shards"]]
        trainer.step = ckpt["step"]
        return trainer


def train_single_process(examples, steps=30, lr=0.05, momentum=0.5):
    """Reference trainer: full vectors, same arithmetic per element."""
    validate_examples(examples)
    params = initial_params()
    velocity = [0.0] * NUM_PARAMS
    log = []
    for step in range(1, steps + 1):
        loss_before = compute_loss(params, examples)
        grad = compute_gradient(params, examples)
        velocity = [momentum * v - lr * g for v, g in zip(velocity, grad)]
        params = [p + v for p, v in zip(params, velocity)]
        log.append({"step": step, "loss_before_update": loss_before,
                    "params_after_update": list(params)})
    return {"log": log, "final_params": params,
            "initial_loss": compute_loss(initial_params(), examples),
            "final_loss": compute_loss(params, examples)}


def run_strategy(strategy, examples, checkpoint_path, steps=30, lr=0.05,
                 momentum=0.5, num_ranks=2, checkpoint_step=12):
    """Run one strategy uninterrupted, and again with a restart.

    The restarted run trains to checkpoint_step, saves a checkpoint
    to checkpoint_path, builds a FRESH trainer from that file, and
    continues to the final step. Returns measured comparisons
    against the single-process reference and the uninterrupted run.
    """
    ref = train_single_process(examples, steps=steps, lr=lr, momentum=momentum)
    trainer = ShardedTrainer(strategy, examples, steps=steps, lr=lr,
                             momentum=momentum, num_ranks=num_ranks,
                             checkpoint_step=checkpoint_step)
    initial_loss = compute_loss(trainer.full_params(), examples)
    trainer.run_to(steps)
    final_params = trainer.full_params()
    final_loss = compute_loss(final_params, examples)

    restarted = ShardedTrainer(strategy, examples, steps=steps, lr=lr,
                               momentum=momentum, num_ranks=num_ranks,
                               checkpoint_step=checkpoint_step)
    restarted.run_to(checkpoint_step)
    ckpt = restarted.save_checkpoint(checkpoint_path)
    fresh = ShardedTrainer.from_checkpoint(checkpoint_path, examples)
    fresh.run_to(steps)
    restarted_final = fresh.full_params()

    parts = partition_indices(NUM_PARAMS, num_ranks)
    shard_sizes = [len(p) for p in parts]
    return {
        "strategy": strategy,
        "num_ranks": num_ranks,
        "shard_sizes": shard_sizes,
        "steps": steps,
        "checkpoint_step": checkpoint_step,
        "learning_rate": lr,
        "momentum": momentum,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_params": final_params,
        "log": trainer.log,
        "max_diff_vs_single_process": max_abs_diff(final_params, ref["final_params"]),
        "restarted_final_params": restarted_final,
        "restart_max_diff_vs_uninterrupted": max_abs_diff(restarted_final, final_params),
        "checkpoint_config_hash": ckpt["config_hash"],
        "communication": communication_accounting(NUM_PARAMS, shard_sizes, steps),
        "memory": memory_accounting(NUM_PARAMS, shard_sizes),
    }
