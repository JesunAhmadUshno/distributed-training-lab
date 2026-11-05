"""Simulated data-parallel training, Level 1 Learn.

This module teaches data-parallel training mechanics in one process
with the Python standard library only. Two simulated ranks hold
identical copies of a tiny 3-parameter linear model. Each rank
computes the gradient on its own shard of an 8-example synthetic
dataset, an all-reduce step averages the rank gradients in code,
and one synchronous update is applied to every replica.

The ranks are simulated in one process. This is NOT real PyTorch
DDP, it uses NO GPU, it starts no processes, and it moves no bytes
over any network. Communication volume is accounting only: a byte
count computed from the parameter count, never a measured transfer.
Real DDP on 2 GPUs is the SPEC Learn target on real hardware, and
the TensorFlow MirroredStrategy bridge run named in the SPEC is
Build and Deep scope, deferred at this level. Deep later follows
Addendum B: the 2-GPU Kaggle exercise set. This Learn level makes
no GPU claim of any kind.

Model: prediction = w1 * x1 + w2 * x2 + b, parameters [w1, w2, b].
Loss: mean squared error over the examples in scope. Gradient of
the loss for one example with error e = prediction - y is
2 * e * [x1, x2, 1], averaged over the examples in scope.
"""


class DataParallelError(Exception):
    """Named error for invalid data, shards, and training inputs."""


BYTES_PER_PARAM = 4  # float32 accounting only, nothing is transferred


def build_dataset():
    """The Learn dataset: 8 synthetic examples, hand-checkable.

    Every target follows y = 2 * x1 - 1 * x2 + 1 exactly, so the
    true parameters are [2.0, -1.0, 1.0] and the data has no noise.
    """
    rows = [
        (1.0, 0.0, 3.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 2.0),
        (2.0, 0.0, 5.0),
        (0.0, 2.0, -1.0),
        (2.0, 1.0, 4.0),
        (1.0, 2.0, 1.0),
        (3.0, 1.0, 6.0),
    ]
    return [{"x": [r[0], r[1]], "y": r[2]} for r in rows]


def validate_params(params):
    if (not isinstance(params, list) or len(params) != 3
            or not all(isinstance(v, (int, float)) for v in params)):
        raise DataParallelError("params must be a list of 3 numbers [w1, w2, b]")


def validate_examples(examples):
    if not isinstance(examples, list) or not examples:
        raise DataParallelError("examples must be a non-empty list")
    for ex in examples:
        if (not isinstance(ex.get("x"), list) or len(ex["x"]) != 2
                or not all(isinstance(v, (int, float)) for v in ex["x"])
                or not isinstance(ex.get("y"), (int, float))):
            raise DataParallelError("each example needs x of 2 numbers and a numeric y")


def predict(params, x):
    validate_params(params)
    return params[0] * x[0] + params[1] * x[1] + params[2]


def compute_loss(params, examples):
    """Mean squared error of params over examples."""
    validate_params(params)
    validate_examples(examples)
    total = 0.0
    for ex in examples:
        err = predict(params, ex["x"]) - ex["y"]
        total += err * err
    return total / len(examples)


def compute_gradient(params, examples):
    """Mean gradient of the MSE loss over examples.

    Per example the gradient is 2 * e * [x1, x2, 1] with
    e = prediction - y. The result is the mean over examples.
    """
    validate_params(params)
    validate_examples(examples)
    grad = [0.0, 0.0, 0.0]
    for ex in examples:
        err = predict(params, ex["x"]) - ex["y"]
        grad[0] += 2.0 * err * ex["x"][0]
        grad[1] += 2.0 * err * ex["x"][1]
        grad[2] += 2.0 * err
    n = float(len(examples))
    return [g / n for g in grad]


def shard_dataset(examples, num_ranks):
    """Split examples into num_ranks contiguous shards.

    Shard sizes differ by at most one: the first shards take one
    extra example when the count does not divide evenly. Shards
    are returned in dataset order.
    """
    validate_examples(examples)
    if not isinstance(num_ranks, int) or num_ranks < 1:
        raise DataParallelError("num_ranks must be a positive integer")
    if num_ranks > len(examples):
        raise DataParallelError("num_ranks cannot exceed the example count")
    base = len(examples) // num_ranks
    extra = len(examples) % num_ranks
    shards = []
    start = 0
    for rank in range(num_ranks):
        size = base + (1 if rank < extra else 0)
        shards.append(examples[start:start + size])
        start += size
    return shards


def all_reduce_average(gradients):
    """Average one gradient vector per rank, element by element.

    This is the simulated all-reduce: every rank ends the step with
    this same averaged gradient. With equal shard sizes the average
    equals the single-process gradient on the full batch. With
    unequal shard sizes it does not: see the imbalance experiment.
    """
    if not isinstance(gradients, list) or not gradients:
        raise DataParallelError("gradients must be a non-empty list")
    for grad in gradients:
        validate_params(grad)
    out = [0.0, 0.0, 0.0]
    for grad in gradients:
        for i in range(3):
            out[i] += grad[i]
    return [v / len(gradients) for v in out]


def weighted_average(gradients, weights):
    """Average gradients weighted by shard size (the correct form)."""
    if len(gradients) != len(weights) or not gradients:
        raise DataParallelError("gradients and weights must match and be non-empty")
    if any(w <= 0 for w in weights) or sum(weights) == 0:
        raise DataParallelError("weights must be positive")
    total = float(sum(weights))
    out = [0.0, 0.0, 0.0]
    for grad, w in zip(gradients, weights):
        validate_params(grad)
        for i in range(3):
            out[i] += grad[i] * w
    return [v / total for v in out]


def max_abs_diff(a, b):
    validate_params(a)
    validate_params(b)
    return max(abs(x - y) for x, y in zip(a, b))


def communication_accounting(num_params, num_ranks, steps):
    """Logical all-reduce byte accounting. Nothing is transferred.

    Each rank contributes one gradient vector per step. The payload
    per rank per step is num_params * BYTES_PER_PARAM bytes. The
    logical total per step is that payload times num_ranks, and the
    run total is the per-step total times steps.
    """
    if num_params < 1 or num_ranks < 1 or steps < 1:
        raise DataParallelError("accounting inputs must be positive")
    per_rank = num_params * BYTES_PER_PARAM
    return {
        "num_params": num_params,
        "bytes_per_param": BYTES_PER_PARAM,
        "payload_bytes_per_rank_per_step": per_rank,
        "bytes_per_step_all_ranks": per_rank * num_ranks,
        "steps": steps,
        "total_bytes_all_ranks": per_rank * num_ranks * steps,
        "total_bytes_per_rank": per_rank * steps,
    }


def train_data_parallel(examples, num_ranks=2, steps=20, lr=0.05,
                        initial_params=None):
    """Run synchronous simulated data-parallel training.

    Every replica starts from the same parameters. Each step, each
    simulated rank computes its shard gradient, the all-reduce
    average is formed, its difference from the single-process
    full-batch gradient is recorded, and the same update is applied
    to every replica, so replicas stay identical by construction
    and the run verifies that fact at every step.
    """
    validate_examples(examples)
    if not isinstance(steps, int) or steps < 1:
        raise DataParallelError("steps must be a positive integer")
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise DataParallelError("lr must be a positive number")
    params0 = list(initial_params) if initial_params is not None else [0.0, 0.0, 0.0]
    validate_params(params0)
    shards = shard_dataset(examples, num_ranks)
    replicas = [list(params0) for _ in range(num_ranks)]
    log = []
    max_grad_diff = 0.0
    max_replica_diff = 0.0
    for step in range(1, steps + 1):
        rank_grads = [compute_gradient(replicas[r], shards[r])
                      for r in range(num_ranks)]
        rank_losses = [compute_loss(replicas[r], shards[r])
                       for r in range(num_ranks)]
        averaged = all_reduce_average(rank_grads)
        full = compute_gradient(replicas[0], examples)
        grad_diff = max_abs_diff(averaged, full)
        max_grad_diff = max(max_grad_diff, grad_diff)
        combined_loss = compute_loss(replicas[0], examples)
        for r in range(num_ranks):
            replicas[r] = [p - lr * g for p, g in zip(replicas[r], averaged)]
        replica_diff = 0.0
        for r in range(1, num_ranks):
            replica_diff = max(replica_diff, max_abs_diff(replicas[0], replicas[r]))
        max_replica_diff = max(max_replica_diff, replica_diff)
        log.append({
            "step": step,
            "rank_losses": rank_losses,
            "rank_gradients": rank_grads,
            "averaged_gradient": averaged,
            "full_batch_gradient": full,
            "grad_max_abs_diff": grad_diff,
            "combined_loss_before_update": combined_loss,
            "params_after_update": list(replicas[0]),
            "replica_max_abs_diff_after_update": replica_diff,
        })
    accounting = communication_accounting(3, num_ranks, steps)
    summary = {
        "num_ranks": num_ranks,
        "num_examples": len(examples),
        "shard_sizes": [len(s) for s in shards],
        "steps": steps,
        "learning_rate": lr,
        "initial_params": params0,
        "initial_loss": compute_loss(params0, examples),
        "final_params": list(replicas[0]),
        "final_loss": compute_loss(replicas[0], examples),
        "max_grad_diff_vs_single_process": max_grad_diff,
        "max_replica_diff": max_replica_diff,
        "communication": accounting,
    }
    return {"log": log, "summary": summary, "replicas_final": replicas}


def imbalance_experiment(examples, initial_params=None):
    """Show the shard imbalance effect at the initial parameters.

    Splits the 8 examples as 6 and 2. The naive all-reduce average
    gives the small shard the same weight as the large shard, so it
    differs from the full-batch gradient. The size-weighted average
    matches the full-batch gradient exactly.
    """
    params0 = list(initial_params) if initial_params is not None else [0.0, 0.0, 0.0]
    validate_params(params0)
    validate_examples(examples)
    if len(examples) != 8:
        raise DataParallelError("the imbalance experiment is defined on 8 examples")
    shards = [examples[:6], examples[6:]]
    grads = [compute_gradient(params0, s) for s in shards]
    naive = all_reduce_average(grads)
    weighted = weighted_average(grads, [len(s) for s in shards])
    full = compute_gradient(params0, examples)
    return {
        "shard_sizes": [6, 2],
        "rank_gradients": grads,
        "naive_average": naive,
        "weighted_average": weighted,
        "full_batch_gradient": full,
        "naive_max_abs_diff_vs_full": max_abs_diff(naive, full),
        "weighted_max_abs_diff_vs_full": max_abs_diff(weighted, full),
    }
