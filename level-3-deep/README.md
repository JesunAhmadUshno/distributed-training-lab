# Level 3, Deep: two GPU nodes over 1 GbE (P19, CLOSED)

Status: CLOSED 2026-11-07. Measured on two NVIDIA GB10 nodes (DGX Spark),
NCCL 2.29.7 over TCP on a 1000 Mb/s Ethernet link, torch 2.12.0+cu130.
Pre-registered in PREREG.md before any main-grid run. 5 of 7 hypotheses
pass; H5 and H6 missed their pass lines narrowly and are reported as
missed.

## Question

On two GPU nodes joined by commodity 1 GbE, when does data-parallel
training beat one node, and can a two-parameter link model measured in
two minutes predict the answer before the training runs?

## Answer

Almost never, and yes. Two nodes are slower than one in 9 of 12 cells,
down to 0.07x for the 92M-parameter model at accumulation 1. They win
only at accumulation 64 (1.31x to 1.47x). The link model
t(S) = alpha + S / beta, fitted on all-reduce timings alone, predicted
every two-node step time within 2.7% before any two-node training ran.

## Hypotheses, as scored by analysis.py

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

## Speedup of two nodes over one (DDP, fp32 gradients)

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

## What the misses mean

- H5: the 30% line was set for every cell the model marks as at least
  half communication. S at k=16 sits nearest that boundary (predicted
  comm share 0.56), so halving the bytes cut 29.4%, not 30%. The loss
  side of H5 held everywhere (relative difference under 2e-6).
- H6: FSDP2 saved 34.7% of peak memory on L, just short of the 35%
  needed. Activations and the per-block all-gather buffer are not
  sharded, and at 92M parameters they are a large share of the peak.
  FSDP2 was also the slowest mode (4.60 s per step vs 3.19 s for DDP),
  because it moves parameters as well as gradients over the same link.

## Design (summary; full text in PREREG.md)

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

## Threats and limits

- Two nodes only. Nothing here says how the curve bends at 4 or 8.
- Shared office switch. Mitigated by shuffled order, repeats, medians,
  slower-rank timing; intervals are tight (most under 5%).
- Node 1 hosts an idle vLLM server (about 83 GB of unified memory). Its
  GPU utilisation was 0% at every repeat (runs/main-gpu-util.log).
- Throughput only: 5 to 10 timed steps per run, no convergence claim.
- DeepSpeed was not run; the comparison is DDP, compressed DDP, FSDP2.
- Harness bring-up runs (runs/bringup/) and one relaunch of E1 FSDP2
  after a harness bug are logged in runs/RUNLOG.md and excluded.

## Reproduce

Hardware: two CUDA nodes on one subnet, torch with NCCL. Set MASTER and
NCCL_SOCKET_IFNAME in cluster/launch.sh, then from cluster/:

```
./launch.sh 2 e1_correct.py --mode ddp --out ../runs/main.jsonl
python3 run_grid.py single ../runs/main.jsonl
python3 analysis.py predict ../runs/main.jsonl
python3 run_grid.py two ../runs/main.jsonl
python3 analysis.py report ../runs/main.jsonl
```

The scoring alone needs only Python's standard library:
`python3 analysis.py report ../runs/main.jsonl` reproduces
results_deep.json from the committed evidence, and
`python3 -m unittest test_analysis` checks the analysis on known answers.

## Files

- PREREG.md: question, hypotheses, pass lines, design, analysis plan.
- predictions.json: locked per-cell predictions. results_deep.json: scores.
- runs/main.jsonl: every result line, each stamped with hosts, versions, link.
- runs/RUNLOG.md: everything that happened, including failures.
- cluster/: the harness (see cluster/README.md).
- PREP-README.md, DEEP-SPEC.md, KAGGLE-RUN.md, kaggle/: the superseded
  2026-11-05 Kaggle route, kept for the record.
