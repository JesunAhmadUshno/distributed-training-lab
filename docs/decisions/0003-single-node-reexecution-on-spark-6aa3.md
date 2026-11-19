# 0003: Single-node arm re-executed on spark-6aa3 (gap closure, 2026-11-17)

Date: 2026-11-17. Status: accepted (gap-closure re-execution record).

Context. The Deep study's paper review verified every number by
recomputation from the recorded run lines in runs/main.jsonl, but no GPU
run was ever re-executed. Jesun ordered the machine-window gap-closure
runs on 2026-11-17. Boundary for this run: spark-6aa3 only. spark-31c4
is not contacted in any way: it stays org-testing-only, and the
2026-11-11 boundary lift covers spark-6aa3 only. The recorded single-node
arm ran on spark-6aa3 itself (env stamps in runs/main.jsonl: hosts
["spark-6aa3"], torch 2.12.0+cu130, CUDA 13.0, NVIDIA GB10), and this
machine's stack is unchanged, so the single-node arm can be re-executed
on the same host and stack as the record.

Decision. Re-execute the E3 single-node arm (sizes S, M, L x accumulation
1, 4, 16, 64 x 3 repeats, micro-batch 8, warmup and timed steps exactly
as PREREG.md and run_grid.py fix them) on spark-6aa3. Cells are launched
through launch.sh with 1 node, in run_grid.py's per-repeat shuffled
order (random.Random(rep) over the same cell list). run_grid.py itself
is not used, because its per-repeat GPU-utilisation probe ssh's to
spark-31c4; the probe is omitted, nothing else changes. Fresh run lines
go to the new file runs/rerun-2026-11-17.jsonl. runs/main.jsonl,
predictions.json, results_deep.json and PREREG.md are not modified.
Analysis uses this repo's analysis.py (cell_stat, comm_time) unchanged.
Three comparisons are reported: (a) fresh T_single against the committed
cell values in results_deep.json; (b) an H3/H4 re-check: predictions
recomputed from the fresh T_single values and the RECORDED E2 link fit
(alpha 0.00039636 s, beta 117.4365 MB/s, from predictions.json), scored
against the COMMITTED two-node step times; (c) fresh peak memory against
the committed peaks.

Not re-executed, with reasons. E2 measures the spark-6aa3 to spark-31c4
1 GbE path; a one-node all-reduce measures loopback, not the registered
link, so E2 cannot be re-run under the boundary. The E3 ddp, ddp_bf16
and fsdp cells, the E1 two-node arms and E4 all need the second node.
The DeepSpeed arm stays parked (decision 0001, REPORT.md future work);
a re-execution does not widen the registered study.

Consequences. The rerun record stands beside the original record. No
frozen verdict changes whatever the fresh numbers say; any divergence
is reported as measured, with both values.
