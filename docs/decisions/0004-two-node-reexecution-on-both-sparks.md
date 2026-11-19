# 0004: Two-node arms re-executed on spark-6aa3 and spark-31c4 (gap closure, 2026-11-17)

Date: 2026-11-17. Status: accepted (owner ruling, gap-closure re-execution record).

Context. Decision 0003 re-executed the E3 single-node arm on spark-6aa3
and left E1, E2, the 25 two-node E3 cells and E4 unrun, because
spark-31c4 was out of bounds (org-testing-only). On 2026-11-17 at
21:41 EST Jesun lifted the spark-31c4 boundary for this work only, with
a window of tonight until 2026-11-18 06:00 EST. Node roles and stack
verified tonight against the committed env stamps in runs/main.jsonl:
node 0 is spark-6aa3 (enP7s7 192.168.99.170, 1000 Mb/s), node 1 is
spark-31c4 (enP7s7 192.168.99.153, 1000 Mb/s), both NVIDIA GB10,
torch 2.12.0+cu130, CUDA 13.0, NCCL 2.29.7. Node 1 matches the threat
PREREG.md names in advance: an idle vLLM server (0% GPU utilisation at
check time, about 27 GB of memory available). Its services are not
touched; all work on both machines stays inside ~/Jesun_Gallea_Dev.

Known condition on node 0. spark-6aa3 also hosts the finetune-lab
confirmatory run (started 2026-11-17 about 21:29 EST; about 92 GB held,
about 95% GPU utilisation at check time). It is separate owner-ordered
work and is not paused or disturbed. E1 and E4 score exactness (hashes
and parameter differences), not time, so they are valid under
co-tenancy. E2 and the E3 cells are timed, so before the grid starts,
two probe cells (ddp L at k=1, the most communication-bound cell, and
ddp S at k=64, the most compute-sensitive two-node cell, one repeat
each) run into a scratch file that is not part of this record. If both
probe values land within 10% of their committed cell values in
results_deep.json, the grid runs into the record file below, with
per-repeat GPU-utilisation samples of both nodes logged beside it. If
either probe is off by more, the grid runs into the scratch file
instead and this record reports the measured contention as the reason
its timings are not a clean re-execution.

Decision. Re-execute, in the order of record: E1 (ddp twice, fsdp
once, exactly as Phase 1 of runs/RUNLOG.md), E2, the E3 two-node cells
(ddp and ddp_bf16 on sizes S, M, L x accumulation 1, 4, 16, 64, plus
fsdp on L at accumulation 1; 25 cells x 3 repeats, micro-batch 8,
warmup and timed steps exactly as PREREG.md and run_grid.py fix them,
cell order shuffled per repeat with run_grid.py's seeds), and E4
(clean run, then faults at steps 13, 25 and 37, P19_PG_TIMEOUT=20 and
torchrun --max-restarts=2 as runs/RUNLOG.md records, one checkpoint
directory per launch under level-3-deep/ckpt/). Fresh run lines go to
the new file runs/twonode-2026-11-17.jsonl. runs/main.jsonl,
predictions.json, results_deep.json and PREREG.md are not modified.
The driver is cluster/rerun_twonode.py; it mirrors run_grid.py's cell
list, shuffle seeds and launch path, and differs only in error
handling (a failed cell is logged and the grid continues, where
run_grid.py exits) and in sampling both nodes' GPU utilisation per
repeat. On node 1, launch.sh's REMOTE is pointed at the clone inside
~/Jesun_Gallea_Dev (REMOTE=Jesun_Gallea_Dev/distributed-training-lab/
level-3-deep/cluster) instead of the default p19-run/cluster, so
node 1's mirrored runs and ckpt paths also stay inside
~/Jesun_Gallea_Dev; the code, rendezvous and interface are unchanged.

Comparisons reported (cluster/rerun_twonode_compare.py, analysis.py
used unchanged, output runs/twonode-2026-11-17-comparison.json and
runs/twonode-2026-11-17-rescore.json): (a) fresh two-node cell values
against the committed cells in results_deep.json; (b) the fresh E2
link fit against the committed fit and the H2 pass lines; (c) the H3
headline re-check: predictions recomputed from the fresh E2 fit and
the fresh single-node times of decision 0003, scored against the
fresh two-node times, with the maximum absolute relative error set
beside the committed 2.7%; the fresh two-node times are also scored
against the locked predictions.json; (d) E1 parameter differences and
hashes against the committed E1 lines; (e) E4 hashes against the
committed E4 lines; (f) a full fresh H1 to H7 re-score computed in
memory by analysis.py from the fresh rows and fresh predictions.

Consequences. The rerun record stands beside the original record. No
frozen verdict changes whatever the fresh numbers say; any divergence
is reported as measured, with both values. Nothing is pushed.
