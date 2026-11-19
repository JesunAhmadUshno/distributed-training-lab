# 0005: Clean two-node E3 grid and E4 finish on both Sparks (gap closure, 2026-11-19)

Date: 2026-11-19. Status: accepted (owner finish-everything order, 2026-11-19).

Context. Decision 0004's timed grid was routed to scratch because node 0
was co-tenanted by the finetune confirmatory run: the probe's
communication-bound cell landed +3.07% (within the 10% line) and its
compute-bound cell +117.08% (past it). The grid reached 14 of 25 rep-0
cells into runs/probe-2026-11-17.jsonl (scratch) before the machine
saturated; spark-6aa3 then refused SSH sessions on every route from
2026-11-17 about 22:10 EST, three further rep-0 cells timed out at
1200 s, and E4 never ran. On 2026-11-19 the owner ordered every open
item finished. State found this morning: spark-6aa3 reachable again
(direct tailnet route and via the spark-31c4 LAN relay), load average
0.00, GPU idle on both nodes; node 1's vLLM server is resident at 0%
GPU utilisation and is not touched, as decision 0004 fixes. The
finetune confirmatory run is not running: its driver log ends
2026-11-17 21:59 EST with eval/base complete (GSM8K accuracy 0.5080,
670 of 1319, 0.179 GPU-hours) and h1/full/seed0 dead at model load with
no checkpoint written. ORDER A (pause the finetune run, owner order
2026-11-17 23:51 EST) therefore has nothing to pause; the run stays
stopped for this grid and is resumed after it, per the same order set.

Decision. Re-execute the E3 two-node grid and E4 exactly as decision
0004 fixes them: same cell list (ddp and ddp_bf16 on S, M, L x
accumulation 1, 4, 16, 64, plus fsdp on L at accumulation 1; 25 cells
x 3 repeats), same per-repeat shuffle seeds, same warmup and timed
steps, same launch path, E4 as clean run then faults at steps 13, 25
and 37 with P19_PG_TIMEOUT=20 and --max-restarts=2. The probe rule is
unchanged: the two probe cells (ddp L k=1, ddp S k=64, one repeat each)
run first into the new scratch file runs/probe-2026-11-19.jsonl; if
both land within 10% of their committed cells in results_deep.json,
the grid runs into the new record file runs/twonode-2026-11-19.jsonl,
otherwise into the scratch file, labelled scratch. E1 and E2 are not
re-run; their 2026-11-17 fresh records stand and are inputs to the
comparison. The driver is cluster/rerun_twonode_20261119.py, a copy of
rerun_twonode.py differing only in the two dated output paths. The
comparison runs through a dated copy of rerun_twonode_compare.py using
analysis.py unchanged, writing runs/twonode-2026-11-19-comparison.json
and runs/twonode-2026-11-19-rescore.json. runs/main.jsonl,
predictions.json, results_deep.json and PREREG.md are not modified.

Consequences. The clean rerun record stands beside the contended
scratch record; the scratch cells stay labelled scratch and are not
scored as a re-execution. No frozen verdict changes whatever the fresh
numbers say; any divergence is reported as measured, with both values.
