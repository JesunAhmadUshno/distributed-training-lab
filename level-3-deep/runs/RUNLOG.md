# Run log (P19 Deep, two DGX Spark nodes)

- 2026-11-07, bring-up (excluded from results, files in bringup/): NCCL smoke test; c10d elastic
  rendezvous hung on this pair, switched to static rendezvous; first crash-recovery attempt hung because
  the surviving rank waited on the 10-minute default collective timeout (fixed: P19_PG_TIMEOUT, 20 s for E4);
  one 4-step single-node timing probe for S and L to size the grid.
- PREREG.md committed and pushed (branch deep/spark-2node) before any main-grid run.
- Phase 1 (main.jsonl): E1 ddp x2, E2, single-node grid (36 cells x repeats). Node 1 GPU utilisation 0% at
  every repeat (main-gpu-util.log).
- E1 fsdp first launch crashed on a harness bug (hashing a tensor that still required grad, no result line
  written); fixed with .detach() and relaunched once. No other E1 launch was repeated.
- predictions.json computed by `analysis.py predict` from E2 and the single-node arm, committed and pushed
  before any two-node E3 run.
