# cluster: two-node Deep harness (P19)

Runs the pre-registered study in ../PREREG.md on two DGX Spark nodes.

| File | Role |
| --- | --- |
| common.py | model sizes, deterministic data, process-group setup, environment stamp |
| e1_correct.py | H1: DDP and FSDP2 against a single-process replay |
| e2_comm.py | H2: all-reduce and broadcast timing, 4 KiB to 256 MiB |
| e3_scale.py | H3 to H6: one grid cell (mode, size, accumulation, repeat) |
| e4_fault.py | H7: hard crash of node 1, elastic restart, bit-identical resume |
| run_grid.py | the E3 grid in shuffled order, resumable |
| analysis.py | link fit, predictions, report (standard library only); tests in test_analysis.py |
| launch.sh | starts a script on both nodes (static rendezvous, NCCL over the 1 GbE interface) |

Order of record: E1, E2, single-node grid, `analysis.py predict` and commit predictions.json,
two-node grid, E4, `analysis.py report`.
