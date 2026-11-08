# 0001: Deep runs on two DGX Spark nodes, not Kaggle T4 x2

Date: 2026-11-07. Status: accepted (Owner ruling).

Context. Deep was defined on 2026-11-05 as the 2-GPU Kaggle T4 x2
exercise set (Addendum B; level-3-deep/DEEP-SPEC.md, KAGGLE-RUN.md,
kaggle/). Those scripts were syntax-checked on CPU and never ran.

Decision. Deep moves to two NVIDIA DGX Spark nodes, one GB10 GPU each,
joined by 1 GbE through an office switch, NCCL over TCP sockets (no
InfiniBand, no RoCE). Recorded in PREREG.md, Route change.

Consequences. The link is about 100 times slower than NVLink-class
links, so communication, not compute, sets the result; the study asks
when two nodes beat one and whether a link model predicts it. The Kaggle
route stays in the repo as superseded. DeepSpeed, more than two nodes
and the ConnectX interconnect are left for v1.1 (REPORT.md, Future work).
