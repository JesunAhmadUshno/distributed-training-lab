# SPEC: distributed-training-lab (P19)

Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 4 (Learn and Build).
Hiring evidence: 5 of 5 (distributed systems in 31/56, the second highest count) [confirmed, fleet and scoring pass 2026-11-04, or the decision pack for P23 to P29].

## Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

distributed-training-lab: APPROVED by Jesun 2026-10-19 (Section 3 ruling)

Subject: distributed training and cluster scale.
- Learn: data parallel training of the P01 GPT on 2 GPUs with PyTorch DDP, logged end to end (the CEO next-15 list, step 6, names this exact run) [confirmed, cross-match Part 5]. One bridge run: a TensorFlow MirroredStrategy run logged beside it (register item 229).
- Build: the same run under FSDP and under DeepSpeed, with checkpoint save and restart mid-run. Measure interconnect bandwidth with NCCL tests before trusting the scaling plan (register item 542).
- Deep: a scaling table (tokens per second and cost per run against GPU count), a failure log (what breaks at each scale), and a short write-up that compares DDP, FSDP, and DeepSpeed on the same model [confirmed shape, career documentation Section D proposal]. A JAX sharding stretch note (mesh, partition specs) rides here from register item 220.
Gate: this project starts only when the GPU access plan exists (round 2 addition A4) and any rental fits the spend gate [confirmed, board plan decision 11].
Fan-out: Book chapter "Two Cards, One Model". Paper angle: the DDP, FSDP, DeepSpeed same-model comparison. Course module: P19-distributed-training-lab.

## Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.
