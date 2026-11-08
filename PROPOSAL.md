# PROPOSAL: distributed-training-lab (P19)

## The locked proposal

The proposal of record is **level-3-deep/PREREG.md**, committed in
f617be6 on 2026-11-07 before any Deep run result (the first result
lines arrive in a181dd2) and tagged **proposal-v1**. It fixes the
question, hypotheses H1 to H7 with their pass lines, the design, the
analysis plan and the threats named in advance. It keeps its name and
folder because the harness and the analysis refer to it
(docs/decisions/0002). tests/test_layout.py pins its SHA-256; any
change after the lock fails CI and belongs in docs/decisions/ instead.

The per-cell predictions (level-3-deep/predictions.json) were computed
from the link fit and the single-node arm and committed in a181dd2,
before any two-node grid run, as PREREG.md requires.

## Scope before the lock (history)

The three files below were written on 2026-11-04, before any code, as
the project's scaffold. They set a wider scope (DDP, FSDP and DeepSpeed
on the P01 GPT, a TensorFlow bridge run, a JAX note) than the study that
was finally locked. Where they differ from PREREG.md, PREREG.md wins:
DDP, bf16-compressed DDP and FSDP2 were measured; DeepSpeed, JAX,
TensorFlow and more than two nodes were not (REPORT.md, Future work).
They are moved here verbatim from SPEC.md, NOVELTY.md and DOSSIER.md.

### From SPEC.md


Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 4 (Learn and Build).
Hiring evidence: 5 of 5 (distributed systems in 31/56, the second highest count) [confirmed, fleet and scoring pass 2026-11-04, or the decision pack for P23 to P29].

#### Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

distributed-training-lab: APPROVED by Jesun 2026-10-19 (Section 3 ruling)

Subject: distributed training and cluster scale.
- Learn: data parallel training of the P01 GPT on 2 GPUs with PyTorch DDP, logged end to end (the CEO next-15 list, step 6, names this exact run) [confirmed, cross-match Part 5]. One bridge run: a TensorFlow MirroredStrategy run logged beside it (register item 229).
- Build: the same run under FSDP and under DeepSpeed, with checkpoint save and restart mid-run. Measure interconnect bandwidth with NCCL tests before trusting the scaling plan (register item 542).
- Deep: a scaling table (tokens per second and cost per run against GPU count), a failure log (what breaks at each scale), and a short write-up that compares DDP, FSDP, and DeepSpeed on the same model [confirmed shape, career documentation Section D proposal]. A JAX sharding stretch note (mesh, partition specs) rides here from register item 220.
Gate: this project starts only when the GPU access plan exists (round 2 addition A4) and any rental fits the spend gate [confirmed, board plan decision 11].
Fan-out: Book chapter "Two Cards, One Model". Paper angle: the DDP, FSDP, DeepSpeed same-model comparison. Course module: P19-distributed-training-lab.

#### Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.

### From NOVELTY.md


Status: DRAFT, PROPOSED. Not verified. This text does not move into any public post, paper, or profile until the Jev claim_check gate passes on the exact text.

#### Proposed claim

Against distributed training docs (PyTorch DDP, FSDP, DeepSpeed tutorials): this project aims to differ in the way its SPEC.md section text defines (a from-zero or self-hosted build, measured in public, on a named home fleet, with the failure modes written down). The exact difference is stated at the project's wave, after the prior art is re-read.

#### Prior art table

| Prior art | What it does | What this project does differently | Measured difference |
| --- | --- | --- | --- |
| distributed training docs (PyTorch DDP, FSDP, DeepSpeed tutorials) | See DOSSIER.md, Past | PROPOSED, see the claim above | NO MEASUREMENT YET |

#### Honest size

If the novelty proves small, this file will say so and name the real value instead (a cleaner reproduction, a better benchmark, a first public dataset). No novelty is invented to fill this section.

### From DOSSIER.md


Written under the documentation and publishing standard (past, present, future). Updated at ship time.

#### Past

Prior art for this project is named in NOVELTY.md (distributed training docs (PyTorch DDP, FSDP, DeepSpeed tutorials)). The history of the problem, the key papers and systems, and what was tried before are recorded in the Roadmap v2 section text in SPEC.md and will be expanded with dated sources at the project's wave, before Build opens.

#### Present

Current demand evidence: hiring value 5 of 5 (distributed systems in 31/56, the second highest count) [confirmed, fleet and scoring pass 2026-11-04; for P23 to P29, the decision pack 2026-11-04]. The Round 3 research (2026-11-04) found the market centre at post-training, evals, environments, and inference depth [confirmed as the research finding; the report is at ~/workspace/research_notes/roadmap-v3-revaluation-research-20261104-0949/report.md]. Tools and versions are pinned at Build time, not in this v1.

#### Future

Open problems and the project's next steps sit in the Deep level scope in SPEC.md. Any forecast about this field is [unverified] unless a dated primary source is added here at the wave update.

