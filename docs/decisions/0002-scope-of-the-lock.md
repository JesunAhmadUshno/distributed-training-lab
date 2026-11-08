# 0002: What the proposal-v1 lock covers

Date: 2026-11-08. Status: accepted (standard v2 migration).

Context. The portfolio standard v2 tags a proposal proposal-v1 only
where git history proves the plan came before the results it governs.
In this repo the Learn and Build CPU simulations (2026-11-04 to
2026-11-07) were built without a written plan. level-3-deep/PREREG.md
was committed in f617be6 (2026-11-07 16:11 EST) together with the
harness and before any Deep result line; the first result lines are in
a181dd2 (16:26 EST). Bring-up runs made before it are logged and
excluded (level-3-deep/runs/RUNLOG.md).

Decision. proposal-v1 points at f617be6 and covers the Deep study only.
Learn and Build are labelled not pre-registered and are reported as
background, never as results. PREREG.md keeps its name and folder,
because the harness, the analysis, CI and the dashboard refer to its
neighbours by relative path; README's layout table maps it to the
proposal role. tests/test_layout.py pins its SHA-256 as committed in
f617be6.

Consequences. Any change to PREREG.md fails CI; an amendment is a new
record here. A future confirmatory cycle (for example DeepSpeed or four
nodes) needs its own locked proposal and tag.
