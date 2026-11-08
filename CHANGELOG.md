# Changelog

All notable changes to this project. Format: Keep a Changelog 1.1.
Versions follow Semantic Versioning.

## [Unreleased]

### Changed
- 2026-11-08: Adopted the portfolio standard v2. README is the abstract
  (question, headline with its interval, field, job family, verdict,
  reproduce, lifecycle, layout). REPORT.md carries the Deep results,
  the CPU background, the earlier review (moved from
  docs/research-review.md), future work (moved from ROADMAP.md) and the
  glossary. PROPOSAL.md points at the locked level-3-deep/PREREG.md and
  holds SPEC.md, NOVELTY.md and DOSSIER.md verbatim. docs/index.md (the
  retired Jekyll page) and PUBLICITY.md are folded into this file.
  Grades leave every public file; they are kept internally.
- 2026-11-08: CI runs the whole test suite (`unittest discover`, which
  also picks up the sweep and ring tests the named-module steps
  missed), declares read-only permissions, and fails if a results file
  changes when the runners regenerate it. CodeQL also scans the
  dashboard's JavaScript; Dependabot also watches site/ (npm).

### Added
- 2026-11-08: tests/test_layout.py (required files, no public grade, and
  the SHA-256 of PREREG.md as committed in f617be6) and
  tests/test_review.py (re-derives the headline numbers from
  runs/main.jsonl without the study's analysis code). Decision records
  0001 (route change) and 0002 (scope of the lock). CITATION.cff,
  SECURITY.md, this file.

## [1.0] - 2026-11-07

### Added
- Deep, Level 3: DONE. Route changed by Owner ruling from Kaggle T4 x2
  to two DGX Spark nodes over 1 GbE (docs/decisions/0001).
  Pre-registered study (PREREG.md), 5 of 7 hypotheses pass, H5 and H6
  missed narrowly. Write-up: level-3-deep/README.md. CI re-scores the
  evidence with a 1e-9 tolerance (Python 3.11 and 3.12 sum floats
  differently in the last digits).
- Live dashboard in site/ with in-browser trainers and a build-time
  known-answer check; Pages publishes it instead of the Jekyll docs page.
- Learn rank sweep (1 to 8 ranks) and every 2-rank shard split
  (results_sweep.json); Build chunked ring all-reduce and stage-aware
  DDP/ZeRO memory counts (results_scaling.json).

### Security
- Vite upgraded to 6.4.5 so npm audit reports no high or critical
  advisories.

## [0.5] - 2026-11-05

### Added
- Build, Level 2: DONE. Simulated FSDP-class and ZeRO-class sharding,
  checkpoint restart proved on CPU.
- Deep PREP: Kaggle-ready Addendum B package with a CPU equivalence
  check (superseded 2026-11-07, never run).
- GitHub Actions polish pack and Pages (notes below).

## [0.1] - 2026-11-05

### Added
- Learn, Level 1: DONE. Simulated data-parallel training, all-reduce
  equivalence proved on CPU.

## Notes moved from PUBLICITY.md (2026-11-05, Actions polish pack)

Date: 2026-11-05. Prepared by the ROADMAP-V3-BATCH loop. The loop
prepares drafts and never posts to social.

Project state on record: Learn DONE and Build DONE (2026-11-05).
Deep is NOT closed: the Deep prep package is Kaggle-ready, Deep is
defined as the 2-GPU Kaggle exercise set (Addendum B), and the
4-GPU scope is the parked v1.1 remainder in ROADMAP.md. Releases on
record: v0.1 (Learn) and v0.5 (Build). No fan-out set for a finished
level is claimed, graded, or shipped in this file; this file logs
only the Actions polish pack application below.

### Actions polish pack (2026-11-05)

Applied through a pull request on 2026-11-05, following the Actions
polish pack standard. The pack adds: CI on a Python 3.11 and 3.12
matrix with a ruff lint step (the pre-existing CI step, the scaffold
smoke test, is preserved, and the matrix CI also runs the Learn,
Build, and Deep prep test suites and their runners), CodeQL
scanning, Dependabot for pip and GitHub Actions, a release workflow
that attaches a source tarball named
distributed-training-lab-<tag>.tar.gz to v* tag releases, Labeler,
Greetings (snake_case inputs for actions/first-interaction@v3:
repo_token, issue_message, pr_message), Stale in comment-only mode
(days-before-close: -1, never closes), and a docs summary page at
docs/index.md for GitHub Pages. Ruff scope is recorded in
ruff.toml (E, F, I, line-length 120). Ruff fixes in this pull
request were style only: 40 initial findings (import sorting and
wrapping, multiple imports on one line, one unused import, line
length, and semicolon statements in the Kaggle exercises), fixed
with no change to behaviour. All three results files
(results_learn.json, results_build.json, results_deep_prep.json)
stayed byte-identical across the runners, and the full test suite
passes locally (50 tests). requirements.txt records the
standard-library-only fact for CI and notes that torch is provided
by the Kaggle image for the Deep exercises, so Dependabot pip has
a manifest to watch.

| Channel | Outcome on 2026-11-05 |
| --- | --- |
| GitHub Pages | Docs page added at docs/index.md in this pull request. Pages enablement and live build status are verified after merge and recorded in the batch ledger; no live claim is made in this file before that check. |
| PyPI publishing | Held: needs a PyPI token, a new credential. Jesun's tap only. Not set up. |
| Social channels (LinkedIn, X, Facebook, Instagram, Threads) | Held: no posts made. The loop never posts to social; Threads waits for Jesun's tap per post under the standing gates. |
