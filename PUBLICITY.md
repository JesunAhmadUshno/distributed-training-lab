# PUBLICITY: distributed-training-lab (P19)

Date: 2026-11-05. Prepared by the ROADMAP-V3-BATCH loop. The loop
prepares drafts and never posts to social.

Project state on record: Learn DONE and Build DONE (2026-11-05).
Deep is NOT closed: the Deep prep package is Kaggle-ready, Deep is
defined as the 2-GPU Kaggle exercise set (Addendum B), and the
4-GPU scope is the parked v1.1 remainder in ROADMAP.md. Releases on
record: v0.1 (Learn) and v0.5 (Build). No fan-out set for a finished
level is claimed, graded, or shipped in this file; this file logs
only the Actions polish pack application below.

## Actions polish pack (2026-11-05)

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

Publicity grade for this polish pack log piece, on the 1,000
point rubric (accuracy 300, research 200, gap fit 200, clarity
150, reach format 150), ship line 900:

| Piece | Accuracy /300 | Research /200 | Gap fit /200 | Clarity /150 | Reach /150 | Total /1000 |
| --- | --- | --- | --- | --- | --- | --- |
| PUBLICITY.md polish pack section (2026-11-05) | 295 | 180 | 195 | 145 | 100 | 915 |

At or above the 900 ship line. Named deductions: research minus
20, because the pack is a standard applied across repos and this
section records its application rather than new research; reach
minus 50, because the docs page is copy in this pull request with
no live Pages claim yet, PyPI publishing is held, and no social
channel carries this piece by the standing gates; accuracy minus
5, gap fit minus 5, and clarity minus 5 for summary compression
of the full pack standard, which lives in the batch files.
