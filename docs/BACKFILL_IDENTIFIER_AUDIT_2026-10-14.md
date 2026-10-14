# October 14, 2026 backfill identifier audit

Baseline main: `8b1a88a6072fb17387a971faf2a0c352c6f3a6d2`.
Baseline tree: `76e3504744f7f8dc97e4c3c9536ea4d314076e5c`.

## Failure and cause

The October 13 calendar correction worked:

- [October 14 production 38624403253](https://github.com/salemmorelli1/Weekly-U.S.-Gas-Prices/actions/runs/38624403253)
  succeeded using source `f2376c273318d5dc70bd7521dac1b1dd6471f46d`.
- It published baseline `8b1a88a6072fb17387a971faf2a0c352c6f3a6d2`.
- [Pages 38624783776](https://github.com/salemmorelli1/Weekly-U.S.-Gas-Prices/actions/runs/38624783776)
  succeeded.
- [Backfill 38632519108](https://github.com/salemmorelli1/Weekly-U.S.-Gas-Prices/actions/runs/38632519108),
  job `116087443789`, fetched the October 12 GASREGCOVW observation at $4.150,
  scored one additional outcome, and computed four eligible realized rows.
  Its `gas_release_manifest.py backfill` command then rejected changed provenance.

This failure is a CSV serialization bug, not an exchange market closure or a
missing EIA observation. [FRED's GASREGCOVW series](https://fred.stlouisfed.org/series/GASREGCOVW)
reported October 12 at $4.150 when checked October 14. The holiday release date
was October 14 in [EIA's publication schedule](https://www.eia.gov/petroleum/gasdiesel/schedule.php).
The Ubuntu image-migration notice did not cause the failure.

`gas_backfill_realized.py` read the seven-row CSV with inferred types. Blank
legacy IDs forced `pipeline_run_id` to floating point. The subsequent CSV write
changed the latest ID from `38624403253` to `38624403253.0`. The manifest's
immutable-input guard correctly rejected that change before publication.

Other historic IDs already end in `.0`. Their exact text is published provenance;
this patch preserves it. It does not reconstruct or normalize old IDs.

The audit also reproduced a related weakness: pandas' default frame comparison
tolerance accepted a `1e-9` change to a frozen forecast. Frozen-input comparison
must be exact, independently of the CSV identifier correction.

## Corrections

- Centralize prediction-log reading in `gas_time_contract.py`. Explicit text
  converters preserve blanks, NA-looking labels, leading zeros, large identifiers,
  run attempts, source SHAs, dates, and other textual fields. Numeric columns use
  round-trip float parsing and retain numeric missing-value handling.
- Use that reader in backfill, governance, attribution, and manifest validation.
  Converters, rather than a string dtype alone, prevent pandas' default NA
  interpretation from erasing opaque text labels.
- Compare immutable input columns with `check_exact=True`. Backfill may update
  only `actual`, `actual_date`, `mae`, `rmse`, `mape`, and `direction_correct`.
  Row additions, deletions, reordering, and frozen column changes are rejected.
- Validate every temporary CSV cell before replacing the canonical ledger.
  Reject already inferred numeric identifiers. Failed writes or serialization
  checks preserve the previous canonical bytes and remove the temporary file.
- Apply the verified writer to both governance and backfill, so subsequent
  forecast appends cannot reintroduce identifier or float conversion.
- Add a repeatable offline replay and run it in an isolated backfill environment
  on both CI operating systems, independently of the CPU model environment.

No committed forecasts, outcomes, publication data, artifacts, or release-manifest
bytes are changed. Existing target selection, release calendar, prospective
eligibility, sleeve gates, persistence fallback, and 52-observation threshold remain.

## Verification

- Baseline: 111 tests passed in the locked Python 3.12 CPU environment.
- Three initial regressions failed before correction: identifier preservation,
  the complete backfill publication sequence, and a tiny frozen forecast rewrite.
- Corrected suite: **143 tests passed**, including CPU model training and the
  disposable pipeline integration test. The 32 added tests cover raw identifier
  tokens, NA markers, leading zeros, values above `2**53`, exact floats, governance
  appends, immutable row/cell rejection, failed writes, and actual CLI sequencing.
- A fresh environment with only `requirements-backfill-lock.txt` passed
  `python gas_replay_backfill.py --require-minimal`. Torch, sklearn, XGBoost,
  hmmlearn, and yfinance were absent.
- The replay scores the October 12 target at $4.150, leaves October 19 unscored,
  verifies the staged manifest, preserves all frozen inputs and raw identifiers,
  and confirms a second backfill leaves identical ledger bytes. Attribution
  remains `INSUFFICIENT_DATA`, with 4 eligible rows and a minimum of 52.
- Both environments passed `pip check`. Fatal-error lint, compilation, workflow
  YAML parsing, committed manifest verification, and unchanged publication data
  are also checked locally and by the Linux/Windows PR workflow.

The pinned fixture is the exact seven-row ledger from baseline main, SHA-256
`43e7058cd95bab936493f82e5bf06770dde86c18e5c295175d33bcaaa56e8154`.
Replay first verifies the current published manifest, copies its release into a
temporary directory, substitutes this historical ledger, and refreshes only the
temporary manifest's ledger hash. It then runs the actual backfill, attribution,
and staging code against the observed $4.150 fixture. This fixture setup is not
a newly signed production release. No API requests or source-data writes occur.

## Recovery after review and merge

Start a **new** `weekly-backfill.yml` workflow dispatch on `main`, with
`dry_run=false` and `force=false`. The workflow will fetch current authoritative
observations, score eligible exact targets, verify immutable forecast inputs,
commit the realized-evidence surface, and deploy Pages. No new forecast needs to
be issued to repair this failed backfill.

Do not use "Re-run failed jobs" on run 38632519108: it reuses the old source
commit and therefore the faulty reader. Until the PR is merged, a main-branch
backfill still uses the old code. The feature branch's CI replay verifies the
correction without changing production outcomes.
