# Weekly U.S. Gas Prices

Forecasts next week's U.S. regular conventional gasoline price (the weekly EIA series,
`GASREGCOVW`). Runs after EIA's normal Tuesday publication of the Monday
observation, logs an immutable forecast for the following Monday, then scores
eligible forecasts after their exact target observation becomes available.

Dashboard: https://salemmorelli1.github.io/Weekly-U.S.-Gas-Prices/

## How it works

The pipeline runs in order:

```
gas_part0    FRED + yfinance weekly history (gas prices, WTI, RBOB, macro)
gas_part0c   EIA Open Data API (gasoline stocks, demand, refinery utilization)
gas_part6    descriptive HMM/GMM regimes (not predictive validation features)
gas_part1    point-in-time feature engineering (lags, momentum, spreads, seasonality)
gas_part2    sklearn ensemble - the primary forecaster
gas_part2b   XGBoost sleeve (optional)
gas_part2a   LSTM sleeve (optional, needs torch)
gas_part3    fuses the sleeves and writes prediction_log.csv
gas_part9    live stats (MAE, MAPE, HAC/HLN Diebold-Mariano vs persistence, drift)
```

The most recent week has no realized target yet, so Part 1 flags it as the
live row. Models train on everything before it and predict it — that's the
actual forecast, targeting the following Monday's price observation. The prediction
log is keyed by target date. A target can be appended once; re-runs cannot
revise its forecast or provenance.

The core ensemble is validated with expanding-origin blocked forecasts and
must beat same-date persistence in RMSE, a one-sided HAC/HLN-corrected paired
loss test, and at least 75% of validation folds. Until that gate passes, the
published forecast is fail-closed to persistence. Optional XGBoost and LSTM
sleeves cannot activate unless the core gate has already passed; missing
evidence fails closed.

Regular FRED monthly and quarterly macro histories are current-vintage series:
their period labels predate publication and past values can be revised. They
remain available for diagnostics but are excluded from predictive features
until point-in-time vintages are supplied. Full-history HMM states are also
diagnostic-only.

A prediction only counts once it's in `prediction_log.csv` before the answer
is known. Wednesday's backfill fills in the realized price and error metrics;
Part 9 keeps a running paired test against the persistence value stored when
each forecast was issued. Legacy same-day and hindsight rows are retained for
audit history but excluded from the eligible live cohort. General live-health
claims require at least 52 eligible realized observations. A recorded UTC issue
time must agree with the Eastern decision date; each scored outcome must have
a finite price, a finite frozen persistence comparator and its exact target date.

## Running it

Two free API keys:

- `FRED_API_KEY` (required) — https://fred.stlouisfed.org/docs/api/api_key.html
- `EIA_API_KEY` (recommended, powers the fundamentals) — https://www.eia.gov/opendata/register.php

Locally:

```bash
pip install -r requirements.txt
export FRED_API_KEY="..."
export EIA_API_KEY="..."
python gas_run_weekly_prediction.py --force        # bypass calendar/day guard only
```

Add `--with-backfill` to also fetch realized prices for past predictions.
Everything also works in Colab — set `GASPRICE_ROOT` to a Drive path first.

PyTorch is intentionally outside the production lock. Install a separately
reviewed CPU-only build only when evaluating the optional LSTM sleeve.

## GitHub Actions

Three workflows:

- `weekly-production.yml` — Tuesdays at 11:30 AM ET, with an idempotent
  Wednesday holiday/delay fallback. The shared EIA calendar skips the Tuesday
  before a published Wednesday holiday release. Publishes the small release ledger and
  retains the full research bundle for 90 days as a workflow artifact.
- `weekly-backfill.yml` — Wednesdays at 1:00 PM ET. Fills only exact target
  observations and re-runs the eligible live cohort.
- `pages.yml` — verifies and deploys only the content-hashed committed data.

The workflows use GitHub's schedule timezone field directly. The release
preflight checks the earliest scheduled EIA publication time, rather than a
narrow execution window: late jobs remain eligible. A due date does not prove
that FRED has updated; Part 0 still rejects stale source weeks and past targets.

`gas_release_schedule.py` uses the EIA holiday table reviewed October 13, 2026,
including its two 2025 Tuesday exceptions. Automatic preflight covers reference
Mondays January 6, 2025 through January 18, 2027. Refresh the pinned table and
coverage when EIA publishes subsequent dates; unreviewed weeks fail explicitly.
Scheduled retries skip a source/target pair already marked successful. Manual
dispatches respect the earliest release time and may deliberately retry a due
week; direct `--force` bypasses only the runner's calendar/day guard.

Offline diagnostics (no downloads, training, or forecasts):

```bash
python gas_release_schedule.py --event schedule --schedule '30 11 * * 2' \
  --now '2026-10-13T16:46:00-04:00'
python gas_release_schedule.py --event schedule --schedule '30 11 * * 3' \
  --now '2026-10-14T11:30:00-04:00'
```

See [the October 13 holiday-release audit](docs/HOLIDAY_RELEASE_AUDIT_2026-10-13.md)
for the failed-run evidence and regression tests.

Setup on a fork: add the two keys as Actions secrets, set Pages source to
"GitHub Actions", and give workflows read/write permission under
Settings → Actions → General.

## Where things land

Large model, parquet, and database artifacts stay out of Git history and are
retained in immutable workflow bundles. The reviewable publication surface is
content-hashed in `data/release_manifest.json`. The main files are:

- `artifacts_part3/prediction_log.csv` — the record that matters
- `artifacts_part9/live_attribution_report.json` — health, DM test, drift
- `data/release_manifest.json` — hashes, run identity, source date, and target
- `data/gas_oof_predictions.csv` — expanding-origin blocked validation tape

The dashboard's release-path graphic shows the observed source week, learned
candidate, governed publication, and next-week target directly from committed
telemetry. Its forecast chart defaults to `oos_val=1`; full fitted history is
available only as an explicitly labeled diagnostic view.

Health thresholds Part 9 watches: MAPE over 3% is a warning, over 6% is a
stop signal; direction accuracy under 50% is a warning; recent-vs-historical
RMSE ratio over 1.5x flags drift.

## October 11, 2026 source and sleeve audit

The direct EIA fallback now uses `PET.EMM_EPMRU_PTE_NUS_DPG.W`, the
regular conventional series matching `GASREGCOVW`. It previously requested
diesel. Prices retain their exact Monday labels and missing surveys remain
missing. Targets and lag features follow calendar weeks. Gasoline days of
supply use stocks in thousand barrels divided by demand in thousand barrels
per day; demand is already a daily rate.

XGBoost and LSTM stopping parameters are selected on an inner chronological
window before each scored outer block. Their OOF dates, targets and
persistence values must match the core. Governance verifies the OOF and
forecast tape hashes, source/target dates and feature matrix before admitting
a sleeve, then evaluates fusion weights chosen from prior scored blocks.
A failed fusion retains the validated core; a failed core retains persistence.

The regression suite includes CPU training and an entirely disposable pipeline
through publication staging and exact-target backfill. CI checks Linux and
Windows. Existing published forecasts and the 52-observation health threshold
remain unchanged. See `docs/SOURCE_SLEEVE_AUDIT_2026-10-11.md`.
