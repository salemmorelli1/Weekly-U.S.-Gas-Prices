# Weekly U.S. Gas Prices

Forecasts next week's U.S. average regular gas price (the weekly EIA number,
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
actual forecast, targeting the following Monday's EIA release. The prediction
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
claims require at least 52 eligible realized observations.

## Running it

Two free API keys:

- `FRED_API_KEY` (required) — https://fred.stlouisfed.org/docs/api/api_key.html
- `EIA_API_KEY` (recommended, powers the fundamentals) — https://www.eia.gov/opendata/register.php

Locally:

```bash
pip install -r requirements.txt
export FRED_API_KEY="..."
export EIA_API_KEY="..."
python gas_run_weekly_prediction.py --force        # --force = run on any day
```

Add `--with-backfill` to also fetch realized prices for past predictions.
Everything also works in Colab — set `GASPRICE_ROOT` to a Drive path first.

PyTorch is intentionally outside the production lock. Install a separately
reviewed CPU-only build only when evaluating the optional LSTM sleeve.

## GitHub Actions

Three workflows:

- `weekly-production.yml` — Tuesdays at 11:30 AM ET, with an idempotent
  Wednesday holiday/delay fallback. Publishes the small release ledger and
  retains the full research bundle for 90 days as a workflow artifact.
- `weekly-backfill.yml` — Wednesdays at 1:00 PM ET. Fills only exact target
  observations and re-runs the eligible live cohort.
- `pages.yml` — verifies and deploys only the content-hashed committed data.

The workflows use GitHub's schedule timezone field directly. They do not use
delay-sensitive wall-clock gates, so scheduler delay cannot turn a failed
publication into a green skip.

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
