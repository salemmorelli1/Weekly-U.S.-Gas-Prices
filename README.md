# Weekly U.S. Gas Prices

Forecasts next week's U.S. average regular gas price (the weekly EIA number,
`GASREGCOVW`). Runs on GitHub Actions every Monday morning after the EIA
release, logs its prediction, then scores itself on Wednesday once the next
release is out.

Dashboard: https://salemmorelli1.github.io/Weekly-U.S.-Gas-Prices/

## How it works

The pipeline runs in order:

```
gas_part0    FRED + yfinance weekly history (gas prices, WTI, RBOB, macro)
gas_part0c   EIA Open Data API (gasoline stocks, demand, refinery utilization)
gas_part6    HMM regime detection (NORMAL / SUPPLY_SHOCK / DEMAND_SURGE / DEFLATION)
gas_part1    feature engineering (lags, momentum, crack spread, seasonality, regime)
gas_part2    sklearn ensemble - the primary forecaster
gas_part2b   XGBoost sleeve (optional)
gas_part2a   LSTM sleeve (optional, needs torch)
gas_part3    fuses the sleeves and writes prediction_log.csv
gas_part9    live performance stats (MAE, MAPE, Diebold-Mariano vs naive, drift)
```

The most recent week has no realized target yet, so Part 1 flags it as the
live row. Models train on everything before it and predict it — that's the
actual forecast, targeting the following Monday's EIA release. The prediction
log is keyed by target date, so re-runs update in place instead of
duplicating rows.

The XGBoost and LSTM sleeves only count if they beat the sklearn ensemble on
the same trailing 52-week window. The LSTM additionally won't run at all
unless the XGBoost sleeve earned its spot first (it checks the Part 2b
summary at runtime). Missing baselines fail closed.

A prediction only counts once it's in `prediction_log.csv` before the answer
is known. Wednesday's backfill fills in the realized price and error metrics;
Part 9 keeps a running Diebold-Mariano test against just predicting last
week's price. That comparison is the whole ballgame — takes about 8 weeks of
realized data before it means anything.

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

For the LSTM sleeve, uncomment `torch` in requirements.txt. It's safe to
install eagerly; the sleeve gates itself.

## GitHub Actions

Three workflows:

- `weekly-production.yml` — Mondays ~10:35 AM ET. Full pipeline, commits
  artifacts, which triggers the Pages deploy.
- `weekly-backfill.yml` — Wednesdays ~8:00 AM ET. Fills realized prices,
  re-runs Part 9.
- `pages.yml` — deploys the dashboard from the committed data.

The schedules use paired UTC crons plus a time gate because GitHub ignores
timezone keys on cron triggers (learned that one the hard way).

Setup on a fork: add the two keys as Actions secrets, set Pages source to
"GitHub Actions", and give workflows read/write permission under
Settings → Actions → General.

## Where things land

Artifacts are gitignored; the workflows force-add the ones worth keeping.
The main ones:

- `artifacts_part3/prediction_log.csv` — the record that matters
- `artifacts_part9/live_attribution_report.json` — health, DM test, drift
- `artifacts_part0/gas_weekly_master.parquet` — the assembled weekly dataset

Health thresholds Part 9 watches: MAPE over 3% is a warning, over 6% is a
stop signal; direction accuracy under 50% is a warning; recent-vs-historical
RMSE ratio over 1.5x flags drift.
