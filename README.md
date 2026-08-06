# ⛽ GasPriceForecast

> **Weekly U.S. average regular gas price forecasting model.**
> Ensemble of sklearn, XGBoost, and LSTM sleeves fused via inverse-RMSE weighting.
> Data: EIA Open Data API, FRED, yfinance commodities.
> Runs automatically every Monday at 10:35 AM ET, after the EIA weekly release.

[![Weekly Forecast](https://github.com/YOUR_USERNAME/GasPriceForecast/actions/workflows/weekly-production.yml/badge.svg)](https://github.com/YOUR_USERNAME/GasPriceForecast/actions/workflows/weekly-production.yml)

---

## 📊 Live Dashboard

[**→ View Dashboard**](https://YOUR_USERNAME.github.io/GasPriceForecast/)

The dashboard shows:
- This week's fusion forecast vs. EIA realized prices
- Rolling MAPE (4w / 8w / 13w) and all-time model health
- Active sleeve comparison (sklearn / XGB / LSTM)
- Regime detection (NORMAL / SUPPLY_SHOCK / DEMAND_SURGE / DEFLATION)
- Diebold-Mariano significance test vs. naive carry benchmark
- Concept drift monitor

---

## 🏗️ Architecture

```
gas_part0   ─── FRED + yfinance weekly data (WTI, RBOB, FRED gas prices)
gas_part0c  ─── EIA Open Data API v2 (stocks, demand, refinery utilization)
    ↓
gas_part6   ─── HMM regime engine (4 regimes: NORMAL / SUPPLY_SHOCK / DEMAND_SURGE / DEFLATION)
    ↓
gas_part1   ─── Feature builder (lags, momentum, crack spread, seasonal, macro, regime)
    ↓
gas_part2   ─── sklearn ensemble (HistGBM + RF + GBM + ElasticNet + Ridge)
gas_part2b  ─── XGBoost sleeve (3 configs, gated on val RMSE)         [optional]
gas_part2a  ─── LSTM sleeve (2-layer, sequence_length=16)              [optional]
    ↓
gas_part3   ─── Governance + inverse-RMSE fusion + prediction_log.csv
    ↓
gas_part9   ─── Live attribution: MAE/MAPE/DM test/drift detection
```

**Target variable:** Next week's U.S. average regular gas price ($/gal) — `GASREGCOVW` from EIA/FRED.

**Cadence:** Weekly, Monday. EIA releases weekly gas price data Monday mornings.

**Live-row contract:** Part 1 keeps the most recent week (whose next-week target
is not yet realized) as a flagged `is_live` row. Every sleeve trains on labeled
rows only and predicts the live row — so the logged forecast genuinely targets
`target_date = anchor week + 7 days`, the next EIA release. The prediction log
is keyed by `target_date` and each row carries `is_live_forecast` so
retrospective (--force off-cycle) runs are distinguishable from real forecasts.

**Common gate window:** all three sleeves score on the same trailing 52 labeled
weeks, so the inverse-RMSE fusion gates in Part 3 compare like with like.

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/GasPriceForecast.git
cd GasPriceForecast
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **LSTM sleeve** requires PyTorch. Uncomment `torch` in `requirements.txt` to enable.
> The activation gate is enforced **in code**: `gas_part2a_lstm_sleeve.py` reads
> `gas_part2b_summary.json` each run and skips itself unless
> `xgb_sleeve_recommended: true` (override with `GASPRICE_FORCE_LSTM=1`).

### 3. Set API keys

All three API keys are **free**:

```bash
# Required (core FRED/EIA gas price data)
export FRED_API_KEY="your_fred_key"
# ↳ Free at: https://fred.stlouisfed.org/docs/api/api_key.html

# Strongly recommended (EIA fundamentals: stocks, demand, refinery util —
# these feed both the regime engine and the forecaster)
export EIA_API_KEY="your_eia_key"
# ↳ Free at: https://www.eia.gov/opendata/register.php
```

### 4. Run the pipeline

```bash
# Full weekly run
python gas_run_weekly_prediction.py

# Force run on any day
python gas_run_weekly_prediction.py --force

# Run pipeline AND backfill realized prices
python gas_run_weekly_prediction.py --force --with-backfill
```

---

## 📁 Pipeline Files

| File | Description |
|------|-------------|
| `gas_part0_data_infrastructure.py` | Core data: FRED + yfinance, DuckDB store |
| `gas_part0c_eia_fetcher.py` | EIA weekly petroleum data (API v2) |
| `gas_part6_regime_engine.py` | HMM regime detection (4 states) |
| `gas_part1_feature_builder.py` | Feature engineering (lags, seasonal, crack spread, macro) |
| `gas_part2_forecaster.py` | sklearn ensemble forecaster (primary) |
| `gas_part2b_xgb_ensemble.py` | XGBoost sleeve (optional, gated) |
| `gas_part2a_lstm_sleeve.py` | LSTM deep learning sleeve (optional, gated) |
| `gas_part3_governance.py` | Governance, fusion, prediction log writer |
| `gas_part9_live_attribution.py` | Live performance diagnostics |
| `gas_backfill_realized.py` | Backfill EIA realized prices into prediction log |
| `gas_run_weekly_prediction.py` | Weekly pipeline runner (orchestrator) |

---

## 📦 Artifacts

All artifacts are written to subdirectories and ignored by `.gitignore`; the GitHub
Actions workflows commit the accumulating subset with `git add -f` and stage the
dashboard's JSON/CSV copies into `data/`.

| Artifact | Path |
|----------|------|
| Master weekly parquet | `artifacts_part0/gas_weekly_master.parquet` |
| Feature matrix | `artifacts_part1/gas_feature_matrix.parquet` |
| Forecast tape | `artifacts_part2/gas_forecast_tape.parquet` |
| **Prediction log** | `artifacts_part3/prediction_log.csv` |
| Fusion tape | `artifacts_part3/gas_fusion_tape.parquet` |
| Regime tape | `artifacts_part6/gas_regime_tape.parquet` |
| Live attribution report | `artifacts_part9/live_attribution_report.json` |

---

## ⚙️ GitHub Actions Setup

Two production workflows (dual-UTC-cron pattern — GitHub ignores `timezone:` keys):

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `weekly-production.yml` | Mondays 10:35 AM ET (14:35 & 15:35 UTC + time gate) | Full forecast pipeline; commits artifacts; triggers Pages |
| `weekly-backfill.yml` | Wednesdays 8:00 AM ET (12:00 & 13:00 UTC + time gate) | Backfills realized EIA prices; re-runs Part 9; commits with `[skip ci]` |
| `pages.yml` | On production commit / manual | Deploys the dashboard with the committed data files |

### Required secrets (Settings → Secrets → Actions)

| Secret | Required | Description |
|--------|----------|-------------|
| `FRED_API_KEY` | ✅ Yes | Core gas price + macro data |
| `EIA_API_KEY` | 🟢 Recommended | EIA fundamentals: stocks, demand, refinery data |

### Enable GitHub Pages

1. Go to **Settings → Pages**
2. Set Source to **GitHub Actions** (the `pages.yml` workflow deploys the site;
   `configure-pages` with `enablement: true` will also self-enable on first run)
3. Your dashboard will be live at `https://YOUR_USERNAME.github.io/GasPriceForecast/`

The production workflow stages `prediction_log.csv` and the summary JSONs into
`data/` on each run; `pages.yml` copies them flat into the site root, which is
where `index.html` fetches them from.

---

## 🔮 Model Details

### Feature Families

| Family | Examples |
|--------|---------|
| **LAG** | `gas_us_avg_lag_1w` through `lag_52w` |
| **MOMENTUM** | 1w/4w/12w/26w price-to-rolling-mean ratio |
| **VOLATILITY** | 4w/8w/13w rolling return std |
| **CRUDE** | WTI price, returns, crude-to-gas ratio |
| **RBOB** | RBOB gasoline futures, RBOB crack spread |
| **EIA** | Stocks z-score, days-of-supply, demand trend, refinery util deviation |
| **SEASONAL** | Month dummies, driving season, hurricane season, winter demand |
| **MACRO** | USD index, 10Y Treasury, S&P 500, XLE energy ETF, CPI energy |
| **REGIME** | One-hot HMM regime (prior week) |

### Sleeve Gates

All gate RMSEs are computed on the **common gate window** (last 52 labeled weeks)
and **fail closed** — a missing baseline means the gate does not pass.

The LSTM sleeve activates only when **both** gates pass:
1. `xgb_sleeve_recommended = true` (XGB RMSE < sklearn ensemble RMSE) —
   enforced at runtime by `gas_part2a_lstm_sleeve.py` itself
2. `lstm_sleeve_recommended = true` (LSTM RMSE < both sklearn and XGB)

### Confidence Flags

- **HIGH_CONF**: Sleeve forecasts agree within 2% AND regime ≠ SUPPLY_SHOCK
- **LOW_CONF**: Sleeves disagree > 2% OR regime = SUPPLY_SHOCK

### Regimes

| Regime | Characteristics |
|--------|----------------|
| NORMAL | Balanced supply/demand, moderate volatility |
| SUPPLY_SHOCK | Refinery outages, hurricane season — spiking prices, high volatility |
| DEMAND_SURGE | Summer driving, economic expansion — low stocks, rising demand |
| DEFLATION | Demand collapse, oil glut — falling prices, stock build |

---

## 📊 Google Colab Usage

All parts support Google Colab + Drive. Set `GASPRICE_ROOT` to your Drive path:

```python
import os
os.environ["GASPRICE_ROOT"] = "/content/drive/MyDrive/GasPriceForecast"
os.environ["FRED_API_KEY"] = "your_key"
os.environ["EIA_API_KEY"] = "your_key"

# Then run each part:
%run gas_part0_data_infrastructure.py
# ... etc.
```

---

## 📈 Backfilling Realized Prices

EIA releases weekly gas prices every Monday morning. The `weekly-backfill.yml`
workflow runs each Wednesday (2 days of publication buffer); locally:

```bash
python gas_backfill_realized.py

# Audit without writing
python gas_backfill_realized.py --dry-run
```

The backfill script:
1. Fetches EIA `GASREGCOVW` weekly price history via FRED (or EIA API as fallback)
2. Matches each `target_date` in the prediction log to the EIA release (±3 days)
3. Computes MAE, MAPE, and direction accuracy (vs the prior week's realized
   EIA price) for each matured row
4. Is idempotent — already-realized rows are skipped unless `--force` is passed

---

## 🧪 Validation

After accumulating at least 8 weeks of realized predictions, `gas_part9` will compute:

- **All-time MAE, RMSE, MAPE** vs. naive carry benchmark
- **Rolling MAPE** (4w, 8w, 13w, 26w)
- **Diebold-Mariano test** (is the model significantly better than naively predicting last week's price?)
- **Concept drift detection** (recent RMSE / historical RMSE ratio)
- **Direction accuracy** (did we correctly call price direction each week?)

Health thresholds:
| Metric | Warning | Stop Signal |
|--------|---------|-------------|
| MAPE | > 3% | > 6% |
| Direction accuracy | < 50% | < 40% |
| RMSE ratio (drift) | > 1.5× | > 2.0× |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GASPRICE_ROOT` | ✅ | Project root directory |
| `FRED_API_KEY` | ✅ | FRED API key for gas price + macro data |
| `EIA_API_KEY` | 🟢 | EIA Open Data API key for petroleum fundamentals |

---

## 📜 License

MIT — see LICENSE for details.

---

*Built following the same architecture as [PriceCallProject](https://github.com/YOUR_USERNAME/PriceCallProject)
but targeting gas price regression rather than equity tail risk classification.*
