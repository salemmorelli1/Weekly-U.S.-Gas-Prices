# ⛽ GasPriceForecast

> **Weekly U.S. average regular gas price forecasting model.**
> Ensemble of sklearn, XGBoost, and LSTM sleeves fused via inverse-RMSE weighting.
> Data: EIA, FRED, CollectAPI live prices, yfinance commodities.
> Runs automatically every Monday after the EIA weekly release (~10 AM ET).

[![Weekly Forecast](https://github.com/YOUR_USERNAME/GasPriceForecast/actions/workflows/weekly_forecast.yml/badge.svg)](https://github.com/YOUR_USERNAME/GasPriceForecast/actions/workflows/weekly_forecast.yml)

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
gas_part0b  ─── CollectAPI live gas prices by region + weather severity
gas_part0c  ─── EIA API v2 direct (stocks, demand, refinery utilization)
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

> **LSTM sleeve** requires PyTorch. Uncomment `torch` in `requirements.txt` to enable,
> but only after `gas_part2b_summary.json` confirms `xgb_sleeve_recommended: true`.

### 3. Set API keys

All three API keys are **free**:

```bash
# Required (core FRED/EIA gas price data)
export FRED_API_KEY="your_fred_key"
# ↳ Free at: https://fred.stlouisfed.org/docs/api/api_key.html

# Optional (live weekly gas prices by region + weather)
export COLLECTAPI_KEY="your_collectapi_key"
# ↳ Free at: https://collectapi.com  (subscribe to Gas Prices + Weather APIs)

# Optional (direct EIA series for stocks, demand, refinery util)
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
| `gas_part0b_collectapi_fetcher.py` | Live gas prices by region (CollectAPI) |
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

All artifacts are written to subdirectories and **not committed to git** (see `.gitignore`).
Summary JSONs and the prediction log are copied to `docs/` by the GitHub Actions workflow.

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

The workflow (`.github/workflows/weekly_forecast.yml`) runs every Monday at 14:00 UTC.

### Required secrets (Settings → Secrets → Actions)

| Secret | Required | Description |
|--------|----------|-------------|
| `FRED_API_KEY` | ✅ Yes | Core gas price + macro data |
| `EIA_API_KEY` | ⚪ Optional | EIA stocks, demand, refinery data |
| `COLLECTAPI_KEY` | ⚪ Optional | Live regional gas prices + weather |

### Enable GitHub Pages

1. Go to **Settings → Pages**
2. Set Source to **Deploy from a branch**
3. Select branch `main`, folder `/docs`
4. Your dashboard will be live at `https://YOUR_USERNAME.github.io/GasPriceForecast/`

The workflow automatically copies `prediction_log.csv` and JSON summaries to `docs/` on each run.

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

The LSTM sleeve activates only when **both** gates pass:
1. `xgb_sleeve_recommended = true` (XGB RMSE < sklearn RMSE)
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
os.environ["COLLECTAPI_KEY"] = "your_key"

# Then run each part:
%run gas_part0_data_infrastructure.py
%run gas_part0b_collectapi_fetcher.py
# ... etc.
```

---

## 📈 Backfilling Realized Prices

EIA releases weekly gas prices every Monday morning. Run the backfill script
weekly (or let the GitHub Actions workflow do it via `--with-backfill`):

```bash
python gas_backfill_realized.py

# Audit without writing
python gas_backfill_realized.py --dry-run
```

The backfill script:
1. Fetches EIA `GASREGCOVW` weekly price history via FRED (or EIA API as fallback)
2. Matches each `target_date` in the prediction log to the EIA release
3. Computes MAE, MAPE, and direction accuracy for each matured row

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
| `EIA_API_KEY` | ⚪ | EIA API v2 key for petroleum supply data |
| `COLLECTAPI_KEY` | ⚪ | CollectAPI key for live regional prices + weather |

---

## 📜 License

MIT — see LICENSE for details.

---

*Built following the same architecture as [PriceCallProject](https://github.com/YOUR_USERNAME/PriceCallProject)
but targeting gas price regression rather than equity tail risk classification.*
