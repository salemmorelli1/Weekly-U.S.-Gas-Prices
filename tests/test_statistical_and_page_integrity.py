import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import gas_part1_feature_builder as part1
import gas_part2_forecaster as part2
import gas_part9_live_attribution as part9
from gas_forecast_statistics import (
    TEST_METHOD,
    compare_squared_errors,
    one_sided_hac_mean_test,
)
from gas_time_contract import strict_json_dump

ROOT = Path(__file__).resolve().parents[1]


def test_hac_test_accounts_for_serial_correlation():
    loss_advantage = np.repeat(
        np.array([0.02, 0.03, 0.01, 0.04, 0.02, 0.03, -0.01, 0.02]),
        6,
    )
    result = one_sided_hac_mean_test(loss_advantage)
    iid_p_value = stats.ttest_1samp(
        loss_advantage,
        0.0,
        alternative="greater",
    ).pvalue

    assert result["test_method"] == TEST_METHOD
    assert result["hac_lags"] > 0
    assert result["p_value"] > iid_p_value
    assert result["interpretation"] == "MODEL_SIGNIFICANTLY_BETTER"


def test_committed_oof_candidate_remains_fail_closed_under_hac_test():
    frame = pd.read_csv(ROOT / "data/gas_oof_predictions.csv")
    result = compare_squared_errors(
        frame["actual"].to_numpy(),
        frame["pred_ensemble"].to_numpy(),
        frame["pred_persistence"].to_numpy(),
    )
    wrapped = part9.diebold_mariano_test(
        frame["actual"].to_numpy(),
        frame["pred_ensemble"].to_numpy(),
        frame["pred_persistence"].to_numpy(),
    )

    assert result["n"] == 52
    assert result["mean_loss_advantage"] < 0
    assert result["p_value"] > 0.90
    assert result["interpretation"] == "MODEL_WORSE_THAN_NAIVE"
    assert wrapped["dm_method"] == TEST_METHOD
    assert wrapped["dm_hac_lags"] == result["hac_lags"]


def test_part2_gate_records_hac_evidence():
    n_rows = 48
    x_axis = np.arange(n_rows, dtype=float)
    target = 3.2 + 0.003 * x_axis + 0.08 * np.sin(x_axis / 4)
    frame = pd.DataFrame(
        {
            "week_date": pd.date_range("2025-01-06", periods=n_rows, freq="W-MON"),
            "signal": np.cos(x_axis / 5),
            "gas_us_avg_current": np.r_[target[0], target[:-1]],
            "is_live": 0,
        }
    )
    cfg = part2.Part2Config(
        val_weeks=12,
        min_train_weeks=12,
        hgb_max_iter=5,
        rf_n_estimators=5,
        gbm_n_estimators=5,
    )
    _, _, metrics, oof, _ = part2.walk_forward_train(
        frame,
        pd.Series(target),
        cfg,
    )

    assert len(oof) == 12
    assert np.isfinite(metrics["paired_test_statistic"])
    assert np.isfinite(metrics["paired_p_value"])
    assert metrics["paired_hac_lags"] >= 1


def test_nonvintage_macro_releases_are_not_predictive_features():
    frame = pd.DataFrame(
        {
            "cpi_energy": np.linspace(100.0, 110.0, 16),
            "unemployment": np.linspace(4.0, 4.5, 16),
            "gdp_growth": np.linspace(1.0, 2.0, 16),
            "treasury_10y": np.linspace(3.0, 4.0, 16),
        }
    )
    result = part1.add_macro_features(frame.copy())
    assert "treasury_level" in result
    assert "cpi_energy_chg_4w" not in result
    assert not any(
        name.startswith(("unemployment_", "gdp_growth_"))
        for name in result.columns
    )


def test_strict_json_bytes_are_lf_only(tmp_path):
    path = tmp_path / "artifact.json"
    strict_json_dump({"alpha": 1, "nested": {"beta": 2}}, path)
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r\n" not in raw
    assert json.loads(raw) == {"alpha": 1, "nested": {"beta": 2}}


def test_pages_hero_and_chart_match_forecast_objective():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    identifiers = re.findall(r'\bid="([^"]+)"', html)

    assert len(identifiers) == len(set(identifiers))
    assert "Forecast next week’s" in html
    assert "Observed Monday → next Monday target" in html
    assert 'id="heroObserved"' in html
    assert 'id="heroCandidate"' in html
    assert 'id="heroForecast"' in html
    assert 'id="heroPublicationMode"' in html
    assert 'id="fitScope"' in html
    assert 'fitRows.filter(row => row.oos_val === 1)' in html
    assert "Full fitted history (diagnostic)" in html
    assert "price-orbit" not in html
    assert 'class="pump"' not in html
    assert "$3.776" not in html


def test_repository_declares_lf_checkout_policy():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in attributes
