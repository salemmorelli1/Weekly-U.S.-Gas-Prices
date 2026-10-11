from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gas_backfill_realized as backfill
import gas_part0_data_infrastructure as part0
import gas_part0c_eia_fetcher as part0c
import gas_part1_feature_builder as part1
import gas_part9_live_attribution as attribution
import gas_release_manifest as release
from gas_source_contract import EIA_TARGET_SERIES, OBSERVATION_CONTRACT, eia_price_records, observed_monday_prices
from gas_time_contract import protocol_eligible_record, sha256_file, strict_json_dump


def ledger_row(**overrides):
    row = dict(decision_date='2026-09-08', anchor_week='2026-09-07', target_date='2026-09-14',
               is_live_forecast=1, pred_fusion=4.0, pred_persistence=3.9, actual=np.nan,
               schema_version='V2_RELEASE_LEDGER', run_utc='2026-09-08T19:00:00Z',
               pipeline_run_id='audit', source_code_sha='a'*40)
    return {**row, **overrides}


def test_regular_conventional_series_is_shared_by_ingestion_and_backfill(monkeypatch):
    assert EIA_TARGET_SERIES == 'PET.EMM_EPMRU_PTE_NUS_DPG.W'
    assert part0c.EIA_SERIES['eia_gas_us_regular'] == backfill.EIA_API_SERIES == EIA_TARGET_SERIES
    assert part0c.EIA_SERIES['eia_gas_us_diesel'] == 'PET.EMD_EPD2D_PTE_NUS_DPG.W'
    called = []
    def get(url, **kwargs):
        called.append(url)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'response': {'data': [{'period': '2026-09-14', 'value': '4.1',
                                   'series': 'EMM_EPMRU_PTE_NUS_DPG', 'units': '$/GAL'}]}})
    monkeypatch.setenv('EIA_API_KEY', 'dummy-for-mocked-request')
    monkeypatch.setattr(backfill.requests, 'get', get)
    series = backfill.fetch_gas_history_eia_api('2026-09-14', '2026-09-14')
    assert called == ['https://api.eia.gov/v2/seriesid/' + EIA_TARGET_SERIES]
    assert series.loc['2026-09-14'] == 4.1


@pytest.mark.parametrize('reported,units', [('EMD_EPD2D_PTE_NUS_DPG', '$/GAL'),
                                          ('EMM_EPMRU_PTE_NUS_DPG', 'percent')])
def test_provider_product_and_units_must_match_the_target(reported, units):
    with pytest.raises(ValueError):
        eia_price_records([dict(period='2026-09-14', value=4.1, series=reported, units=units)], EIA_TARGET_SERIES)


def test_missing_fred_release_is_not_forward_filled(monkeypatch):
    raw = pd.Series([3.8, np.nan], index=pd.to_datetime(['2026-09-07', '2026-09-14']))
    fred = SimpleNamespace(get_series=lambda *args, **kwargs: raw)
    result = part0.fetch_fred_series(fred, 'GASREGCOVW', '2026-09-07', '2026-09-14')
    assert pd.isna(result.iloc[-1])
    monkeypatch.setattr(part0, 'eastern_today', lambda: pd.Timestamp('2026-09-15'))
    frame = result.rename('gas_us_avg').rename_axis('week_date').reset_index()
    assert not part0.check_freshness(frame, part0.Part0Config(min_history_weeks=1))['data_freshness_ok']


@pytest.mark.parametrize('dates', [['2026-09-08'], ['2026-09-07', '2026-09-07']])
def test_observed_prices_require_exact_unique_mondays(dates):
    with pytest.raises(ValueError, match='Monday'):
        observed_monday_prices(pd.Series([4.0]*len(dates), index=pd.to_datetime(dates)))


def test_gap_does_not_change_calendar_horizon_or_weekly_lags():
    frame = pd.DataFrame({'week_date': pd.to_datetime(['2026-09-07','2026-09-21','2026-09-28']),
                          'gas_us_avg':[3.8,4.0,4.1]})
    target = part1.build_target(frame, part1.Part1Config())
    assert pd.isna(target.iloc[0]) and target.iloc[1] == 4.1
    features, actual = part1.build_feature_matrix(frame, None, part1.Part1Config())
    sept21 = features.loc[features.week_date.eq(pd.Timestamp('2026-09-21'))].iloc[0]
    assert pd.isna(sept21['gas_us_avg_lag_1w'])
    assert sept21['gas_us_avg_lag_2w'] == 3.8
    assert features.loc[features.is_live.eq(1), 'week_date'].tolist() == [pd.Timestamp('2026-09-28')]
    assert pd.isna(actual.iloc[-1])


def test_days_supply_uses_daily_demand_units():
    frame = pd.DataFrame({'eia_gas_stocks_total':[210000.]*3, 'eia_gas_demand':[9000.]*3})
    derived = part0c.compute_eia_derived_features(frame, part0c.Part0cConfig())
    assert derived.eia_gas_days_supply.iloc[-1] == pytest.approx(23.3333333)
    assert part1.add_eia_ratio_features(frame).eia_days_supply.iloc[-1] == pytest.approx(23.3333333)


@pytest.mark.parametrize('changes', [dict(run_utc='2026-09-15T19:00:00Z'), dict(run_utc='2026-09-08'),
                                     dict(schema_version='V1_WEEKLY'), dict(pred_persistence=np.inf),
                                     dict(source_code_sha=''), dict(decision_date='2026-09-06')])
def test_live_eligibility_rejects_hindsight_and_unqualified_rows(changes):
    assert protocol_eligible_record(ledger_row())
    assert not protocol_eligible_record(ledger_row(**changes))


def test_live_metrics_use_only_finite_exact_target_pairs(tmp_path):
    path = tmp_path/'ledger.csv'
    rows = [ledger_row(actual=4.1, actual_date='2026-09-14'),
            ledger_row(actual=np.inf, actual_date='2026-09-14'),
            ledger_row(actual=4.2, actual_date='2026-09-15')]
    pd.DataFrame(rows).to_csv(path, index=False)
    assert len(attribution.load_realized_rows(path)) == 1
    pd.DataFrame([rows[0], rows[0]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        attribution.load_realized_rows(path)


def test_direction_uses_frozen_issue_time_persistence(monkeypatch):
    monkeypatch.setattr(backfill, 'eastern_today', lambda: pd.Timestamp('2026-09-15'))
    prices = pd.Series([4.3,4.1], index=pd.to_datetime(['2026-09-07','2026-09-14']))
    # At issue time the last known price was 3.9; a revised 4.3 would reverse direction.
    result, _, _ = backfill.backfill(pd.DataFrame([ledger_row()]), prices)
    assert result.direction_correct.iloc[0] == 1
    assert result.pred_persistence.iloc[0] == 3.9


def test_unverified_legacy_master_cannot_supply_realized_prices(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill, 'PROJECT_DIR', tmp_path)
    monkeypatch.setattr(backfill, 'fetch_gas_history_fred', lambda *a: pd.Series(dtype=float))
    monkeypatch.setattr(backfill, 'fetch_gas_history_eia_api', lambda *a: pd.Series(dtype=float))
    folder = tmp_path/'artifacts_part0'; folder.mkdir()
    path=folder/'gas_weekly_master.parquet'
    pd.DataFrame({'week_date':[pd.Timestamp('2026-09-14')], 'gas_us_avg':[4.1]}).to_parquet(path)
    assert backfill.fetch_gas_history_master('2026-09-14','2026-09-14').empty
    strict_json_dump({'price_observation_contract':OBSERVATION_CONTRACT,
                      'master_parquet_sha256':sha256_file(path)}, folder/'part0_summary.json')
    assert backfill.fetch_gas_history_master('2026-09-14','2026-09-14').iloc[0] == 4.1


def test_backfill_manifest_refuses_to_resign_a_changed_forecast(tmp_path):
    data=tmp_path/'data'; data.mkdir()
    pd.DataFrame([ledger_row()]).to_csv(data/'prediction_log.csv',index=False)
    strict_json_dump({'ok':True},data/'live_attribution_report.json')
    pd.DataFrame({'n':[1]}).to_csv(data/'live_attribution_tape.csv',index=False)
    strict_json_dump({'schema_version':release.SCHEMA,'files':{
        p.name:{'sha256':sha256_file(p),'size_bytes':p.stat().st_size}
        for p in data.iterdir()}},data/'release_manifest.json')
    folder=tmp_path/'artifacts_part3'; folder.mkdir()
    pd.DataFrame([ledger_row(pred_fusion=9.0)]).to_csv(folder/'prediction_log.csv',index=False)
    before=(data/'release_manifest.json').read_bytes()
    with pytest.raises(ValueError,match='immutable'):
        release.stage_backfill(tmp_path)
    assert (data/'release_manifest.json').read_bytes() == before
