import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

import gas_backfill_realized as backfill
import gas_part0_data_infrastructure as part0
import gas_part0c_eia_fetcher as part0c
import gas_part1_feature_builder as part1
import gas_part2_forecaster as part2
import gas_part2a_lstm_sleeve as lstm
import gas_part2b_xgb_ensemble as xgb
import gas_part3_governance as governance
import gas_part6_regime_engine as regimes
import gas_part9_live_attribution as attribution
import gas_release_manifest as release
from gas_time_contract import (
    protocol_eligible_record, read_prediction_log, sha256_file, write_prediction_log_atomic,
)


def test_disposable_pipeline_trains_stages_and_backfills_without_rewriting_forecasts(tmp_path, monkeypatch):
    monkeypatch.setenv('GASPRICE_ROOT',str(tmp_path))
    monkeypatch.setenv('GASPRICE_PIPELINE_RUN_ID','synthetic-release-audit')
    monkeypatch.setenv('GITHUB_SHA','a'*40)
    monkeypatch.setenv('GASPRICE_FORCE_LSTM','1')
    decision=pd.Timestamp('2026-10-06')
    monkeypatch.setattr(part0,'eastern_today',lambda:decision)
    monkeypatch.setattr(governance,'eastern_today',lambda:decision)
    monkeypatch.setattr(attribution,'eastern_today',lambda:decision)
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):
            value=datetime(2026,10,6,17,tzinfo=timezone.utc)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)
    monkeypatch.setattr(governance,'datetime',Clock)
    n=200
    dates=pd.date_range(end='2026-10-05',periods=n,freq='W-MON')
    rng=np.random.default_rng(922)
    prices=3.5+np.cumsum(rng.normal(0,.025,n))
    crude=65+np.cumsum(rng.normal(0,.4,n))
    def fred_series(series,*args,**kwargs):
        values=prices if series in {'GASREGCOVW','GASMIDCOVW'} else crude
        return pd.Series(values,index=dates)
    monkeypatch.setattr(part0,'get_fred_client',lambda:SimpleNamespace(get_series=fred_series))
    monkeypatch.setattr(part0,'fetch_yfinance_weekly',lambda *args:pd.DataFrame({'CL=F':crude,'RB=F':prices*.7},index=dates))
    cfg0=part0.Part0Config(history_start=str(dates[0].date()),history_end='2026-10-06',min_history_weeks=52)
    monkeypatch.setattr(part0,'Part0Config',lambda:cfg0)
    eia=pd.DataFrame({'week_date':dates,'eia_gas_us_regular':prices,
                      'eia_gas_stocks_total':210000+np.arange(n)*10.,'eia_gas_demand':9000+np.sin(np.arange(n))*100,
                      'eia_refinery_util':85+np.sin(np.arange(n)/4)})
    monkeypatch.setattr(part0c.EIAClient,'fetch_all_series',lambda *args:eia.copy())
    cfg6=regimes.Part6Config(n_regimes=2,hmm_n_iter=10,hmm_min_train_rows=20)
    monkeypatch.setattr(regimes,'Part6Config',lambda:cfg6)
    cfg2=part2.Part2Config(val_weeks=24,min_train_weeks=20,hgb_max_iter=5,rf_n_estimators=8,gbm_n_estimators=8)
    monkeypatch.setattr(part2,'Part2Config',lambda:cfg2)
    cfgx=xgb.Part2bConfig(val_weeks=24,xgb_configs=({'n_estimators':8,'max_depth':2,'learning_rate':.1,'n_jobs':1},))
    monkeypatch.setattr(xgb,'Part2bConfig',lambda:cfgx)
    cfgl=lstm.Part2aConfig(val_weeks=24,sequence_length=4,epochs=1,batch_size=64,
                          lstm_hidden_size_1=4,lstm_hidden_size_2=3,dense_hidden_size=3,dropout_rate=0,device='cpu')
    monkeypatch.setattr(lstm,'Part2aConfig',lambda:cfgl)
    for module in [part0,part0c,regimes,part1,part2,xgb,lstm,governance,attribution]:
        assert module.main()==0, module.__name__
    assert (tmp_path/'artifacts_part2a/gas_lstm_oof.csv').is_file()
    assert (tmp_path/'artifacts_part2b/gas_xgb_oof.csv').is_file()
    checkpoint=lstm.torch.load(tmp_path/'artifacts_part2a/gas_lstm_model.pt',weights_only=False)
    features=pd.read_parquet(tmp_path/'artifacts_part1/gas_feature_matrix.parquet')
    restored=lstm._new_model(len(lstm.get_feature_cols(features)),cfgl)
    restored.load_state_dict(checkpoint['model_state_dict'])
    bundle=(checkpoint['imputer'],checkpoint['scaler'],checkpoint['y_mean'],checkpoint['y_std'])
    replay=lstm._lstm_predict(restored,bundle,features[lstm.get_feature_cols(features)].to_numpy(),
                              np.array([len(features)-1]),cfgl)
    recorded=pd.read_parquet(tmp_path/'artifacts_part2a/gas_lstm_tape.parquet')
    np.testing.assert_allclose(replay[0],recorded.pred_lstm.iloc[-1],atol=1e-6)
    release.stage_release(tmp_path)
    release.verify_manifest(tmp_path)
    ledger_path=tmp_path/'artifacts_part3/prediction_log.csv'
    before=read_prediction_log(ledger_path)
    assert len(before)==1 and protocol_eligible_record(before.iloc[0])
    summary=json.loads((tmp_path/'artifacts_part3/gas_part3_summary.json').read_text())
    if not summary['operator_validated']:
        assert summary['publication_mode']=='FAIL_CLOSED_PERSISTENCE'
        assert before.pred_fusion.iloc[0]==before.pred_persistence.iloc[0]
    else:
        assert summary['fusion_operator_validation']['operator_validated']
    monkeypatch.setattr(backfill,'eastern_today',lambda:pd.Timestamp('2026-10-13'))
    monkeypatch.setattr(attribution,'eastern_today',lambda:pd.Timestamp('2026-10-13'))
    updated,matured,new=backfill.backfill(before.copy(),pd.Series([prices[-1]+.02],index=pd.to_datetime(['2026-10-12'])))
    assert matured==1 and new==1
    immutable=[key for key in before if key not in {'actual','actual_date','mae','rmse','mape','direction_correct'}]
    pd.testing.assert_frame_equal(before[immutable],updated[immutable])
    write_prediction_log_atomic(updated,ledger_path)
    assert attribution.main()==0
    release.stage_backfill(tmp_path)
    release.verify_manifest(tmp_path)
    report=json.loads((tmp_path/'data/live_attribution_report.json').read_text())
    assert report['n_realized']==1 and report['model_health']['min_required']==52
    manifest=json.loads((tmp_path/'data/release_manifest.json').read_text())
    assert manifest['files']['prediction_log.csv']['sha256']==sha256_file(ledger_path)
