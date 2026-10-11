import numpy as np
import pandas as pd
import pytest

import gas_part2a_lstm_sleeve as lstm
import gas_part2b_xgb_ensemble as xgb
import gas_part3_governance as governance
from gas_validation import NESTED_PROTOCOL, aligned_oof, candidate_gate, prequential_fusion


def training_frame(n=96):
    rng=np.random.default_rng(79)
    actual=3.5 + .1*np.sin(np.arange(n)/8) + rng.normal(0,.02,n)
    frame=pd.DataFrame({'week_date':pd.date_range('2024-01-01',periods=n,freq='W-MON'),
                        'gas_us_avg_current':np.r_[actual[0],actual[:-1]],
                        'signal':np.cos(np.arange(n)/9), 'is_live':0})
    return frame,pd.Series(actual)


def test_xgboost_stopping_cannot_read_the_outer_scored_outcomes():
    assert xgb.HAVE_XGB, 'Install the pinned CPU XGBoost dependency'
    X,y=training_frame()
    cfg=xgb.Part2bConfig(val_weeks=24,xgb_configs=({'n_estimators':8,'max_depth':2,
                                                 'learning_rate':.1,'n_jobs':1},))
    *_, scored=xgb.train_xgb_ensemble(X,y,cfg)
    altered=y.copy(); altered.iloc[-24:]+=20
    *_, changed=xgb.train_xgb_ensemble(X,altered,cfg)
    first=scored.fold.eq(1)
    np.testing.assert_allclose(scored.loc[first,'pred_xgb_ensemble'],
                               changed.loc[first,'pred_xgb_ensemble'],rtol=0,atol=0)
    assert len(scored)==24 and aligned_oof(scored,scored)


def test_lstm_stopping_cannot_read_the_outer_scored_outcomes():
    assert lstm.HAVE_TORCH, 'Install the pinned CPU PyTorch dependency'
    X,y=training_frame()
    cfg=lstm.Part2aConfig(sequence_length=4,val_weeks=24,epochs=1,batch_size=32,
                         lstm_hidden_size_1=4,lstm_hidden_size_2=3,dense_hidden_size=3,
                         dropout_rate=0,device='cpu')
    _,_,scored,epochs=lstm.nested_lstm_forecasts(X,y,cfg)
    altered=y.copy(); altered.iloc[-24:]+=20
    _,_,changed,_=lstm.nested_lstm_forecasts(X,altered,cfg)
    first=scored.fold.eq(1)
    np.testing.assert_allclose(scored.loc[first,'pred_lstm'],changed.loc[first,'pred_lstm'],rtol=0,atol=0)
    assert len(scored)==24 and epochs==[1]*4


def test_lstm_refuses_to_compress_missing_calendar_weeks():
    X,y=training_frame()
    with pytest.raises(ValueError,match='calendar-week'):
        lstm.nested_lstm_forecasts(X.drop(20).reset_index(drop=True),y.drop(20).reset_index(drop=True),
                                   lstm.Part2aConfig())


def test_oof_alignment_rejects_same_size_but_different_targets():
    core=pd.DataFrame({'week_date':pd.date_range('2024-01-01',periods=12,freq='W-MON'),
                       'actual':4.,'pred_persistence':3.9,'fold':np.repeat([1,2,3,4],3)})
    assert aligned_oof(core,core)
    for field in ['actual','pred_persistence','fold']:
        changed=core.copy(); changed.loc[0,field]+=1
        assert not aligned_oof(core,changed)
    changed=core.copy(); changed.week_date+=pd.Timedelta(days=7)
    assert not aligned_oof(core,changed)


def test_fusion_weights_do_not_use_their_own_future_block():
    frame=pd.DataFrame({'week_date':pd.date_range('2024-01-01',periods=24,freq='W-MON'),
                        'actual':4.+.1*np.sin(np.arange(24)), 'pred_persistence':3.9,
                        'fold':np.repeat([1,2,3,4],6), 'a':4.1,'b':4.2})
    first=prequential_fusion(frame,['a','b'])
    changed=frame.copy(); changed.loc[changed.fold.ge(2),'actual']+=100
    second=prequential_fusion(changed,['a','b'])
    np.testing.assert_allclose(first.loc[first.fold.le(2),'pred_fusion'],
                               second.loc[second.fold.le(2),'pred_fusion'])


def test_bad_optional_fusion_cannot_replace_a_valid_core():
    actual=4.+.1*np.sin(np.arange(52))
    frame=pd.DataFrame({'week_date':pd.date_range('2024-01-01',periods=52,freq='W-MON'),
                        'actual':actual,'pred_persistence':actual+.5+.1*np.cos(np.arange(52)),
                        'fold':np.repeat([1,2,3,4],13),'pred_part2':actual+.01})
    frames={'part2':frame,'part2b':frame.assign(pred_part2b=actual+10)}
    summaries={'part2':{'val_metrics':{'ensemble_rmse':.01},'pipeline_run_id':'run'},
               'part2b':{'xgb_sleeve_recommended':True,'xgb_val_rmse':10.,
                          'pipeline_run_id':'run','validation_protocol':NESTED_PROTOCOL}}
    weights,active,scored,validation=governance.governed_fusion_weights(summaries,frames)
    assert active==['part2'] and weights=={'part2':1.}
    assert validation['operator_validated']
    np.testing.assert_allclose(scored.pred_fusion,frame.pred_part2)


def test_missing_or_nonfinite_scored_rows_fail_the_operator_gate():
    frame=pd.DataFrame({'actual':[4.]*12,'pred_persistence':[4.1]*12,
                        'fold':np.repeat([1,2,3,4],3),'pred':[4.]*12})
    frame.loc[0,'pred']=np.inf
    assert not candidate_gate(frame,'pred')['operator_validated']
    assert not candidate_gate(frame.drop(columns='actual'),'pred')['operator_validated']


@pytest.mark.parametrize('tamper', ['none','run','source_date','feature_hash','oof_hash','tape_hash','protocol'])
def test_optional_evidence_requires_hashes_lineage_and_exact_live_clock(tmp_path,tamper):
    from gas_time_contract import sha256_file
    cfg=governance.Part3Config()
    for folder in ['artifacts_part1',cfg.part2_dir_name,cfg.part2b_dir_name]:
        (tmp_path/folder).mkdir()
    matrix=tmp_path/'artifacts_part1/gas_feature_matrix.parquet'
    pd.DataFrame({'x':[1.]}).to_parquet(matrix,index=False)
    actual=4.+.1*np.sin(np.arange(52))
    core=pd.DataFrame({'week_date':pd.date_range(end='2026-09-07',periods=52,freq='W-MON'),
                       'actual':actual,'pred_persistence':actual+.5+.1*np.cos(np.arange(52)),
                       'fold':np.repeat([1,2,3,4],13),'pred_ensemble':actual+.1})
    core_path=tmp_path/cfg.part2_dir_name/'gas_oof_predictions.csv'; core.to_csv(core_path,index=False)
    optional=core.assign(pred_xgb_ensemble=actual+.01)
    oof_path=tmp_path/cfg.part2b_dir_name/'gas_xgb_oof.csv'; optional.to_csv(oof_path,index=False)
    tape_path=tmp_path/cfg.part2b_dir_name/'gas_xgb_tape.parquet'
    pd.DataFrame({'week_date':[pd.Timestamp('2026-09-14')], 'pred_xgb_ensemble':[4.1],
                  'is_live':[1]}).to_parquet(tape_path,index=False)
    summaries={'part2':{'pipeline_run_id':'current','forecast_anchor_week':'2026-09-14',
                       'forecast_target_week':'2026-09-21','oof_sha256':sha256_file(core_path),
                       'val_metrics':{'ensemble_rmse':.1}},
               'part2b':{'pipeline_run_id':'current','validation_protocol':NESTED_PROTOCOL,
                          'xgb_sleeve_recommended':True,'xgb_val_rmse':.01,
                          'feature_matrix_sha256':sha256_file(matrix),'oof_sha256':sha256_file(oof_path),
                          'forecast_tape_sha256':sha256_file(tape_path),
                          'latest_forecast':{'anchor_week':'2026-09-14','target_week':'2026-09-21',
                                             'is_true_live_forecast':True}}}
    if tamper=='run': summaries['part2b']['pipeline_run_id']='other'
    elif tamper=='source_date': summaries['part2b']['latest_forecast']['anchor_week']='2026-09-07'
    elif tamper=='feature_hash': summaries['part2b']['feature_matrix_sha256']='wrong'
    elif tamper=='oof_hash': summaries['part2b']['oof_sha256']='wrong'
    elif tamper=='tape_hash': summaries['part2b']['forecast_tape_sha256']='wrong'
    elif tamper=='protocol': summaries['part2b']['validation_protocol']='TUNED_SCORED_HOLDOUT'
    qualified,frames=governance.qualified_sleeve_evidence(tmp_path,cfg,summaries)
    assert qualified['part2b']['xgb_sleeve_recommended'] == (tamper=='none')
    assert ('part2b' in frames) == (tamper=='none')
