"""Provenance and common-denominator checks for stabilized reporting figures."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as a


def fixture(root):
    rows=[]
    for rep in range(2):
        row=dict(scenario='steady_rr150',family='steady',method='OCFR Gaussian',
            replicate=rep,alarm=True,alarm_day=99,true_change=91,tau_hat_at_alarm=90,
            fit_success=True,pi_left=.02,pi_right=.03,risk_ratio=1.5)
        for target in a.ESTIMATION_TARGETS:
            truth=row[target]
            row.update({target+'_estimate':truth*1.2,target+'_point_valid':True,
                target+'_lower':truth*.8,target+'_upper':truth*1.4,target+'_interval_valid':True,
                'nishiura_'+target+'_estimate':truth*1.1,'nishiura_'+target+'_point_valid':True,
                'nishiura_'+target+'_admissible':True})
        rows.append(row)
    base=pd.DataFrame(rows)
    base.loc[0,'nishiura_risk_ratio_point_valid']=False
    base.loc[0,'nishiura_risk_ratio_estimate']=np.nan
    source=root/'estimation-reference';source.mkdir()
    base.to_csv(source/a.RAW_FILE,index=False)
    (source/a.MANIFEST_FILE).write_text('{}')
    raw=base.copy()
    for target in a.ESTIMATION_TARGETS:
        for suffix in ('estimate','point_valid','lower','upper','interval_valid'):
            raw['regularized_'+target+'_'+suffix]=base[target+'_'+suffix]
        raw['regularized_'+target+'_estimate']=base[target]*1.05
    stage=root/'estimation-stabilized';stage.mkdir()
    raw.to_csv(stage/a.RAW_FILE,index=False)
    records={'methods.py':'a'*64}
    manifest=dict(status='complete',stage='estimation-stabilized',mode='full',
        post_alarm_estimation_config={'precision':1.,'covariance':'sandwich'},
        original_reference_csv_sha256=a._sha256(source/a.RAW_FILE),
        original_reference_manifest_sha256=a._sha256(source/a.MANIFEST_FILE),
        postprocessor_code_files=records,
        postprocessor_code_hash=hashlib.sha256(a._canonical_json(records).encode('ascii')).hexdigest(),
        artifacts={a.RAW_FILE:a._sha256(stage/a.RAW_FILE)})
    (stage/a.MANIFEST_FILE).write_text(json.dumps(manifest))
    return raw,manifest,stage


def test_stabilized_all_alarm_and_three_way_paired_denominators(tmp_path):
    fixture(tmp_path)
    tables,meta=a._compact_load_stabilized_estimation(tmp_path)
    summary=tables['compact_stabilized_summary'].set_index('target')
    assert summary.loc['risk_ratio','n_point']==2
    assert summary.loc['risk_ratio','relative_rmse']==pytest.approx(.05)
    paired=tables['compact_stabilized_reference_paired'].set_index('target')
    assert paired.loc['risk_ratio','n_paired']==1
    assert paired.loc['pi_left','n_paired']==2
    assert paired.loc['risk_ratio','mle_paired_relative_rmse']==pytest.approx(.2)
    assert paired.loc['risk_ratio','reference_paired_relative_rmse']==pytest.approx(.1)
    assert paired.loc['risk_ratio','ocfr_paired_relative_rmse']==pytest.approx(.05)
    assert meta['precision']==1 and meta['status']=='complete'


@pytest.mark.parametrize('mutation',['source','config','raw','original'])
def test_reject_stale_or_changed_stabilized_evidence(tmp_path,mutation):
    raw,manifest,stage=fixture(tmp_path)
    if mutation=='source':
        manifest['original_reference_csv_sha256']='0'*64
    elif mutation=='config':
        manifest['post_alarm_estimation_config']['precision']=2.
    elif mutation=='raw':
        manifest['artifacts'][a.RAW_FILE]='0'*64
    else:
        raw.loc[0,'pi_left_estimate']=.9
        raw.to_csv(stage/a.RAW_FILE,index=False)
        manifest['artifacts'][a.RAW_FILE]=a._sha256(stage/a.RAW_FILE)
    (stage/a.MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(a.EvidenceError):
        a._compact_load_stabilized_estimation(tmp_path)


def test_optional_absent_stage_and_rejected_demo(tmp_path):
    assert a._compact_load_stabilized_estimation(tmp_path)==({}, {'status':'absent'})
    _,manifest,stage=fixture(tmp_path)
    manifest['mode']='demo'
    (stage/a.MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(a.EvidenceError,match='full'):
        a._compact_load_stabilized_estimation(tmp_path)
