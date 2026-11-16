"""Portable synthetic reporting/interval checks; no DGP draws or native backend."""
from pathlib import Path
from dataclasses import asdict, replace
import json
import hashlib

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

import analysis as a
from config import (get_compact_release_study_config, get_post_alarm_validation_config,
                    PostAlarmEstimationConfig, PostAlarmLSConfig)
from simulation import _planned_looks
from methods import GAUSSIAN_SPENDING_POLICY


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True), encoding='utf-8')


def csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return a._sha256(path)


def source_rows(config):
    pieces = []
    for scenario in config.scenarios:
        if scenario.split != 'alternative' or scenario.misspecification != 'on_model':
            continue
        rep = np.arange(1000)
        alarm = rep < 20
        available = alarm & (rep != 0)
        roots = (rep % 20-10)/5
        frame = pd.DataFrame(dict(scenario=scenario.name, replicate=rep, method='OCFR',
            family=scenario.family, horizon=scenario.horizon, true_change=scenario.true_change,
            pi_left=scenario.pi_left, pi_right=scenario.pi_right, risk_ratio=scenario.risk_ratio,
            alarm=alarm, alarm_day=np.where(alarm, 104, np.nan),
            tau_hat_at_alarm=np.where(alarm, 40, np.nan), regularized_status='fixture',
            case_hash='a'*64, death_hash='b'*64, true_delay_hash='c'*64, assumed_delay_hash='d'*64,
            source_payload_hash='e'*64))
        for target, se_name in zip(a.ESTIMATION_TARGETS, a.INTERVAL_SE_NAMES):
            truth = getattr(scenario, target)
            center = np.log(truth) if target == 'risk_ratio' else logit(truth)
            transformed = center+.1*roots
            estimate = np.exp(transformed) if target == 'risk_ratio' else expit(transformed)
            low = np.exp(transformed-.196) if target == 'risk_ratio' else expit(transformed-.196)
            high = np.exp(transformed+.196) if target == 'risk_ratio' else expit(transformed+.196)
            for prefix in ('', 'regularized_'):
                frame[prefix+target+'_estimate'] = np.where(alarm, estimate, np.nan)
                frame[prefix+target+'_point_valid'] = alarm if prefix else (alarm & (rep != 2))
                frame[prefix+target+'_lower'] = np.where(available, low, np.nan)
                frame[prefix+target+'_upper'] = np.where(available, high, np.nan)
                frame[prefix+target+'_interval_valid'] = available
            frame[se_name] = np.where(available, .1, np.nan)
        days = [look-1 for look in _planned_looks(scenario, config)]
        no_alarm_trace = json.dumps([dict(day=day, available=False) for day in days])
        alarm_trace = [dict(day=day, available=False) for day in days if day < 104]
        alarm_trace.append(dict(day=104, available=True, statistic=2., boundary=1.))
        frame['trace_json'] = np.where(alarm, json.dumps(alarm_trace), no_alarm_trace)
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True).sort_values(['scenario', 'replicate'], ignore_index=True)


def complete_fixture(root):
    config = get_compact_release_study_config(backend='python')
    records = {name: str(index)*64 for index, name in enumerate(
        ('config.py', 'methods.py', 'simulation.py', 'csrc/ocfr_kernels.c'), start=1)}
    code_hash = hashlib.sha256(a._canonical_json(records).encode('ascii')).hexdigest()
    scientific = config.scientific_payload()
    digest = hashlib.sha256(a._canonical_json(dict(schema_version=3, code_hash=code_hash,
        scientific=scientific)).encode('ascii')).hexdigest()[:20]
    train = source_rows(config)
    evaluation = train[['scenario', 'replicate', 'method', 'alarm_day', 'tau_hat_at_alarm']].copy()
    raw_hash = csv(root/'evaluation'/a.RAW_FILE, evaluation)
    em = dict(schema_version=3, protocol='compact-v2', status='complete', stage='evaluation', mode='full',
        scientific_config=scientific, scientific_digest=digest, code_hash=code_hash, code_files=records,
        artifacts={a.RAW_FILE:raw_hash})
    write_json(root/'evaluation'/a.MANIFEST_FILE, em)
    reference = train[[column for column in train if not column.startswith('regularized_')]].copy()
    rh = csv(root/'estimation-reference'/a.RAW_FILE, reference)
    write_json(root/'estimation-reference'/a.MANIFEST_FILE,
        dict(status='complete', stage='estimation-reference', mode='full', artifacts={a.RAW_FILE:rh}))
    common = dict(status='complete', mode='full', original_manifest_sha256=a._sha256(root/'evaluation'/a.MANIFEST_FILE),
        original_csv_sha256=raw_hash, original_scientific_digest=digest,
        postprocessor_code_files=records, postprocessor_code_hash=code_hash,
        post_alarm_estimation_config=asdict(PostAlarmEstimationConfig()))
    for bank in ('original', 'validation'):
        cfg = config if bank == 'original' else replace(config, master_seed=2026111405)
        stage = 'estimation-stabilized' if bank == 'original' else 'estimation-validation'
        frame = train.copy() if bank == 'original' else source_rows(cfg)
        names = sorted(frame.scenario.unique())
        manifest = dict(common, stage=stage, scientific_config=cfg.scientific_payload(),
            scenarios=names, replications_by_scenario={name:1000 for name in names})
        if bank == 'original':
            manifest.update(original_code_files=records, original_code_hash=code_hash, scientific_digest=digest,
                original_reference_manifest_sha256=a._sha256(root/'estimation-reference'/a.MANIFEST_FILE),
                original_reference_csv_sha256=rh)
        else:
            manifest.update(validation_config=asdict(get_post_alarm_validation_config('compact-v2')),
                scientific_digest=hashlib.sha256(a._canonical_json(cfg.scientific_payload()).encode('ascii')).hexdigest()[:20])
        manifest['artifacts'] = {a.RAW_FILE:csv(root/stage/a.RAW_FILE, frame)}
        write_json(root/stage/a.MANIFEST_FILE, manifest)
        ls_stage = 'estimation-ls-'+bank
        ls = frame.copy()
        for target in a.ESTIMATION_TARGETS:
            ls['ls_'+target+'_estimate'] = ls['regularized_'+target+'_estimate']*1.01
            if target != 'risk_ratio':
                ls['ls_'+target+'_estimate'] = ls['ls_'+target+'_estimate'].clip(upper=.999)
            ls['ls_'+target+'_point_valid'] = ls.alarm & ls.replicate.ne(1)
        for column, value in dict(ls_status='success:fixture', ls_seconds=0., ls_rss=1.,
            ls_active_set='free', ls_kkt_residual=0., ls_kkt_tolerance=1e-6).items():
            ls[column] = value
        provenance = dict(source_manifest_sha256=a._sha256(root/stage/a.MANIFEST_FILE),
            source_csv_sha256=manifest['artifacts'][a.RAW_FILE], source_bank=bank,
            source_scientific_digest=manifest['scientific_digest'], scientific_config=cfg.scientific_payload(),
            postprocessor_code_hash=code_hash, postprocessor_code_files=records, postprocessor_native_files={},
            ls_estimation_config=asdict(PostAlarmLSConfig()))
        lm = dict(provenance, stage=ls_stage, status='complete', mode='full',
            scientific_digest=manifest['scientific_digest'], scenarios=names,
            replications_by_scenario={name:1000 for name in names},
            cache_key=hashlib.sha256(a._canonical_json(provenance).encode('ascii')).hexdigest()[:20],
            artifacts={a.RAW_FILE:csv(root/ls_stage/a.RAW_FILE, ls)})
        write_json(root/ls_stage/a.MANIFEST_FILE, lm)
    return config


@pytest.fixture(scope='module')
def full_bank(tmp_path_factory):
    root = tmp_path_factory.mktemp('v2-full-bank')
    complete_fixture(root)
    return root


def test_shortest_roots_missing_denominators_and_ties():
    cells = {'a':np.arange(20.), 'b':np.arange(1.,21.)}
    assert a.shortest_empirical_interval(cells, {'a':20,'b':20}) == {'lower':1., 'upper':19.}
    with pytest.raises(RuntimeError, match='infeasible'):
        a.shortest_empirical_interval({'a':np.arange(18.)}, {'a':20})
    assert a.shortest_empirical_interval({'a':np.arange(19.)}, {'a':20}) == {'lower':0.,'upper':18.}
    assert a.shortest_empirical_interval({'a':np.ones(20)}, {'a':20}) == {'lower':1.,'upper':1.}


def test_infinite_interval_is_available_and_missing_is_not():
    low, high, valid = a.truth_free_apply(np.array([0.,1000.,0.]), np.array([1000.,1.,1.]),
        np.array([True,True,False]), 'risk_ratio', {'lower':-1.,'upper':1.})
    assert low[0] == 0 and np.isposinf(high[0]) and valid[0]
    assert np.isposinf(low[1]) and np.isposinf(high[1]) and valid[1]
    assert not valid[2] and np.isnan(low[2])


def test_shorter_interval_matches_independent_exhaustive_search():
    # Exhaustively check finite support pairs, rather than reusing the optimized search.
    cells = {'left':np.array([-9., *range(-2, 17)]),
             'right':np.array([*range(-3, 16), 40.])}
    counts = {'left':20, 'right':20}
    support = np.unique(np.concatenate(list(cells.values())))
    candidates = [(upper-lower, lower, upper) for lower in support for upper in support
        if upper >= lower and all(np.sum((values >= lower) & (values <= upper)) >= 19
                                  for values in cells.values())]
    _, lower, upper = min(candidates)
    assert a.shortest_empirical_interval(cells, counts) == {'lower':lower, 'upper':upper}
    # A missing alarm interval still occupies one of the twenty denominator slots.
    missing = {'left':cells['left'][1:], 'right':cells['right'][:-1]}
    support = np.unique(np.concatenate(list(missing.values())))
    candidates = [(upper-lower, lower, upper) for lower in support for upper in support
        if upper >= lower and all(np.sum((values >= lower) & (values <= upper)) >= 19
                                  for values in missing.values())]
    _, lower, upper = min(candidates)
    assert a.shortest_empirical_interval(missing, counts) == {'lower':lower, 'upper':upper}


def test_calibrate_never_opens_validation_and_apply_requires_hash(full_bank, monkeypatch):
    original = Path.open
    def guard(path, *args, **kwargs):
        if 'estimation-validation' in path.parts:
            raise AssertionError('calibration opened validation')
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', guard)
        frozen = a.run_interval_calibration(full_bank)
    with pytest.raises(a.EvidenceError, match='before validation'):
        a.run_interval_application(full_bank, 'f'*64)
    result = a.run_interval_application(full_bank, frozen['cutoffs_sha256'])
    assert result['intervals_truth_free']
    for target in a.ESTIMATION_TARGETS:
        for cell in result['reports'][target].values():
            assert cell['alarms'] == 20 and cell['available'] == 19
            assert cell['unavailable_counted_as_noncoverage'] == 1


def test_source_grid_and_hash_mutations_rejected(full_bank, tmp_path):
    # Copy only the affected stage manifest, then point all other reads to the original bank.
    manifest_path = full_bank/'estimation-stabilized'/a.MANIFEST_FILE
    saved = manifest_path.read_bytes()
    manifest = json.loads(saved)
    manifest['postprocessor_code_hash'] = '0'*64
    try:
        write_json(manifest_path, manifest)
        with pytest.raises(a.EvidenceError, match='source digest mismatch'):
            a._v2_selected_source(full_bank, 'original')
    finally:
        manifest_path.write_bytes(saved)
    assert manifest['scientific_config']['compact_contract']['boundary_policy'] == GAUSSIAN_SPENDING_POLICY
    for mutate in ('seed', 'policy'):
        changed = json.loads(saved)
        if mutate == 'seed':
            changed['scientific_config']['master_seed'] = 2026110702
        else:
            changed['scientific_config']['compact_contract']['boundary_policy'] = 'historical-alpha-over-J'
        try:
            write_json(manifest_path, changed)
            with pytest.raises(a.EvidenceError):
                a._v2_selected_source(full_bank, 'original')
        finally:
            manifest_path.write_bytes(saved)
    raw = full_bank/'estimation-stabilized'/a.RAW_FILE
    saved_csv = raw.read_bytes()
    frame = pd.read_csv(raw).iloc[:-1]
    try:
        manifest = json.loads(saved);manifest['artifacts'][a.RAW_FILE] = csv(raw, frame)
        write_json(manifest_path, manifest)
        with pytest.raises(a.EvidenceError, match='grid'):
            a._v2_selected_source(full_bank, 'original')
    finally:
        raw.write_bytes(saved_csv);manifest_path.write_bytes(saved)


def test_compatible_frozen_artifact_and_target_specific_pairing(full_bank):
    # A compatible external producer can add its own provenance without changing
    # scientific bindings. The reporting loader must preserve its frozen bytes.
    a.run_interval_calibration(full_bank)
    cutoff = full_bank/'interval-calibration/CUTOFFS.json'
    record = json.loads(cutoff.read_text())
    record['training_source'].update(snapshot_sha256='a'*64, calibrator_sha256='b'*64)
    a._interval_atomic_json(cutoff, record)
    a._interval_atomic_json(cutoff.parent/'FROZEN.json',
        dict(cutoffs_sha256=a._sha256(cutoff), source=record['training_source']))
    before = cutoff.read_bytes()
    a.run_interval_application(full_bank, a._sha256(cutoff))
    interval_manifest = full_bank/'interval-validation/manifest.json'
    applied = json.loads(interval_manifest.read_text())
    applied['source'].update(snapshot_sha256='a'*64, calibrator_sha256='b'*64)
    a._interval_atomic_json(interval_manifest, applied)
    tables, metadata = a._compact_load_v2_postalarm(full_bank)
    assert cutoff.read_bytes() == before
    assert metadata['status'] == 'complete'
    assert len(tables['compact_v2_estimation_cells']) == 16
    assert len(tables['compact_v2_interval_summary']) == 16*3*2
    assert set(tables['compact_v2_interval_summary'].n_alarm) == {20}
    paired = tables['compact_v2_ls_summary']
    assert set(paired.method) == {'OCFR','Full-delay LS'}
    # Replicate 0 has no interval but valid paired points and must be retained.
    # Replicate 1 lacks LS; replicate 2 lacks only the unpenalized NB comparator.
    assert set(paired.n_common_paired) == {19}
    assert set(tables['compact_ls_summary'].n_common_paired) == {18}


def test_infinite_width_artifact_counts_as_available(tmp_path):
    row = dict(scenario='one', replicate=0, alarm=True, family='steady', true_change=91,
               pi_left=.02, pi_right=.03, risk_ratio=1.5)
    for target, se in zip(a.ESTIMATION_TARGETS, a.INTERVAL_SE_NAMES):
        row['regularized_'+target+'_estimate'] = row[target]
        row['regularized_'+target+'_point_valid'] = True
        row[se] = 1000. if target == 'risk_ratio' else 1.
        row['regularized_'+target+'_lower'] = np.nan
        row['regularized_'+target+'_upper'] = np.nan
        row['regularized_'+target+'_interval_valid'] = False
    frame = pd.DataFrame([row])
    cutoffs = {target:{'lower':-1.,'upper':1.} for target in a.ESTIMATION_TARGETS}
    record = a.apply_interval_cutoffs(frame, cutoffs, {}, tmp_path, 'a'*64)
    saved = record['reports']['risk_ratio']['one']
    assert saved['available']==1 and saved['covered']==1 and saved['infinite_widths']==1
    assert np.isposinf(saved['mean_available_width']) and np.isposinf(saved['median_available_width'])
    assert 'Infinity' in (tmp_path/'interval-validation/manifest.json').read_text()
    intervals = pd.read_csv(tmp_path/'interval-validation/intervals.csv', float_precision='round_trip')
    reporting = a._v2_interval_reporting(frame, intervals)
    rr = reporting[reporting.target.eq('risk_ratio') & reporting.interval_rule.eq('Global calibrated')].iloc[0]
    assert rr.n_available==1 and rr.coverage==1 and rr.n_infinite_width==1


def test_available_unknown_trace_and_changed_selection_rejected(full_bank):
    path=full_bank/'estimation-validation'/a.MANIFEST_FILE
    raw=path.parent/a.RAW_FILE
    original_manifest=path.read_bytes();original_csv=raw.read_bytes()
    try:
        manifest=json.loads(original_manifest);frame=pd.read_csv(raw)
        trace=json.loads(frame.loc[0,'trace_json'])
        trace[-1]['statistic']=float('nan')
        frame.loc[0,'trace_json']=json.dumps(trace)
        manifest['artifacts'][a.RAW_FILE]=csv(raw,frame);write_json(path,manifest)
        with pytest.raises(a.EvidenceError, match='does not replay'):
            a._v2_selected_source(full_bank,'validation')
    finally:
        raw.write_bytes(original_csv);path.write_bytes(original_manifest)
    path=full_bank/'estimation-stabilized'/a.MANIFEST_FILE;raw=path.parent/a.RAW_FILE
    original_manifest=path.read_bytes();original_csv=raw.read_bytes()
    try:
        manifest=json.loads(original_manifest);frame=pd.read_csv(raw)
        frame.loc[0,'tau_hat_at_alarm']=41
        manifest['artifacts'][a.RAW_FILE]=csv(raw,frame);write_json(path,manifest)
        with pytest.raises(a.EvidenceError, match='first selection'):
            a._v2_selected_source(full_bank,'original')
    finally:
        raw.write_bytes(original_csv);path.write_bytes(original_manifest)


def test_truth_changes_assessment_not_interval_construction(full_bank, tmp_path):
    frame, _, provenance, _ = a._v2_selected_source(full_bank, 'validation')
    cutoffs = {target:{'lower':-2.,'upper':2.} for target in a.ESTIMATION_TARGETS}
    one=tmp_path/'one';two=tmp_path/'two'
    a.apply_interval_cutoffs(frame, cutoffs, provenance, one, 'a'*64)
    changed = frame.copy()
    changed['pi_left'] = .9;changed['pi_right'] = .8;changed['risk_ratio'] = 9.
    a.apply_interval_cutoffs(changed, cutoffs, provenance, two, 'a'*64)
    first = pd.read_csv(one/'interval-validation/intervals.csv')
    second = pd.read_csv(two/'interval-validation/intervals.csv')
    columns = [column for column in first if not column.endswith('_covered')]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert not first[[t+'_covered' for t in a.ESTIMATION_TARGETS]].equals(
        second[[t+'_covered' for t in a.ESTIMATION_TARGETS]])


def test_cli_and_old_cutoff_schema_rejection(full_bank, tmp_path, monkeypatch):
    assert a.main(['--protocol','compact-v2','--input',str(full_bank),'--calibrate-intervals']) == 0
    cutoff=full_bank/'interval-calibration/CUTOFFS.json'
    assert a.main(['--protocol','compact-v2','--input',str(full_bank),'--apply-intervals',
                   '--expected-cutoff-sha256',a._sha256(cutoff)]) == 0
    old=tmp_path/'interval-calibration/CUTOFFS.json'
    write_json(old,dict(schema='historical-old-policy-cutoffs',training_source={}))
    write_json(old.parent/'FROZEN.json',dict(cutoffs_sha256=a._sha256(old),source={}))
    monkeypatch.setattr(a,'_v2_selected_source',lambda *args:pytest.fail('old cutoff opened a source bank'))
    with pytest.raises(a.EvidenceError, match='schema'):
        a.run_interval_application(tmp_path,a._sha256(old))


def test_apply_rejects_rehashed_wrong_cutoffs_before_validation(full_bank, monkeypatch):
    cutoff=full_bank/'interval-calibration/CUTOFFS.json';freeze=cutoff.parent/'FROZEN.json'
    original_cutoff=cutoff.read_bytes();original_freeze=freeze.read_bytes()
    try:
        record=json.loads(original_cutoff);record['targets']['risk_ratio']={'lower':-100.,'upper':100.}
        write_json(cutoff,record)
        frozen=json.loads(original_freeze);frozen['cutoffs_sha256']=a._sha256(cutoff);write_json(freeze,frozen)
        original_open=Path.open
        def guard(path,*args,**kwargs):
            if 'estimation-validation' in path.parts:pytest.fail('wrong cutoff opened validation')
            return original_open(path,*args,**kwargs)
        monkeypatch.setattr(Path,'open',guard)
        with pytest.raises(a.EvidenceError,match='different frozen'):
            a.run_interval_application(full_bank,a._sha256(cutoff))
    finally:
        cutoff.write_bytes(original_cutoff);freeze.write_bytes(original_freeze)


def test_compact_five_panel_detection_layout(monkeypatch, tmp_path):
    power=pd.DataFrame([dict(family=family,change_day=92,risk_ratio=ratio,method=method,
        estimate=.5,ci_low=.4,ci_high=.6) for family in ('steady','sparse','wave')
        for ratio in (2/3,1.25,1.5,2.) for method in ('OCFR','NB-GLR','Page-CUSUM')])
    curves=pd.DataFrame([dict(stage='evaluation',misspecification='on_model',family=family,
        change_day=day,risk_ratio=1.5,method='OCFR',delay_day=delay,estimate=delay/50)
        for family in ('steady','wave') for day in (47,92,137) for delay in (0,15,30,45)])
    def inspect(fig,directory,name):
        assert len(fig.axes)==5
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for ax in fig.axes:
            lo,hi=ax.get_xlim()
            assert len([tick for tick in ax.get_xticks() if lo<=tick<=hi])>=3
            if ax.get_ylabel():
                box=ax.yaxis.label.get_window_extent(renderer)
                assert not any(box.overlaps(other.get_window_extent()) for other in fig.axes if other is not ax)
        for ax in fig.axes[-2:]:
            assert set(line.get_color() for line in ax.lines)=={a.COMPACT_COLORS['OCFR']}
            assert set(line.get_linestyle() for line in ax.lines)=={'-','--',':'}
        texts=[text for ax in fig.axes for text in [*ax.texts,*ax.get_xticklabels(),*ax.get_yticklabels(),
               ax.xaxis.label,ax.yaxis.label,ax.title] if text.get_text()]
        texts += [text for legend in fig.legends for text in [*legend.get_texts(),legend.get_title()] if text.get_text()]
        assert min(text.get_fontsize() for text in texts)*5.892/fig.get_figwidth()>=8
        a.plt.close(fig)
    monkeypatch.setattr(a,'_compact_save',inspect)
    a._compact_style()
    assert 'A-C' in a._compact_v2_detection_figure(power,curves,tmp_path)
