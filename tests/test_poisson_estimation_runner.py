"""Frozen original/validation reanalysis: prefixes, provenance and cache replay."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
import methods as m
import simulation as sim


@pytest.fixture
def episode(monkeypatch):
    config = cfg.get_compact_experiment_config('demo', backend='python', protocol="compact-v1")
    scenario = next(s for s in config.scenarios if s.split == 'alternative')
    cases, deaths, delay = np.full(40, 300.), np.r_[np.full(12, 2.), np.full(28, 6.)], np.array([1.])
    hashes = dict(cases='a', deaths='b', true_delay='c', assumed_delay='d')
    source = dict(scenario=scenario.name, replicate=0, alarm_day=25, tau_hat_at_alarm=12,
        source_payload_hash='p', case_hash='a', death_hash='b', true_delay_hash='c', assumed_delay_hash='d',
        trace_json=json.dumps([dict(day=18, available=True, statistic=0., boundary=1.),
                               dict(day=25, available=True, statistic=2., boundary=1.)]),
        stream_manifest_json=json.dumps(dict(hashes=hashes, seeds=dict(case=3, death=4, multiplier=5))))
    streams = dict(hashes=hashes, seeds=dict(case=3, death=4, multiplier=5))
    payload = dict(payload_hash='p', rows=[dict(method='OCFR Gaussian', trace=[
        dict(day=25, tau_hat=12, status='available', statistic=1.)])], stream_manifest=dict(hashes=hashes))
    monkeypatch.setattr(sim, '_load_cache', lambda *a, **kw: payload)
    monkeypatch.setattr(sim, '_compact_generate_components', lambda *a: (cases, deaths, delay, streams))
    def forbidden(*a, **kw):
        pytest.fail('Poisson postprocessing replayed detection or NB estimation')
    for name in ('score_scan', 'multiplier_statistics_from_draws', '_validation_first_alarm'):
        monkeypatch.setattr(sim, name, forbidden)
    monkeypatch.setattr(m, 'fit_nb_identity', forbidden)
    return scenario, config, source, cases, deaths, payload


@pytest.mark.parametrize('bank', ['original', 'validation'])
def test_same_saved_prefix_and_no_future_inputs(episode, bank):
    scenario, config, source, cases, deaths, _ = episode
    args = (scenario, 0, config, bank, 'digest', Path('cache'), source, cfg.PostAlarmPoissonConfig())
    first = sim._poisson_episode(*args)
    assert first['poisson_regularized_risk_ratio_point_valid']
    assert first['poisson_unpenalized_risk_ratio_point_valid']
    assert not any('lower' in c or 'upper' in c or 'interval' in c for c in first)
    cases[26:], deaths[26:] = 1e9, 1e9
    second = sim._poisson_episode(*args)
    for key in first:
        if not key.endswith('_seconds'):
            assert second[key] == first[key]


@pytest.mark.parametrize('bank', ['original', 'validation'])
def test_input_hash_and_saved_alarm_mismatch_rejected(episode, bank):
    scenario, config, source, *_ = episode
    args = (scenario, 0, config, bank, 'digest', Path('cache'), source)
    source['case_hash'] = 'changed'
    with pytest.raises(ValueError, match='hash mismatch'):
        sim._poisson_observations(*args)
    source['case_hash'], source['alarm_day'] = 'a', 24
    with pytest.raises(ValueError, match='does not replay'):
        sim._poisson_observations(*args)
    source['alarm_day'], source['tau_hat_at_alarm'] = 25, 26
    with pytest.raises(ValueError, match='index'):
        sim._poisson_observations(*args)


def test_failure_is_retained_without_valid_or_available_points(episode, monkeypatch):
    scenario, config, source, *_ = episode
    monkeypatch.setattr(m, 'fit_poisson_identity', lambda *a, **kw:
        m.PoissonFit(np.array([.01, .03]), -1., False, 'nonconvergence:forced', 3))
    row = sim._poisson_episode(scenario, 0, config, 'validation', 'digest', Path('cache'), source,
                               cfg.PostAlarmPoissonConfig())
    for prefix in ('poisson_regularized_', 'poisson_unpenalized_'):
        assert row[prefix+'status'] == 'nonconvergence:forced'
        for target in ('pi_left', 'pi_right', 'risk_ratio'):
            assert not row[prefix+target+'_point_valid']
            assert row[prefix+target+'_estimate'] is None


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    output = tmp_path/'out'
    config = cfg.get_compact_experiment_config('full', backend='python', protocol="compact-v1")
    selected = [s for s in config.scenarios if s.split == 'alternative' and s.misspecification == 'on_model']
    for bank, stage in (('original', 'estimation-stabilized'), ('validation', 'estimation-validation')):
        folder = output/stage
        folder.mkdir(parents=True)
        bank_config = config if bank == 'original' else replace(config, master_seed=2026110702)
        records = [dict(scenario=s.name, replicate=r, alarm_day=None, tau_hat_at_alarm=None,
            source_payload_hash='p', case_hash='a', death_hash='b', true_delay_hash='c', assumed_delay_hash='d',
            trace_json='[]', stream_manifest_json='{}', regularized_status='no_alarm',
            old_custom_column='preserved') for s in selected for r in range(1000)]
        sim._atomic_csv(pd.DataFrame(records), folder/'replicate_results.csv')
        files = {'frozen.py': 'old'}
        code = sim._sha256_bytes(sim._canonical_json(files).encode('ascii'))
        digest = sim._sha256_bytes(sim._canonical_json(dict(schema_version=sim.SCHEMA_VERSION,
            code_hash=code, scientific=bank_config.scientific_payload())).encode('ascii'))[:20]
        if bank == 'validation':
            digest = sim._sha256_bytes(sim._canonical_json(bank_config.scientific_payload()).encode('ascii'))[:20]
        manifest = dict(stage=stage, status='complete', mode='full', scientific_config=bank_config.scientific_payload(),
            scientific_digest=digest, original_code_files=files, original_code_hash=code,
            post_alarm_estimation_config=asdict(cfg.PostAlarmEstimationConfig()),
            validation_config=asdict(cfg.PostAlarmValidationConfig()),
            replications_by_scenario={s.name: 1000 for s in selected},
            artifacts={'replicate_results.csv': sim._file_hash(folder/'replicate_results.csv')})
        sim._atomic_json(manifest, folder/'manifest.json')
    calls = []
    def observations(scenario, replicate, *args):
        calls.append((scenario.name, replicate))
        return np.ones(3), np.zeros(3), np.ones(1), None, None
    monkeypatch.setattr(sim, '_poisson_observations', observations)
    return output, calls


def test_both_banks_resume_reset_source_preservation_and_parallel(frozen, monkeypatch):
    output, calls = frozen
    hashes = {p: sim._file_hash(p) for p in output.glob('*/replicate_results.csv')}
    first = sim.run_compact_estimation_poisson(output=output)
    assert len(first) == 320 and first.old_custom_column.eq('preserved').all()
    assert set(first.estimation_bank) == {'original', 'validation'}
    assert first.poisson_regularized_status.eq('no_alarm').all()
    before = {b: (output/('estimation-poisson-'+b)/'replicate_results.csv').read_bytes()
              for b in ('original', 'validation')}
    original_episode = sim._poisson_episode
    monkeypatch.setattr(sim, '_poisson_episode', lambda *a: pytest.fail('resume refit an episode'))
    sim.run_compact_estimation_poisson(output=output, reset_scope='derived')
    assert len(calls) == 640
    for bank in ('original', 'validation'):
        stage = output/('estimation-poisson-'+bank)
        manifest = json.loads((stage/'manifest.json').read_text())
        assert manifest['cached_tasks_loaded'] == 160 and manifest['tasks_computed'] == 0
        assert (stage/'replicate_results.csv').read_bytes() == before[bank]
        assert not any('analysis' in c for c in manifest['postprocessor_code_files'])
    monkeypatch.setattr(sim, '_poisson_episode', original_episode)
    monkeypatch.setattr(sim, 'ProcessPoolExecutor', ThreadPoolExecutor)
    sim.run_compact_estimation_poisson(output=output, bank='original', reset_scope='selected', jobs=2)
    assert (output/'estimation-poisson-original'/'replicate_results.csv').read_bytes() == before['original']
    assert all(sim._file_hash(p) == h for p, h in hashes.items())


def test_source_tampering_incomplete_grid_and_override_ambiguity(frozen):
    output, _ = frozen
    path = output/'estimation-validation'/'replicate_results.csv'
    manifest_path = path.parent/'manifest.json'
    with pytest.raises(ValueError, match='ambiguous'):
        sim.run_compact_estimation_poisson(output=output, source_manifest=manifest_path)
    frame = pd.read_csv(path).iloc[:-1]
    sim._atomic_csv(frame, path)
    with pytest.raises(ValueError, match='hash mismatch'):
        sim.run_compact_estimation_poisson(output=output, bank='validation')
    manifest = json.loads(manifest_path.read_text())
    manifest['artifacts']['replicate_results.csv'] = sim._file_hash(path)
    sim._atomic_json(manifest, manifest_path)
    with pytest.raises(ValueError, match='grid is incomplete'):
        sim.run_compact_estimation_poisson(output=output, bank='validation')


def test_demo_full_prefix_parity_with_bounded_noalarm_run(frozen, monkeypatch):
    output, _ = frozen
    source = sim._poisson_source
    def bounded(*args):
        manifest, config, selected, base = source(*args)
        config = replace(config, replications=10 if config.mode == 'demo' else 12)
        return manifest, config, selected, base.loc[base.replicate < config.replications]
    monkeypatch.setattr(sim, '_poisson_source', bounded)
    demo = sim.run_compact_estimation_poisson(output=output, bank='validation')
    full = sim.run_compact_estimation_poisson(output=output, bank='validation', mode='full')
    pd.testing.assert_frame_equal(demo, full.loc[full.replicate < 10].reset_index(drop=True))
    manifest = json.loads((output/'estimation-poisson-validation'/'manifest.json').read_text())
    assert manifest['cached_tasks_loaded'] == 160 and manifest['tasks_computed'] == 32


def test_cli_and_config_contract():
    args = sim._parser().parse_args(['--stage', 'estimation-poisson'])
    assert args.estimation_bank == 'both'
    assert asdict(cfg.PostAlarmPoissonConfig())['primary_precision'] == 1.
    assert 'poisson' not in json.dumps(cfg.get_compact_experiment_config('full', protocol="compact-v1").scientific_payload())
    with pytest.raises(ValueError):
        cfg.PostAlarmPoissonConfig(primary_precision=2.)
