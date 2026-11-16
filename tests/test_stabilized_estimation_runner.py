"""Frozen-source replay, caching, and post-alarm reporting checks."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
import methods as m
import simulation as sim


def test_reporting_config_is_separate_from_frozen_scientific_payload():
    before = cfg.get_compact_experiment_config('full', protocol="compact-v1").scientific_payload()
    reporting = cfg.PostAlarmEstimationConfig()
    assert asdict(reporting)['precision'] == 1.
    assert 'post_alarm_estimation' not in json.dumps(before)
    assert before == cfg.get_compact_experiment_config('full', protocol="compact-v1").scientific_payload()
    with pytest.raises(ValueError):
        cfg.PostAlarmEstimationConfig(precision=2.)
    args = sim._parser().parse_args(['--stage', 'estimation-stabilized'])
    assert args.stage == 'estimation-stabilized'


@pytest.fixture
def episode(monkeypatch):
    config = cfg.get_compact_experiment_config('demo', backend='python', protocol="compact-v1")
    scenario = next(s for s in config.scenarios if s.split == 'alternative')
    hashes = dict(cases='a', deaths='b', true_delay='c', assumed_delay='d')
    cases, deaths, delay = np.full(35, 300.), np.r_[np.ones(12), np.full(23, 5.)], np.array([1.])
    source = dict(scenario=scenario.name, replicate=0, alarm_day=25, tau_hat_at_alarm=12,
                  source_payload_hash='p', case_hash='a', death_hash='b',
                  true_delay_hash='c', assumed_delay_hash='d')
    payload = dict(payload_hash='p', rows=[dict(method='OCFR Gaussian', trace=[
        dict(day=25, tau_hat=12, status='available', statistic=1.)])],
        stream_manifest=dict(hashes=hashes))
    total, post = m.exposure_components(cases[:26], delay)
    active = total > m.MU_FLOOR
    b = post[:, 11]
    ordinary = m.fit_nb_identity(deaths[:26][active], np.zeros(active.sum()),
                                 np.column_stack([total-b, b])[active])
    source.update(pi_left_estimate=ordinary.rates[0], pi_right_estimate=ordinary.rates[1],
                  phi_estimate=ordinary.phi, fit_loglik=ordinary.loglik, fit_success=ordinary.success,
                  fit_status=ordinary.status, fit_iterations=ordinary.nit)
    monkeypatch.setattr(sim, '_load_cache', lambda *a, **kw: payload)
    monkeypatch.setattr(sim, '_compact_generate_components', lambda *a: (cases, deaths, delay, dict(hashes=hashes)))
    def forbidden(*a, **kw):
        raise AssertionError('postprocessing reran the detector or ordinary fit')
    monkeypatch.setattr(sim, 'score_scan', forbidden)
    monkeypatch.setattr(sim, 'multiplier_statistics_from_draws', forbidden)
    monkeypatch.setattr(m, 'fit_nb_identity', forbidden)
    args = (scenario, 0, config, 'original', Path('cache'), source, cfg.PostAlarmEstimationConfig())
    return args, source, cases, deaths, payload


def test_replay_uses_stored_fit_and_only_prefix(episode):
    args, source, cases, deaths, _ = episode
    first = sim._stabilized_episode(*args)
    assert first['regularized_pi_left_point_valid']
    assert first['regularized_pi_left_interval_valid']
    assert first['regularized_penalized_objective_gain'] >= -1e-10
    cases[26:], deaths[26:] = 1e9, 1e9
    second = sim._stabilized_episode(*args)
    for key in first:
        if key != 'regularized_seconds':
            assert second[key] == first[key]


def test_replay_rejects_changed_observations_and_first_crossing(episode):
    args, source, _, _, payload = episode
    source['case_hash'] = 'changed'
    with pytest.raises(ValueError, match='hash mismatch'):
        sim._stabilized_episode(*args)
    source['case_hash'], source['tau_hat_at_alarm'] = 'a', 13
    with pytest.raises(ValueError, match='does not replay'):
        sim._stabilized_episode(*args)
    source['tau_hat_at_alarm'], payload['payload_hash'] = 12, 'changed'
    with pytest.raises(ValueError, match='cache does not match'):
        sim._stabilized_episode(*args)


def test_failed_regularized_fit_has_no_valid_point_or_interval(episode, monkeypatch):
    args, source, *_ = episode
    failed = m.NBFit(np.array([.01, .02]), 10., -1., False,
                     'nonconvergence:regularized_baseline_fallback', 1)
    monkeypatch.setattr(m, 'fit_nb_regularized_identity', lambda *a, **kw: failed)
    result = sim._stabilized_episode(*args)
    assert result['regularized_status'].startswith('nonconvergence')
    for target in ('pi_left', 'pi_right', 'risk_ratio'):
        assert not result['regularized_'+target+'_point_valid']
        assert not result['regularized_'+target+'_interval_valid']
        assert result['regularized_'+target+'_lower'] is None


@pytest.fixture
def frozen_sources(tmp_path, monkeypatch):
    output = tmp_path/'out'
    evaluation, reference = output/'evaluation', output/'estimation-reference'
    evaluation.mkdir(parents=True)
    reference.mkdir()
    config = cfg.get_compact_experiment_config('full', backend='python', protocol="compact-v1")
    scientific = config.scientific_payload()
    files = {'old.py': 'frozen'}
    code = sim._sha256_bytes(sim._canonical_json(files).encode('ascii'))
    digest = sim._sha256_bytes(sim._canonical_json(dict(schema_version=sim.SCHEMA_VERSION,
        code_hash=code, scientific=scientific)).encode('ascii'))[:20]
    (evaluation/'replicate_results.csv').write_text('frozen evaluation\n')
    original = dict(stage='evaluation', status='complete', mode='full', scientific_config=scientific,
        scientific_digest=digest, code_files=files, code_hash=code,
        artifacts={'replicate_results.csv': sim._file_hash(evaluation/'replicate_results.csv')})
    sim._atomic_json(original, evaluation/'manifest.json')
    selected = [s for s in config.scenarios if s.split == 'alternative' and s.misspecification == 'on_model']
    rows = []
    for scenario in selected:
        for rep in range(1000):
            row = dict(scenario=scenario.name, replicate=rep, alarm_day=None, tau_hat_at_alarm=None,
                source_payload_hash='p', case_hash='a', death_hash='b', true_delay_hash='c',
                assumed_delay_hash='d', fit_success=False, fit_status='no_alarm', fit_iterations=0,
                fit_loglik=None, phi_estimate=None, nishiura_status='no_alarm', old_custom_column='preserved')
            for target in ('pi_left', 'pi_right', 'risk_ratio'):
                for suffix in ('estimate', 'lower', 'upper', 'point_valid', 'interval_valid'):
                    row[target+'_'+suffix] = False if suffix.endswith('valid') else None
                for suffix in ('estimate', 'point_valid', 'admissible'):
                    row['nishiura_'+target+'_'+suffix] = None if suffix == 'estimate' else False
            rows.append(row)
    sim._atomic_csv(pd.DataFrame(rows), reference/'replicate_results.csv')
    manifest = dict(stage='estimation-reference', status='complete', mode='full',
        original_manifest_sha256=sim._file_hash(evaluation/'manifest.json'),
        original_csv_sha256=sim._file_hash(evaluation/'replicate_results.csv'),
        original_scientific_digest=digest,
        artifacts={'replicate_results.csv': sim._file_hash(reference/'replicate_results.csv')})
    sim._atomic_json(manifest, reference/'manifest.json')
    calls = []
    def observations(scenario, replicate, *args):
        calls.append((scenario.name, replicate))
        return np.ones(3), np.zeros(3), np.ones(1), None, None
    monkeypatch.setattr(sim, '_stabilized_observations', observations)
    return output, calls, original


def test_fresh_resume_and_derived_reset_preserve_sources(frozen_sources, monkeypatch):
    output, calls, original = frozen_sources
    reference = output/'estimation-reference'/'replicate_results.csv'
    source_hash = sim._file_hash(reference)
    first = sim.run_compact_estimation_stabilized(output=output, jobs=1)
    assert len(first) == 160
    assert first.old_custom_column.eq('preserved').all()
    stage = output/'estimation-stabilized'
    before = (stage/'replicate_results.csv').read_bytes()
    def forbidden(*args):
        raise AssertionError('resume recomputed a cached fit')
    monkeypatch.setattr(sim, '_stabilized_episode', forbidden)
    sim.run_compact_estimation_stabilized(output=output, reset_scope='derived')
    manifest = json.loads((stage/'manifest.json').read_text())
    assert manifest['tasks_computed'] == 0
    assert manifest['cached_tasks_loaded'] == 160
    assert manifest['scientific_digest'] == original['scientific_digest']
    assert manifest['post_alarm_estimation_config']['covariance'] == 'sandwich'
    assert len(calls) == 320
    assert (stage/'replicate_results.csv').read_bytes() == before
    assert sim._file_hash(reference) == source_hash


def test_changed_source_hash_and_incomplete_source_rejected(frozen_sources):
    output, _, _ = frozen_sources
    path = output/'estimation-reference'/'replicate_results.csv'
    source = pd.read_csv(path).iloc[:-1]
    sim._atomic_csv(source, path)
    with pytest.raises(ValueError, match='provenance mismatch'):
        sim.run_compact_estimation_stabilized(output=output)
    manifest_path = path.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['artifacts']['replicate_results.csv'] = sim._file_hash(path)
    sim._atomic_json(manifest, manifest_path)
    with pytest.raises(ValueError, match='grid is incomplete'):
        sim.run_compact_estimation_stabilized(output=output)


def test_real_data_reports_regularized_fit_and_sandwich(monkeypatch):
    import real_data as real
    seen = {}
    regularized = real.fit_nb_regularized_identity
    intervals = real.regularized_parameter_intervals
    def record_fit(*args, **kwargs):
        seen['initial'] = kwargs['initial_fit']
        return regularized(*args, **kwargs)
    def record_interval(*args, **kwargs):
        seen['covariance'] = kwargs['covariance']
        return intervals(*args, **kwargs)
    monkeypatch.setattr(real, 'fit_nb_regularized_identity', record_fit)
    monkeypatch.setattr(real, 'regularized_parameter_intervals', record_interval)
    result = real._two_rate_intervals(np.full(40, 300.), np.r_[np.full(20, 2.), np.full(20, 6.)],
                                      np.array([1.]), 20, 1)
    assert isinstance(seen['initial'], m.NBFit)
    assert seen['covariance'] == 'sandwich'
    assert result['estimator'] == 'regularized_log_rr'
    assert result['estimator_precision'] == 1.
    assert result['interval_status'] == 'success:regularized_sandwich_approximate'


def test_real_data_nonconvergence_is_explicit(monkeypatch):
    import real_data as real
    monkeypatch.setattr(real, 'fit_nb_regularized_identity', lambda *a, **kw:
        m.NBFit(np.array([.01, .02]), 10., -1., False, 'nonconvergence', 1))
    result = real._two_rate_intervals(np.full(40, 300.), np.ones(40), np.array([1.]), 20, 1)
    assert not result['point_valid']
    assert result['interval_status'] == 'failed:regularized_fit_nonconvergence'


def test_validation_policy_uses_full_bank_all_looks_and_strict_first_crossing(monkeypatch):
    from types import SimpleNamespace
    config = cfg.get_compact_experiment_config('demo', backend='python', protocol="compact-v1")
    scenario = next(s for s in config.scenarios if s.split == 'alternative')
    bank = np.arange(4*scenario.horizon).reshape(4, scenario.horizon)
    config = __import__('dataclasses').replace(config,
        methods=__import__('dataclasses').replace(config.methods, n_multiplier=4))
    monkeypatch.setattr(sim, '_compact_generate_components', lambda *a:
        (np.ones(scenario.horizon), np.zeros(scenario.horizon), np.ones(1), {'seeds': {'multiplier': 13}}))
    monkeypatch.setattr(sim, 'make_sampler', lambda *a, **kw:
        SimpleNamespace(standard_normal=lambda shape: bank if shape == bank.shape else None))
    monkeypatch.setattr(sim, '_planned_looks', lambda *a: [8, 15, 22, 29])
    statistics, prefixes, slices, allocations = iter([0., 2., 3., 100.]), [], [], []
    def scan(cases, deaths, delay, **kwargs):
        prefixes.append(len(cases))
        return SimpleNamespace(candidates=np.array([3]), statistic=next(statistics),
                               tau_hat=3, influence=np.ones((len(cases), 1)))
    def maxima(draws, influence, **kwargs):
        slices.append(draws.copy())
        return np.ones(4)
    def critical(values, alpha):
        allocations.append(alpha)
        return 2.
    monkeypatch.setattr(sim, 'score_scan', scan)
    monkeypatch.setattr(sim, 'multiplier_statistics_from_draws', maxima)
    monkeypatch.setattr(sim, 'monte_carlo_critical_value', critical)
    *_, alarm, tau, trace = sim._validation_first_alarm(scenario, 0, config)
    assert (alarm, tau) == (21, 3)
    assert prefixes == [8, 15, 22]
    assert allocations == [config.methods.alpha/4]*3
    for draw, prefix in zip(slices, prefixes):
        np.testing.assert_array_equal(draw, bank[:, :prefix])
    assert len(trace) == 3


def test_validation_demo_resume_and_seed_contract(frozen_sources, monkeypatch):
    output, _, _ = frozen_sources
    calls = []
    def episode(scenario, rep, config, reporting):
        calls.append((scenario.name, rep, config.master_seed, config.replications))
        return dict(scenario=scenario.name, replicate=rep, alarm=False,
                    regularized_status='no_alarm', case_hash=str(config.master_seed))
    monkeypatch.setattr(sim, '_validation_episode', episode)
    first = sim.run_compact_estimation_validation(output=output)
    assert len(first) == 160
    assert all(seed == 2026110702 and count == 10 for _, _, seed, count in calls)
    def forbidden(*args):
        raise AssertionError('validation resume repeated Gaussian detection')
    monkeypatch.setattr(sim, '_validation_episode', forbidden)
    second = sim.run_compact_estimation_validation(output=output, reset_scope='derived')
    pd.testing.assert_frame_equal(first, second)
    manifest = json.loads((output/'estimation-validation'/'manifest.json').read_text())
    assert manifest['tasks_computed'] == 0
    assert manifest['scientific_config']['master_seed'] == 2026110702
    assert manifest['validation_config']['validation_master_seed'] == 2026110702
