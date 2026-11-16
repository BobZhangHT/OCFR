from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import config as c
import methods as m
import simulation as s


def test_modes_release_and_stage_contract():
    demo, full = [c.get_compact_experiment_config(mode) for mode in ('demo', 'full')]
    assert demo.scientific_payload() == full.scientific_payload()
    for split in ('calibration', 'size_audit', 'holdout', 'alternative'):
        assert (demo.replications_for(split), full.replications_for(split)) == (10, 1000)
    release = c.get_compact_release_study_config()
    assert demo.methods.n_multiplier == full.methods.n_multiplier == release.methods.n_multiplier == 1999
    assert c.get_compact_experiment_config('demo', protocol='compact-v1').methods.n_multiplier == 10000
    assert c.MethodConfig().n_multiplier == c.REAL_DATA.n_multiplier == 10000
    import inspect
    assert inspect.signature(m.OnlineGaussianProcessScore).parameters['n_multiplier'].default == 10000
    assert [release.replications_for(x) for x in ('calibration', 'size_audit', 'holdout', 'alternative')] == [10000, 5000, 1000, 1000]
    assert release.scientific_payload() != full.scientific_payload()
    assert release.master_seed == 2026111404
    assert c.get_post_alarm_validation_config('compact-v2').validation_master_seed == 2026111405
    assert demo.stage_methods('calibration') == ('NB-GLR', 'Page-CUSUM')
    assert demo.stage_methods('audit') == demo.stage_methods('evaluation') == ('OCFR', 'NB-GLR', 'Page-CUSUM')


def test_legacy_payload_and_exact_source_reconstruction():
    # Frozen canonical scientific JSON hash of the archived v1 public payload.
    # This fixture has no dependency on another checkout or private files.
    import hashlib
    legacy = c.get_compact_experiment_config('full', protocol='compact-v1')
    frozen_sha256 = '1d620dfc4e09f5213e03e2444957b1a32dd038e641e58456ca02cf4c63df7a6d'
    assert hashlib.sha256(s._canonical_json(legacy.scientific_payload()).encode('ascii')).hexdigest() == frozen_sha256
    for cfg in (legacy, c.get_compact_release_study_config()):
        manifest = dict(status='complete', stage='evaluation', scientific_config=cfg.scientific_payload())
        assert s._estimation_source_config(manifest, 'full').scientific_payload() == cfg.scientific_payload()


def _stub_scan(cases, deaths, delay, **kwargs):
    return SimpleNamespace(candidates=np.array([1]), statistic=100., influence=np.ones((len(cases), 1))/len(cases),
        tau_hat=1, direction=1, null_fit=SimpleNamespace(success=True, status='success', pi=.02, phi=10.))


def test_compact_and_validation_agree_and_ignore_future(monkeypatch):
    cfg = c.get_compact_experiment_config('demo', backend='python')
    cfg = replace(cfg, methods=replace(cfg.methods, n_multiplier=399))
    scene = next(x for x in cfg.scenarios if x.name == 'steady_rr150')
    monkeypatch.setattr(s, 'score_scan', _stub_scan)
    def benchmarks(cases, deaths, delay, looks, **kwargs):
        return {name: [SimpleNamespace(statistic=1., estimated_cohort_change_day=1, direction=1, fit_status='success') for _ in looks] for name in ('NB-GLR', 'Page-CUSUM')}
    monkeypatch.setattr(m, 'surveillance_benchmark_traces', benchmarks)
    row = next(x for x in s._compact_evaluate_one(scene, 0, cfg, 'test')['rows'] if x['method'] == 'OCFR')
    validation = s._validation_first_alarm(scene, 0, cfg)
    first = next(x for x in row['trace'] if x['statistic'] > 0)
    assert validation[4] == first['day']
    assert validation[6][-1]['boundary'] == first['boundary']
    original = s._compact_generate_components
    def perturbed(*args):
        cases, deaths, delay, streams = original(*args)
        cases[42:] += 100000
        deaths[42:] += 100000
        return cases, deaths, delay, streams
    monkeypatch.setattr(s, '_compact_generate_components', perturbed)
    changed = next(x for x in s._compact_evaluate_one(scene, 0, cfg, 'test')['rows'] if x['method'] == 'OCFR')
    assert changed['trace'][0] == row['trace'][0]


def test_calibration_does_not_fit_scores(monkeypatch):
    cfg = c.get_compact_experiment_config('demo', backend='python')
    scene = next(x for x in cfg.scenarios if x.split == 'calibration')
    monkeypatch.setattr(s, 'score_scan', lambda *a, **k: pytest.fail('calibration fitted score'))
    monkeypatch.setattr(m, 'surveillance_benchmark_traces', lambda cases, deaths, delay, looks, **kw: {
        name: [SimpleNamespace(statistic=1., estimated_cohort_change_day=1, direction=1, fit_status='success') for _ in looks] for name in ('NB-GLR', 'Page-CUSUM')})
    assert [x['method'] for x in s._compact_evaluate_one(scene, 0, cfg, 'test')['rows']] == ['NB-GLR', 'Page-CUSUM']


def test_python_v2_parallel_resume_and_complete_cache(tmp_path):
    # Two tiny actual episodes check spawned imports, source-bound config,
    # deterministic banks and cache replay without a costly experiment.
    cfg = c.get_compact_experiment_config('demo', backend='python')
    scene = replace(next(x for x in cfg.scenarios if x.name == 'steady_null'), horizon=42, change_fraction=20/42)
    cfg = replace(cfg, scenarios=(scene,), replications=2,
                  methods=replace(cfg.methods, n_multiplier=399))
    digest = s._scientific_digest(cfg)[0]
    for directory in (tmp_path/'sequential', tmp_path/'parallel'):
        stage = directory/'calibration'
        stage.mkdir(parents=True)
        frame = pd.DataFrame([dict(null_family='steady', method=name,
            n_calibration=2, critical_value=10., threshold_version='fixture')
            for name in cfg.stage_methods('calibration')])
        s._atomic_csv(frame, stage/'thresholds.csv')
        s._atomic_json(dict(scientific_digest=digest, stage='calibration', status='complete',
            artifacts={'thresholds.csv': s._file_hash(stage/'thresholds.csv')}), stage/'manifest.json')
    first, _ = s.run_compact_simulations(cfg, output=tmp_path/'sequential', jobs=1)
    parallel, _ = s.run_compact_simulations(cfg, output=tmp_path/'parallel', jobs=2)
    resumed, _ = s.run_compact_simulations(cfg, output=tmp_path/'sequential', jobs=1)
    columns = [name for name in first if name != 'runtime_seconds']
    pd.testing.assert_frame_equal(first[columns], parallel[columns])
    pd.testing.assert_frame_equal(first, resumed)
    cache = s._cache_path(tmp_path/'sequential'/'cache', digest, scene.name, 0)
    payload = s._load_cache(cache, digest=digest, scenario=scene, replicate=0, config=cfg)
    assert payload is not None and set(payload['expected_methods']) == set(cfg.stage_methods('evaluation'))
    payload['rows'] = payload['rows'][:-1]
    payload['payload_hash'] = s._payload_hash(payload)
    s._atomic_json(payload, cache)
    assert s._load_cache(cache, digest=digest, scenario=scene, replicate=0, config=cfg) is None


@pytest.mark.parametrize('stage', ['stabilized', 'validation'])
def test_transformed_se_exports_existing_interval_result(monkeypatch, stage):
    cfg = c.get_compact_experiment_config('demo', backend='python')
    scene = next(x for x in cfg.scenarios if x.split == 'alternative')
    cases, deaths, delay = np.full(35, 300.), np.r_[np.ones(12), np.full(23, 5.)], np.array([1.])
    total, post = m.exposure_components(cases[:26], delay)
    b, active = post[:, 11], total > m.MU_FLOOR
    ordinary = m.fit_nb_identity(deaths[:26][active], np.zeros(active.sum()), np.column_stack([total-b, b])[active])
    source = dict(source_payload_hash='fixture', pi_left_estimate=ordinary.rates[0], pi_right_estimate=ordinary.rates[1],
        phi_estimate=ordinary.phi, fit_loglik=ordinary.loglik, fit_success=ordinary.success,
        fit_status=ordinary.status, fit_iterations=ordinary.nit)
    alarm = [25]
    monkeypatch.setattr(s, '_stabilized_observations', lambda *a: (cases, deaths, delay, alarm[0], 12))
    monkeypatch.setattr(s, '_validation_first_alarm', lambda *a: (cases, deaths, delay,
        dict(hashes=dict(cases='a', deaths='b', true_delay='c', assumed_delay='d')), alarm[0], 12, []))
    actual_intervals, captured = m.regularized_parameter_intervals, []
    def intervals(*args, **kwargs):
        ci = actual_intervals(*args, **kwargs)
        captured.append(ci)
        return ci
    monkeypatch.setattr(m, 'regularized_parameter_intervals', intervals)
    def run():
        if stage == 'stabilized':
            return s._stabilized_episode(scene, 0, cfg, 'digest', None, source, c.PostAlarmEstimationConfig())
        return s._validation_episode(scene, 0, cfg, c.PostAlarmEstimationConfig())
    row = run()
    names = ('regularized_pi_left_logit_se', 'regularized_pi_right_logit_se', 'regularized_log_rr_se')
    np.testing.assert_array_equal([row[name] for name in names],
        [*captured[0].cfr_logit_standard_errors, captured[0].log_risk_ratio_standard_error])
    alarm[0] = None
    assert all(run()[name] is None for name in names)
    alarm[0] = 25
    def unavailable(*args, **kwargs):
        raise np.linalg.LinAlgError('fixture unavailable')
    monkeypatch.setattr(m, 'regularized_parameter_intervals', unavailable)
    assert all(run()[name] is None for name in names)


@pytest.mark.parametrize('available', [False, True])
def test_validation_serializes_unavailable_and_unresolved_boundaries(monkeypatch, tmp_path, available):
    import json
    cfg = c.get_compact_experiment_config('demo', backend='python')
    # A one-path bank cannot resolve the .05 ranks. This deliberately exercises
    # an available but unresolved look as well as a returned unavailable fit.
    cfg = replace(cfg, methods=replace(cfg.methods, n_multiplier=1))
    scene = next(x for x in cfg.scenarios if x.split == 'alternative')
    hashes = dict(cases='a', deaths='b', true_delay='c', assumed_delay='d')
    monkeypatch.setattr(s, '_compact_generate_components', lambda *a: (
        np.ones(scene.horizon), np.zeros(scene.horizon), np.array([1.]),
        dict(seeds={'multiplier': 13}, hashes=hashes)))
    monkeypatch.setattr(s, 'make_sampler', lambda *a, **kw: SimpleNamespace(standard_normal=lambda shape: np.zeros(shape)))
    monkeypatch.setattr(s, 'score_scan', lambda cases, *a, **kw: SimpleNamespace(
        candidates=np.array([1]) if available else np.array([], dtype=int), statistic=2.,
        null_fit=SimpleNamespace(success=available), influence=np.zeros((len(cases), 1)), tau_hat=1))
    monkeypatch.setattr(s, 'multiplier_statistics_from_draws', lambda *a, **kw: np.array([1.]))
    row = s._validation_episode(scene, 0, cfg, c.PostAlarmEstimationConfig())
    trace = json.loads(row['trace_json'])
    assert row['no_alarm'] and not row['alarm']
    assert len(trace) == len(s._planned_looks(scene, cfg))
    assert all(x['available'] == available and not x['resolved'] and np.isposinf(x['boundary']) for x in trace)
    with pytest.raises(ValueError):
        s._canonical_json(trace)
    payload = dict(trace=trace, statistic=-np.inf)
    path = tmp_path/'diagnostic.json'
    s._atomic_json(payload, path)
    assert json.loads(path.read_text()) == payload
