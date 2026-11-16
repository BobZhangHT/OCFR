"""First-alarm estimation recipe and frozen-source integrity checks."""
import sys
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))
import config
import methods
import simulation as sim


def episode(monkeypatch, *, alarm=8, tau=3, corrupt=False):
    cfg = config.get_compact_experiment_config('demo', backend='python', protocol="compact-v1")
    scenario = next(s for s in cfg.scenarios if s.split == 'alternative')
    scenario = replace(scenario, change_fraction=.5)
    trace = [] if alarm is None else [dict(day=alarm, tau_hat=tau, status='available', statistic=1.)]
    hashes = dict(cases='a', deaths='b', true_delay='c', assumed_delay='d')
    payload = dict(rows=[dict(method='OCFR Gaussian', trace=trace)],
                   stream_manifest=dict(hashes={**hashes, 'cases': 'wrong'} if corrupt else hashes), payload_hash='original')
    monkeypatch.setattr(sim, '_load_cache', lambda *a, **kw: payload)
    monkeypatch.setattr(sim, '_compact_generate_components', lambda *a: (
        np.arange(1., 183.), np.arange(182), np.array([1.]), dict(hashes=hashes)))
    args = (scenario, 0, cfg, 'old-digest', Path('cache'),
            dict(alarm_day=alarm, tau_hat_at_alarm=tau if alarm is not None else None), config.EstimationConfig())
    return args


def test_zero_based_prefix_and_tau_recipe_premature(monkeypatch):
    args = episode(monkeypatch)
    calls = {}
    original = methods.exposure_components
    def exposure(cases, delay):
        calls['prefix'] = len(cases)
        total, post = original(cases, delay)
        calls['design'] = np.column_stack((total-post[:, 2], post[:, 2]))[total > methods.MU_FLOOR]
        calls['active'] = total > methods.MU_FLOOR
        return total, post
    def fit(y, offset, design):
        np.testing.assert_array_equal(design, calls['design'])
        np.testing.assert_array_equal(y, np.arange(9)[calls['active']])
        np.testing.assert_array_equal(offset, np.zeros(len(y)))
        return methods.NBFit(np.array([.02, .03]), 10., -10., True, 'ok', 2)
    monkeypatch.setattr(methods, 'exposure_components', exposure)
    monkeypatch.setattr(methods, 'fit_nb_identity', fit)
    row = sim._estimation_episode(*args)
    assert calls['prefix'] == 9
    assert row['premature_alarm'] and row['fit_success'] and row['interval_valid']
    assert row['risk_ratio_estimate'] == 1.5


def test_no_alarm_never_refits(monkeypatch):
    args = episode(monkeypatch, alarm=None)
    def fail(*a, **kw):
        raise AssertionError('no-alarm episode must not refit')
    monkeypatch.setattr(methods, 'fit_nb_identity', fail)
    row = sim._estimation_episode(*args)
    assert row['no_alarm'] and row['pi_left_estimate'] is None


def test_ci_failure_preserves_points(monkeypatch):
    args = episode(monkeypatch)
    monkeypatch.setattr(methods, 'fit_nb_identity', lambda *a: methods.NBFit(np.array([.02, .03]), 7., -1., True, 'ok', 1))
    def singular(*a, **kw):
        raise np.linalg.LinAlgError('singular')
    monkeypatch.setattr(methods, 'fixed_tau_parameter_intervals', singular)
    row = sim._estimation_episode(*args)
    assert row['fit_success'] and not row['interval_valid']
    assert row['pi_right_estimate'] == .03 and row['phi_estimate'] == 7.


def test_stream_corruption_is_infrastructure_failure(monkeypatch):
    with pytest.raises(ValueError, match='hash mismatch'):
        sim._estimation_episode(*episode(monkeypatch, corrupt=True))


def test_first_alarm_not_maximum_or_later_alarm():
    trace = [dict(day=34, tau_hat=10, status='available', statistic=1.),
             dict(day=99, tau_hat=70, status='available', statistic=20.)]
    assert sim._estimation_first_alarm(dict(alarm_day='34.0', tau_hat_at_alarm='10'), trace) == (34, 10)
    with pytest.raises(ValueError, match='does not replay'):
        sim._estimation_first_alarm(dict(alarm_day=99, tau_hat_at_alarm=70), trace)


def test_source_deserialization_and_demo_full_seed_reuse():
    cfg = config.get_compact_experiment_config('full', backend='python', protocol="compact-v1")
    manifest = dict(status='complete', stage='evaluation', scientific_config=cfg.scientific_payload())
    demo = sim._estimation_source_config(manifest, 'demo')
    full = sim._estimation_source_config(manifest, 'full')
    assert (demo.replications, full.replications) == (10, 1000)
    assert demo.scientific_payload() == full.scientific_payload()
    assert sim._component_seeds(demo.scenarios[0], demo, 9) == sim._component_seeds(full.scenarios[0], full, 9)
    manifest['scientific_config']['methods']['look_every'] = 11
    assert sim._estimation_source_config(manifest, 'demo').methods.look_every == 11


def test_unexpected_interval_programming_error_propagates(monkeypatch):
    args = episode(monkeypatch)
    monkeypatch.setattr(methods, 'fit_nb_identity', lambda *a: methods.NBFit(np.array([.02, .03]), 7., -1., True, 'ok', 1))
    def bug(*a, **kw):
        raise ValueError('malformed design')
    monkeypatch.setattr(methods, 'fixed_tau_parameter_intervals', bug)
    with pytest.raises(ValueError, match='malformed design'):
        sim._estimation_episode(*args)


@pytest.mark.parametrize('success,rates', [(False, [.02, .03]), (True, [.02, 1.01])])
def test_point_unavailability_is_retained_without_intervals(monkeypatch, success, rates):
    args = episode(monkeypatch)
    monkeypatch.setattr(methods, 'fit_nb_identity', lambda *a: methods.NBFit(np.array(rates), 7., -1., success, 'fit_status', 1))
    def forbidden(*a, **kw):
        raise AssertionError('invalid point fits must not get intervals')
    monkeypatch.setattr(methods, 'fixed_tau_parameter_intervals', forbidden)
    row = sim._estimation_episode(*args)
    assert row['alarm'] and not row['point_available'] and not row['interval_valid']
    assert not any(row[f'{target}_point_valid'] for target in ('pi_left', 'pi_right', 'risk_ratio'))
    assert row['pi_left_estimate'] == .02 and row['fit_status'] == 'fit_status'


def test_rr_overflow_retains_finite_cfr_intervals_and_log_endpoints(monkeypatch):
    args = episode(monkeypatch)
    monkeypatch.setattr(methods, 'fit_nb_identity', lambda *a: methods.NBFit(np.array([.02, .03]), 7., -1., True, 'ok', 1))
    monkeypatch.setattr(methods, 'fixed_tau_parameter_intervals', lambda *a, **kw: SimpleNamespace(
        cfr_lower=np.array([0., 0.]), cfr_upper=np.array([1., 1.]),
        risk_ratio_lower=0., risk_ratio_upper=float('inf'),
        log_risk_ratio_lower=-800., log_risk_ratio_upper=800.))
    row = sim._estimation_episode(*args)
    assert row['point_available'] and not row['interval_valid']
    assert row['pi_left_interval_valid'] and row['pi_right_interval_valid']
    assert not row['risk_ratio_interval_valid']
    assert row['risk_ratio_upper'] == float('inf') and row['log_risk_ratio_upper'] == 800.
    assert row['pi_left_interval_status'] == 'success'


def test_infrastructure_fit_error_propagates(monkeypatch):
    args = episode(monkeypatch)
    def infrastructure(*a):
        raise OSError('native runtime missing')
    monkeypatch.setattr(methods, 'fit_nb_identity', infrastructure)
    with pytest.raises(OSError, match='native runtime missing'):
        sim._estimation_episode(*args)

def test_all_episodes_resume_and_derived_reset_without_detection(tmp_path, monkeypatch):
    import csv
    cfg = config.get_compact_experiment_config('full', backend='python', protocol="compact-v1")
    evaluation = [s for s in cfg.scenarios if s.split in {'holdout', 'alternative'}]
    selected = [s for s in evaluation if s.split == 'alternative' and s.misspecification == 'on_model']
    stage = tmp_path / 'evaluation'
    stage.mkdir()
    source = stage / 'replicate_results.csv'
    with source.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['scenario', 'replicate', 'method', 'status', 'alarm_day', 'tau_hat_at_alarm', 'trace_json'])
        writer.writeheader()
        for s in evaluation:
            for rep in range(1000):
                for method in config.COMPACT_METHODS:
                    writer.writerow(dict(scenario=s.name, replicate=rep, method=method, status='success', alarm_day='', tau_hat_at_alarm='', trace_json='[]'))
    manifest = dict(status='complete', stage='evaluation', mode='full', scientific_digest='frozen',
        code_hash='original', code_files={'simulation.py': 'old'}, scientific_config=cfg.scientific_payload(),
        scenarios=[s.name for s in evaluation], methods=list(config.COMPACT_METHODS),
        replications_by_scenario={s.name: 1000 for s in evaluation}, artifacts={'replicate_results.csv': sim._file_hash(source)})
    manifest['code_hash'] = sim._sha256_bytes(sim._canonical_json(manifest['code_files']).encode('ascii'))
    manifest['scientific_digest'] = sim._sha256_bytes(sim._canonical_json(dict(schema_version=sim.SCHEMA_VERSION,
        code_hash=manifest['code_hash'], scientific=cfg.scientific_payload())).encode('ascii'))[:20]
    (stage / 'manifest.json').write_text(json.dumps(manifest))
    calls = []
    def calculate(s, rep, *args):
        calls.append((s.name, rep))
        return dict(scenario=s.name, replicate=rep, no_alarm=True, source_payload_hash='source')
    monkeypatch.setattr(sim, '_estimation_episode', calculate)
    monkeypatch.setattr(sim, '_load_cache', lambda *a, **kw: dict(payload_hash='source', rows=[dict(method='OCFR Gaussian', trace=[])]))
    def forbidden(*a, **kw):
        raise AssertionError('postprocessing must not scan or draw Gaussian banks')
    monkeypatch.setattr(sim, 'score_scan', forbidden)
    monkeypatch.setattr(sim, '_compact_evaluate_one', forbidden)
    monkeypatch.setattr(sim, 'multiplier_statistics_from_draws', forbidden)
    first = sim.run_compact_estimation(output=tmp_path)
    assert len(first) == 160 and first.no_alarm.all() and len(calls) == 160
    assert set(first.scenario) == {s.name for s in selected}
    result_manifest = json.loads((tmp_path / 'estimation' / 'manifest.json').read_text())
    assert result_manifest['artifacts']['replicate_results.csv'] == sim._file_hash(tmp_path / 'estimation' / 'replicate_results.csv')
    assert result_manifest['original_scientific_digest'] == manifest['scientific_digest']
    again = sim.run_compact_estimation(output=tmp_path, reset_scope='derived')
    assert len(again) == 160 and len(calls) == 160
    sim.run_compact_estimation(output=tmp_path, reset_scope='selected')
    assert len(calls) == 320
    assert (stage / 'manifest.json').is_file() and source.is_file()
    monkeypatch.setattr(sim, '_load_cache', lambda *a, **kw: None)
    with pytest.raises(ValueError, match='original episode changed'):
        sim.run_compact_estimation(output=tmp_path)
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='CSV hash mismatch'):
        sim.run_compact_estimation(output=tmp_path)
