"""Independent arithmetic checks for the raw two-stage moment reference."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import methods
import simulation as sim


def independent_exposures(cases, delay, tau):
    # Explicit cohort/report summation, independent of the production matrix.
    total = np.zeros(len(cases))
    post = np.zeros(len(cases))
    for report in range(len(cases)):
        for cohort in range(report):
            lag = report - cohort - 1
            if lag < len(delay):
                value = cases[cohort] * delay[lag]
                total[report] += value
                if cohort >= tau:
                    post[report] += value
    return total, post


@pytest.mark.parametrize('left,right', [(.02, .04), (.04, .02), (.3, .3)])
def test_noiseless_means_recover_both_risks(left, right):
    cases, delay, tau = np.arange(20., 35.), np.array([.2, .5, .3]), 6
    total, post = independent_exposures(cases, delay, tau)
    means = left * (total-post) + right * post
    result = methods.nishiura_two_stage_moments(cases, means, delay, tau=tau)
    assert result['pi_left_estimate'] == pytest.approx(left)
    assert result['pi_right_estimate'] == pytest.approx(right)
    assert result['risk_ratio_estimate'] == pytest.approx(right/left)


def test_report_index_and_contamination_subtraction():
    cases = np.array([10., 20., 30., 40., 50., 60., 70.])
    deaths, delay, tau = np.array([0., 1., 3., 8., 2., 1., 9.]), np.array([.4, .6]), 4
    x, b = independent_exposures(cases, delay, tau)
    result = methods.nishiura_two_stage_moments(cases, deaths, delay, tau=tau)
    left = sum(deaths[:tau]) / sum(x[:tau])
    right = (sum(deaths[tau:])-left*sum(x[tau:]-b[tau:])) / sum(b[tau:])
    assert result['pre_exposure'] == pytest.approx(sum(x[:tau]))
    assert result['pre_contamination'] == pytest.approx(sum(x[tau:]-b[tau:]))
    assert result['pi_left_estimate'] == pytest.approx(left)
    assert result['pi_right_estimate'] == pytest.approx(right)
    assert not np.isclose(right, sum(deaths[tau:])/sum(b[tau:]))


def test_alarm_prefix_ignores_future_observations():
    cases, deaths = np.arange(10., 22.), np.arange(12.)
    one = methods.nishiura_two_stage_moments(cases, deaths, [.3, .7], tau=4, alarm=8)
    cases[9:], deaths[9:] = 999999., 999999.
    two = methods.nishiura_two_stage_moments(cases, deaths, [.3, .7], tau=4, alarm=8)
    assert one == two
    assert one == methods.nishiura_two_stage_moments(cases[:9], deaths[:9], [.3, .7], tau=4)


def test_negative_and_greater_than_one_points_are_retained():
    cases, deaths = np.full(9, 10.), np.zeros(9)
    deaths[1:3] = 20.
    result = methods.nishiura_two_stage_moments(cases, deaths, [.5, .5], tau=4)
    assert result['pi_left_estimate'] > 1
    assert result['pi_right_estimate'] < 0
    assert result['risk_ratio_estimate'] < 0
    for target in ('pi_left', 'pi_right', 'risk_ratio'):
        assert result[target + '_point_valid']
        assert not result[target + '_admissible']


def test_zero_pre_risk_keeps_cfr_points_but_ratio_is_unavailable():
    result = methods.nishiura_two_stage_moments(np.full(9, 10.), np.zeros(9), [1.], tau=4)
    assert result['pi_left_estimate'] == result['pi_right_estimate'] == 0
    assert result['pi_left_point_valid'] and result['pi_right_point_valid']
    assert result['pi_left_admissible'] and result['pi_right_admissible']
    assert result['risk_ratio_estimate'] is None
    assert not result['risk_ratio_point_valid']
    assert result['status'] == 'zero_pre_risk'


def test_zero_right_risk_has_admissible_zero_ratio():
    cases, delay, tau = np.full(12, 10.), np.array([1.]), 5
    x, b = independent_exposures(cases, delay, tau)
    result = methods.nishiura_two_stage_moments(cases, .1*(x-b), delay, tau=tau)
    assert result['risk_ratio_estimate'] == pytest.approx(0)
    assert result['risk_ratio_admissible']


def test_zero_exposure_and_missing_post_exposure():
    result = methods.nishiura_two_stage_moments(np.zeros(9), np.zeros(9), [1.], tau=4)
    assert result['status'] == 'missing_pre_exposure'
    assert all(result[t + '_estimate'] is None for t in ('pi_left', 'pi_right', 'risk_ratio'))
    cases = np.array([10., 10., 10., 10., 0., 0., 0., 0., 0.])
    result = methods.nishiura_two_stage_moments(cases, np.zeros(9), [1.], tau=4)
    assert result['pi_left_point_valid']
    assert not result['pi_right_point_valid']
    assert result['status'] == 'missing_post_exposure'


@pytest.mark.parametrize('tau,alarm', [(0, 8), (9, 8), (4, 1), (4, 9), (4.5, 8)])
def test_invalid_indices(tau, alarm):
    with pytest.raises(ValueError):
        methods.nishiura_two_stage_moments(np.ones(9), np.zeros(9), [1.], tau=tau, alarm=alarm)


def test_reference_episode_checks_observation_hash_and_saved_alarm(monkeypatch):
    from dataclasses import replace
    cfg = sim.get_compact_experiment_config('demo', backend='python', protocol="compact-v1")
    scenario = next(s for s in cfg.scenarios if s.split == 'alternative')
    hashes = dict(cases='a', deaths='b', true_delay='c', assumed_delay='d')
    trace = [dict(day=8, tau_hat=4, status='available', statistic=1.)]
    payload = dict(payload_hash='p', rows=[dict(method='OCFR Gaussian', trace=trace)],
                   stream_manifest=dict(hashes=hashes))
    source = dict(alarm_day=8, tau_hat_at_alarm=4, source_payload_hash='p',
                  case_hash='a', death_hash='b', true_delay_hash='c', assumed_delay_hash='d')
    monkeypatch.setattr(sim, '_load_cache', lambda *a, **kw: payload)
    monkeypatch.setattr(sim, '_compact_generate_components', lambda *a: (
        np.full(12, 100.), np.ones(12), np.array([1.]), dict(hashes=hashes)))
    args = (scenario, 0, cfg, 'original', Path('cache'), source)
    result = sim._reference_episode(*args)
    assert result['nishiura_pi_left_point_valid']
    assert not any('lower' in key or 'upper' in key for key in result)
    source['case_hash'] = 'changed'
    with pytest.raises(ValueError, match='hash mismatch'):
        sim._reference_episode(*args)
    source['case_hash'], source['tau_hat_at_alarm'] = 'a', 5
    with pytest.raises(ValueError, match='does not replay'):
        sim._reference_episode(*args)
