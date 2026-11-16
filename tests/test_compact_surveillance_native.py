"""Native frozen-nuisance chart traces against the original Python reference."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import time
import json
import os
import hashlib

import numpy as np
import pytest

from test_compact_native_multiplier import _isolated_module


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    cached = os.environ.get('OCFR_SURVEILLANCE_TEST_BUILD')
    if cached:
        # Reuse only an explicitly supplied, source-identical isolated build.
        # This preserves OS policy enforcement and avoids needless recompilation.
        stage = Path(cached)
        root = Path(__file__).resolve().parents[1]
        assert hashlib.sha256((stage / 'csrc' / 'ocfr_kernels.c').read_bytes()).digest() == hashlib.sha256((root / 'csrc' / 'ocfr_kernels.c').read_bytes()).digest()
        spec = importlib.util.spec_from_file_location('ocfr_surveillance_current_test', root / 'methods.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        library = stage / ('ocfr_fast.dll' if os.name == 'nt' else 'libocfr_fast.so')
        module._library_path = lambda: library
    else:
        module = _isolated_module(tmp_path_factory.mktemp('surveillance-native'), 'ocfr_surveillance_test')
    module._get_native_kernels()
    return module


def _reference(module, deaths, mean, residuals, phi, looks, reference=8, allowance=.5):
    statistics, locations, directions = [], [], []
    positive = negative = 0.
    positive_start = negative_start = reference
    previous = reference
    for prefix in looks:
        best, calendar, scale = -np.inf, reference, 1.
        for change in range(reference, prefix):
            candidate_scale, statistic = module._profile_nb_scale(deaths[change:prefix], mean[change:prefix], phi)
            if statistic > best:
                best, calendar, scale = statistic, change, candidate_scale
        for day in range(previous, prefix):
            value = float(residuals[day])
            if positive + value - allowance <= 0.:
                positive, positive_start = 0., day + 1
            else:
                positive += value - allowance
            if negative - value - allowance <= 0.:
                negative, negative_start = 0., day + 1
            else:
                negative += -value - allowance
        if positive <= 0. and negative <= 0.:
            page, page_calendar, page_direction = 0., prefix - 1, 0
        elif positive >= negative:
            page, page_calendar, page_direction = positive, positive_start, 1
        else:
            page, page_calendar, page_direction = negative, negative_start, -1
        statistics.append([best, page])
        locations.append([calendar, page_calendar])
        directions.append([int(np.sign(scale - 1.)), page_direction])
        previous = prefix
    return np.asarray(statistics), np.asarray(locations), np.asarray(directions)


@pytest.mark.parametrize('phi', [2.5, 10., 1e8, np.inf])
@pytest.mark.parametrize('pattern', ['null', 'zero', 'shift', 'large', 'clip_low', 'clip_high', 'ties'])
def test_frozen_traces(native, phi, pattern):
    n = 65
    mean = np.linspace(2., 120., n)
    deaths = np.random.default_rng(347).poisson(mean).astype(float)
    if pattern == 'zero':
        deaths[:] = 0.
    elif pattern == 'shift':
        deaths[40:] *= 4.
    elif pattern == 'large':
        mean *= 10000.
        deaths = np.rint(mean * np.linspace(.5, 2., n))
    elif pattern == 'clip_low':
        mean[:] = 1e5
        deaths[:] = 1.
    elif pattern == 'clip_high':
        mean[:] = 1e-12
        deaths[:] = 1.
    elif pattern == 'ties':
        mean[:] = 4.
        deaths[:] = 4.
    residuals = (deaths - mean) / np.sqrt(mean + mean * mean / phi)
    looks = (9, 16, 32, 47, 65)
    expected = _reference(native, deaths, mean, residuals, phi, looks)
    actual = native.FastKernels().surveillance_benchmark_traces(deaths, mean, residuals, phi, 8, looks, .5)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=1e-8)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    # Common thresholds away from numerical ties must yield identical decisions.
    thresholds = np.array([20., 5.])
    np.testing.assert_array_equal(actual[0] > thresholds, expected[0] > thresholds)
    # Causality: altered future data cannot affect an earlier look.
    modified = deaths.copy()
    modified[32:] += 500.
    modified_mean = mean.copy()
    modified_mean[32:] *= 3.
    modified_residuals = residuals.copy()
    modified_residuals[32:] += 50.
    future = native.FastKernels().surveillance_benchmark_traces(
        modified, modified_mean, modified_residuals, phi, 8, looks, .5)
    np.testing.assert_array_equal(actual[0][:3], future[0][:3])
    np.testing.assert_array_equal(actual[1][:3], future[1][:3])
    np.testing.assert_array_equal(actual[2][:3], future[2][:3])


@pytest.mark.parametrize('family', ['steady', 'sparse', 'wave'])
def test_public_helper_families(native, family):
    days = np.arange(120)
    cases = np.full(120, 250. if family == 'sparse' else 1000.)
    if family == 'wave':
        cases *= 1. + .6 * np.sin(days / 14.)
    delay = native.gamma_delay_pmf(14., maximum_lag=30)
    deaths = native.simulate_nb_deaths(cases, delay, np.full(120, .02), 10., np.random.default_rng(581))
    deaths[75:] *= 3
    looks = (35, 56, 84, 120)
    a = native.surveillance_benchmark_traces(cases, deaths, delay, looks, backend='python')
    b = native.surveillance_benchmark_traces(cases, deaths, delay, looks, backend='c')
    for method in a:
        np.testing.assert_allclose([x.statistic for x in a[method]],
                                   [x.statistic for x in b[method]], rtol=0, atol=1e-8)
        for x, y in zip(a[method], b[method]):
            assert (x.method, x.day, x.estimated_calendar_change_day,
                    x.estimated_cohort_change_day, x.direction) == (
                        y.method, y.day, y.estimated_calendar_change_day,
                        y.estimated_cohort_change_day, y.direction)
    # Prefix helper versus full helper (same frozen reference and causal exposure).
    early = native.surveillance_benchmark_traces(cases[:56], deaths[:56], delay, looks[:2], backend='c')
    for method in b:
        assert b[method][:2] == early[method]


def test_page_is_current_look_not_historical_max(native):
    mean = np.full(16, 10.)
    deaths = mean.copy()
    residuals = np.zeros(16)
    residuals[8:10] = 3.
    values, locations, directions = native.FastKernels().surveillance_benchmark_traces(
        deaths, mean, residuals, 10., 8, (10, 16), .5)
    assert values[0, 1] == 5.
    assert values[1, 1] == 2.
    assert locations[:, 1].tolist() == [8, 8]
    assert directions[:, 1].tolist() == [1, 1]



@pytest.mark.parametrize('phi', [1., 2.5, 10., 1e7, 1e8, np.inf])
@pytest.mark.parametrize('outcomes,means', [([0, 0, 1000], [10, 10, 100]),
    ([0, 0, 0, 100], [10, 10, 10, 100]), ([0], [20]),
    ([1, 0, 2], [1e8, 1e8, 1e8]), ([1, 3, 2], [1e-12, 2e-12, 1e-12])])
def test_profile_independent_optimum(native, phi, outcomes, means):
    from scipy.optimize import brentq
    methods = native
    y, mu = np.asarray(outcomes, dtype=float), np.asarray(means, dtype=float)
    gamma, lr = methods._profile_nb_scale(y, mu, phi)
    if np.sum(y) == 0:
        assert gamma == 0.
        expected_lr = 2 * (np.sum(mu) if phi >= 1e7 else np.sum(phi * np.log1p(mu / phi)))
    elif phi >= 1e7:
        expected = np.sum(y) / np.sum(mu)
        np.testing.assert_allclose(gamma, expected, rtol=1e-12)
        expected_lr = 2 * np.sum(y * np.log(expected) - (expected - 1) * mu)
    else:
        def score(eta):
            fitted = np.exp(eta) * mu
            return np.sum(phi * (y - fitted) / (phi + fitted))
        eta = brentq(score, -100., 100., xtol=1e-13)
        expected = np.exp(eta)
        np.testing.assert_allclose(gamma, expected, rtol=1e-12)
        expected_lr = 2 * np.sum(y * eta - (y + phi) *
            (np.log1p(expected * mu / phi) - np.log1p(mu / phi)))
    # At LR around 6e8 one float64 ULP already exceeds 1e-8.
    # Keep the absolute criterion elsewhere, and allow four ULPs only there.
    np.testing.assert_allclose(lr, expected_lr, rtol=0,
                               atol=max(1e-8, 4 * np.spacing(abs(expected_lr))))
    if phi == 10. and outcomes == [0, 0, 1000]:
        np.testing.assert_allclose(lr, 78.33745395776782, rtol=0, atol=1e-9)


@pytest.mark.parametrize('backend', ['python', 'c'])
def test_unavailable_fit_retains_every_look(native, monkeypatch, backend):
    def fail(*args, **kwargs):
        raise RuntimeError('benchmark reference fit failed: test')
    monkeypatch.setattr(native, '_baseline_residuals', fail)
    trace = native.surveillance_benchmark_traces(np.ones(40), np.ones(40), [1.], (30, 40), backend=backend)
    assert set(trace) == {'NB-GLR', 'Page-CUSUM'}
    for rows in trace.values():
        assert len(rows) == 2
        assert all(np.isnan(row.statistic) and row.estimated_calendar_change_day is None
                   and row.fit_status.startswith('unavailable:') for row in rows)



def test_native_infrastructure_errors_propagate(native, monkeypatch):
    def fail():
        raise RuntimeError('native infrastructure test')
    monkeypatch.setattr(native, '_baseline_residuals',
                        lambda *args, **kwargs: (np.zeros(40), np.ones(40), 1., 10.))
    monkeypatch.setattr(native, '_get_native_kernels', fail)
    with pytest.raises(RuntimeError, match='native infrastructure'):
        native.surveillance_benchmark_traces(np.ones(40), np.ones(40), [1.], (30, 40))


def test_baseline_invalid_argument_propagates(native, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('invalid reference argument')
    monkeypatch.setattr(native, '_baseline_residuals', fail)
    with pytest.raises(ValueError, match='invalid reference'):
        native.surveillance_benchmark_traces(np.ones(40), np.ones(40), [1.], (30, 40))


def benchmark(path):
    stage = Path(tempfile.mkdtemp(prefix='ocfr-surveillance-benchmark-'))
    module = _isolated_module(stage, 'ocfr_surveillance_benchmark')
    days = np.arange(180)
    cases = 1000. * (1. + .6 * np.sin(days / 14.))
    delay = module.gamma_delay_pmf(14., maximum_lag=45)
    deaths = module.simulate_nb_deaths(cases, delay, np.full(180, .02), 10., np.random.default_rng(992))
    looks = tuple(range(35, 181, 7))
    if looks[-1] != 180:
        looks += (180,)
    residuals, mean, _, phi = module._baseline_residuals(cases, deaths, delay, 28)
    times = {}
    for backend in ('python', 'c'):
        module.surveillance_benchmark_traces(cases, deaths, delay, looks, backend=backend)
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            result = module.surveillance_benchmark_traces(cases, deaths, delay, looks, backend=backend)
            timings.append(time.perf_counter() - start)
        times[backend] = float(np.median(timings))
    result = {'n': len(cases), 'looks': len(looks), 'phi': phi,
              'median_seconds_with_shared_fit': times, 'temporary_build_root': str(stage)}
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    benchmark(Path(sys.argv[1]))
