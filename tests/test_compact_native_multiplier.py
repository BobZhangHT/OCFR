"""Isolated native transform parity; never loads or overwrites the release DLL."""
from __future__ import annotations

import ctypes
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import pytest


LEGACY_MULTIPLIER = r'''
OCFR_API int ocfr_multiplier_maxima(
    int n_multiplier, int n_obs, int n_candidates,
    const double *multipliers, const double *influence, double *out
) {
    if (!multipliers || !influence || !out || n_multiplier <= 0 ||
        n_obs <= 0 || n_candidates <= 0) return 1;
    for (int b = 0; b < n_multiplier; ++b) {
        double maximum = 0.0;
        const double *xi = multipliers + (size_t)b * (size_t)n_obs;
        for (int k = 0; k < n_candidates; ++k) {
            double process = 0.0;
            for (int s = 0; s < n_obs; ++s)
                process += xi[s] * influence[(size_t)s * (size_t)n_candidates + (size_t)k];
            const double squared = process * process;
            if (k == 0 || squared > maximum) maximum = squared;
        }
        out[b] = maximum;
    }
    return 0;
}
'''


def _isolated_module(parent: Path, name: str, *, legacy: bool = False):
    """Build by the public build_native path, using a distinct library filename."""
    root = Path(__file__).resolve().parents[1]
    stage = parent / name
    (stage / 'csrc').mkdir(parents=True)
    shutil.copy2(root / 'methods.py', stage / 'methods.py')
    source = (root / 'csrc' / 'ocfr_kernels.c').read_text()
    if legacy:
        start = source.index('OCFR_API int ocfr_multiplier_maxima(')
        end = source.index('\n}\n', start) + 3
        source = source[:start] + LEGACY_MULTIPLIER + source[end:]
    (stage / 'csrc' / 'ocfr_kernels.c').write_text(source)
    spec = importlib.util.spec_from_file_location(name, stage / 'methods.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.build_native(force=True)
    return module


@pytest.fixture(scope='module')
def kernels(tmp_path_factory):
    parent = tmp_path_factory.mktemp('compact-native-multiplier')
    return (_isolated_module(parent, 'ocfr_speed_old', legacy=True),
            _isolated_module(parent, 'ocfr_speed_new'))


@pytest.mark.parametrize('n,k,B', [(1, 1, 1), (7, 1, 19), (17, 3, 37),
                                 (64, 8, 399), (91, 17, 400),
                                 (180, 80, 2000), (180, 178, 399),
                                 (13, 0, 19)])
def test_fixed_draw_transform(kernels, n, k, B):
    old, new = kernels
    rng = np.random.default_rng(n * 17 + k)
    # Noncontiguous views also exercise the unchanged public copying contract.
    draws = rng.standard_normal((B, 2 * n))[:, ::2]
    weights = rng.standard_normal((n, 2 * k))[:, ::2] / np.sqrt(n)
    a = old.multiplier_statistics_from_draws(draws, weights)
    b = new.multiplier_statistics_from_draws(draws, weights)
    ref = new.multiplier_statistics_from_draws(draws, weights, backend='python')
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(b, ref, rtol=0, atol=1e-10, equal_nan=True)


def _real_data(module):
    rng = np.random.default_rng(318)
    cases = np.full(180, 10000.)
    delay = module.gamma_delay_pmf(12, maximum_lag=30)
    deaths = module.simulate_nb_deaths(cases, delay, np.full(180, .02), 30, rng)
    return cases, deaths, delay


@pytest.mark.parametrize("shift", [False, True])
def test_legacy_common_draw_boundaries_and_alarm(kernels, shift):
    """Historical alpha/J utility parity, independent of the online default."""
    old, new = kernels
    cases, deaths, delay = _real_data(new)
    if shift:
        deaths = deaths.copy()
        deaths[90:] *= 4
    common = np.random.default_rng(510).standard_normal((2000, 180))
    crossed = [np.zeros(2000, dtype=bool) for _ in range(3)]
    alarms = []
    for prefix in (60, 90, 180):
        scan = new.score_scan(cases[:prefix], deaths[:prefix], delay,
                              backend='c', use_lindeberg_gate=False)
        assert scan.influence.shape[1] > 0
        maxima = [old.multiplier_statistics_from_draws(common[:, :prefix], scan.influence),
                  new.multiplier_statistics_from_draws(common[:, :prefix], scan.influence),
                  new.multiplier_statistics_from_draws(common[:, :prefix], scan.influence,
                                                       backend='python')]
        np.testing.assert_array_equal(maxima[0], maxima[1])
        np.testing.assert_allclose(maxima[1], maxima[2], rtol=0, atol=1e-10)
        thresholds = [new.monte_carlo_critical_value(x, .05 / 3) for x in maxima]
        assert thresholds[0] == thresholds[1]
        np.testing.assert_allclose(thresholds, thresholds[0], rtol=0, atol=1e-10)
        assert len({bool(scan.statistic > t) for t in thresholds}) == 1
        alarms.append(bool(scan.statistic > thresholds[1]))
        for crossed_path, values, threshold in zip(crossed, maxima, thresholds):
            crossed_path |= values > threshold
        np.testing.assert_array_equal(crossed[0], crossed[1])
        np.testing.assert_array_equal(crossed[1], crossed[2])
    assert any(alarms) == shift


@pytest.mark.parametrize("shift", [False, True])
def test_spending_common_draw_boundaries_and_alarm(kernels, shift):
    """The v2 shared boundary must agree across native and Python transforms."""
    old, new = kernels
    cases, deaths, delay = _real_data(new)
    if shift:
        deaths = deaths.copy()
        deaths[90:] *= 4
    common = np.random.default_rng(510).standard_normal((2000, 180))
    states = [new.GaussianSpendingBoundary(2000, 3, .05) for _ in range(3)]
    for prefix in (60, 90, 180):
        scan = new.score_scan(cases[:prefix], deaths[:prefix], delay,
                              backend='c', use_lindeberg_gate=False)
        assert scan.influence.shape[1] > 0
        maxima = [old.multiplier_statistics_from_draws(common[:, :prefix], scan.influence),
                  new.multiplier_statistics_from_draws(common[:, :prefix], scan.influence),
                  new.multiplier_statistics_from_draws(common[:, :prefix], scan.influence,
                                                       backend='python')]
        looks = [state.advance(values) for state, values in zip(states, maxima)]
        assert looks[0] == looks[1]
        assert looks[2].rank == looks[0].rank
        np.testing.assert_allclose([look.boundary for look in looks], looks[0].boundary,
                                   rtol=0, atol=1e-10)
        for state in states[1:]:
            np.testing.assert_array_equal(state.eligible, states[0].eligible)
            np.testing.assert_array_equal(state.support_discarded, states[0].support_discarded)
            np.testing.assert_array_equal(state.strict_crossed, states[0].strict_crossed)
        assert len({bool(scan.statistic > look.boundary) for look in looks}) == 1


def test_report_day_weights(kernels):
    old, new = kernels
    cases, deaths, delay = _real_data(new)
    aligned = new.score_scan(cases, deaths, delay, backend='c', use_lindeberg_gate=False)
    scan = new.report_day_score_scan(aligned, deaths, use_lindeberg_gate=False)
    draws = np.random.default_rng(542).standard_normal((399, len(cases)))
    a = old.multiplier_statistics_from_draws(draws, scan.influence)
    b = new.multiplier_statistics_from_draws(draws, scan.influence)
    ref = new.multiplier_statistics_from_draws(draws, scan.influence, backend='python')
    np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
    np.testing.assert_allclose(b, ref, rtol=0, atol=1e-10)


def test_native_abi_dimension_guard(kernels):
    _, new = kernels
    huge = np.lib.stride_tricks.as_strided(np.zeros(1), shape=(2**31, 1), strides=(0, 0))
    with pytest.raises(ValueError, match='C int ABI limit'):
        new.FastKernels().multiplier_maxima(huge, np.ones((1, 1)))
    one = np.ones(1)
    ptr = one.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    assert new.FastKernels().lib.ocfr_multiplier_maxima(1, 1, 0, ptr, ptr, ptr) == 1


def benchmark(output: Path):
    # Run this file directly in a fresh process. Temporary builds survive only
    # under their distinct paths, so Windows DLL caching cannot select old code.
    parent = Path(tempfile.mkdtemp(prefix='ocfr-multiplier-benchmark-'))
    old = _isolated_module(parent, 'ocfr_bench_old', legacy=True)
    new = _isolated_module(parent, 'ocfr_bench_new')
    cases, deaths, delay = _real_data(new)
    real = new.score_scan(cases, deaths, delay, backend='c', use_lindeberg_gate=False)
    report_day = new.report_day_score_scan(real, deaths, use_lindeberg_gate=False)
    rng = np.random.default_rng(873)
    rows = []
    for label, n, k, B in [('synthetic', 7, 1, 19), ('synthetic', 64, 8, 399),
                            ('synthetic', 180, 80, 2000),
                            ('real_score', 180, real.influence.shape[1], 2000),
                            ('real_score', 180, real.influence.shape[1], 10000),
                            ('report_day', 180, report_day.influence.shape[1], 10000)]:
        draws = rng.standard_normal((B, n))
        weights = (real.influence if label == 'real_score' else report_day.influence
                   if label == 'report_day' else rng.standard_normal((n, k)) / np.sqrt(n))
        seconds, values = {}, {}
        for name, module, backend in [('old_c', old, 'c'), ('new_c', new, 'c'),
                                       ('python', new, 'python')]:
            module.multiplier_statistics_from_draws(draws, weights, backend=backend)
            timings = []
            for _ in range(5):
                start = time.perf_counter()
                values[name] = module.multiplier_statistics_from_draws(draws, weights, backend=backend)
                timings.append(time.perf_counter() - start)
            seconds[name] = float(np.median(timings))
        rows.append(dict(label=label, n=n, k=k, B=B, median_seconds=seconds,
                         bitwise_old_new=np.array_equal(values['old_c'].view(np.uint64), values['new_c'].view(np.uint64)),
                         max_abs_old_new=float(np.max(np.abs(values['old_c'] - values['new_c']))),
                         max_abs_new_python=float(np.max(np.abs(values['new_c'] - values['python'])))))
    output.write_text(json.dumps(dict(temporary_build_root=str(parent), rows=rows), indent=2))
    print(output.read_text())


if __name__ == '__main__':
    benchmark(Path(sys.argv[1]))
