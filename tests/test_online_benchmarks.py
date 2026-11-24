"""Prefix measurability and backend parity for real-data benchmarks."""

import numpy as np
import pytest

from methods import fast_available
from real_data import OnlineBenchmarkCalibration


@pytest.mark.parametrize('backend', ['python', 'c'])
def test_online_benchmarks_use_only_the_current_prefix(backend):
    if backend == 'c' and not fast_available():
        pytest.skip('native kernel is unavailable')
    cases = np.rint(np.linspace(80, 260, 30))
    deaths = np.random.default_rng(87).poisson(0.04 * cases)
    future_cases, future_deaths = cases.copy(), deaths.copy()
    future_cases[22:] += 2500
    future_deaths[22:] += 75
    config = dict(planned_looks=(15, 22, 30), observation_start=3,
                  reference_end=11, lag_offset=0, n_paths=89, n_audit=33, seed=5)
    delay = [0.3, 0.4, 0.3]
    detector = OnlineBenchmarkCalibration(delay, **config, backend=backend)
    reference_detector = OnlineBenchmarkCalibration(delay, **config, backend='python')
    altered = OnlineBenchmarkCalibration(delay, **config, backend=backend, workers=3)

    with pytest.raises(ValueError, match='next planned look'):
        detector.advance(cases, deaths)

    for prefix in config['planned_looks']:
        current = detector.advance(cases[:prefix], deaths[:prefix])
        reference = reference_detector.advance(cases[:prefix], deaths[:prefix])
        changed = altered.advance(future_cases[:prefix], future_deaths[:prefix])
        for actual, expected in zip(current, reference):
            for key in actual:
                if key in ('statistic', 'cutoff'):
                    np.testing.assert_allclose(actual[key], expected[key], atol=2e-8, rtol=2e-9)
                else:
                    assert actual[key] == expected[key]
        assert (current == changed) == (prefix <= 22)
