"""Scientific parity for shared reference fits and separate direction screens."""
from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import methods as m


@pytest.mark.parametrize("level", [250, 1000])
def test_shared_reference_trace_equals_prefix_charts(level):
    sampler = m.make_sampler(1892 + level, backend="python")
    cases = m.make_correlated_epidemic_cases(84, "constant", level, sampler)
    delay = m.gamma_delay_pmf(14)
    deaths = m.simulate_nb_deaths(cases, delay, np.full(84, .02), 10, sampler)
    looks = [35, 56, 84]
    traces = m.surveillance_benchmark_traces(cases, deaths, delay, looks, backend="python")
    for j, n in enumerate(looks):
        for name, chart in [("NB-GLR", m.nb_surveillance_glr), ("Page-CUSUM", m.page_cusum)]:
            expected = chart(cases[:n], deaths[:n], delay)
            got = traces[name][j]
            assert got.statistic == pytest.approx(expected.statistic, abs=1e-10)
            assert got.estimated_calendar_change_day == expected.estimated_calendar_change_day
            assert got.direction == expected.direction
    changed_future = deaths.copy()
    changed_future[56:] += 20
    perturbed = m.surveillance_benchmark_traces(cases, changed_future, delay, looks, backend="python")
    for name in traces:
        assert traces[name][:2] == perturbed[name][:2]


def test_report_direction_does_not_inherit_cohort_screen_when_grid_supplied():
    sampler = m.make_sampler(123, backend="python")
    cases = m.make_correlated_epidemic_cases(84, "constant", 1000, sampler)
    delay = m.gamma_delay_pmf(14)
    deaths = m.simulate_nb_deaths(cases, delay, np.full(84, .02), 10, sampler)
    scan = m.score_scan(cases, deaths, delay, backend="python")
    initial = np.arange(1, 84)
    expected = m.report_day_score_scan(scan, deaths, initial_candidates=initial)
    emptied = replace(scan, candidates=np.empty(0, dtype=np.int64))
    got = m.report_day_score_scan(emptied, deaths, initial_candidates=initial)
    assert got.statistic == expected.statistic
    np.testing.assert_array_equal(got.candidates, expected.candidates)
    assert np.isnan(m.report_day_score_scan(emptied, deaths).statistic)
