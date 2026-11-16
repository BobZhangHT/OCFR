"""Synthetic/analytic checks of the alternative-blinded secondary amendment."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from analysis import (COMPACT_METHOD_ORDER, EvidenceError, _compact_standardized_power,
                      _compact_trace_maxima, analyze_compact_simulation,
                      compact_null_quantile_bracket, compact_standardized_power_envelope)
from test_compact_analysis import _fixture, _hash, _rewrite_stage


def test_prespecified_5000_quantile_ranks_and_exact_tail_allocations():
    result = compact_null_quantile_bracket(np.arange(5000, dtype=float))
    assert (result["point_rank"], result["lower_rank"], result["upper_rank"]) == (4750, 4709, 4790)
    assert (result["cutoff"], result["cutoff_low"], result["cutoff_high"]) == (4749, 4708, 4789)
    assert binom.cdf(4708, 5000, .95) <= .005 < binom.cdf(4709, 5000, .95)
    assert binom.sf(4789, 5000, .95) <= .005 < binom.sf(4788, 5000, .95)
    # Exact binomial tail equality is admissible, rather than one rank wider.
    tied_tail = compact_null_quantile_bracket([0., 1.], q=.5, error=.5)
    assert (tied_tail["lower_rank"], tied_tail["upper_rank"]) == (1, 2)


@pytest.mark.parametrize("n", [1, 3, 10, 200])
def test_non5000_ranks_and_infinite_sentinels(n):
    result = compact_null_quantile_bracket(np.arange(n, dtype=float))
    assert result["point_rank"] == int(np.ceil(.95*n))
    assert result["quantile_lower_tail"] + result["quantile_upper_tail"] <= .01
    if result["lower_rank"] == 0:
        assert np.isneginf(result["cutoff_low"])
    if result["upper_rank"] == n+1:
        assert np.isposinf(result["cutoff_high"])
    if n == 1:
        assert result["lower_rank"] == 0 and result["upper_rank"] == 2


def test_quantile_bracket_coverage_against_known_discrete_population():
    # Population: P(Q=0)=.949, P(Q=1)=.051, so its generalized .95 quantile
    # is exactly 1. Sum over the entire sample distribution, not random runs.
    n, coverage = 200, 0.
    for zeros in range(n+1):
        sample = np.r_[np.zeros(zeros), np.ones(n-zeros)]
        bracket = compact_null_quantile_bracket(sample)
        if bracket["cutoff_low"] <= 1 <= bracket["cutoff_high"]:
            coverage += binom.pmf(zeros, n, .949)
    assert .99 <= coverage <= 1+1e-12
    assert 1-.949 > .05  # c=0 would violate the target; c=1 has strict size 0.


def test_ties_strict_crossing_and_first_alarm_equivalence():
    for scores in itertools.product([0., 1., 2.], repeat=3):
        trace = json.dumps([dict(day=day, status="available", raw_statistic=score)
                            for day, score in enumerate(scores)])
        a, b, _ = _compact_trace_maxima(trace, 1, 3, planned_looks=[0, 1, 2])
        for cutoff in [0., 1., 2.]:
            first = next((day for day, score in enumerate(scores) if score > cutoff), None)
            oracle = first is not None and first >= 1
            result = compact_standardized_power_envelope([a], [b], cutoff=cutoff,
                cutoff_low=cutoff, cutoff_high=cutoff)
            assert result["standardized_events"] == int(oracle)
    tied = compact_null_quantile_bracket(np.ones(5000))
    assert tied["cutoff"] == tied["cutoff_low"] == tied["cutoff_high"] == 1.


def test_premature_alarm_nonmonotonicity_needs_envelopes():
    # Useful at c=3, but not at either c=1 (premature) or c=5 (absent).
    a, b = [2.], [4.]
    endpoint_events = []
    for cutoff in [1., 3., 5.]:
        endpoint_events.append(compact_standardized_power_envelope(a, b, cutoff=cutoff,
            cutoff_low=cutoff, cutoff_high=cutoff)["standardized_events"])
    assert endpoint_events == [0, 1, 0]
    result = compact_standardized_power_envelope(a, b, cutoff=3., cutoff_low=1., cutoff_high=5.)
    assert result["lower_envelope_events"] == 0 and result["upper_envelope_events"] == 1
    # Exhaustively check the logical envelope for all integer thresholds
    # within this bracket, independently of the unknown true threshold.
    for a, b in itertools.product([-np.inf, 0., 1., 2., 4., 5., 6.], repeat=2):
        result = compact_standardized_power_envelope([a], [b], cutoff=3., cutoff_low=1., cutoff_high=5.)
        for c in [1., 2., 3., 4., 5.]:
            oracle = int(a <= c < b)
            assert result["lower_envelope_events"] <= oracle <= result["upper_envelope_events"]


def test_unavailable_and_empty_maxima_retained_but_missing_computation_rejected():
    a, b, unavailable = _compact_trace_maxima(json.dumps([
        dict(day=0, status="unavailable"), dict(day=2, status="unavailable", raw_statistic=None)]),
        1, 3, planned_looks=[0, 2])
    assert np.isneginf(a) and np.isneginf(b) and unavailable == 2
    result = compact_standardized_power_envelope([a], [b], cutoff=1., cutoff_low=0., cutoff_high=2.)
    assert result["alternative_n"] == 1 and result["standardized_events"] == 0
    with pytest.raises(EvidenceError, match="every planned look"):
        _compact_trace_maxima("[]", 1, 3, planned_looks=[0, 2])
    with pytest.raises(EvidenceError, match="explicitly"):
        _compact_trace_maxima('[{"day": 0}]', 1, 3)
    with pytest.raises(EvidenceError, match="missing or nonfinite"):
        _compact_trace_maxima('[{"day": 0, "status": "available"}]', 1, 3)
    assert _compact_trace_maxima("[]", 0, 3)[:2] == (-np.inf, -np.inf)


def test_one_sided_cp_endpoints_and_oracle_binomial_coverage():
    n = 12
    zero = compact_standardized_power_envelope(np.zeros(n), np.zeros(n),
        cutoff=1., cutoff_low=1., cutoff_high=1.)
    one = compact_standardized_power_envelope(np.zeros(n), np.full(n, 2.),
        cutoff=1., cutoff_low=1., cutoff_high=1.)
    assert zero["mc_ci_low"] == 0 and zero["mc_ci_high"] == pytest.approx(1-.02**(1/n))
    assert one["mc_ci_high"] == 1 and one["mc_ci_low"] == pytest.approx(.02**(1/n))
    # Known alternative distribution has useful probability .4 at c=1.
    # Enumerate its binomial law to assess interval coverage analytically.
    coverage = 0.
    for k in range(n+1):
        after = np.r_[np.full(k, 2.), np.zeros(n-k)]
        result = compact_standardized_power_envelope(np.zeros(n), after,
            cutoff=1., cutoff_low=1., cutoff_high=1.)
        if result["mc_ci_low"] <= .4 <= result["mc_ci_high"]:
            coverage += binom.pmf(k, n, .4)
    assert coverage >= .96


def _add_traces(frame):
    traces, maxima = [], []
    for row in frame.itertuples(index=False):
        if row.replicate == 2:
            trace = [dict(day=0, status="unavailable"),
                     dict(day=int(row.true_change), status="unavailable")]
            maximum = -np.inf
        else:
            a = 0. if row.replicate == 0 else 2.
            b = 2. if row.replicate == 0 else 3.
            trace = [dict(day=0, status="available", raw_statistic=a),
                     dict(day=int(row.true_change), status="available", raw_statistic=b)]
            maximum = max(a, b)
        traces.append(json.dumps(trace))
        maxima.append(maximum)
    frame["trace_json"], frame["episode_statistic"] = traces, maxima
    return frame


def test_synthetic_stream_integration_roundtrip_and_artifact_provenance(tmp_path):
    _fixture(tmp_path)
    _rewrite_stage(tmp_path, "evaluation", _add_traces)
    def audit_scores(frame):
        frame["episode_statistic"] = float.fromhex("0x1.0000000000001p+0")
        return frame
    _rewrite_stage(tmp_path, "audit", audit_scores)
    source_hashes = {path: _hash(path) for stage in ("calibration", "audit", "evaluation")
                     for path in (tmp_path/stage).iterdir() if path.is_file()}
    tables = analyze_compact_simulation(tmp_path)
    result = tables["compact_standardized_power"]
    assert len(result) == 4 and set(result.method) == set(COMPACT_METHOD_ORDER[1:])
    assert set(result.alternative_n) == {3} and set(result.standardized_events) == {1}
    assert set(result.cutoff) == {float.fromhex("0x1.0000000000001p+0")}
    assert set(result.empty_after_n) == {1} and set(result.unavailable_looks) == {2}
    report = json.loads((tmp_path/"analysis"/"compact_analysis_manifest.json").read_text())
    secondary = report["secondary_analyses"]["null_quantile_standardized_useful_power"]
    assert secondary["status"] == "complete" and not secondary["operational_thresholds_changed"]
    assert report["files"]["compact_standardized_power.csv"] == _hash(tmp_path/"analysis"/"compact_standardized_power.csv")
    assert not (tmp_path/"analysis"/"figures").exists()
    assert all(_hash(path) == digest for path, digest in source_hashes.items())


def test_old_synthetic_fixture_explicitly_skips_missing_traces(tmp_path):
    _fixture(tmp_path)
    analyze_compact_simulation(tmp_path)
    report = json.loads((tmp_path/"analysis"/"compact_analysis_manifest.json").read_text())
    secondary = report["secondary_analyses"]["null_quantile_standardized_useful_power"]
    assert secondary["status"] == "skipped_synthetic_fixture_without_traces"
    assert "compact_standardized_power.csv" not in report["files"]


def test_real_data_cannot_skip_missing_traces(tmp_path):
    _fixture(tmp_path)
    em = json.loads((tmp_path/"evaluation"/"manifest.json").read_text())
    am = json.loads((tmp_path/"audit"/"manifest.json").read_text())
    em["synthetic_fixture"] = False
    with pytest.raises(EvidenceError, match="secondary raw evidence lacks"):
        _compact_standardized_power(tmp_path, pd.DataFrame(), em, am)


def test_stream_selects_all_64_cells_and_rejects_missing_trace_rows(tmp_path):
    # Complete synthetic scope fixture for the stream reducer only. Its helper
    # receives metadata standing in for the already-validated primary grid.
    from config import get_compact_experiment_config
    (tmp_path/"audit").mkdir()
    (tmp_path/"evaluation").mkdir()
    alternatives = [s for s in get_compact_experiment_config("demo", protocol="compact-v1").scenarios
                    if s.split == "alternative" and s.misspecification == "on_model"]
    rows = []
    for s in alternatives:
        for method in COMPACT_METHOD_ORDER[1:]:
            rows.append(dict(scenario=s.name, family=s.family, method=method, replicate=0,
                status="success", split="alternative", misspecification="on_model",
                true_change=s.true_change, change_day=s.true_change+1, horizon=s.horizon,
                alternative=True, risk_ratio=s.risk_ratio, episode_statistic=2.,
                trace_json=json.dumps([dict(day=s.true_change, status="available", raw_statistic=2.)])))
    evaluation = pd.DataFrame(rows)
    evaluation.to_csv(tmp_path/"evaluation"/"replicate_results.csv", index=False)
    audit = pd.DataFrame([dict(family=family, method=method, replicate=0, status="success",
        episode_statistic=1.) for family in ["steady", "sparse", "wave"] for method in COMPACT_METHOD_ORDER[1:]])
    audit.to_csv(tmp_path/"audit"/"replicate_results.csv", index=False)
    em = dict(synthetic_fixture=True, artifacts={"replicate_results.csv": _hash(tmp_path/"evaluation"/"replicate_results.csv")})
    am = dict(synthetic_fixture=True, artifacts={"replicate_results.csv": _hash(tmp_path/"audit"/"replicate_results.csv")})
    result, metadata = _compact_standardized_power(tmp_path, evaluation, em, am)
    assert len(result) == 64 and result.scenario.nunique() == 16 and metadata["rows"] == 64
    evaluation.iloc[1:].to_csv(tmp_path/"evaluation"/"replicate_results.csv", index=False)
    with pytest.raises(EvidenceError, match="omit or duplicate"):
        _compact_standardized_power(tmp_path, evaluation, em, am)
