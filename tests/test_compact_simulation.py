from __future__ import annotations

from dataclasses import replace
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
import simulation as sim


def test_compact_frozen_grid_and_mode_contract():
    demo = config.get_compact_experiment_config("demo", protocol="compact-v1")
    full = config.get_compact_experiment_config("full", protocol="compact-v1")
    assert demo.scientific_payload() == full.scientific_payload()
    assert (demo.replications, full.replications) == (10, 1000)
    assert demo.replications_for("calibration") == full.replications_for("calibration") == 10000
    assert demo.replications_for("size_audit") == full.replications_for("size_audit") == 5000
    assert len(sim._compact_stage_scenarios(demo, "evaluation")) == 27
    assert len(sim._compact_stage_scenarios(demo, "calibration")) == 3
    assert len(sim._compact_stage_scenarios(demo, "audit")) == 3
    assert all(s.pi_left == .02 and s.phi == 10 for s in demo.scenarios)
    assert demo.methods.n_multiplier == full.methods.n_multiplier == 10000
    assert all(s.true_change == 91 for s in demo.scenarios if "day47" not in s.name and "day137" not in s.name)
    assert sim._planned_looks(demo.scenarios[0], demo) == tuple(range(35, 183, 7))
    assert "OCFR Gaussian" not in sim._compact_method_names(next(s for s in demo.scenarios if s.split == "size_audit"))


def test_compact_independent_splits_and_exact_ascertainment_pairing():
    cfg = config.get_compact_experiment_config("demo", backend="python", protocol="compact-v1")
    scenes = {s.name: s for s in cfg.scenarios}
    seeds = [sim._component_seeds(scenes[n], cfg, 0) for n in
             ("wave_calibration", "wave_audit", "wave_null", "wave_rr150")]
    assert len({s["case"] for s in seeds}) == 4
    assert len({s["death"] for s in seeds}) == 4
    cases, deaths, _, manifest = sim._compact_generate_components(scenes["wave_null"], cfg, 0)
    for name, scale in (("wave_ascertainment75", .75), ("wave_ascertainment125", 1.25)):
        observed, other_deaths, _, streams = sim._compact_generate_components(scenes[name], cfg, 0)
        np.testing.assert_array_equal(deaths, other_deaths)
        np.testing.assert_array_equal(observed[:91], cases[:91])
        np.testing.assert_array_equal(observed[91:], cases[91:] * scale)
        assert streams["latent_case_hash"] == manifest["latent_case_hash"]


def test_compact_sensitivity_generator_native_reference_parity():
    py = config.get_compact_experiment_config("demo", backend="python", protocol="compact-v1")
    native = config.get_compact_experiment_config("demo", backend="c", protocol="compact-v1")
    for name in ("steady_null", "wave_null", "wave_delay11_rr150", "wave_binomial_null", "wave_binomial_rr150"):
        scenario = next(s for s in py.scenarios if s.name == name)
        left = sim._compact_generate_components(scenario, py, 0)
        right = sim._compact_generate_components(scenario, native, 0)
        for a, b in zip(left[:3], right[:3]):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)


def test_raw_calibration_retains_unavailable_and_uses_b_plus_one():
    frame = pd.DataFrame({"split": ["calibration"] * 20,
        "null_family": ["steady"] * 20, "method": ["OCFR empirical"] * 20,
        "replicate": range(20), "status": ["success"] * 20,
        "episode_statistic": [-np.inf] + list(range(19))})
    thresholds = sim._compact_calibrate(frame, .05, "digest", expected_count=20)
    assert thresholds.iloc[0].order_rank == 20
    assert thresholds.iloc[0].n_calibration == 20
    assert thresholds.iloc[0].critical_value == 18
    with pytest.raises(ValueError, match="allocated"):
        sim._compact_calibrate(frame.iloc[1:], .05, "digest", expected_count=20)


def test_strict_crossing_and_endpoint_decomposition():
    traces = [
        [{"day": 83, "statistic": 3., "tau_hat": 70, "status": "available"}],
        [{"day": 91, "statistic": 3., "tau_hat": 91, "status": "available"}],
        [{"day": 181, "statistic": 3., "tau_hat": 170, "status": "available"}],
        [{"day": 91, "statistic": 2., "tau_hat": 91, "status": "available"}],
        [{"day": 91, "statistic": -np.inf, "tau_hat": None, "status": "unavailable"}],
    ]
    frame = pd.DataFrame([{"null_family": "steady", "method": "OCFR empirical",
        "split": "alternative", "status": "success", "true_change": 91,
        "horizon": 182, "trace_json": json.dumps(t)} for t in traces])
    thresholds = pd.DataFrame([{"null_family": "steady", "method": "OCFR empirical",
        "critical_value": 2., "threshold_version": "test"}])
    result = sim._compact_apply_thresholds(frame, thresholds, 28)
    assert result.alarm.tolist() == [True, True, True, False, False]
    assert result.premature_alarm.tolist() == [True, False, False, False, False]
    assert result.postchange_detection.tolist() == [False, True, True, False, False]
    assert result.detected_within_28.tolist() == [False, True, False, False, False]
    np.testing.assert_array_equal(result.premature_alarm.astype(int) + result.postchange_detection.astype(int)
                                  + result.no_alarm.astype(int), np.ones(5))


def _write_threshold_fixture(output: Path, cfg):
    digest = sim._scientific_digest(cfg)[0]
    directory = output / "calibration"
    directory.mkdir(parents=True)
    thresholds = pd.DataFrame([{"null_family": "steady", "family": "steady", "method": method,
        "alpha": .05, "n_calibration": 10000, "critical_value": 10.,
        "threshold_version": "fixture", "scientific_digest": digest}
        for method in config.COMPACT_EMPIRICAL_METHODS])
    path = directory / "thresholds.csv"
    thresholds.to_csv(path, index=False)
    sim._atomic_json({"stage": "calibration", "status": "complete", "scientific_digest": digest,
        "artifacts": {"thresholds.csv": sim._file_hash(path)}}, directory / "manifest.json")


def test_compact_cold_resume_parallel_and_cache_validation(tmp_path, monkeypatch):
    # Other workers may edit unrelated scientific sources during this fixture.
    # Pin its initial fingerprint; production intentionally rejects such drift.
    source_fingerprint = sim._code_hash()
    monkeypatch.setattr(sim, "_code_hash", lambda: source_fingerprint)
    cfg = config.get_compact_experiment_config("demo", backend="python", protocol="compact-v1")
    scenario = replace(next(s for s in cfg.scenarios if s.name == "steady_null"), horizon=42,
                       change_fraction=20 / 42)
    cfg = replace(cfg, scenarios=(scenario,), replications=2)
    sequential, parallel = tmp_path / "one", tmp_path / "two"
    for output in (sequential, parallel):
        _write_threshold_fixture(output, cfg)
    first, _ = sim.run_compact_simulations(cfg, output=sequential, jobs=1)
    second, _ = sim.run_compact_simulations(cfg, output=parallel, jobs=2)
    columns = [name for name in first.columns if name != "runtime_seconds"]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    resumed, _ = sim.run_compact_simulations(cfg, output=sequential, jobs=1)
    pd.testing.assert_frame_equal(first, resumed)
    manifest = json.loads((sequential / "evaluation" / "manifest.json").read_text())
    assert manifest["tasks_computed"] == 0 and manifest["cached_tasks_loaded"] == 2
    cache = sim._cache_path(sequential / "cache", manifest["scientific_digest"], scenario.name, 0)
    payload = json.loads(cache.read_text())
    payload["rows"][0]["method"] = "wrong"
    payload["payload_hash"] = sim._payload_hash(payload)
    sim._atomic_json(payload, cache)
    assert sim._load_cache(cache, digest=manifest["scientific_digest"], scenario=scenario,
                           replicate=0, config=cfg) is None
    sentinel = sequential / "outside.txt"
    sentinel.write_text("keep")
    sim.run_compact_simulations(cfg, output=sequential, jobs=1, reset_scope="all")
    assert sentinel.read_text() == "keep"


def test_compact_requires_frozen_matching_threshold_artifact(tmp_path):
    cfg = config.get_compact_experiment_config("demo", protocol="compact-v1")
    with pytest.raises(ValueError, match="calibration"):
        sim.run_compact_simulations(cfg, output=tmp_path)
    _write_threshold_fixture(tmp_path, cfg)
    path = tmp_path / "calibration" / "thresholds.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="mismatch"):
        sim._compact_read_thresholds(path, sim._scientific_digest(cfg)[0], {"steady"})


def _synthetic_payload(scenario, replicate, cfg, digest):
    """Fixed traces test serialization/aggregation, without detector execution."""
    seeds = sim._component_seeds(scenario, cfg, replicate)
    names = sim._compact_method_names(scenario)
    rows = []
    for method in names:
        unavailable = method == "Report-day score" and replicate % 3 == 0
        trace = []
        for prefix in sim._planned_looks(scenario, cfg):
            raw = float(prefix % 13 + replicate) if not unavailable else -np.inf
            boundary = 2.5 if method == "OCFR Gaussian" else None
            trace.append({"day": prefix - 1, "manuscript_day": prefix, "raw_statistic": raw,
                "boundary": boundary, "statistic": raw - boundary if boundary else raw,
                "tau_hat": prefix - 5 if not unavailable else None,
                "direction": 1 if not unavailable else 0, "fit_status": "success",
                "status": "unavailable" if unavailable else "available"})
        best = max(trace, key=lambda look: look["statistic"])
        rows.append({"scenario": scenario.name, "family": scenario.family,
            "null_family": scenario.null_family, "split": scenario.split, "purpose": scenario.purpose,
            "method": method, "replicate": replicate, "status": "success",
            "misspecification": scenario.misspecification, "horizon": scenario.horizon,
            "risk_ratio": scenario.risk_ratio, "true_change": scenario.true_change,
            "true_change_day": scenario.true_change + 1, "case_seed": seeds["case"],
            "death_seed": seeds["death"], "multiplier_seed": seeds["multiplier"],
            "episode_statistic": best["statistic"], "tau_hat_at_max": best["tau_hat"],
            "all_looks_unavailable": unavailable, "unavailable_looks": len(trace) if unavailable else 0,
            "trace": trace})
    streams = {"rng_contract_version": cfg.rng_contract_version, "seeds": seeds,
        "groups": {"case": scenario.case_stream_group, "death": scenario.outcome_stream_group,
                   "multiplier": scenario.name},
        "hashes": {key: hashlib.sha256(f"fixture:{replicate}:{key}".encode()).hexdigest()
            for key in ("cases", "deaths", "true_delay", "assumed_delay", "multipliers")}}
    payload = {"schema_version": sim.SCHEMA_VERSION, "scientific_digest": digest,
        "scenario_digest": sim._scenario_digest(scenario), "scenario": scenario.name,
        "replicate": replicate, "expected_methods": names, "runtime_seconds": replicate + .125,
        "stream_manifest": streams, "rows": rows}
    payload["payload_hash"] = sim._payload_hash(payload)
    return payload


def _aggregation_fixture(tmp_path, monkeypatch, replications=9):
    fingerprint = sim._code_hash()
    monkeypatch.setattr(sim, "_code_hash", lambda: fingerprint)
    cfg = config.get_compact_experiment_config("demo", protocol="compact-v1")
    scenario = next(s for s in cfg.scenarios if s.name == "steady_rr150")
    cfg = replace(cfg, scenarios=(scenario,), replications=replications)
    _write_threshold_fixture(tmp_path, cfg)
    monkeypatch.setattr(sim, "_compact_evaluate_one", _synthetic_payload)
    return cfg


def test_batched_artifact_equals_ordinary_transform_and_derived_resume(tmp_path, monkeypatch):
    cfg = _aggregation_fixture(tmp_path, monkeypatch)
    sizes = []
    ordinary_flatten = sim._flatten
    def measured_flatten(payloads):
        sizes.append(len(payloads))
        return ordinary_flatten(payloads)
    monkeypatch.setattr(sim, "_flatten", measured_flatten)
    full, thresholds = sim.run_compact_simulations(cfg, output=tmp_path, aggregation_batch_size=2)
    assert max(sizes) == 2 and sum(sizes) == 9
    assert {"trace_json", "stream_manifest_json"}.issubset(full.columns)
    assert full.case_seed.dtype == np.dtype("uint64")
    assert int(full.case_seed.max()) > 2**63
    assert full.tau_hat_at_max.isna().any() and full.all_looks_unavailable.any()
    digest = sim._scientific_digest(cfg)[0]
    payloads = [_synthetic_payload(cfg.scenarios[0], r, cfg, digest) for r in range(9)]
    ordinary = sim._compact_apply_thresholds(ordinary_flatten(payloads).sort_values(
        ["scenario", "replicate", "method"], ignore_index=True), thresholds, cfg.dgp.timely_window)
    for column in ("stream_manifest_json", "trace_json"):
        ordinary[column] = ordinary[column].map(lambda text: json.dumps(json.loads(text),
            sort_keys=True, separators=(",", ":")))
    ordinary = ordinary.reindex(sorted(ordinary.columns), axis=1)
    expected = tmp_path / "ordinary.csv"
    sim._atomic_csv(ordinary, expected)
    result = tmp_path / "evaluation/replicate_results.csv"
    before = result.read_bytes()
    assert before == expected.read_bytes()
    pd.testing.assert_frame_equal(full, ordinary)
    def no_reexecution(*args):
        raise AssertionError("derived reset must not run a detector")
    monkeypatch.setattr(sim, "_compact_evaluate_one", no_reexecution)
    for batch_size in (1, 3, 20):
        slim, _ = sim.run_compact_simulations(cfg, output=tmp_path, reset_scope="derived",
            aggregation_batch_size=batch_size, return_mode="slim")
        assert not {"trace_json", "stream_manifest_json"}.intersection(slim.columns)
        pd.testing.assert_frame_equal(slim, full.drop(columns=["trace_json", "stream_manifest_json"]))
        assert result.read_bytes() == before
        manifest = json.loads((tmp_path / "evaluation/manifest.json").read_text())
        assert manifest["status"] == "complete" and manifest["tasks_computed"] == 0
        assert manifest["cached_tasks_loaded"] == 9
        assert manifest["aggregation"]["batch_episodes"] == batch_size


def test_compact_limits_pending_futures_and_cli_returns_slim(tmp_path, monkeypatch):
    cfg = _aggregation_fixture(tmp_path, monkeypatch, replications=11)
    outstanding = peak = 0
    class TrackedFuture(Future):
        def result(self, *args, **kwargs):
            nonlocal outstanding
            outstanding -= 1
            return super().result(*args, **kwargs)
    class TrackedPool:
        def __init__(self, max_workers):
            assert max_workers == 2
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, function, *args):
            nonlocal outstanding, peak
            outstanding += 1
            peak = max(peak, outstanding)
            future = TrackedFuture()
            future.set_result(function(*args))
            return future
    monkeypatch.setattr(sim, "ProcessPoolExecutor", TrackedPool)
    result, _ = sim.run_compact_simulations(cfg, output=tmp_path, jobs=2,
                                           aggregation_batch_size=3, return_mode="slim")
    assert len(result) == 55 and outstanding == 0 and peak == 4
    def cli_stub(config, **kwargs):
        assert kwargs["return_mode"] == "slim"
        return pd.DataFrame(), pd.DataFrame()
    monkeypatch.setattr(sim, "run_compact_simulations", cli_stub)
    assert sim.main(["--protocol", "compact-v1", "--output", str(tmp_path)]) == 0


def test_aggregation_cache_loss_cannot_mark_stage_complete(tmp_path, monkeypatch):
    cfg = _aggregation_fixture(tmp_path, monkeypatch, replications=2)
    original_write = sim._compact_write_results
    def lose_cache(config, selected, cache_root, digest, *args):
        sim._cache_path(cache_root, digest, selected[0].name, 0).unlink()
        return original_write(config, selected, cache_root, digest, *args)
    monkeypatch.setattr(sim, "_compact_write_results", lose_cache)
    with pytest.raises(RuntimeError, match="cache became unavailable"):
        sim.run_compact_simulations(cfg, output=tmp_path, return_mode="slim")
    manifest = json.loads((tmp_path / "evaluation/manifest.json").read_text())
    assert manifest["status"] == "running" and "failed_method_rows" not in manifest
