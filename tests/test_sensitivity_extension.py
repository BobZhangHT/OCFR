"""Ten-cell extension contract, immutable-parent validation and bounded resume."""
from concurrent.futures import Future
from dataclasses import replace
import ast
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

import config
import simulation as sim


def test_ten_cell_grid_and_demo_full_contract():
    original = config.get_compact_release_study_config(backend="python")
    before = original.scientific_payload()
    full = config.get_sensitivity_extension_config(original)
    demo = config.get_sensitivity_extension_config(replace(original, mode="demo", replications=10))
    assert full.scientific_payload() == demo.scientific_payload()
    assert original.scientific_payload() == before
    assert len(sim._compact_stage_scenarios(original, "evaluation")) == 27
    assert len(full.scenarios) == 10
    assert not ({s.name for s in full.scenarios} & {s.name for s in original.scenarios})
    assert {s.true_delay_mean_days for s in full.scenarios if s.misspecification == "delay_error"} == {12., 13., 15., 16.}
    assert all(s.true_change == 91 and s.case_stream_group == "wave" and s.assumed_delay_mean_days == 14 for s in full.scenarios)
    assert all(s.true_delay_shape == 4 and s.assumed_delay_shape == 4 for s in full.scenarios)
    assert full.master_seed == 2026111404 and full.methods.n_multiplier == 1999
    assert len(sim._planned_looks(full.scenarios[0], full)) == 22
    assert full.stage_methods("evaluation") == ("OCFR", "NB-GLR", "Page-CUSUM")
    assert {full.replications_for(s.split) for s in full.scenarios} == {1000}
    assert {demo.replications_for(s.split) for s in demo.scenarios} == {10}
    assert config.get_compact_experiment_config("full").methods.backend == "c"


def test_new_ascertainment_pairing_and_delay_stream_contract():
    base = config.get_compact_experiment_config("demo", backend="python")
    cfg = config.get_sensitivity_extension_config(base)
    control = next(s for s in base.scenarios if s.name == "wave_null")
    cases, deaths, _, streams = sim._compact_generate_components(control, base, 3)
    for scenario in cfg.scenarios:
        seeds = sim._component_seeds(scenario, cfg, 3)
        group_control = next(s for s in base.scenarios if s.name == ("wave_null" if scenario.split == "holdout" else "wave_rr150"))
        assert seeds["case"] == sim._component_seeds(group_control, base, 3)["case"]
        if scenario.misspecification.startswith("exposure_scale_"):
            scale = float(scenario.misspecification.split("_")[-1])
            observed, generated_deaths, _, other = sim._compact_generate_components(scenario, cfg, 3)
            np.testing.assert_array_equal(deaths, generated_deaths)
            np.testing.assert_array_equal(cases[:91], observed[:91])
            np.testing.assert_array_equal(cases[91:] * scale, observed[91:])
            assert other["latent_case_hash"] == streams["latent_case_hash"]
            assert scenario.outcome_stream_group == "wave_null"
            assert seeds["death"] == streams["seeds"]["death"]
        else:
            assert scenario.outcome_stream_group == scenario.name
            assert seeds["death"] != sim._component_seeds(group_control, base, 3)["death"]


@pytest.fixture
def parent(tmp_path, monkeypatch):
    """A real complete demo grid with lightweight raw columns and exact sources."""
    cfg = config.get_compact_experiment_config("demo", backend="python")
    base, source = tmp_path/"parent", tmp_path/"source"
    code_hash, records = sim._code_hash()
    # Freeze this fixture's source identity while independent agents work.
    monkeypatch.setattr(sim, "_code_hash", lambda: (code_hash, records))
    for name in records:
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(sim.__file__).parent/name, target)
    digest = sim._scientific_digest(cfg)[0]
    for stage in ("calibration", "evaluation"):
        directory = base/stage
        directory.mkdir(parents=True)
        scenarios = sim._compact_stage_scenarios(cfg, stage)
        methods = cfg.stage_methods(stage)
        counts = {s.name: cfg.replications_for(s.split) for s in scenarios}
        pd.DataFrame([dict(scenario=s.name, replicate=r, method=m, status="success")
            for s in scenarios for r in range(counts[s.name]) for m in methods]).to_csv(directory/"replicate_results.csv", index=False)
        artifacts = {"replicate_results.csv": sim._file_hash(directory/"replicate_results.csv")}
        if stage == "calibration":
            pd.DataFrame([dict(null_family=f, method=m, critical_value=2.,
                threshold_version="fixture-strict", scientific_digest=digest, n_calibration=10)
                for f in ("steady", "sparse", "wave") for m in methods]).to_csv(directory/"thresholds.csv", index=False)
            artifacts["thresholds.csv"] = sim._file_hash(directory/"thresholds.csv")
        manifest = dict(stage=stage, status="complete", mode="demo", protocol=cfg.protocol,
            scientific_config=cfg.scientific_payload(), scientific_digest=digest,
            code_hash=code_hash, code_files=records, methods=list(methods),
            scenarios=[s.name for s in scenarios], replications_by_scenario=counts,
            artifacts=artifacts)
        if stage == "evaluation":
            manifest["threshold_artifact"] = dict(sha256=sim._file_hash(base/"calibration"/"thresholds.csv"))
        sim._atomic_json(manifest, directory/"manifest.json")
    return base, source, cfg


def test_parent_source_and_threshold_tampering_rejected(parent, monkeypatch):
    base, source, cfg = parent
    loaded, thresholds, provenance = sim._sensitivity_base_inputs(base, base_source=source)
    assert loaded.scientific_payload() == cfg.scientific_payload()
    assert len(thresholds) == 6
    assert provenance["threshold_artifact"]["producer_scientific_digest"] == sim._scientific_digest(cfg)[0]
    threshold = base/"calibration"/"thresholds.csv"
    original = threshold.read_bytes()
    threshold.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        sim._sensitivity_base_inputs(base, base_source=source)
    threshold.write_bytes(original)
    (source/"methods.py").write_text("tampered")
    with pytest.raises(ValueError, match="source hash mismatch"):
        sim._sensitivity_base_inputs(base, base_source=source)


def test_numeric_method_and_transitive_ast_rejected(parent, monkeypatch):
    _, source, _ = parent
    code, records = sim._code_hash()
    changed = dict(records, **{"methods.py": "0" * 64})
    monkeypatch.setattr(sim, "_code_hash", lambda: (code, changed))
    with pytest.raises(ValueError, match="methods/C"):
        sim._sensitivity_source_compatibility(source, records)
    monkeypatch.setattr(sim, "_code_hash", lambda: (code, records))
    # Reconstruct a pre-extension snapshot: the checker must accept pure additions.
    changed = dict(records)
    for filename in ("config.py", "simulation.py"):
        path = source/filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
            (node.name.startswith("_sensitivity_") or node.name in {
                "get_sensitivity_extension_config", "run_sensitivity_extension", "load_sensitivity_extension"}))
            and not (isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "SENSITIVITY_EXTENSION_VERSION" for target in node.targets))]
        path.write_text(ast.unparse(tree), encoding="utf-8")
        changed[filename] = sim._file_hash(path)
    assert sim._sensitivity_source_compatibility(source, changed)["ast_checks"]["simulation.py"]["all_original_non_cli_nodes_identical"]
    # A correctly rehashed snapshot with a changed helper must still fail AST checks.
    path = source/"simulation.py"
    text = path.read_text(encoding="utf-8").replace('return int.from_bytes(hashlib.sha256', 'return 1 + int.from_bytes(hashlib.sha256', 1)
    path.write_text(text, encoding="utf-8")
    changed["simulation.py"] = sim._file_hash(path)
    with pytest.raises(ValueError, match="AST mismatch"):
        sim._sensitivity_source_compatibility(source, changed)


def fake_episode(scenario, rep, cfg, digest):
    seeds = sim._component_seeds(scenario, cfg, rep)
    streams = dict(rng_contract_version=cfg.rng_contract_version, seeds=seeds,
        groups=dict(case=scenario.case_stream_group, death=scenario.outcome_stream_group, multiplier=scenario.name),
        hashes={name: "a"*64 for name in ("cases", "deaths", "true_delay", "assumed_delay", "multipliers")})
    rows = [dict(scenario=scenario.name, family="wave", null_family="wave", replicate=rep,
        method=m, status="success", split=scenario.split, true_change=91, horizon=182,
        all_looks_unavailable=False, trace=[dict(day=97, tau_hat=91, statistic=1. if m == "OCFR" else 3.,
            status="available", boundary=0.)]) for m in cfg.stage_methods("evaluation")]
    payload = dict(schema_version=sim.SCHEMA_VERSION, scientific_digest=digest,
        scenario_digest=sim._scenario_digest(scenario), scenario=scenario.name, replicate=rep,
        expected_methods=list(cfg.stage_methods("evaluation")), stream_manifest=streams,
        runtime_seconds=0., rows=rows)
    payload["payload_hash"] = sim._payload_hash(payload)
    return payload


class ImmediatePool:
    """Deterministic executor exercising the production bounded scheduling branch."""
    def __init__(self, **kwargs):
        self.futures = []
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)
        self.futures.append(future)
        return future


def test_extension_resume_parallel_reset_and_artifact_validation(parent, tmp_path, monkeypatch):
    base, source, _ = parent
    base_hashes = {str(p.relative_to(base)): sim._file_hash(p) for p in base.rglob("*") if p.is_file()}
    calls, pending_sizes = [], []
    def evaluate(*args):
        calls.append((args[0].name, args[1]))
        return fake_episode(*args)
    original_wait = sim.wait
    def bounded_wait(pending, **kwargs):
        pending_sizes.append(len(pending))
        return original_wait(pending, **kwargs)
    monkeypatch.setattr(sim, "_compact_evaluate_one", evaluate)
    monkeypatch.setattr(sim, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(sim, "wait", bounded_wait)
    sequential, parallel = tmp_path/"one", tmp_path/"two"
    kwargs = dict(base_output=base, base_source=source, aggregation_batch_size=7)
    first = sim.run_sensitivity_extension(output=sequential, **kwargs)
    assert len(first) == 300 and len(calls) == 100
    second = sim.run_sensitivity_extension(output=parallel, jobs=3, **kwargs)
    pd.testing.assert_frame_equal(first, second)
    assert max(pending_sizes) <= 6
    calls.clear()
    resumed = sim.run_sensitivity_extension(output=sequential, **kwargs)
    assert not calls
    pd.testing.assert_frame_equal(first, resumed)
    manifest_path = sequential/"sensitivity-extension"/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["tasks_computed"] == 0 and manifest["cached_tasks_loaded"] == 100
    cache = next((sequential/"sensitivity-extension"/"cache").rglob("00000.json"))
    payload = json.loads(cache.read_text())
    payload["rows"][0]["status"] = "failure"
    sim._atomic_json(payload, cache)
    sim.run_sensitivity_extension(output=sequential, **kwargs)
    assert len(calls) == 1
    calls.clear()
    sentinel = sequential/"keep.txt"
    sentinel.write_text("keep")
    sim.run_sensitivity_extension(output=sequential, reset_scope="all", **kwargs)
    assert len(calls) == 100 and sentinel.read_text() == "keep"
    cfg, frame, manifest = sim.load_sensitivity_extension(sequential, base_output=base, base_source=source)
    assert len(frame) == 300 and len(cfg.scenarios) == 10
    assert not {"trace_json", "stream_manifest_json"} & set(frame.columns)
    _, full, _ = sim.load_sensitivity_extension(sequential, base_output=base, base_source=source, return_mode="full")
    assert {"trace_json", "stream_manifest_json"} <= set(full.columns)
    with pytest.raises(ValueError, match="return mode"):
        sim.load_sensitivity_extension(sequential, base_output=base, base_source=source, return_mode="invalid")
    assert manifest["threshold_artifact"]["sha256"] == sim._file_hash(base/"calibration"/"thresholds.csv")
    assert base_hashes == {str(p.relative_to(base)): sim._file_hash(p) for p in base.rglob("*") if p.is_file()}
    raw = sequential/"sensitivity-extension"/"replicate_results.csv"
    raw.write_bytes(raw.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="raw artifact hash mismatch"):
        sim.load_sensitivity_extension(sequential, base_output=base, base_source=source)
    with pytest.raises(ValueError, match="disjoint"):
        sim.run_sensitivity_extension(output=base, **kwargs)


def test_infrastructure_failure_is_incomplete_and_full_requires_full_parent(parent, tmp_path, monkeypatch):
    base, source, _ = parent
    with pytest.raises(ValueError, match="full parent"):
        sim.run_sensitivity_extension(base_output=base, base_source=source, output=tmp_path/"full", mode="full")
    def fail(*args):
        raise OSError("fixture infrastructure failure")
    monkeypatch.setattr(sim, "_compact_evaluate_one", fail)
    output = tmp_path/"failed"
    with pytest.raises(OSError, match="infrastructure"):
        sim.run_sensitivity_extension(base_output=base, base_source=source, output=output)
    manifest = json.loads((output/"sensitivity-extension"/"manifest.json").read_text())
    assert manifest["status"] == "running"
    assert not (output/"sensitivity-extension"/"replicate_results.csv").exists()


def test_cli_inherits_parent_mode_and_rejects_overrides(tmp_path, monkeypatch):
    received = []
    monkeypatch.setattr(sim, "run_sensitivity_extension", lambda **kwargs: received.append(kwargs) or [])
    common = ["--stage", "sensitivity-extension", "--base-study", str(tmp_path/"base"), "--output", str(tmp_path/"new"), "--base-code", str(tmp_path/"source")]
    assert sim.main(common) == 0
    assert received[-1]["mode"] is None
    assert sim.main(common+["--mode", "demo"]) == 0
    assert received[-1]["mode"] == "demo"
    with pytest.raises(SystemExit, match="inherits"):
        sim.main(common+["--backend", "python"])
