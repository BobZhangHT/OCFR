"""Synthetic-only checks of compact denominators, pairing, and evidence gates."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from analysis import (COMPACT_METHOD_ORDER, EvidenceError, analyze_compact_simulation,
                      compact_binomial_interval, compact_summaries, _compact_read_summary_csv,
                      _compact_export_metadata)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root):
    """Not study evidence: three deliberately chosen episodes per method."""
    scientific = {"protocol": "compact-v1", "scenarios": []}
    digest = hashlib.sha256(json.dumps({"schema_version": 3, "scientific": scientific},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:20]
    for stage in ("calibration", "audit", "evaluation"):
        directory = root / stage
        directory.mkdir()
        methods = COMPACT_METHOD_ORDER if stage == "evaluation" else COMPACT_METHOD_ORDER[1:]
        scenarios = ["steady_null", "steady_rr150"] if stage == "evaluation" else [f"steady_{stage}"]
        rows = []
        for scenario in scenarios:
            alternative = "rr150" in scenario
            for method in methods:
                # At tau, one day premature, absent. Report-day has no at-tau alarm.
                alarms = [91., 90., np.nan] if alternative else [np.nan]*3
                if method == "Report-day score" and alternative:
                    alarms[0] = np.nan
                for rep, alarm in enumerate(alarms):
                    rows.append(dict(scenario=scenario, family="steady", method=method,
                        replicate=rep, status="success", alarm_day=alarm, true_change=91,
                        true_change_day=92, horizon=182, risk_ratio=1.5 if alternative else 1.,
                        split="alternative" if alternative else "holdout",
                        misspecification="on_model", unavailable_looks=22 if rep == 2 else 0,
                        scientific_digest=digest))
        pd.DataFrame(rows).to_csv(directory / "replicate_results.csv", index=False)
        artifacts = {"replicate_results.csv": _hash(directory / "replicate_results.csv")}
        if stage == "calibration":
            thresholds = pd.DataFrame([dict(family="steady", method=method,
                n_calibration=3, critical_value=1., scientific_digest=digest)
                for method in methods])
            thresholds.to_csv(directory / "thresholds.csv", index=False)
            artifacts["thresholds.csv"] = _hash(directory / "thresholds.csv")
        manifest = dict(protocol="compact-v1", stage=stage, status="complete",
            synthetic_fixture=True, schema_version=3, scientific_config=scientific,
            scientific_digest=digest, scenarios=scenarios, methods=list(methods),
            replications_by_scenario={s:3 for s in scenarios}, artifacts=artifacts)
        if stage != "calibration":
            manifest["threshold_artifact"] = dict(sha256=_hash(root/"calibration"/"thresholds.csv"))
        (directory/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _rewrite_stage(root, stage, mutate):
    directory = root/stage
    path = directory/"replicate_results.csv"
    frame = mutate(pd.read_csv(path))
    frame.to_csv(path, index=False)
    manifest = json.loads((directory/"manifest.json").read_text())
    manifest["artifacts"][path.name] = _hash(path)
    (directory/"manifest.json").write_text(json.dumps(manifest))


def test_unconditional_power_and_delay_include_premature_absent_and_unavailable(tmp_path):
    _fixture(tmp_path)
    tables = analyze_compact_simulation(tmp_path)
    summary = tables["compact_summary"]
    row = summary[(summary.scenario == "steady_rr150") &
        (summary.method == "OCFR empirical") & (summary.metric == "useful_power")].iloc[0]
    assert row.n == 3 and row.events == 1 and row.estimate == pytest.approx(1/3)
    assert row.premature_alarm_n == 1 and row.absent_alarm_n == 1
    curve = tables["compact_cumulative_detection"]
    curve = curve[(curve.method == "OCFR empirical") & (curve.scenario == "steady_rr150")]
    assert set(curve.n) == {3}
    assert set(curve.events) == {1}  # exact-tau alarm is useful at d=0
    assert len(curve) == 91
    failures = tables["compact_failures"]
    assert failures.invalid_looks.sum() > 0
    assert not (tmp_path/"analysis"/"figures").exists()
    report = json.loads((tmp_path/"analysis"/"compact_analysis_manifest.json").read_text())
    assert report["synthetic_fixture"] and report["status"] == "complete"


def test_paired_mcse_uses_episode_discordance(tmp_path):
    _fixture(tmp_path)
    paired = analyze_compact_simulation(tmp_path)["compact_paired_contrasts"]
    row = paired[(paired.scenario == "steady_rr150") &
        (paired.method_a == "OCFR empirical") & (paired.method_b == "Report-day score") &
        (paired.metric == "useful_power")].iloc[0]
    assert row.n == 3 and row.a_only_n == 1 and row.b_only_n == 0
    assert row.difference == pytest.approx(1/3)
    assert row.paired_mcse == pytest.approx(1/3)


def test_recorded_failure_stays_in_denominator(tmp_path):
    _fixture(tmp_path)
    def mutate(frame):
        frame.loc[frame.replicate == 2, "status"] = "failed:synthetic"
        return frame
    _rewrite_stage(tmp_path, "evaluation", mutate)
    tables = analyze_compact_simulation(tmp_path)
    useful = tables["compact_summary"].query("metric == 'useful_power'")
    assert set(useful.n) == {3}
    assert set(useful.failed_n) == {1}


@pytest.mark.parametrize("error", ["hash", "missing", "duplicate", "stage", "threshold", "failed_alarm"])
def test_evidence_rejects_corruption_and_incomplete_jobs(tmp_path, error):
    _fixture(tmp_path)
    directory = tmp_path/"evaluation"
    manifest_path = directory/"manifest.json"
    if error == "hash":
        with (directory/"replicate_results.csv").open("a") as handle:
            handle.write("\n")
    elif error == "missing":
        _rewrite_stage(tmp_path, "evaluation", lambda frame: frame.iloc[1:])
    elif error == "duplicate":
        _rewrite_stage(tmp_path, "evaluation", lambda frame: pd.concat([frame, frame.iloc[:1]]))
    elif error == "failed_alarm":
        def mutate(frame):
            frame.loc[frame.alarm_day.notna(), "status"] = "failed:synthetic"
            return frame
        _rewrite_stage(tmp_path, "evaluation", mutate)
    else:
        manifest = json.loads(manifest_path.read_text())
        if error == "stage":
            manifest["status"] = "running"
        else:
            manifest["threshold_artifact"]["sha256"] = "0"*64
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EvidenceError):
        analyze_compact_simulation(tmp_path)


def test_synthetic_fixture_cannot_export_publication_figures(tmp_path):
    _fixture(tmp_path)
    with pytest.raises(EvidenceError, match="synthetic"):
        analyze_compact_simulation(tmp_path, figure_dir=tmp_path/"manuscript")
    assert not (tmp_path/"manuscript").exists()


def test_all_unavailable_episode_cannot_alarm(tmp_path):
    _fixture(tmp_path)
    def mutate(frame):
        frame["all_looks_unavailable"] = frame.alarm_day.notna()
        return frame
    _rewrite_stage(tmp_path, "evaluation", mutate)
    with pytest.raises(EvidenceError, match="unavailable"):
        analyze_compact_simulation(tmp_path)


def test_validation_failure_removes_stale_completion_marker(tmp_path):
    _fixture(tmp_path)
    analyze_compact_simulation(tmp_path)
    report = tmp_path/"analysis"/"compact_analysis_manifest.json"
    assert report.exists()
    _rewrite_stage(tmp_path, "evaluation", lambda frame: frame.iloc[1:])
    with pytest.raises(EvidenceError):
        analyze_compact_simulation(tmp_path)
    assert not report.exists()


def test_exact_binomial_endpoint_cases():
    lo, hi = compact_binomial_interval(0, 3)
    assert lo == 0 and hi == pytest.approx(1-.025**(1/3))
    lo, hi = compact_binomial_interval(3, 3)
    assert hi == 1 and lo == pytest.approx(.025**(1/3))
    with pytest.raises(ValueError):
        compact_binomial_interval(0, 0)


def test_chunked_metadata_preserves_seed_ids_and_omits_long_traces(tmp_path):
    path = tmp_path/"long_raw.csv"
    values = [1, 2**63+1, 2**64-1]
    pd.DataFrame({"replicate": range(3), "case_seed": values,
        "death_seed": list(reversed(values)), "alarm_day": [91., np.nan, 97.],
        "trace_json": ["x"*100000]*3}).to_csv(path, index=False)
    frame = _compact_read_summary_csv(path,
        {"replicate", "case_seed", "death_seed", "alarm_day"}, chunksize=1)
    assert frame.case_seed.tolist() == [str(value) for value in values]
    assert frame.death_seed.tolist() == [str(value) for value in reversed(values)]
    assert frame.replicate.tolist() == [0, 1, 2]
    assert frame.alarm_day.iloc[[0,2]].tolist() == [91., 97.]
    assert pd.isna(frame.alarm_day.iloc[1])
    assert "trace_json" not in frame


def test_publication_export_metadata_validates_actual_rasters(tmp_path):
    image = Image.new("RGB", (384, 468), "white")
    image.save(tmp_path/"CompactFig3_detection.png", dpi=(600, 600))
    image.save(tmp_path/"Fig3.tif", dpi=(600, 600), compression="tiff_lzw")
    metadata = _compact_export_metadata(tmp_path, ["CompactFig3_detection"])
    figure = metadata["figures"]["CompactFig3_detection"]
    assert figure["physical_inches"] == [.64, .78]
    assert figure["tiff"]["mode"] == "RGB" and figure["tiff"]["compression"] == 5
    json.dumps(metadata)
    image.save(tmp_path/"Fig3.tif", dpi=(300, 300), compression="tiff_lzw")
    with pytest.raises(EvidenceError, match="publication export"):
        _compact_export_metadata(tmp_path, ["CompactFig3_detection"])
