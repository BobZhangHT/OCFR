"""Synthetic denominator and provenance tests; these are not study outcomes."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as a
import config


def row(rep=0, alarm=True, fit=True, interval=True):
    result = dict(scenario="test", family="steady", method="OCFR Gaussian", replicate=rep,
        true_change=91, horizon=182, alarm=alarm, alarm_day=90 if alarm else np.nan,
        tau_hat_at_alarm=85 if alarm else np.nan, fit_success=fit, confidence=.95,
        pi_left=.02, pi_right=.03, risk_ratio=1.5)
    for target in a.ESTIMATION_TARGETS:
        truth = result[target]
        result.update({f"{target}_estimate": truth*1.1 if alarm else np.nan,
            f"{target}_point_valid": alarm and fit,
            f"{target}_lower": truth*.9 if alarm and interval else np.nan,
            f"{target}_upper": truth*1.2 if alarm and interval else np.nan,
            f"{target}_interval_valid": alarm and fit and interval})
    return result


def test_distinct_denominators_premature_failed_fit_and_interval_failure():
    rows = [row(0), row(1, interval=False), row(2, fit=False, interval=False), row(3, alarm=False)]
    rows[1]["pi_left_estimate"] = .024
    result = a.compact_estimation_summaries(pd.DataFrame(rows))
    s = result["compact_estimation_summary"].set_index("target").loc["pi_left"]
    assert (s.N_requested, s.n_alarm, s.n_no_alarm, s.n_point, s.n_interval, s.n_premature) == (4, 3, 1, 2, 1, 3)
    assert s.relative_rmse == pytest.approx(np.sqrt((.1**2+.2**2)/2))
    assert s.relative_bias == pytest.approx(.15)
    assert s.all_alarm_containment == pytest.approx(1/3)
    assert s.valid_only_containment == 1
    assert s.missing_interval_sensitivity_upper == 1
    assert (s.containment_ci_low, s.containment_ci_high) == pytest.approx(a.compact_binomial_interval(1, 3))
    loc = result["compact_estimation_location"].iloc[0]
    assert loc.n_location == 3 and loc.location_mae_days == 6 and loc.location_bias_days == -6


def test_zero_alarm_is_unavailable_not_zero():
    result = a.compact_estimation_summaries(pd.DataFrame([row(alarm=False)]))
    s = result["compact_estimation_summary"]
    assert s[["relative_rmse", "all_alarm_containment", "containment_ci_low", "median_relative_width"]].isna().all().all()
    assert np.isnan(result["compact_estimation_location"].iloc[0].location_mae_days)


def test_target_specific_overflow_and_finite_extreme_width_are_preserved():
    rows = [row(0), row(1)]
    rows[0].update(risk_ratio_upper=np.inf, risk_ratio_interval_valid=False)
    rows[1].update(risk_ratio_upper=1e100)
    result = a.compact_estimation_summaries(pd.DataFrame(rows))["compact_estimation_summary"].set_index("target")
    assert result.loc["pi_left", "n_interval"] == 2
    assert result.loc["risk_ratio", "n_interval"] == 1
    assert result.loc["risk_ratio", "width_max"] == pytest.approx(1e100)
    assert result.loc["risk_ratio", "median_relative_width"] == pytest.approx(1e100/1.5)


def fixture(root, mode="demo"):
    cfg = config.get_compact_experiment_config("full", backend="python", protocol="compact-v1")
    selected = [s for s in cfg.scenarios if s.split == "alternative" and s.misspecification == "on_model"]
    n = 10 if mode == "demo" else 1000
    rows = []
    for scenario in selected:
        for rep in range(n):
            r = row(rep)
            r.update(scenario=scenario.name, family=scenario.family, true_change=scenario.true_change,
                pi_left=scenario.pi_left, pi_right=scenario.pi_right, risk_ratio=scenario.risk_ratio)
            # Valid points/intervals can be numerically poor and are retained.
            rows.append(r)
    frame = pd.DataFrame(rows)
    evaluation = frame.copy()
    evaluation["alarm_day"] += 1
    evaluation["alternative"] = True
    evaluation["misspecification"] = "on_model"
    evaluation["any_alarm"] = True
    original = dict(scientific_digest="frozen0123456789", scientific_config=cfg.scientific_payload(),
        artifacts={a.RAW_FILE:"a"*64})
    (root/"evaluation").mkdir()
    (root/"evaluation"/a.MANIFEST_FILE).write_text(json.dumps(original))
    directory = root/"estimation"
    directory.mkdir()
    frame.to_csv(directory/a.RAW_FILE, index=False)
    records = {"simulation.py":"b"*64}
    manifest = dict(stage="estimation", status="complete", mode=mode,
        original_manifest_sha256=a._sha256(root/"evaluation"/a.MANIFEST_FILE),
        original_csv_sha256="a"*64, original_scientific_digest=original["scientific_digest"],
        scientific_digest=original["scientific_digest"], scientific_config=original["scientific_config"],
        postprocessor_code_files=records,
        postprocessor_code_hash=hashlib.sha256(a._canonical_json(records).encode()).hexdigest(),
        scenarios=[s.name for s in selected], replications_by_scenario={s.name:n for s in selected},
        artifacts={a.RAW_FILE:a._sha256(directory/a.RAW_FILE)})
    (directory/a.MANIFEST_FILE).write_text(json.dumps(manifest))
    return evaluation, original, manifest


@pytest.mark.parametrize("mode", ["demo", "full"])
def test_complete_same_scenario_grid_both_modes(tmp_path, mode):
    evaluation, original, _ = fixture(tmp_path, mode)
    tables, metadata = a._compact_load_estimation(tmp_path, evaluation, original)
    assert len(tables["compact_estimation_summary"]) == 48
    assert len(tables["compact_estimation_location"]) == 16
    assert metadata["confidence_level"] == .95
    assert set(tables["compact_estimation_summary"].N_requested) == {10 if mode == "demo" else 1000}


@pytest.mark.parametrize("bad", ["incomplete", "digest", "source_hash", "csv_hash", "grid"])
def test_rejects_incomplete_or_unbound_evidence(tmp_path, bad):
    evaluation, original, manifest = fixture(tmp_path)
    directory = tmp_path/"estimation"
    if bad == "incomplete":
        manifest["status"] = "running"
    elif bad == "digest":
        manifest["original_scientific_digest"] = "different"
    elif bad == "source_hash":
        manifest["postprocessor_code_hash"] = "c"*64
    elif bad == "csv_hash":
        manifest["artifacts"][a.RAW_FILE] = "c"*64
    else:
        frame = pd.read_csv(directory/a.RAW_FILE).iloc[1:]
        frame.to_csv(directory/a.RAW_FILE, index=False)
        manifest["artifacts"][a.RAW_FILE] = a._sha256(directory/a.RAW_FILE)
    (directory/a.MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(a.EvidenceError):
        a._compact_load_estimation(tmp_path, evaluation, original)


def test_missing_location_is_integrity_error():
    r = row()
    r["tau_hat_at_alarm"] = np.nan
    with pytest.raises(a.EvidenceError, match="location"):
        a.compact_estimation_summaries(pd.DataFrame([r]))


def test_figure_contract_without_study_data(tmp_path, monkeypatch):
    evaluation, original, _ = fixture(tmp_path)
    tables, _ = a._compact_load_estimation(tmp_path, evaluation, original)
    captures = {}
    def capture(fig, directory, name):
        captures.update(size=fig.get_size_inches(), axes=fig.axes, name=name)
        a.plt.close(fig)
    monkeypatch.setattr(a, "_compact_save", capture)
    caption = a._compact_estimation_figure(tables, tmp_path)
    assert captures["name"] == "CompactFig5_estimation"
    np.testing.assert_allclose(captures["size"], [6.4, 7.8])
    assert len(captures["axes"]) == 4
    assert captures["axes"][1].get_ylim() == (0, 1.04)
    assert "premature" in caption and "nominal reference" in caption
