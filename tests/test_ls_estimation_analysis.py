"""Portable synthetic checks for the point-only full-delay LS supplement."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as a
import config


def episode(rep=0, alarm=True, scenario="test"):
    row = dict(scenario=scenario, family="steady", replicate=rep, true_change=91,
        alarm=alarm, alarm_day=90 if alarm else np.nan, tau_hat_at_alarm=85 if alarm else np.nan,
        pi_left=.02, pi_right=.03, risk_ratio=1.5, case_hash="c"*64, death_hash="d"*64,
        fit_success=False, ls_status="success" if alarm else "no_alarm",
        ls_seconds=.01 if alarm else 0., ls_rss=10. if alarm else np.nan,
        ls_active_set="interior" if alarm else "none", ls_kkt_residual=0. if alarm else np.nan,
        ls_kkt_tolerance=1e-7 if alarm else np.nan)
    for _, prefix in a.LS_ESTIMATION_METHODS:
        for target in a.ESTIMATION_TARGETS:
            row[prefix+target+"_estimate"] = row[target]*1.1 if alarm else np.nan
            row[prefix+target+"_point_valid"] = alarm
            # Neither old interval availability nor global optimizer success
            # determines parameter-specific point availability.
            row[prefix+target+"_ci_available"] = False if prefix != "ls_" else None
    for target in a.ESTIMATION_TARGETS:
        row.pop("ls_"+target+"_ci_available")
    return row


def test_target_specific_pairing_availability_and_analytic_loss_contrast():
    rows = [episode(i) for i in range(4)]+[episode(4, False)]
    rows[1]["ls_pi_left_point_valid"] = False
    rows[2]["regularized_pi_right_point_valid"] = False
    rows[0]["ls_pi_left_estimate"] = .024
    result = a.compact_ls_summaries({"Original bank": pd.DataFrame(rows)})
    left = result["compact_ls_summary"].query("target == 'pi_left'").set_index("method")
    assert set(left.n_alarm) == {4}
    assert set(left.n_common_paired) == {3}
    assert left.loc["Full-delay LS", "point_availability"] == .75
    assert left.loc["Full-delay LS", "relative_bias"] == pytest.approx((.2+.1+.1)/3)
    assert left.loc["Full-delay LS", "relative_mse"] == pytest.approx((.2**2+2*.1**2)/3)
    assert set(result["compact_ls_summary"].query("target == 'risk_ratio'").n_common_paired) == {4}
    contrast = result["compact_ls_contrasts"].query(
        "target == 'pi_left' and method_a == 'Full-delay LS' and method_b == 'Unpenalized NB' "
        "and cohort == 'pairwise_available'").iloc[0]
    differences = np.array([.2**2-.1**2, 0, 0])
    assert contrast.n_paired == 3
    assert contrast.mean_squared_relative_error_difference == pytest.approx(differences.mean())
    assert contrast.paired_mcse == pytest.approx(differences.std(ddof=1)/np.sqrt(3))


def test_equal_cell_mean_loss_and_extreme_finite_rr_are_retained():
    rows = [episode(i) for i in range(4)]
    other = episode(scenario="other")
    other["ls_risk_ratio_estimate"] = other["risk_ratio"]*1e8
    rows.append(other)
    result = a.compact_ls_summaries({"Original bank": pd.DataFrame(rows)})
    row = result["compact_ls_aggregate"].query("target == 'risk_ratio' and method == 'Full-delay LS'").iloc[0]
    assert row.n_cells == 2
    assert row.common_paired_relative_rmse == pytest.approx(np.sqrt((.1**2+(1e8-1)**2)/2))
    assert not any("coverage" in c or "ci_" in c for c in result["compact_ls_summary"])


def test_no_alarms_are_missing_errors_not_zero_error():
    result = a.compact_ls_summaries({"Original bank": pd.DataFrame([episode(alarm=False)])})
    assert result["compact_ls_summary"].relative_rmse.isna().all()
    assert result["compact_ls_summary"].point_availability.isna().all()
    assert result["compact_ls_aggregate"].common_paired_relative_rmse.isna().all()


def test_main_pair_ignores_nb_failures_but_keeps_ls_ocfr_intersection():
    rows = [episode(i) for i in range(4)]
    rows[0]["pi_left_point_valid"] = False
    rows[0]["ls_pi_left_estimate"] = .028
    rows[1]["ls_pi_left_point_valid"] = False
    rows[2]["regularized_pi_left_point_valid"] = False
    frame = pd.DataFrame(rows)
    three = a.compact_ls_summaries({"Original bank": frame})["compact_ls_summary"]
    two = a.compact_ls_main_pair_summary({"Original bank": frame})
    assert set(two.method) == {"Full-delay LS", "OCFR"}
    assert set(three.query("target == 'pi_left'").n_common_paired) == {1}
    left = two.query("target == 'pi_left'").set_index("method")
    assert set(left.n_common_paired) == {2}
    assert left.loc["Full-delay LS", "common_paired_relative_mse"] == pytest.approx((.4**2+.1**2)/2)
    assert set(two.query("target == 'risk_ratio'").n_common_paired) == {4}


@pytest.mark.parametrize("bad", ["nonfinite", "outside_alarm", "bounds", "truth"])
def test_invalid_points_are_rejected(bad):
    row = episode(alarm=bad != "outside_alarm")
    if bad == "nonfinite":
        row["ls_risk_ratio_estimate"] = np.inf
    elif bad == "outside_alarm":
        row["ls_pi_left_point_valid"] = True
    elif bad == "bounds":
        row["ls_pi_left_estimate"] = 1.
    else:
        row["pi_left"] = 0.
    with pytest.raises(a.EvidenceError):
        a.compact_ls_summaries({"Original bank": pd.DataFrame([row])})


def _cache_key(manifest):
    keys = ("source_manifest_sha256", "source_csv_sha256", "source_bank", "source_scientific_digest",
        "scientific_config", "postprocessor_code_hash", "postprocessor_code_files",
        "postprocessor_native_files", "ls_estimation_config")
    return hashlib.sha256(a._canonical_json({key: manifest[key] for key in keys}).encode("ascii")).hexdigest()[:20]


def fixture(root):
    scenarios = [s for s in config.COMPACT_SCENARIOS
                 if s.split == "alternative" and s.misspecification == "on_model"]
    rows = []
    for scenario in scenarios:
        for rep in range(10):
            row = episode(rep, scenario=scenario.name)
            row.update(family=scenario.family, true_change=scenario.true_change,
                pi_left=scenario.pi_left, pi_right=scenario.pi_right, risk_ratio=scenario.risk_ratio)
            for _, prefix in a.LS_ESTIMATION_METHODS:
                for target in a.ESTIMATION_TARGETS:
                    row[prefix+target+"_estimate"] = row[target]*1.1
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["scenario", "replicate"], ignore_index=True)
    original = frame[[c for c in frame if not c.startswith("ls_")]]
    for bank, source_stage in (("original", "estimation-stabilized"), ("validation", "estimation-validation")):
        stage = "estimation-ls-"+bank
        source, directory = root/source_stage, root/stage
        source.mkdir(); directory.mkdir()
        original.to_csv(source/a.RAW_FILE, index=False)
        sm = dict(status="complete", stage=source_stage, mode="full", scientific_digest="historical",
            scientific_config={"master_seed": 123}, scenarios=[s.name for s in scenarios],
            artifacts={a.RAW_FILE: a._sha256(source/a.RAW_FILE)})
        (source/a.MANIFEST_FILE).write_text(json.dumps(sm))
        frame.to_csv(directory/a.RAW_FILE, index=False)
        files = {"historical_simulation.py": "a"*64}
        m = dict(status="complete", stage=stage, mode="demo", source_bank=bank,
            source_scientific_digest=sm["scientific_digest"], scientific_digest=sm["scientific_digest"],
            scientific_config=sm["scientific_config"], source_manifest_sha256=a._sha256(source/a.MANIFEST_FILE),
            source_csv_sha256=a._sha256(source/a.RAW_FILE), scenarios=[s.name for s in scenarios],
            replications_by_scenario={s.name: 10 for s in scenarios},
            artifacts={a.RAW_FILE: a._sha256(directory/a.RAW_FILE)}, postprocessor_code_files=files,
            postprocessor_code_hash=hashlib.sha256(a._canonical_json(files).encode("ascii")).hexdigest(),
            postprocessor_native_files={}, ls_estimation_config=asdict(config.PostAlarmLSConfig()))
        m["cache_key"] = _cache_key(m)
        (directory/a.MANIFEST_FILE).write_text(json.dumps(m))
    return root/"estimation-ls-original"


def test_historical_producer_and_limited_refresh_preserve_existing_files(tmp_path):
    fixture(tmp_path)
    destination = tmp_path/"analysis"
    destination.mkdir()
    sentinel = destination/"compact_analysis_manifest.json"
    sentinel.write_text("existing")
    old_fig = destination/"Fig5.tif"
    old_fig.write_bytes(b"old")
    tables = a.analyze_compact_ls_estimation(tmp_path)
    assert len(tables["compact_ls_summary"]) == 2*16*3*3
    assert sentinel.read_text() == "existing" and old_fig.read_bytes() == b"old"
    assert (destination/"compact_ls_analysis_manifest.json").exists()
    assert not (destination/"ls_figures").exists()
    with pytest.raises(a.EvidenceError, match="demo"):
        a.analyze_compact_ls_estimation(tmp_path, tmp_path/"figures")


@pytest.mark.parametrize("bad", ["artifact", "source", "producer", "cache", "settings", "grid", "frozen", "ci", "incomplete"])
def test_provenance_and_frozen_grid_rejected(tmp_path, bad):
    directory = fixture(tmp_path)
    path = directory/a.MANIFEST_FILE
    manifest = json.loads(path.read_text())
    if bad == "artifact":
        manifest["artifacts"][a.RAW_FILE] = "b"*64
    elif bad == "source":
        manifest["source_manifest_sha256"] = "b"*64
    elif bad == "producer":
        manifest["postprocessor_code_hash"] = "b"*64
    elif bad == "cache":
        manifest["cache_key"] = "stale"
    elif bad == "settings":
        manifest["ls_estimation_config"]["penalty"] = 1.
        manifest["cache_key"] = _cache_key(manifest)
    elif bad == "incomplete":
        manifest["status"] = "running"
    else:
        frame = pd.read_csv(directory/a.RAW_FILE)
        if bad == "grid":
            frame = frame.iloc[1:]
        elif bad == "frozen":
            frame.loc[0, "tau_hat_at_alarm"] += 1
        else:
            frame["ls_pi_left_ci_lower"] = .01
        frame.to_csv(directory/a.RAW_FILE, index=False)
        manifest["artifacts"][a.RAW_FILE] = a._sha256(directory/a.RAW_FILE)
    path.write_text(json.dumps(manifest))
    with pytest.raises(a.EvidenceError):
        a._compact_load_ls_estimation(tmp_path)


@pytest.mark.parametrize("extreme_banks", [("Independent bank",), ("Original bank", "Independent bank"), ()])
def test_s4_layout_plain_ticks_and_point_only_caption(tmp_path, monkeypatch, extreme_banks):
    rows = []
    for bank in ("Original bank", "Independent bank"):
        for target in a.ESTIMATION_TARGETS:
            for index, (method, _) in enumerate(a.LS_ESTIMATION_METHODS):
                value = (1e7 if index == 1 else .2) if bank in extreme_banks and target == "risk_ratio" else .1*(index+1)
                rows.append(dict(bank=bank, target=target, method=method,
                    n_cells=16, common_paired_relative_rmse=value))
    captured = []
    monkeypatch.setattr(a, "_compact_save", lambda fig, directory, name: captured.append(fig))
    caption = a._compact_ls_comparison_figure({"compact_ls_aggregate": pd.DataFrame(rows)}, tmp_path)
    fig = captured[0]
    assert len(fig.axes) == 6
    for index, ax in enumerate(fig.axes):
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["LS", "NB 0", "OCFR"]
        assert min(tick.get_fontsize() for tick in ax.get_xticklabels()) >= 9.5
        logarithmic = index >= 4 and ("Original bank" if index == 4 else "Independent bank") in extreme_banks
        assert ax.get_yscale() == ("log" if logarithmic else "linear")
        if not logarithmic:
            assert ax.get_ylim()[0] == 0
        assert not any("e+" in tick.get_text() for tick in ax.get_yticklabels())
        assert not any("," in tick.get_text() for tick in ax.get_yticklabels())
        if logarithmic:
            assert len(ax.get_yticks()) >= 3
            assert any("1e" in tick.get_text() for tick in ax.get_yticklabels())
    assert "no LS confidence intervals" in caption
    assert "all 16 conditions" in caption
    assert "fixed before computing LS bank results" in caption
    a.plt.close(fig)


def main_tables():
    rows, interval, historical_points, historical_paired = [], [], [], []
    scenarios = [s for s in config.COMPACT_SCENARIOS
                 if s.split == "alternative" and s.misspecification == "on_model"]
    for scenario in scenarios:
        for target in a.ESTIMATION_TARGETS:
            meta = dict(bank="Original bank", scenario=scenario.name, family=scenario.family,
                change_day=scenario.true_change+1, risk_ratio=scenario.risk_ratio, target=target)
            for method, value in (("Full-delay LS", .11), ("Unpenalized NB", .22), ("OCFR", .33)):
                rows.append(dict(meta, method=method, common_paired_relative_rmse=value,
                    common_paired_relative_rmse_mcse=.01, relative_rmse=.99,
                    N_requested=1000, n_alarm=900, n_point=900, n_common_paired=900))
            interval.append(dict(scenario=scenario.name, target=target, all_alarm_containment=.82,
                containment_ci_low=.79, containment_ci_high=.85, n_alarm=900))
            for method in ("OCFR", "Nishiura extension"):
                historical_points.append(dict(meta, method=method))
            record = dict(scenario=scenario.name, target=target)
            for prefix in ("ocfr", "mle", "reference"):
                record[prefix+"_paired_relative_rmse"] = .88
                record[prefix+"_paired_relative_rmse_mcse"] = .02
            historical_paired.append(record)
    summary = pd.DataFrame(rows)
    return dict(compact_ls_summary=summary,
        compact_ls_main_pair_summary=summary[summary.method.isin(["Full-delay LS", "OCFR"])].copy(),
        compact_stabilized_summary=pd.DataFrame(interval),
        compact_stabilized_reference_summary=pd.DataFrame(historical_points),
        compact_stabilized_reference_paired=pd.DataFrame(historical_paired))


def test_main_ls_uses_nine_common_cohort_points_and_identical_coverage(tmp_path, monkeypatch):
    tables = main_tables()
    captured = []
    monkeypatch.setattr(a, "_compact_save", lambda fig, directory, name: captured.append(fig))
    a._compact_style()
    old_caption = a._compact_estimation_comparison_figure(tables, tmp_path)
    new_caption = a._compact_estimation_comparison_figure(tables, tmp_path, ls=True)
    old, new = captured
    assert [t.get_text() for t in new.legends[0].get_texts()] == ["Full-delay LS", "OCFR"]
    for index in (0, 2, 4):
        ax = new.axes[index]
        assert len(ax.containers) == 2
        for container, value in zip(ax.containers, (.11, .33)):
            np.testing.assert_allclose(np.asarray(container.lines[0].get_ydata(), float), [value]*9)
            segments = container.lines[2][0].get_segments()
            np.testing.assert_allclose([segment[:, 1] for segment in segments],
                                      [[value-1.96*.01, value+1.96*.01]]*9)
        assert ax.get_ylim()[0] == 0
    for index in (1, 3, 5):
        old_bar, new_bar = old.axes[index].containers[0], new.axes[index].containers[0]
        np.testing.assert_array_equal(old_bar.lines[0].get_ydata(), new_bar.lines[0].get_ydata())
        np.testing.assert_array_equal(old_bar.lines[2][0].get_segments(), new_bar.lines[2][0].get_segments())
        np.testing.assert_array_equal(np.asarray(new_bar.lines[0].get_ydata(), float), [.82]*9)
    assert "Nishiura" in old_caption
    assert "LS supplies point estimates only" in new_caption and "not selection-adjusted" in new_caption
    assert "target-specific common available" in new_caption
    assert min(t.get_fontsize() for ax in new.axes for t in ax.get_xticklabels()) >= 9.5
    a.plt.close(old); a.plt.close(new)


def test_main_precedence_and_historical_fallback(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(a, "_compact_estimation_comparison_figure",
        lambda tables, directory, **kwargs: calls.append(kwargs) or "caption")
    tables = {"compact_ls_summary": pd.DataFrame(), "compact_poisson_summary": pd.DataFrame(),
              "compact_estimation_reference_summary": pd.DataFrame()}
    a._compact_estimation_figure(tables, tmp_path)
    assert calls[-1] == {"ls": True}
    del tables["compact_ls_summary"]
    a._compact_estimation_figure(tables, tmp_path)
    assert calls[-1] == {"poisson": True}
    del tables["compact_poisson_summary"]
    a._compact_estimation_figure(tables, tmp_path)
    assert calls[-1] == {}


def test_limited_main_export_preserves_old_artifacts_and_binds_inputs(tmp_path, monkeypatch):
    tables = main_tables()
    stages = ("estimation-ls-original", "estimation-ls-validation", "estimation-stabilized",
              "estimation-validation", "estimation-reference")
    for stage in stages:
        directory = tmp_path/stage
        directory.mkdir()
        for filename in (a.MANIFEST_FILE, a.RAW_FILE):
            (directory/filename).write_text(stage+filename)
    destination = tmp_path/"analysis"
    destination.mkdir()
    old = {"compact_analysis_manifest.json": b"old manifest", "compact_ls_summary.csv": b"old table",
           "Fig5.tif": b"old fig", "compact_poisson_analysis_manifest.json": b"old poisson"}
    for filename, payload in old.items():
        (destination/filename).write_bytes(payload)
    monkeypatch.setattr(a, "_compact_load_ls_estimation", lambda output, **kwargs:
        ({key: value for key, value in tables.items() if key.startswith("compact_ls_")},
         {"banks": {"original": {"mode": "full"}, "validation": {"mode": "full"}}}))
    monkeypatch.setattr(a, "_compact_load_stabilized_estimation", lambda output:
        ({key: value for key, value in tables.items() if key.startswith("compact_stabilized_")}, {"status": "complete"}))
    def save(fig, directory, name):
        for filename in (name+".pdf", name+".png", "Fig5.tif"):
            (directory/filename).write_bytes(b"new figure")
        a.plt.close(fig)
    monkeypatch.setattr(a, "_compact_save", save)
    monkeypatch.setattr(a, "_compact_export_metadata", lambda *args: {"verified": True})
    result = a.analyze_compact_ls_main_figure(tmp_path)
    assert len(result["compact_ls_main_summary"]) == 9*3*2
    for filename, payload in old.items():
        assert (destination/filename).read_bytes() == payload
    report = json.loads((destination/"compact_ls_main_analysis_manifest.json").read_text())
    assert set(report["source_files_sha256"]) == set(stages)
    assert report["point_methods"] == ["Full-delay LS", "OCFR"]
    assert "independent of unpenalized NB" in report["point_cohort"]
    assert len(report["displayed_conditions"]) == 9
    assert {row["n_alarm"] for row in report["displayed_counts"]} == {900}
    for stage in stages:
        for filename in (a.MANIFEST_FILE, a.RAW_FILE):
            assert report["source_files_sha256"][stage][filename] == a._sha256(tmp_path/stage/filename)
