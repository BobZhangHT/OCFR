"""Synthetic checks of frozen Poisson comparison denominators and provenance."""
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


def episode(rep=0, alarm=True):
    row = dict(scenario="test", family="steady", replicate=rep, true_change=91,
        alarm=alarm, alarm_day=90 if alarm else np.nan, tau_hat_at_alarm=85 if alarm else np.nan,
        pi_left=.02, pi_right=.03, risk_ratio=1.5,
        case_hash="c"*64, death_hash="d"*64, fit_success=alarm, method="OCFR Gaussian")
    for method, prefix, _, _ in a.POISSON_ESTIMATION_METHODS:
        for target in a.ESTIMATION_TARGETS:
            row[prefix+target+"_estimate"] = row[target]*1.1 if alarm else np.nan
            row[prefix+target+"_point_valid"] = alarm
        if prefix.startswith("poisson_"):
            row.update({prefix+"status": "success" if alarm else "no_alarm",
                prefix+"seconds": .01 if alarm else 0., prefix+"loglik": -10. if alarm else np.nan,
                prefix+"iterations": 2 if alarm else 0})
    # Old moment-fit availability must have no influence on new paired cohorts.
    for target in a.ESTIMATION_TARGETS:
        row["nishiura_"+target+"_point_valid"] = False
    return row


def test_target_specific_main_pairing_and_all_alarm_availability():
    rows = [episode(i) for i in range(4)] + [episode(4, alarm=False)]
    rows[1]["poisson_regularized_pi_left_point_valid"] = False
    rows[2]["regularized_pi_right_point_valid"] = False
    rows[3]["poisson_unpenalized_risk_ratio_point_valid"] = False
    rows[0]["pi_left_estimate"] = .024
    result = a.compact_poisson_summaries({"Original bank": pd.DataFrame(rows)})
    summary = result["compact_poisson_summary"]
    left = summary[summary.target.eq("pi_left")].set_index("method")
    assert set(left.n_common_main) == {3}
    assert set(left.n_alarm) == {4}
    assert left.loc["Matched Poisson", "point_availability"] == .75
    assert left.loc["Unpenalized NB", "relative_mse"] == pytest.approx((.2**2+3*.1**2)/4)
    assert left.loc["Unpenalized NB", "common_paired_relative_mse"] == pytest.approx((.2**2+2*.1**2)/3)
    right = summary[summary.target.eq("pi_right")]
    assert set(right.n_common_main) == {3}
    ratio = summary[summary.target.eq("risk_ratio")].set_index("method")
    assert set(ratio.n_common_main) == {4}
    assert set(ratio.n_common_four) == {3}
    assert ratio.loc["OCFR", "n_common_paired"] == 4
    assert ratio.loc["Unpenalized Poisson", "n_common_paired"] == 3
    contrast = result["compact_poisson_contrasts"]
    selected = contrast[contrast.target.eq("pi_left") & contrast.method_a.eq("Unpenalized NB") &
        contrast.method_b.eq("OCFR") & contrast.cohort.eq("pairwise_available")].iloc[0]
    diff = np.array([.2**2-.1**2, 0, 0, 0])
    assert selected.n_paired == 4
    assert selected.mean_squared_relative_error_difference == pytest.approx(diff.mean())
    assert selected.paired_mcse == pytest.approx(diff.std(ddof=1)/2)


def test_equal_cell_aggregation_not_pooled_episode_mse():
    first_rows = [episode(rep) for rep in range(4)]
    second = episode()
    third = episode(1, alarm=False)
    second.update(scenario="other", pi_left=.04, pi_right=.06)
    third.update(scenario="other", pi_left=.04, pi_right=.06)
    for _, prefix, _, _ in a.POISSON_ESTIMATION_METHODS:
        second[prefix+"pi_left_estimate"] = .08
        second[prefix+"pi_right_estimate"] = .06
    result = a.compact_poisson_summaries({"Original bank": pd.DataFrame([*first_rows, second, third])})
    aggregate = result["compact_poisson_aggregate"]
    row = aggregate[aggregate.method.eq("OCFR") & aggregate.target.eq("pi_left")].iloc[0]
    assert row.n_cells == 2 and row.n_alarm == 5 and row.n_point == 5
    assert row.common_paired_relative_rmse == pytest.approx(np.sqrt((.1**2+1)/2))
    assert row.mse_ties_vs_unpenalized_nb == 2


def test_zero_alarms_do_not_become_zero_error():
    result = a.compact_poisson_summaries({"Original bank": pd.DataFrame([episode(alarm=False)])})
    summary = result["compact_poisson_summary"]
    assert summary.relative_rmse.isna().all()
    assert summary.point_availability.isna().all()
    assert result["compact_poisson_aggregate"].common_paired_relative_rmse.isna().all()


@pytest.mark.parametrize("bad", ["nonfinite", "outside_alarm", "truth", "bounds"])
def test_invalid_declared_points_and_truth_raise(bad):
    row = episode(alarm=bad != "outside_alarm")
    if bad == "nonfinite":
        row["poisson_regularized_pi_left_estimate"] = np.inf
    elif bad == "outside_alarm":
        row["poisson_regularized_pi_left_point_valid"] = True
    elif bad == "truth":
        row["pi_left"] = 0.
    else:
        row["poisson_regularized_pi_left_estimate"] = 1.
    with pytest.raises(a.EvidenceError):
        a.compact_poisson_summaries({"Original bank": pd.DataFrame([row])})


def fixture(root):
    scenarios = [s for s in config.get_compact_experiment_config("full", backend="python", protocol="compact-v1").scenarios
                 if s.split == "alternative" and s.misspecification == "on_model"]
    rows = []
    for scenario in scenarios:
        for rep in range(10):
            row = episode(rep)
            row.update(scenario=scenario.name, family=scenario.family, true_change=scenario.true_change,
                pi_left=scenario.pi_left, pi_right=scenario.pi_right, risk_ratio=scenario.risk_ratio)
            for _, prefix, _, _ in a.POISSON_ESTIMATION_METHODS:
                for target in a.ESTIMATION_TARGETS:
                    row[prefix+target+"_estimate"] = row[target]*1.1
            rows.append(row)
    all_rows = pd.DataFrame(rows)
    original = all_rows[[c for c in all_rows if not c.startswith("poisson_")]]
    for stage, source_stage in (("estimation-poisson-original", "estimation-stabilized"),
                                ("estimation-poisson-validation", "estimation-validation")):
        source, directory = root/source_stage, root/stage
        source.mkdir(); directory.mkdir()
        original.to_csv(source/a.RAW_FILE, index=False)
        source_manifest = dict(status="complete", stage=source_stage, mode="full",
            scenarios=[s.name for s in scenarios], scientific_digest="historical-digest",
            scientific_config={"master_seed": 123}, artifacts={a.RAW_FILE:a._sha256(source/a.RAW_FILE)})
        (source/a.MANIFEST_FILE).write_text(json.dumps(source_manifest))
        all_rows.to_csv(directory/a.RAW_FILE, index=False)
        files = {"historical_simulation.py":"a"*64}
        manifest = dict(status="complete", stage=stage, source_bank=stage.removeprefix("estimation-poisson-"), mode="demo",
            source_scientific_digest=source_manifest["scientific_digest"],
            scientific_digest=source_manifest["scientific_digest"], scientific_config=source_manifest["scientific_config"],
            source_manifest_sha256=a._sha256(source/a.MANIFEST_FILE), source_csv_sha256=a._sha256(source/a.RAW_FILE),
            scenarios=[s.name for s in scenarios], replications_by_scenario={s.name:10 for s in scenarios},
            artifacts={a.RAW_FILE:a._sha256(directory/a.RAW_FILE)}, postprocessor_code_files=files,
            postprocessor_code_hash=hashlib.sha256(a._canonical_json(files).encode("ascii")).hexdigest(),
            postprocessor_native_files={}, poisson_estimation_config=dict(primary_precision=1., control_precision=0.,
                likelihood="exact_poisson", interval_scope="point_only"))
        provenance_keys = ("source_manifest_sha256", "source_csv_sha256", "source_bank",
            "source_scientific_digest", "scientific_config", "postprocessor_code_hash",
            "postprocessor_code_files", "postprocessor_native_files", "poisson_estimation_config")
        manifest["cache_key"] = hashlib.sha256(a._canonical_json({k:manifest[k] for k in provenance_keys}).encode("ascii")).hexdigest()[:20]
        (directory/a.MANIFEST_FILE).write_text(json.dumps(manifest))
    return root/"estimation-poisson-original"


def test_historical_producer_hashes_are_not_compared_with_current_code(tmp_path):
    fixture(tmp_path)
    tables, metadata = a._compact_load_poisson_estimation(tmp_path)
    assert len(tables["compact_poisson_summary"]) == 2*16*4*3
    assert len(tables["compact_poisson_aggregate"]) == 2*4*3
    assert metadata["status"] == "complete"
    assert "Nishiura" not in set(tables["compact_poisson_summary"].method)


@pytest.mark.parametrize("bad", ["artifact", "source", "producer", "cache", "settings", "grid", "frozen", "namespace", "incomplete"])
def test_provenance_grid_and_frozen_columns_rejected(tmp_path, bad):
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
        manifest["cache_key"] = "stale-cache-key"
    elif bad == "settings":
        manifest["poisson_estimation_config"]["primary_precision"] = 2.
    elif bad == "incomplete":
        manifest["status"] = "running"
    elif bad == "namespace":
        source = tmp_path/"estimation-stabilized"
        frame = pd.read_csv(source/a.RAW_FILE)
        frame["poisson_reserved"] = 1
        frame.to_csv(source/a.RAW_FILE, index=False)
        source_path = source/a.MANIFEST_FILE
        source_manifest = json.loads(source_path.read_text())
        source_manifest["artifacts"][a.RAW_FILE] = a._sha256(source/a.RAW_FILE)
        source_path.write_text(json.dumps(source_manifest))
        manifest["source_csv_sha256"] = a._sha256(source/a.RAW_FILE)
        manifest["source_manifest_sha256"] = a._sha256(source_path)
        keys = ("source_manifest_sha256", "source_csv_sha256", "source_bank", "source_scientific_digest",
            "scientific_config", "postprocessor_code_hash", "postprocessor_code_files",
            "postprocessor_native_files", "poisson_estimation_config")
        manifest["cache_key"] = hashlib.sha256(a._canonical_json({k:manifest[k] for k in keys}).encode("ascii")).hexdigest()[:20]
    else:
        frame = pd.read_csv(directory/a.RAW_FILE)
        if bad == "grid":
            frame = frame.iloc[1:]
        else:
            frame.loc[0, "tau_hat_at_alarm"] += 1
        frame.to_csv(directory/a.RAW_FILE, index=False)
        manifest["artifacts"][a.RAW_FILE] = a._sha256(directory/a.RAW_FILE)
    path.write_text(json.dumps(manifest))
    with pytest.raises(a.EvidenceError):
        a._compact_load_poisson_estimation(tmp_path)


def test_demo_refresh_is_table_only(tmp_path):
    fixture(tmp_path)
    a.analyze_compact_poisson_estimation(tmp_path)
    assert (tmp_path/"analysis"/"compact_poisson_analysis_manifest.json").is_file()
    assert not (tmp_path/"analysis"/"figures").exists()
    with pytest.raises(a.EvidenceError, match="demo"):
        a.analyze_compact_poisson_estimation(tmp_path, tmp_path/"figures")


def test_main_figure_keeps_coverage_and_s2_keeps_old_points(tmp_path, monkeypatch):
    fixture(tmp_path)
    tables, _ = a._compact_load_poisson_estimation(tmp_path)
    source = tables["compact_poisson_summary"]
    original = source[source.bank.eq("Original bank")]
    point, paired, interval = [], [], []
    for (scenario, target), group in original.groupby(["scenario", "target"]):
        first = group.iloc[0]
        for method in ("OCFR", "Nishiura extension"):
            point.append(dict(scenario=scenario, target=target, method=method, family=first.family,
                risk_ratio=first.risk_ratio, change_day=first.change_day))
        record = dict(scenario=scenario, target=target)
        for prefix, value in (("ocfr", .2), ("mle", .3), ("reference", .4)):
            record[prefix+"_paired_relative_rmse"] = value
            record[prefix+"_paired_relative_rmse_mcse"] = .01
        paired.append(record)
        interval.append(dict(scenario=scenario, target=target, all_alarm_containment=.8,
            containment_ci_low=.7, containment_ci_high=.9))
    tables.update(compact_stabilized_reference_summary=pd.DataFrame(point),
        compact_stabilized_reference_paired=pd.DataFrame(paired),
        compact_stabilized_summary=pd.DataFrame(interval))
    captures = {}
    def capture(fig, directory, name):
        captures[name] = fig
    monkeypatch.setattr(a, "_compact_save", capture)
    old_caption = a._compact_estimation_comparison_figure(tables, tmp_path,
        name="CompactFigS2_estimation_reference")
    new_caption = a._compact_estimation_comparison_figure(tables, tmp_path, poisson=True)
    old, new = captures["CompactFigS2_estimation_reference"], captures["CompactFig5_estimation"]
    assert len(new.axes) == 6
    np.testing.assert_allclose(new.get_size_inches(), [6.8, 7.8])
    for index in (1, 3, 5):
        np.testing.assert_allclose(np.asarray(old.axes[index].lines[0].get_ydata(), dtype=float),
                                   np.asarray(new.axes[index].lines[0].get_ydata(), dtype=float))
    assert list(old.axes[0].lines[0].get_ydata()) == [.2]*9
    np.testing.assert_allclose(np.asarray(new.axes[0].lines[0].get_ydata(), dtype=float), [.1]*9)
    assert "Nishiura" in old_caption and "same fixed log-RR penalty of 1" in new_caption
    assert [text.get_text() for text in new.legends[0].get_texts()] == ["OCFR", "Unpenalized NB", "Matched Poisson"]
    a.plt.close(old); a.plt.close(new)


def test_s3_uses_zero_based_cfr_scales_and_visible_plain_ticks(tmp_path, monkeypatch):
    rows = []
    for bank in ("Original bank", "Independent bank"):
        for target in a.ESTIMATION_TARGETS:
            for method, _, _, precision in a.POISSON_ESTIMATION_METHODS:
                value = (.19 if precision else .25) if target == "pi_left" else .16
                if target == "risk_ratio" and bank == "Independent bank" and not precision:
                    value = 1.3e6
                rows.append(dict(bank=bank, target=target, method=method, four_way_relative_rmse=value))
    capture = {}
    monkeypatch.setattr(a, "_compact_save", lambda fig, directory, name: capture.update(fig=fig))
    caption = a._compact_poisson_comparison_figure({"compact_poisson_aggregate": pd.DataFrame(rows)}, tmp_path)
    fig = capture["fig"]
    fig.canvas.draw()
    for index, ax in enumerate(fig.axes):
        lower, upper = ax.get_ylim()
        visible = [(tick, label.get_text()) for tick, label in zip(ax.get_yticks(), ax.get_yticklabels())
                   if lower <= tick <= upper and label.get_visible() and label.get_text()]
        assert len(visible) >= 3
        assert all("$" not in label and "\\" not in label for _, label in visible)
        assert ax.get_yscale() == ("log" if index == 5 else "linear")
        if index < 5:
            assert lower == 0
            assert "log scale" not in ax.get_ylabel()
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
    assert fig.axes[2].get_ylim() == fig.axes[3].get_ylim()
    assert "Panel F uses a logarithmic axis" in caption
    a.plt.close(fig)
