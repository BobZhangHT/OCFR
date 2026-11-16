"""Frozen grid, paired RNG, statistical null labels and bounded resume."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import config
import simulation as sim
from test_sensitivity_extension import parent


def test_grid_and_paired_components():
    base = config.get_compact_experiment_config("full", backend="python")
    before = base.scientific_payload()
    full = config.get_dispersion_sensitivity_config(base)
    demo = config.get_dispersion_sensitivity_config(replace(base, mode="demo", replications=10))
    assert full.scientific_payload() == demo.scientific_payload()
    assert base.scientific_payload() == before
    assert len(full.scenarios) == 8 and full.master_seed == 2026111604
    assert {s.phi for s in full.scenarios} == {5., 10., 20., 40.}
    cases = set()
    for scenario in full.scenarios:
        assert scenario.split == "alternative" and scenario.true_change == 91
        assert scenario.horizon == 182
        assert scenario.true_delay_mean_days == scenario.assumed_delay_mean_days == 14
        _, _, _, stream = sim._compact_generate_components(scenario, full, 0)
        cases.add(stream["hashes"]["cases"])
    assert len(cases) == 1


def test_real_one_rep_grid_resume_and_null_labels(parent, tmp_path, monkeypatch):
    base, source, _ = parent
    original = config.get_dispersion_sensitivity_config
    monkeypatch.setattr(config, "get_dispersion_sensitivity_config", lambda cfg: replace(original(cfg), replications=1))
    destination = tmp_path / "dispersion"
    first = sim.run_dispersion_sensitivity(base_output=base, base_source=source, output=destination)
    assert len(first) == 24
    null = first[np.isclose(first.risk_ratio, 1.)]
    assert null.split.eq("holdout").all()
    assert not null.postchange_detection.any() and not null.premature_alarm.any()
    resumed = sim.run_dispersion_sensitivity(base_output=base, base_source=source, output=destination)
    pd.testing.assert_frame_equal(first, resumed)
    manifest = json.loads((destination / "dispersion-sensitivity/manifest.json").read_text())
    assert manifest["tasks_computed"] == 0 and manifest["cached_tasks_loaded"] == 8
    _, loaded, _ = sim.load_dispersion_sensitivity(destination, base_output=base, base_source=source)
    pd.testing.assert_frame_equal(first, loaded)


def test_dispersion_panel_actual_grid_and_legacy_default(tmp_path, monkeypatch):
    import analysis as a
    import matplotlib.pyplot as plt
    methods = ("OCFR", "NB-GLR", "Page-CUSUM")
    rows, dispersion = [], []
    for method in methods:
        for delay in (11., 14., 17.):
            for rr, metric in ((1., "alarm_probability"), (1.5, "useful_power")):
                rows.append(dict(family="wave", method=method, risk_ratio=rr,
                    metric=metric, change_day=92, true_delay_mean_days=delay,
                    misspecification="on_model" if delay == 14 else "delay_error",
                    estimate=.5, ci_low=.4, ci_high=.6))
        for rr, metric in ((1., "alarm_probability"), (1.5, "useful_power")):
            rows.append(dict(rows[0], method=method, risk_ratio=rr, metric=metric,
                true_delay_mean_days=14., misspecification="cohort_binomial"))
        for scale in (.75, 1.25):
            rows.append(dict(rows[0], method=method, risk_ratio=1., metric="alarm_probability",
                true_delay_mean_days=14., misspecification=f"exposure_scale_{scale}"))
        for phi in (5., 10., 20., 40.):
            for rr in (1., 1.5):
                for metric in (("alarm_probability",) if rr == 1 else
                        ("alarm_probability", "useful_power", "premature_probability")):
                    dispersion.append(dict(phi=phi, risk_ratio=rr, metric=metric,
                        method=method, estimate=.5, ci_low=.4, ci_high=.6))
    captured = []
    monkeypatch.setattr(a, "_compact_save", lambda fig, *args: captured.append(fig))
    _, caption = a._compact_sensitivity_figure(pd.DataFrame(rows), tmp_path,
        dispersion=pd.DataFrame(dispersion))
    panel = next(ax for ax in captured[-1].axes if "dispersion" in ax.get_xlabel())
    assert panel.get_xscale() == "log"
    assert list(panel.get_xticks()) == [5, 10, 20, 40]
    assert all(list(line.get_xdata()) == [5., 10., 20., 40.]
        for line in panel.lines if len(line.get_xdata()) == 4)
    assert "observed-case" not in caption and "null-error" in caption
    a._compact_dispersion_companions(pd.DataFrame(dispersion), tmp_path)
    assert len(captured[-1].axes) == 1
    assert "Observed case" not in captured[-1].axes[0].get_xlabel()
    _, legacy_caption = a._compact_sensitivity_figure(pd.DataFrame(rows), tmp_path)
    assert "observed-case multipliers" in legacy_caption
    for figure in captured:
        plt.close(figure)
