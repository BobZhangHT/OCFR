"""Scientific denominator checks for the paired delay-adjusted reference."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as a


def episode(rep, alarm=True):
    row = dict(scenario="steady_rr150", family="steady", replicate=rep,
        true_change=91, alarm_day=98 if alarm else np.nan, alarm=alarm,
        pi_left=.02, pi_right=.03, risk_ratio=1.5)
    for target in a.ESTIMATION_TARGETS:
        truth = row[target]
        row.update({f"{target}_estimate": truth if alarm else np.nan,
            f"{target}_point_valid": alarm,
            f"nishiura_{target}_estimate": truth if alarm else np.nan,
            f"nishiura_{target}_point_valid": alarm,
            f"nishiura_{target}_admissible": alarm})
    return row


def test_finite_negative_reference_retained_in_error_and_paired_comparison():
    rows = [episode(0), episode(1), episode(2, False)]
    rows[0].update(nishiura_pi_right_estimate=-.03, nishiura_pi_right_admissible=False,
                   nishiura_risk_ratio_estimate=-1.5, nishiura_risk_ratio_admissible=False)
    tables = a.compact_estimation_reference_summaries(pd.DataFrame(rows))
    summary = tables["compact_estimation_reference_summary"].set_index(["target", "method"])
    value = summary.loc[("pi_right", "Nishiura extension")]
    assert (value.n_alarm, value.n_point, value.n_inadmissible, value.n_unavailable) == (2, 2, 1, 0)
    assert value.relative_rmse == pytest.approx(np.sqrt(2))
    assert value.relative_rmse_mcse > 0
    paired = tables["compact_estimation_reference_paired"].set_index("target").loc["risk_ratio"]
    assert paired.n_paired == 2
    assert paired.mean_squared_relative_error_difference == -2


def test_target_specific_missing_reference_is_not_zero_error():
    rows = [episode(0), episode(1)]
    rows[0].update(nishiura_risk_ratio_estimate=np.nan, nishiura_risk_ratio_point_valid=False,
                   nishiura_risk_ratio_admissible=False)
    rows[1]["nishiura_risk_ratio_estimate"] = 3.
    tables = a.compact_estimation_reference_summaries(pd.DataFrame(rows))
    summary = tables["compact_estimation_reference_summary"].set_index(["target", "method"])
    value = summary.loc[("risk_ratio", "Nishiura extension")]
    assert (value.n_point, value.n_unavailable, value.relative_rmse) == (1, 1, 1.)
    assert summary.loc[("pi_right", "Nishiura extension"), "n_point"] == 2
    assert tables["compact_estimation_reference_paired"].set_index("target").loc["risk_ratio", "n_paired"] == 1


def test_zero_alarm_and_false_finite_declaration():
    summary = a.compact_estimation_reference_summaries(pd.DataFrame([episode(0, False)]))[
        "compact_estimation_reference_summary"]
    assert summary.relative_rmse.isna().all()
    broken = episode(0)
    broken["nishiura_pi_left_estimate"] = np.inf
    with pytest.raises(a.EvidenceError, match="nonfinite"):
        a.compact_estimation_reference_summaries(pd.DataFrame([broken]))


def test_missing_reference_is_optional_but_incomplete_stage_is_rejected(tmp_path):
    assert a._compact_load_estimation_reference(tmp_path) == ({}, {"status": "absent"})
    (tmp_path/"estimation-reference").mkdir()
    with pytest.raises(a.EvidenceError, match="manifest"):
        a._compact_load_estimation_reference(tmp_path)


def test_comparative_figure_preserves_six_panels_and_ocfr_only_coverage(tmp_path, monkeypatch):
    rows = []
    for family in ("steady", "sparse", "wave"):
        for rr in (2/3, 1.25, 1.5):
            row = episode(0)
            row.update(scenario=f"{family}_{rr}", family=family, risk_ratio=rr,
                pi_right=.02*rr, method="OCFR Gaussian", fit_success=True, tau_hat_at_alarm=91)
            for target in a.ESTIMATION_TARGETS:
                truth = row[target]
                row.update({f"{target}_estimate": truth*1.1,
                    f"nishiura_{target}_estimate": truth*1.2,
                    f"{target}_lower": truth*.5, f"{target}_upper": truth*1.5,
                    f"{target}_interval_valid": True})
            rows.append(row)
    frame = pd.DataFrame(rows)
    tables = a.compact_estimation_summaries(frame)
    tables.update(a.compact_estimation_reference_summaries(frame))
    captured = {}
    def capture(fig, directory, name):
        captured.update(axes=fig.axes, name=name)
        a.plt.close(fig)
    monkeypatch.setattr(a, "_compact_save", capture)
    caption = a._compact_estimation_figure(tables, tmp_path)
    assert captured["name"] == "CompactFig5_estimation"
    assert len(captured["axes"]) == 6
    for axis in captured["axes"][1::2]:
        assert axis.get_ylabel() == "OCFR interval coverage"
        assert axis.get_ylim() == (0, 1.04)
    assert "point estimates only" in caption
    assert "inadmissible" in caption
