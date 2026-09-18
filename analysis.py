"""Validate, summarize, and plot the audited OCFR simulation evidence package.

The input directory must contain ``replicate_results.csv`` and ``manifest.json``.
Legacy or partially documented results are rejected.  Figures 2--6 implement
the streamlined PLOS evidence plan.  Alternative detection is always P28: an
alarm from the true change through 28 days afterwards, never arbitrary alarm
probability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, t as student_t


RAW_FILE = "replicate_results.csv"
MANIFEST_FILE = "manifest.json"
REQUIRED_EXPERIMENTS = ("E1", "E2", "E3", "E3M", "E4", "E5", "E6")
METHOD_ORDER = ("OCFR", "OCFR-alpha-matched", "NB-GLR", "Page-CUSUM", "FOCuS")
FAIR_METHOD_ORDER = ("OCFR-alpha-matched", "NB-GLR", "Page-CUSUM", "FOCuS")
COMPARATOR_ORDER = ("NB-GLR", "Page-CUSUM", "FOCuS")
COLORS = {
    "OCFR": "#0F4D92", "OCFR-alpha-matched": "#3775BA",
    "NB-GLR": "#B64342", "Page-CUSUM": "#42949E",
    "FOCuS": "#9A4D8E", "Aligned": "#0F4D92", "Unaligned": "#CF6B5E",
}
MARKERS = {"OCFR": "o", "OCFR-alpha-matched": "P", "NB-GLR": "s", "Page-CUSUM": "^", "FOCuS": "D"}

# Executable raw-data contract.  Nullable columns for other experiments are
# allowed, but every listed field must be populated within its own experiment.
EXPERIMENT_CONTRACT: Mapping[str, tuple[str, ...]] = {
    "E1": ("component", "scenario", "replicate", "nominal_alpha", "reject"),
    "E2": ("scenario", "replicate", "parameter", "true_value", "estimate", "ci_lower", "ci_upper"),
    "E3": ("scenario", "family", "path_id", "replicate", "look_day", "alarm_by_look"),
    "E3M": ("component", "scenario", "multiplier_size", "metric", "estimate", "n"),
    "E4": ("scenario", "family", "replicate", "risk_ratio", "true_change", "alarm_day", "tau_hat_at_alarm"),
    "E5": ("comparison", "pairing_group", "scenario", "purpose", "method", "replicate", "shared_stream_hash", "true_change", "alarm_day"),
    "E6": ("perturbation_type", "perturbation_label", "perturbation_order", "pairing_group", "scenario", "purpose", "method", "replicate", "shared_stream_hash", "true_change", "alarm_day"),
}

NULLABLE_BY_DESIGN = {"alarm_day", "tau_hat_at_alarm"}


class EvidenceError(ValueError):
    """The supplied files cannot support the requested scientific display."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _manifest_file_hash(manifest: Mapping[str, Any], filename: str) -> str:
    entries = manifest.get("files", manifest.get("artifacts"))
    if not isinstance(entries, Mapping) or filename not in entries:
        raise EvidenceError(f"{MANIFEST_FILE} must contain files['{filename}'] with its SHA-256")
    item = entries[filename]
    value = item.get("sha256") if isinstance(item, Mapping) else item
    if not isinstance(value, str) or len(value) != 64:
        raise EvidenceError(f"manifest SHA-256 for {filename} is absent or malformed")
    return value.lower()


def _validate_scientific_digest(manifest: Mapping[str, Any], frame: pd.DataFrame) -> str:
    declared = manifest.get("scientific_digest")
    scientific = manifest.get("scientific_config")
    schema_version = manifest.get("schema_version")
    if not isinstance(declared, str) or len(declared) < 12:
        raise EvidenceError("manifest scientific_digest is absent or too short")
    if scientific is None or schema_version is None:
        raise EvidenceError("manifest must include schema_version and scientific_config")
    digest_payload: dict[str, Any] = {
        "schema_version": schema_version, "scientific": scientific,
    }
    # Release runner schema v2 binds the executable source hash into the
    # scientific digest; schema v1 packages did not.
    if "code_hash" in manifest:
        digest_payload["code_hash"] = manifest["code_hash"]
    calculated = hashlib.sha256(_canonical_json(digest_payload).encode("utf-8")).hexdigest()
    if not calculated.startswith(declared.lower()):
        raise EvidenceError(
            f"scientific digest mismatch: manifest={declared}, "
            f"recomputed={calculated[:len(declared)]}"
        )
    if "scientific_digest" in frame.columns:
        values = set(frame["scientific_digest"].dropna().astype(str))
        if values != {declared}:
            raise EvidenceError(f"row-level scientific digests disagree with manifest: {sorted(values)}")
    return declared


def _as_binary(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    converted = pd.to_numeric(series, errors="coerce")
    bad = converted.isna() | ~converted.isin([0, 1])
    if bad.any():
        raise EvidenceError(f"{name} must contain only 0/1 or booleans; found {series[bad].head(3).tolist()}")
    return converted.astype(int)


def load_evidence(input_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Load and cryptographically validate one fresh E1--E6 evidence package."""
    raw_path, manifest_path = input_dir / RAW_FILE, input_dir / MANIFEST_FILE
    missing = [str(path) for path in (raw_path, manifest_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("required fresh simulation outputs are missing: " + ", ".join(missing))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"invalid JSON in {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise EvidenceError("manifest root must be a JSON object")
    observed, expected = _sha256(raw_path), _manifest_file_hash(manifest, RAW_FILE)
    if observed != expected:
        raise EvidenceError(f"raw-file hash mismatch: expected {expected}, observed {observed}")
    frame = pd.read_csv(
        raw_path, keep_default_na=False, na_values=[""], low_memory=False
    )
    if frame.empty:
        raise EvidenceError(f"{raw_path} is empty")
    if "experiment" not in frame:
        raise EvidenceError("replicate_results.csv lacks required experiment labels E1--E6")
    digest = _validate_scientific_digest(manifest, frame)
    validate_experiment_contract(frame)
    return frame, manifest, digest


def validate_experiment_contract(frame: pd.DataFrame) -> None:
    if "status" not in frame:
        raise EvidenceError("replicate_results.csv lacks required status values")
    failed = frame[~frame["status"].astype(str).eq("success")]
    if not failed.empty:
        examples = failed[["experiment", "scenario", "replicate", "status"]].head(5).to_dict("records")
        raise EvidenceError(
            f"fresh run contains {len(failed)} unsuccessful rows; resolve them before analysis: {examples}"
        )
    observed = set(frame["experiment"].astype(str))
    missing = sorted(set(REQUIRED_EXPERIMENTS) - observed)
    if missing:
        raise EvidenceError(f"fresh evidence is incomplete; missing experiments: {missing}")
    for experiment, columns in EXPERIMENT_CONTRACT.items():
        subset = frame[frame["experiment"] == experiment]
        absent = [column for column in columns if column not in frame]
        if absent:
            raise EvidenceError(f"{experiment} is missing required columns: {absent}")
        for column in columns:
            if column in NULLABLE_BY_DESIGN:
                continue
            if subset[column].isna().any():
                rows = subset.index[subset[column].isna()].tolist()[:3]
                raise EvidenceError(f"{experiment}.{column} is missing at rows {rows}")
    _validate_shared_streams(frame[frame["experiment"].isin(["E5", "E6"])] )


def _validate_shared_streams(frame: pd.DataFrame) -> None:
    """Verify that every claimed matched cell uses exactly one data-stream hash."""
    if frame.empty:
        raise EvidenceError("E5/E6 shared-stream validation received no rows")
    hashes = frame["shared_stream_hash"].astype(str)
    if hashes.str.len().lt(12).any() or hashes.str.lower().isin({"nan", "none", ""}).any():
        raise EvidenceError("E5/E6 shared_stream_hash values must be at least 12 characters")
    local = frame.copy()
    identity = ["experiment", "pairing_group", "purpose", "replicate"]
    comparison = local.get("comparison", pd.Series("", index=local.index)).fillna("").astype(str)
    perturbation = local.get("perturbation_label", pd.Series("", index=local.index)).fillna("").astype(str)
    local["_arm"] = np.where(comparison.eq(""), perturbation, comparison)
    duplicate_key = identity + ["_arm", "scenario", "method"]
    duplicate = local.duplicated(duplicate_key, keep=False)
    if duplicate.any():
        raise EvidenceError("duplicate method rows in paired cells: " + str(local.loc[duplicate, duplicate_key].head().to_dict("records")))
    counts = local.groupby(identity, dropna=False)["shared_stream_hash"].nunique()
    if (counts != 1).any():
        raise EvidenceError(f"matched cells use multiple shared hashes: {counts[counts != 1].head().index.tolist()}")
    # One observed stream may deliberately be reused across named scenario
    # arms. Matching is enforced within each declared group; reuse across
    # groups is not treated as an identity collision.
    # Every replicate in an arm must contain the same method set.  Otherwise a
    # paired contrast can silently become a complete-case comparison.
    arm_key = ["experiment", "pairing_group", "purpose", "_arm"]
    signatures = local.groupby(arm_key + ["replicate"])["method"].apply(
        lambda values: tuple(sorted(values.astype(str)))
    )
    unstable = signatures.groupby(level=arm_key).nunique()
    if (unstable != 1).any():
        raise EvidenceError(f"paired arms have incomplete method sets: {unstable[unstable != 1].index.tolist()[:5]}")


def _exact_probability(values: Sequence[int] | np.ndarray) -> tuple[float, float, float, int]:
    binary = np.asarray(values, dtype=int)
    if binary.size == 0:
        raise EvidenceError("cannot summarize an empty probability cell")
    successes, n = int(binary.sum()), int(binary.size)
    estimate = successes / n
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(beta.ppf(0.975, successes + 1, n - successes))
    return estimate, lower, upper, n


def _mean_ci(values: Sequence[float] | np.ndarray) -> tuple[float, float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        raise EvidenceError("a mean or paired-difference cell needs at least two finite replicates")
    mean = float(array.mean())
    se = float(array.std(ddof=1) / np.sqrt(array.size))
    radius = float(student_t.ppf(0.975, array.size - 1) * se)
    return mean, mean - radius, mean + radius, int(array.size)


def _median_ci(values: Sequence[float] | np.ndarray, seed_text: str) -> tuple[float, float, float, int]:
    """Deterministic percentile-bootstrap interval for a conditional median."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        raise EvidenceError("a conditional-median cell needs at least two finite detections")
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(2000, array.size), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.median(array)), float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975)), int(array.size)


def _p28(frame: pd.DataFrame, window: int) -> pd.Series:
    alarm = pd.to_numeric(frame["alarm_day"], errors="coerce")
    change = pd.to_numeric(frame["true_change"], errors="coerce")
    if change.isna().any():
        raise EvidenceError("true_change must be numeric for P28")
    return (alarm.notna() & alarm.ge(change) & alarm.le(change + window)).astype(int)


def _summarize_binary(frame: pd.DataFrame, groups: Sequence[str], value: str, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, cell in frame.groupby(list(groups), sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        estimate, lower, upper, n = _exact_probability(_as_binary(cell[value], value))
        rows.append(dict(zip(groups, keys)) | {
            "metric": metric, "estimate": estimate, "ci_lower": lower,
            "ci_upper": upper, "n": n,
        })
    return pd.DataFrame(rows)


def derive_figure2(frame: pd.DataFrame) -> pd.DataFrame:
    e1 = frame[frame["experiment"] == "E1"].copy()
    out1 = _summarize_binary(e1, ["component", "scenario", "nominal_alpha"], "reject", "rejection_probability")
    out1["experiment"] = "E1"
    e2 = frame[frame["experiment"] == "E2"].copy()
    e2["covered"] = (pd.to_numeric(e2["ci_lower"]).le(pd.to_numeric(e2["true_value"])) & pd.to_numeric(e2["ci_upper"]).ge(pd.to_numeric(e2["true_value"]))).astype(int)
    coverage = _summarize_binary(e2, ["scenario", "parameter"], "covered", "coverage")
    rows: list[dict[str, Any]] = []
    for keys, cell in e2.groupby(["scenario", "parameter"], sort=True):
        truth = pd.to_numeric(cell["true_value"]).to_numpy(float)
        estimates = pd.to_numeric(cell["estimate"]).to_numpy(float)
        if not np.allclose(truth, truth[0]):
            raise EvidenceError(f"E2 truth varies within {keys}")
        relative = (estimates - truth) / np.where(np.abs(truth) > 0, np.abs(truth), 1.0)
        estimate, lower, upper, n = _mean_ci(relative)
        rows.append({"scenario": keys[0], "parameter": keys[1], "metric": "relative_bias", "estimate": estimate, "ci_lower": lower, "ci_upper": upper, "n": n})
    out2 = pd.concat([coverage, pd.DataFrame(rows)], ignore_index=True)
    out2["experiment"] = "E2"
    return pd.concat([out1, out2], ignore_index=True, sort=False)


def derive_figure3(frame: pd.DataFrame) -> pd.DataFrame:
    e3 = frame[frame["experiment"] == "E3"].copy()
    e3["alarm_by_look"] = _as_binary(e3["alarm_by_look"], "alarm_by_look")
    ordered = e3.sort_values(["scenario", "path_id", "replicate", "look_day"])
    monotone = ordered.groupby(["scenario", "path_id", "replicate"])["alarm_by_look"].apply(lambda x: np.all(np.diff(x.to_numpy(int)) >= 0))
    if not monotone.all():
        raise EvidenceError("E3 alarm_by_look must be cumulative and nondecreasing")
    paths = _summarize_binary(e3, ["scenario", "family", "path_id", "look_day"], "alarm_by_look", "cumulative_episode_error")
    paths["experiment"] = "E3"
    multiplier = frame[frame["experiment"] == "E3M"].copy()
    return pd.concat([paths, multiplier], ignore_index=True, sort=False)


def derive_figure4(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    e4 = frame[frame["experiment"] == "E4"].copy()
    e4["p28"] = _p28(e4, window)
    p28 = _summarize_binary(e4, ["scenario", "family", "risk_ratio"], "p28", f"P{window}")
    detected = e4[e4["p28"] == 1].copy()
    detected["localization_error"] = np.abs(pd.to_numeric(detected["tau_hat_at_alarm"]) - pd.to_numeric(detected["true_change"]))
    detected["detection_delay"] = pd.to_numeric(detected["alarm_day"]) - pd.to_numeric(detected["true_change"])
    rows: list[dict[str, Any]] = []
    groups = e4[["scenario", "family", "risk_ratio"]].drop_duplicates()
    for keys in groups.itertuples(index=False, name=None):
        cell = detected[(detected["scenario"] == keys[0]) &
                        (detected["family"] == keys[1]) &
                        (detected["risk_ratio"] == keys[2])]
        for metric in ("localization_error", "detection_delay"):
            values = pd.to_numeric(cell[metric], errors="coerce").dropna().to_numpy(float)
            if values.size < 2:
                # A cell with fewer than two timely detections has a genuine
                # non-estimable conditional median; retain it transparently
                # instead of dropping the scenario or aborting the full plot.
                estimate = float(np.median(values)) if values.size else np.nan
                lower = upper = np.nan
                n = int(values.size)
            else:
                estimate, lower, upper, n = _median_ci(
                    values, f"{keys}|{metric}|P{window}"
                )
            rows.append({"scenario": keys[0], "family": keys[1], "risk_ratio": keys[2], "metric": f"median_{metric}_given_P{window}", "estimate": estimate, "ci_lower": lower, "ci_upper": upper, "n": n})
    return pd.concat([p28, pd.DataFrame(rows)], ignore_index=True)


def derive_figure5(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    e5 = frame[frame["experiment"] == "E5"].copy()
    e5["outcome"] = np.where(e5["purpose"].eq("null"), pd.to_numeric(e5["alarm_day"], errors="coerce").notna().astype(int), _p28(e5, window))
    absolute = _summarize_binary(e5, ["comparison", "scenario", "purpose", "method"], "outcome", "probability")
    differences: list[dict[str, Any]] = []
    for keys, cell in e5[e5["purpose"] == "alternative"].groupby(["comparison", "scenario"], sort=True):
        pivot = cell.pivot(index=["replicate", "shared_stream_hash"], columns="method", values="outcome")
        reference = "OCFR-alpha-matched"
        if keys[0] == "benchmark" and reference not in pivot:
            raise EvidenceError(
                f"E5 {keys} lacks {reference} for common-size paired P{window} contrasts"
            )
        if keys[0] != "benchmark":
            continue
        for comparator in [name for name in COMPARATOR_ORDER if name in pivot]:
            pair = pivot[[reference, comparator]].dropna()
            estimate, lower, upper, n = _mean_ci(pair[reference] - pair[comparator])
            differences.append({"comparison": keys[0], "scenario": keys[1], "purpose": "alternative", "method": f"OCFR alpha-matched minus {comparator}", "metric": f"paired_P{window}_difference", "estimate": estimate, "ci_lower": lower, "ci_upper": upper, "n": n})
    if not differences:
        raise EvidenceError("E5 contains no paired benchmark alternatives")
    return pd.concat([absolute, pd.DataFrame(differences)], ignore_index=True, sort=False)


def derive_figure6(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    e6 = frame[frame["experiment"] == "E6"].copy()
    e6["outcome"] = np.where(e6["purpose"].eq("alternative"), _p28(e6, window), pd.to_numeric(e6["alarm_day"], errors="coerce").notna().astype(int))
    return _summarize_binary(e6, ["perturbation_type", "perturbation_label", "perturbation_order", "scenario", "purpose"], "outcome", "probability")


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "legend.fontsize": 8.2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.1,
        "lines.linewidth": 1.7, "lines.markersize": 5, "legend.frameon": False,
        "figure.dpi": 160, "savefig.dpi": 400, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })


def _panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.14, 1.07, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
    # Panel meanings are stated in the manuscript caption.  Keeping only the
    # panel letter avoids duplicative in-figure titles and preserves plotting
    # area at the final journal width.
    ax.grid(axis="y", color="0.88", linewidth=0.7)


def _scenario_label(value: Any) -> str:
    """Translate internal cache identifiers into compact reader-facing labels."""
    labels = {
        "S01_null": "Sparse",
        "W01_null": "Wave",
        "T01_null": "Typical",
        "T02_rr067": "RR 0.67",
        "T03_rr125": "RR 1.25",
        "T04_rr150": "RR 1.50",
        "T05_rr200": "RR 2.00",
        "T06_early_rr150": "Early, RR 1.50",
        "T07_late_rr150": "Late, RR 1.50",
        "S02_rr200": "Sparse, RR 2.00",
        "W02_rising_rr150": "Rising wave",
        "W03_falling_rr150": "Falling wave",
        "N1_typical": "Typical",
        "N2_sparse": "Sparse",
        "N5_wave": "Wave",
        "N01_typical": "Typical",
        "N02_sparse_deaths": "Sparse",
        "N05_single_wave": "Wave",
    }
    return labels.get(str(value), str(value).replace("_", " "))


def _estimation_label(scenario: Any, parameter: Any) -> str:
    regime = {"typical": "Typical", "sparse": "Sparse", "late_overdispersed": "Late/OD"}.get(str(scenario), str(scenario))
    quantity = {"pi_left": r"$p_L$", "pi_right": r"$p_R$", "risk_ratio": "RR"}.get(str(parameter), str(parameter))
    return f"{regime}\n{quantity}"


def _comparison_scenario_label(value: Any) -> str:
    labels = {
        "S02_rr200": "Sparse×2",
        "T02_rr067": "×0.67", "T03_rr125": "×1.25",
        "T04_rr150": "×1.50", "T05_rr200": "×2.00",
        "T06_early_rr150": "Early", "T07_late_rr150": "Late",
        "W02_rising_rr150": "Rising", "W03_falling_rr150": "Falling",
    }
    return labels.get(str(value), _scenario_label(value))


def _errorbar(ax: plt.Axes, x: Sequence[Any], data: pd.DataFrame, *, color: str, marker: str = "o", label: str | None = None, connect: bool = False) -> None:
    estimate = data["estimate"].to_numpy(float)
    lower, upper = data["ci_lower"].to_numpy(float), data["ci_upper"].to_numpy(float)
    yerr = np.vstack((estimate - lower, upper - estimate)) if np.isfinite(lower).all() and np.isfinite(upper).all() else None
    ax.errorbar(x, estimate, yerr=yerr, color=color, marker=marker, capsize=2.5, label=label, linestyle="-" if connect else "none")


def _save(fig: plt.Figure, output: Path, basename: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    fig.align_labels()
    fig.tight_layout(pad=1.05)
    paths = [output / f"{basename}.pdf", output / f"{basename}.png"]
    fig.savefig(paths[0], bbox_inches="tight", pad_inches=0.05)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.05, dpi=400)
    plt.close(fig)
    return paths


def figure2_calibration_estimation(data: pd.DataFrame, output: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    fixed = data[(data["experiment"] == "E1") & data["component"].eq("fixed_location")]
    scanned = data[(data["experiment"] == "E1") & data["component"].eq("scanned")]
    if fixed.empty or scanned.empty:
        raise EvidenceError("Fig. 2 requires E1 components fixed_location and scanned")
    for ax, subset, letter, title in ((axes[0, 0], fixed, "A", "Fixed-location null calibration"), (axes[0, 1], scanned, "B", "Scanned-statistic null calibration")):
        subset = subset.sort_values("scenario")
        x = np.arange(len(subset))
        _errorbar(ax, x, subset, color=COLORS["OCFR"])
        ax.scatter(x, subset["nominal_alpha"], facecolors="white", edgecolors="black", marker="s", label="Nominal")
        ax.set_xticks(x, [_scenario_label(v) for v in subset["scenario"]], rotation=0)
        ax.set_ylabel("Null rejection probability")
        ax.set_ylim(bottom=0)
        _panel(ax, letter, title)
    e2 = data[data["experiment"] == "E2"]
    coverage = e2[e2["metric"] == "coverage"].sort_values(["scenario", "parameter"])
    bias = e2[e2["metric"] == "relative_bias"].sort_values(["scenario", "parameter"])
    scenario_order = [s for s in ("typical", "sparse", "late_overdispersed") if s in set(e2["scenario"])]
    parameter_order = [p for p in ("pi_left", "pi_right", "risk_ratio") if p in set(e2["parameter"])]
    parameter_labels = {"pi_left": r"$p_L$", "pi_right": r"$p_R$", "risk_ratio": "RR"}
    parameter_colors = {"pi_left": "#0F4D92", "pi_right": "#42949E", "risk_ratio": "#B64342"}
    for ax, subset, letter, title, ylabel, reference in (
        (axes[1, 0], coverage, "C", "Fixed-location interval estimation", "95% interval coverage", 0.95),
        (axes[1, 1], bias, "D", "Fixed-location point estimation", "Mean relative bias", 0.0),
    ):
        centers = np.arange(len(scenario_order), dtype=float)
        offsets = np.linspace(-0.18, 0.18, len(parameter_order))
        for parameter, offset in zip(parameter_order, offsets):
            group = subset[subset["parameter"] == parameter].set_index("scenario").reindex(scenario_order)
            _errorbar(ax, centers + offset, group, color=parameter_colors[parameter], marker={"pi_left": "o", "pi_right": "s", "risk_ratio": "^"}[parameter], label=parameter_labels[parameter])
        ax.axhline(reference, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(centers, ["Typical", "Sparse", "Late/OD"][:len(scenario_order)])
        ax.set_ylabel(ylabel)
        _panel(ax, letter, title)
    axes[1, 0].legend(title="Parameter", ncol=3, loc="lower left", fontsize=7.5, title_fontsize=7.6)
    axes[1, 0].set_ylim(0, 1.02)
    return _save(fig, output, "Fig2_calibration_estimation")


def figure3_sequential(data: pd.DataFrame, output: Path) -> list[Path]:
    fig, axes_array = plt.subplots(2, 2, figsize=(7.2, 5.4))
    axes = list(axes_array.ravel())
    path_data = data[data["metric"] == "cumulative_episode_error"].copy()
    pooled_rows = []
    for keys, cell in path_data.groupby(["scenario", "family", "look_day"], sort=True):
        pooled_rows.append({"scenario": keys[0], "family": keys[1], "look_day": keys[2], "estimate": np.average(cell["estimate"], weights=cell["n"])})
    pooled = pd.DataFrame(pooled_rows)
    family_styles = {"sparse": ("o", "-"), "typical": ("s", "--"), "wave": ("^", "-.")}
    for family, group in pooled.groupby("family", sort=True):
        group = group.sort_values("look_day")
        marker, linestyle = family_styles.get(str(family), ("o", "-"))
        axes[0].plot(group["look_day"], group["estimate"], marker=marker, linestyle=linestyle, label=str(family).capitalize())
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Monitoring day", ylabel="Cumulative episode error")
    axes[0].legend()
    _panel(axes[0], "A", "Error across monitoring looks")
    final = path_data.loc[path_data.groupby(["scenario", "path_id"])["look_day"].idxmax()].sort_values(["family", "scenario", "path_id"])
    cell_rows = []
    for keys, cell in final.groupby(["family", "scenario"], sort=True):
        cell_rows.append({"family": keys[0], "scenario": keys[1], "estimate": np.average(cell["estimate"], weights=cell["n"]), "minimum": cell["estimate"].min(), "maximum": cell["estimate"].max()})
    cells = pd.DataFrame(cell_rows)
    labels = [str(v).capitalize() for v in cells["family"]]
    x = np.arange(len(cells))
    axes[1].errorbar(x, cells["estimate"], yerr=np.vstack((cells["estimate"] - cells["minimum"], cells["maximum"] - cells["estimate"])), fmt="o", color=COLORS["OCFR"], capsize=3)
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, labels, rotation=0)
    axes[1].set_ylabel("Final episode error")
    _panel(axes[1], "B", "Sentinel null cells")
    families = list(dict.fromkeys(final["family"].astype(str)))
    values = [final.loc[final["family"].astype(str) == family, "estimate"].to_numpy(float) for family in families]
    axes[2].boxplot(values, widths=0.55, patch_artist=True, boxprops={"facecolor": "#DDEAF5"}, medianprops={"color": COLORS["OCFR"]})
    axes[2].set_xticks(np.arange(1, len(families) + 1), families)
    axes[2].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Path-specific episode error")
    _panel(axes[2], "C", "Case-path heterogeneity")
    multiplier = data[data["experiment"] == "E3M"].copy()
    multiplier = multiplier[multiplier["metric"] == "p95_relative_critical_error"]
    if multiplier.empty:
        raise EvidenceError("Fig. 3 lacks the multiplier-resolution diagnostic")
    for scenario, group in multiplier.groupby("scenario", sort=True):
        group = group.sort_values("multiplier_size")
        axes[3].plot(group["multiplier_size"], 100 * group["estimate"], marker="o", label=_scenario_label(str(scenario)))
    axes[3].set(xlabel="Multiplier draws", ylabel="95th percentile relative error (%)")
    axes[3].set_xscale("log", base=2)
    multiplier_ticks = sorted(multiplier["multiplier_size"].dropna().astype(int).unique())
    axes[3].set_xticks(multiplier_ticks, [f"{value:,}" for value in multiplier_ticks])
    axes[3].legend(fontsize=7.2)
    _panel(axes[3], "D", "Multiplier-tail resolution")
    return _save(fig, output, "Fig3_sequential_path_heterogeneity")


def figure4_operating(data: pd.DataFrame, output: Path, window: int) -> list[Path]:
    fig = plt.figure(figsize=(7.2, 5.2))
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    p28 = data[data["metric"] == f"P{window}"]
    primary_effect = {"T02_rr067", "T03_rr125", "T04_rr150", "T05_rr200"}
    p28_curve = p28[p28["scenario"].astype(str).isin(primary_effect)]
    palette = ["#0F4D92", "#42949E", "#B64342", "#9A4D8E"]
    for (family, group), color in zip(p28_curve.groupby("family", sort=True), palette):
        group = group.sort_values("risk_ratio")
        _errorbar(axes[0], group["risk_ratio"], group, color=color, label=family, connect=True)
    axes[0].axvline(1, color="0.45", linestyle=":", linewidth=1)
    axes[0].set(xlabel="Post/pre risk ratio", ylabel=f"Timely detection, P{window}", ylim=(0, 1.02))
    axes[0].legend()
    _panel(axes[0], "A", f"P{window} by effect and regime")
    for ax, prefix, ylabel, letter in ((axes[1], "median_detection_delay", "Detection delay (days)", "B"), (axes[2], "median_localization_error", "Localization error (days)", "C")):
        subset = data[data["metric"].str.startswith(prefix, na=False)].sort_values(["family", "risk_ratio"])
        labels = [_scenario_label(v) for v in subset["scenario"]]
        yerr = np.vstack((subset["estimate"] - subset["ci_lower"], subset["ci_upper"] - subset["estimate"]))
        ax.bar(np.arange(len(subset)), subset["estimate"], yerr=yerr, capsize=2.5, color="#3775BA", edgecolor="black", linewidth=0.6)
        ax.set_xticks(np.arange(len(subset)), labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        _panel(ax, letter, ("Timely detection delay" if letter == "B" else "Timely localization error"))
    return _save(fig, output, "Fig4_OCFR_operating_characteristics")


def figure5_comparison(data: pd.DataFrame, output: Path, window: int) -> list[Path]:
    """Four-panel, scenario-specific size-matched method comparison.

    The primary OCFR rule uses its theoretical Gaussian-process boundary and is
    intentionally conservative; it is retained as the proposed operational
    rule.  OCFR-AM is the size-matched OCFR variant used for fair power contrasts
    against the empirically calibrated benchmark procedures.
    """
    # The source canvas and explicit type sizes are chosen so that all labels
    # remain at least about 8 pt after inclusion at 0.90\textwidth.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8))
    absolute = data[data["metric"] == "probability"]
    null = absolute[(absolute["purpose"] == "null") & absolute["comparison"].eq("benchmark")]
    methods = [m for m in METHOD_ORDER if m in set(null["method"])]
    if set(methods) != set(METHOD_ORDER):
        raise EvidenceError(f"Fig. 5 benchmark/null is incomplete; missing methods: {sorted(set(METHOD_ORDER) - set(methods))}")
    null_scenarios = sorted(null["scenario"].astype(str).unique())
    offsets = np.linspace(-0.22, 0.22, len(methods))
    for index, method in enumerate(methods):
        group = null[null["method"] == method].set_index("scenario").reindex(null_scenarios)
        display_label = "OCFR (primary)" if method == "OCFR" else (
            "OCFR-AM" if method == "OCFR-alpha-matched" else method
        )
        _errorbar(axes[0, 0], np.arange(len(null_scenarios)) + offsets[index],
                  group, color=COLORS[method],
                  marker=["o", "s", "^", "D", "P"][index], label=display_label)
    axes[0, 0].axhspan(0.04, 0.06, color="0.92", zorder=0)
    axes[0, 0].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_xticks(np.arange(len(null_scenarios)),
                          [_scenario_label(v) for v in null_scenarios])
    axes[0, 0].set_ylabel("Episode-level type-I error")
    null_upper = float(pd.to_numeric(null["ci_upper"], errors="coerce").max())
    # Start at zero so conservative primary-OCFR estimates and their intervals
    # remain visible.  A truncated lower limit previously hid the Typical point
    # (0.015) and nearly clipped the Wave point (0.024).
    axes[0, 0].set_ylim(0.0, max(0.075, null_upper + 0.003))
    axes[0, 0].legend(ncol=2, fontsize=9.8, loc="upper center",
                      bbox_to_anchor=(0.5, 1.16))
    axes[0, 0].text(0.98, 0.04, "target .05; grey band .04--.06",
                    transform=axes[0, 0].transAxes, ha="right", va="bottom",
                    fontsize=9.6)
    _panel(axes[0, 0], "A", "Held-out size before power")

    alternative = absolute[(absolute["purpose"] == "alternative") & absolute["comparison"].eq("benchmark")]
    scenarios = sorted(alternative["scenario"].astype(str).unique())
    if set(alternative["method"]) != set(METHOD_ORDER) or not scenarios:
        raise EvidenceError("Fig. 5 benchmark/alternative is incomplete")
    power_matrix = alternative.pivot(index="scenario", columns="method", values="estimate").reindex(index=scenarios, columns=METHOD_ORDER)
    image_b = axes[0, 1].imshow(power_matrix.to_numpy(float), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axes[0, 1].set_yticks(np.arange(len(scenarios)), [_comparison_scenario_label(v) for v in scenarios])
    axes[0, 1].set_xticks(np.arange(len(METHOD_ORDER)), ["OCFR", "OCFR-AM", "NB-GLR", "Page", "FOCuS"])
    axes[0, 1].tick_params(axis="x", labelrotation=28, labelsize=9.4)
    axes[0, 1].tick_params(axis="y", labelsize=9.4)
    plt.setp(axes[0, 1].get_xticklabels(), ha="right", rotation_mode="anchor")
    for row in range(power_matrix.shape[0]):
        for col in range(power_matrix.shape[1]):
            value = float(power_matrix.iloc[row, col])
            axes[0, 1].text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9.6, color="white" if value > 0.55 else "black")
    _panel(axes[0, 1], "B", "Absolute timely detection")
    difference = data[data["metric"] == f"paired_P{window}_difference"].sort_values(["scenario", "method"])
    comparators = ["NB-GLR", "Page-CUSUM", "FOCuS"]
    diff_names = [f"OCFR alpha-matched minus {name}" for name in comparators]
    # Panel B already gives every scenario-specific probability.  Panel C
    # therefore summarizes the 27 paired contrasts in three uncluttered rows:
    # one open circle per alternative, its range, and the across-alternative
    # median.  The derived CSV retains the scenario-specific confidence
    # intervals for readers who need the numerical detail.
    y = np.arange(len(comparators), dtype=float)
    for row, (comparator, diff_name) in enumerate(zip(comparators, diff_names)):
        group = difference[difference["method"] == diff_name].set_index("scenario").reindex(scenarios)
        estimate = group["estimate"].to_numpy(float)
        jitter = np.linspace(-0.11, 0.11, len(estimate))
        axes[1, 0].hlines(row, estimate.min(), estimate.max(),
                          color=COLORS[comparator], linewidth=1.2, alpha=0.65)
        axes[1, 0].scatter(
            estimate, row + jitter, s=19, marker="o", facecolors="white",
            edgecolors=COLORS[comparator], linewidths=0.9, zorder=3,
        )
        axes[1, 0].scatter(
            np.median(estimate), row, s=38, marker="D",
            color=COLORS[comparator], edgecolors="black", linewidths=0.45,
            zorder=4,
        )
    axes[1, 0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_yticks(y, ["NB-GLR", "Page", "FOCuS"])
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel(
        rf"Paired difference in $P_{{{window}}}$" + "\n(OCFR-AM minus comparator)"
    )
    axes[1, 0].grid(axis="x", color="0.88", linewidth=0.7)
    _panel(axes[1, 0], "C", "Paired advantage across alternatives")
    ablation = absolute[(absolute["purpose"] == "alternative") & absolute["comparison"].eq("alignment_ablation")]
    methods = [m for m in ("Aligned", "Unaligned") if m in set(ablation["method"])]
    if len(methods) != 2:
        raise EvidenceError("alignment_ablation requires methods Aligned and Unaligned")
    ablation_matrix = ablation.pivot(index="scenario", columns="method", values="estimate").reindex(index=scenarios, columns=methods)
    image_d = axes[1, 1].imshow(ablation_matrix.to_numpy(float), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axes[1, 1].set_yticks(np.arange(len(scenarios)), [_comparison_scenario_label(v) for v in scenarios])
    for row in range(ablation_matrix.shape[0]):
        for col in range(ablation_matrix.shape[1]):
            value = float(ablation_matrix.iloc[row, col])
            axes[1, 1].text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9.6, color="white" if value > 0.55 else "black")
    axes[1, 1].set_xticks(np.arange(len(methods)), methods)
    axes[1, 1].tick_params(axis="x", labelrotation=0, labelsize=9.8)
    axes[1, 1].tick_params(axis="y", labelsize=9.4)
    plt.setp(axes[1, 1].get_xticklabels(), ha="right", rotation_mode="anchor")
    _panel(axes[1, 1], "D", "Cohort-alignment mechanism ablation")
    return _save(fig, output, "Fig5_size_comparable_comparison")


def figure6_tolerance(data: pd.DataFrame, output: Path, window: int) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    delay = data[data["perturbation_type"] == "delay_error"]
    for ax, purpose, letter, ylabel in ((axes[0, 0], "null", "A", "Episode-level type-I error"), (axes[0, 1], "alternative", "B", f"P{window}")):
        subset = delay[delay["purpose"] == purpose].sort_values("perturbation_order")
        if subset.empty: raise EvidenceError(f"Fig. 6 lacks delay_error/{purpose}")
        _errorbar(ax, subset["perturbation_order"], subset, color=COLORS["OCFR"], connect=True)
        if purpose == "null": ax.axhline(0.05, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(subset["perturbation_order"], subset["perturbation_label"])
        ax.set(xlabel="Assumed minus true delay (days)", ylabel=ylabel, ylim=(0, 1.02))
        _panel(ax, letter, "Delay-distribution tolerance")
    other = data[data["perturbation_type"] != "delay_error"]
    for ax, purpose, letter, title, ylabel in ((axes[1, 0], "null", "C", "Other observation-model checks", "Episode-level type-I error"), (axes[1, 1], "alternative", "D", "Power under matched disturbances", f"P{window}")):
        selected_purposes = [purpose, "negative_control"] if purpose == "null" else [purpose]
        baseline = delay[(delay["purpose"] == purpose)
                         & (delay["perturbation_order"] == 0)].copy()
        baseline["perturbation_label"] = "On-model"
        baseline["perturbation_order"] = -1
        subset = pd.concat(
            [baseline, other[other["purpose"].isin(selected_purposes)]],
            ignore_index=True,
        ).sort_values(["perturbation_order", "perturbation_label"])
        if subset.empty: raise EvidenceError(f"Fig. 6 lacks non-delay/{purpose} rows")
        x = np.arange(len(subset))
        _errorbar(ax, x, subset, color=COLORS["OCFR"], connect=False)
        if purpose == "null": ax.axhline(0.05, color="black", linestyle="--", linewidth=1)
        compact = {"On-model": "On-model", "1-day lag": "1-day\nlag", "Cohort binomial": "Cohort\nbinomial", "+25% ascertainment": "+25%\nascertainment"}
        ax.set_xticks(x, [compact.get(str(v), str(v)) for v in subset["perturbation_label"]], rotation=0)
        if purpose == "null":
            ylabel = "Alarm probability\n(no CFR change)"
        ax.set(ylabel=ylabel, ylim=(0, 1.02))
        _panel(ax, letter, title)
    return _save(fig, output, "Fig6_operational_tolerance")


def analyze(input_dir: Path, output_dir: Path, *, timely_window: int = 28) -> dict[str, pd.DataFrame]:
    if timely_window <= 0:
        raise ValueError("timely_window must be positive")
    frame, source_manifest, digest = load_evidence(input_dir.resolve())
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    derived = {
        "Fig2": derive_figure2(frame), "Fig3": derive_figure3(frame),
        "Fig4": derive_figure4(frame, timely_window),
        "Fig5": derive_figure5(frame, timely_window),
        "Fig6": derive_figure6(frame, timely_window),
    }
    derived_paths: list[Path] = []
    for name, table in derived.items():
        path = output / f"{name}_derived.csv"
        _atomic_csv(table, path)
        derived_paths.append(path)
    _style()
    figures = []
    figures += figure2_calibration_estimation(derived["Fig2"], output)
    figures += figure3_sequential(derived["Fig3"], output)
    figures += figure4_operating(derived["Fig4"], output, timely_window)
    figures += figure5_comparison(derived["Fig5"], output, timely_window)
    figures += figure6_tolerance(derived["Fig6"], output, timely_window)
    outputs = derived_paths + figures
    manifest = {
        "analysis_schema_version": 2,
        "source_directory": str(input_dir.resolve()),
        "source_scientific_digest": digest,
        "source_manifest_sha256": _sha256(input_dir.resolve() / MANIFEST_FILE),
        "source_raw_sha256": _sha256(input_dir.resolve() / RAW_FILE),
        "timely_estimand": f"P{timely_window}: Pr(true_change <= alarm_day <= true_change + {timely_window})",
        "p28_excludes_premature_alarms": True,
        "source_mode": source_manifest.get("mode"),
        "outputs": {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in outputs},
    }
    _atomic_json(manifest, output / "analysis_manifest.json")
    return derived


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).parent / "outputs"
    parser.add_argument("--input", type=Path, default=base / "simulation")
    parser.add_argument("--output", type=Path, default=base / "analysis")
    parser.add_argument("--timely-window", type=int, default=28)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    derived = analyze(args.input, args.output, timely_window=args.timely_window)
    for name, table in derived.items():
        print(f"{name}: {len(table)} derived rows")
    print(f"wrote validated figures, tables, and manifest to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
