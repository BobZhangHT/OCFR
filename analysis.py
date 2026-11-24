"""Validate, summarize, and plot audited OCFR simulation evidence packages.

The input directory must contain ``replicate_results.csv`` and ``manifest.json``.
Old or partially documented results are rejected. The original branch's Figures
2--6 implement the streamlined PLOS evidence plan. Its alternative detection is P28: an
alarm from the true change through 28 days afterwards, never arbitrary alarm
probability.

The named ``compact-v1`` and ``compact-v2`` branches read independent calibration, audit, and
evaluation stages. Its useful detection probability uses the entire remaining
episode and its delay curves retain every requested replicate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import beta, binom, t as student_t
from scipy.special import expit, logit
from config import ORACLE_OCFR, PAIRED_DETECTION_METHODS as COMPACT_PAIRED_METHODS


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
    rule. OCFR with empirical calibration supplies the size-matched contrasts
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
        display_label = "OCFR (Gaussian)" if method == "OCFR" else (
            "OCFR (empirical)" if method == "OCFR-alpha-matched" else method
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
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=3, fontsize=9.0, loc="upper center",
        bbox_to_anchor=(0.5, 1.075), columnspacing=1.15, handletextpad=0.35,
    )
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
    axes[0, 1].set_xticks(np.arange(len(METHOD_ORDER)), ["OCFR\nGauss.", "OCFR\nEmp.", "NB-GLR", "Page", "FOCuS"])
    axes[0, 1].tick_params(axis="x", labelrotation=0, labelsize=9.4)
    axes[0, 1].tick_params(axis="y", labelsize=9.4)
    plt.setp(axes[0, 1].get_xticklabels(), ha="center")
    for row in range(power_matrix.shape[0]):
        for col in range(power_matrix.shape[1]):
            value = float(power_matrix.iloc[row, col])
            axes[0, 1].text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9.6, color="white" if value > 0.55 else "black")
    _panel(axes[0, 1], "B", "Absolute timely detection")
    axes[0, 1].grid(False)
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
        rf"Paired difference in $P_{{{window}}}$" + "\n(OCFR empirical minus comparator)"
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
    axes[1, 1].grid(False)
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


COMPACT_METHOD_ORDER = ("OCFR Gaussian", "OCFR empirical", "NB-GLR", "Page-CUSUM",
                        "Report-day score")
COMPACT_MAIN_METHODS = COMPACT_METHOD_ORDER[:4]
COMPACT_COLORS = dict(zip(COMPACT_METHOD_ORDER,
    ("#0072B2", "#56B4E9", "#D55E00", "#009E73", "#CC79A7")))
COMPACT_MARKERS = dict(zip(COMPACT_METHOD_ORDER, ("o", "s", "^", "D", "v")))
COMPACT_LINESTYLES = dict(zip(COMPACT_METHOD_ORDER, ("-", "--", "-.", ":", "--")))


COMPACT_COLORS["OCFR"] = COMPACT_COLORS["OCFR Gaussian"]
COMPACT_MARKERS["OCFR"] = COMPACT_MARKERS["OCFR Gaussian"]
COMPACT_LINESTYLES["OCFR"] = COMPACT_LINESTYLES["OCFR Gaussian"]

COMPACT_COLORS[ORACLE_OCFR] = "#163D62"
COMPACT_MARKERS[ORACLE_OCFR] = "s"
COMPACT_LINESTYLES[ORACLE_OCFR] = "--"
COMPACT_PAIRED_LINEWIDTHS = {"OCFR": 2.2, ORACLE_OCFR: 1.15}
COMPACT_PAIRED_LABELS = {"OCFR": "Practical OCFR", ORACLE_OCFR: "Oracle OCFR",
                         "Page-CUSUM": "Page CUSUM"}


def _compact_method_handles(methods: Sequence[str]) -> list[plt.Line2D]:
    paired = ORACLE_OCFR in methods
    return [plt.Line2D([], [], color=COMPACT_COLORS[method], marker=COMPACT_MARKERS[method],
                      linestyle=COMPACT_LINESTYLES[method], markersize=4,
                      linewidth=COMPACT_PAIRED_LINEWIDTHS.get(method) if paired else None,
                      label=COMPACT_PAIRED_LABELS.get(method, method) if paired else method)
            for method in methods]


def compact_binomial_interval(k: int, n: int) -> tuple[float, float]:
    """Two-sided 95% Clopper--Pearson interval, including boundary cases."""
    if n < 1 or not 0 <= k <= n:
        raise ValueError("binomial counts require 0 <= k <= n and n > 0")
    return (0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1. if k == n else float(beta.ppf(.975, k+1, n-k)))


def _compact_read_summary_csv(path: Path, columns: set[str], *,
                              chunksize: int = 4096) -> pd.DataFrame:
    """Keep only compact metadata, with bounded parser buffers for long raw rows."""
    if chunksize < 1:
        raise ValueError("summary CSV chunk size must be positive")
    # Seed identifiers are not measurements. Explicit strings preserve all 64
    # bits even if different chunks otherwise infer signed/unsigned integers.
    with pd.read_csv(path, usecols=lambda column: column in columns,
                     dtype={"case_seed": "string", "death_seed": "string"},
                     chunksize=chunksize) as reader:
        pieces = list(reader)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _compact_load_stage(directory: Path, stage: str) -> tuple[pd.DataFrame, dict]:
    """Validate the complete declared grid; failures are observations, not exclusions."""
    manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("protocol") not in {"compact-v1", "compact-v2"}:
        raise EvidenceError(f"{directory}: expected a named compact protocol")
    if manifest.get("scientific_config", {}).get("protocol") != manifest["protocol"]:
        raise EvidenceError("manifest/scientific configuration protocols disagree")
    if manifest.get("stage") != stage or manifest.get("status") != "complete":
        raise EvidenceError(f"{directory}: {stage} stage is not complete")
    if manifest["protocol"] == "compact-v2" and not manifest.get("synthetic_fixture", False):
        contract = manifest.get("scientific_config", {}).get("compact_contract", {})
        if contract.get("boundary_policy") != "gaussian-first-crossing-linear-support-discard-v1":
            raise EvidenceError("v2 spending policy provenance is absent or incompatible")
        if contract.get("stage_methods") != {s: list(_compact_method_contract("compact-v2", s))
                for s in ("calibration", "audit", "evaluation")}:
            raise EvidenceError("v2 scientific stage-method contract mismatch")
    raw = directory / RAW_FILE
    if _sha256(raw) != _manifest_file_hash(manifest, RAW_FILE):
        raise EvidenceError(f"{directory}: raw-file hash mismatch")
    required = {"scenario", "family", "method", "replicate", "status", "alarm_day",
                "true_change", "horizon", "risk_ratio", "split", "misspecification"}
    optional = {"scientific_digest", "true_change_day", "all_looks_unavailable",
                "unavailable_looks", "true_delay_mean_days", "assumed_delay_mean_days",
                "case_seed", "death_seed", "tau_hat_at_alarm", "phi"}
    # Whole-file SHA-256 above still binds traces, but summary generation need
    # not materialize hundreds of thousands of long trace JSON strings.
    frame = _compact_read_summary_csv(raw, required | optional)
    _validate_scientific_digest(manifest, frame)
    missing = required - set(frame)
    if missing or frame.empty:
        raise EvidenceError(f"{directory}: absent compact rows/columns: {sorted(missing)}")
    keys = ["scenario", "method", "replicate"]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise EvidenceError("missing or duplicate compact episode identifiers")
    scenarios = manifest.get("scenarios", [])
    names = {x["name"] if isinstance(x, dict) else x for x in scenarios}
    if names != set(frame.scenario):
        raise EvidenceError("compact manifest/results scenario mismatch")
    expected_methods = _compact_method_contract(manifest["protocol"], stage)
    methods = tuple(manifest.get("methods", expected_methods))
    if set(methods) != set(expected_methods):
        raise EvidenceError("compact stage methods disagree with the declared protocol")
    counts = manifest.get("replications_by_scenario", {})
    default_n = manifest.get("replications_per_scenario", manifest.get("replications"))
    if not manifest.get("synthetic_fixture", False):
        definitions = manifest["scientific_config"].get("scenarios", [])
        expected_names = {s["name"] for s in definitions if
            (s["split"] == "size_audit" if stage == "audit" else
             s["split"] in {"holdout", "alternative"})}
        if names != expected_names:
            raise EvidenceError("publication analysis requires the complete compact scenario bank")
    for scenario, group in frame.groupby("scenario", sort=False):
        n = counts.get(scenario, default_n)
        if not isinstance(n, int) or n < 1:
            raise EvidenceError(f"{scenario}: requested replication count is absent")
        if not manifest.get("synthetic_fixture", False):
            expected_n = _compact_declared_n(manifest, str(group["split"].iloc[0]))
            if n != expected_n:
                raise EvidenceError(f"{scenario}: replication count violates compact mode contract")
        for method in methods:
            ids = group.loc[group.method == method, "replicate"]
            if len(ids) != n or set(ids) != set(range(n)):
                raise EvidenceError(f"{scenario}/{method}: incomplete requested replication grid")
        if set(group.method) != set(methods):
            raise EvidenceError(f"{scenario}: unexpected method")
        metadata = ["family", "true_change", "horizon", "risk_ratio", "split", "misspecification"]
        if any(group[column].nunique(dropna=False) != 1 for column in metadata):
            raise EvidenceError(f"{scenario}: inconsistent scenario metadata")
        for column in ("case_seed", "death_seed"):
            if column in group and (group.groupby("replicate")[column].nunique(dropna=False) != 1).any():
                raise EvidenceError(f"{scenario}: methods do not share paired {column} streams")
    return _compact_episode_metrics(frame), manifest


def _compact_episode_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert zero-based episode indices and compute first-alarm indicators."""
    for column in ("alarm_day", "true_change", "horizon", "risk_ratio"):
        converted = pd.to_numeric(frame[column], errors="coerce")
        if (frame[column].notna() & converted.isna()).any():
            raise EvidenceError(f"invalid numeric compact field {column}")
        frame[column] = converted
    if frame[["true_change", "horizon", "risk_ratio"]].isna().any().any():
        raise EvidenceError("compact scenario truth is missing")
    # All comparisons use manuscript day labels. The cohort index is zero-based.
    frame["change_day"] = frame["true_change"] + 1
    # Runner trace/alarm indexes are zero-based; manuscript labels are one-based.
    frame["alarm_day"] = frame["alarm_day"] + 1
    if "true_change_day" in frame and not np.allclose(frame.true_change_day, frame.change_day):
        raise EvidenceError("zero-based true_change and true_change_day disagree")
    alarm = frame.alarm_day.notna()
    if (alarm & ((frame.alarm_day < 1) | (frame.alarm_day > frame.horizon)
                  | ~np.isfinite(frame.alarm_day))).any():
        raise EvidenceError("alarm lies outside the surveillance episode")
    if "all_looks_unavailable" in frame:
        unavailable = _as_binary(frame.all_looks_unavailable, "all_looks_unavailable").astype(bool)
        if (alarm & unavailable).any():
            raise EvidenceError("an episode with all looks unavailable cannot alarm")
    success = frame.status.eq("success")
    if (alarm & ~success).any():
        raise EvidenceError("failed job cannot contribute an alarm")
    frame["failed"] = ~success
    frame["any_alarm"] = alarm
    frame["alternative"] = frame["split"].eq("alternative")
    frame["premature"] = frame.alternative & alarm & (frame.alarm_day < frame.change_day)
    frame["useful"] = frame.alternative & alarm & (frame.alarm_day >= frame.change_day)
    frame["delay"] = np.where(frame.useful, frame.alarm_day-frame.change_day, np.nan)
    return frame



def _compact_method_contract(protocol: str, stage: str) -> tuple[str, ...]:
    if protocol == "compact-v1":
        return COMPACT_METHOD_ORDER if stage == "evaluation" else COMPACT_METHOD_ORDER[1:]
    if protocol == "compact-v2":
        return ("NB-GLR", "Page-CUSUM") if stage == "calibration" else ("OCFR", "NB-GLR", "Page-CUSUM")
    raise EvidenceError(f"unsupported compact protocol: {protocol}")


def _compact_declared_n(manifest: Mapping[str, Any], split: str) -> int:
    scientific = manifest.get("scientific_config", {})
    plan = scientific.get("split_replications")
    if plan is not None:
        try:
            value = dict(plan)[split]
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceError("invalid explicit replication plan") from exc
        if type(value) is not int or value < 1:
            raise EvidenceError("invalid explicit replication count")
        return value
    mode = manifest.get("mode")
    if mode not in {"demo", "full"}:
        raise EvidenceError("compact mode must be demo or full")
    if manifest.get("protocol") == "compact-v1" and split in {"calibration", "size_audit"}:
        return 10000 if split == "calibration" else 5000
    return 10 if mode == "demo" else 1000

def _compact_probability(values: Sequence) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    n, k = len(values), int(values.sum())
    lo, hi = compact_binomial_interval(k, n)
    estimate = k/n
    return {"n": n, "events": k, "estimate": estimate, "ci_low": lo, "ci_high": hi,
            "mcse": float(np.sqrt(estimate*(1-estimate)/n))}


def compact_null_quantile_bracket(scores: Sequence[float], *, q: float = .95,
                                  error: float = .01) -> dict[str, float]:
    """CDF-inverse estimate and distribution-free order-statistic bracket.

    Ranks are one-based; ranks zero/n+1 denote infinite sentinels. Ties are
    retained literally. This estimates a population null quantile and does not
    train or validate a replacement operational threshold.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or np.isnan(values).any() or np.isposinf(values).any():
        raise EvidenceError("null quantile requires all allocated raw maxima, finite or -infinity")
    if not 0 < q < 1 or not 0 < error < 1:
        raise ValueError("quantile and error probability must lie strictly between zero and one")
    values = np.sort(values)
    n = len(values)
    point_rank = int(np.ceil(q*n))
    lower_rank = int(binom.ppf(error/2, n, q))
    upper_rank = int(binom.ppf(1-error/2, n, q))+1
    # Tail inequalities are authoritative, including exact probability ties
    # where a quantile inverse's equality convention can change a rank by one.
    while lower_rank < n and binom.cdf(lower_rank, n, q) <= error/2:
        lower_rank += 1
    while lower_rank > 0 and binom.cdf(lower_rank-1, n, q) > error/2:
        lower_rank -= 1
    while upper_rank > 1 and binom.sf(upper_rank-2, n, q) <= error/2:
        upper_rank -= 1
    while upper_rank < n+1 and binom.sf(upper_rank-1, n, q) > error/2:
        upper_rank += 1
    # Verify the defining tails after the inverse-function initialization.
    if not (binom.cdf(lower_rank-1, n, q) <= error/2 and
            binom.sf(upper_rank-1, n, q) <= error/2):
        raise EvidenceError("null-quantile rank tails do not meet their allocation")
    return {"null_n": n, "null_quantile": q, "quantile_error": error,
            "point_rank": point_rank, "lower_rank": lower_rank, "upper_rank": upper_rank,
            "cutoff": float(values[point_rank-1]),
            "cutoff_low": -np.inf if lower_rank == 0 else float(values[lower_rank-1]),
            "cutoff_high": np.inf if upper_rank == n+1 else float(values[upper_rank-1]),
            "quantile_lower_tail": float(binom.cdf(lower_rank-1, n, q)),
            "quantile_upper_tail": float(binom.sf(upper_rank-1, n, q))}


def compact_standardized_power_envelope(before: Sequence[float], after: Sequence[float], *,
        cutoff: float, cutoff_low: float, cutoff_high: float) -> dict[str, float]:
    """Useful first-crossing event and its nonmonotone quantile envelopes.

    One-sided 98% Clopper--Pearson bounds on the two envelope probabilities,
    combined with a 99% quantile bracket, give at least 95% pointwise Monte
    Carlo coverage. No contrast test or simultaneous coverage is claimed.
    """
    a, b = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if (a.ndim != 1 or b.shape != a.shape or not len(a) or np.isnan(a).any() or
        np.isnan(b).any() or np.isposinf(a).any() or np.isposinf(b).any()):
        raise EvidenceError("alternative maxima require every allocated episode, finite or -infinity")
    if not cutoff_low <= cutoff <= cutoff_high:
        raise EvidenceError("point cutoff must lie within its order-statistic bracket")
    point = (a <= cutoff) & (b > cutoff)
    lower = (a <= cutoff_low) & (b > cutoff_high)
    upper = (a <= cutoff_high) & (b > cutoff_low)
    n, x, xl, xu = len(a), int(point.sum()), int(lower.sum()), int(upper.sum())
    return {"alternative_n": n, "standardized_events": x, "standardized_power": x/n,
        "lower_envelope_events": xl, "upper_envelope_events": xu,
        "lower_envelope_probability": xl/n, "upper_envelope_probability": xu/n,
        "mc_ci_low": 0. if xl == 0 else float(beta.ppf(.02, xl, n-xl+1)),
        "mc_ci_high": 1. if xu == n else float(beta.ppf(.98, xu+1, n-xu)),
        "pointwise_coverage_at_least": .95, "lower_binomial_error": .02,
        "upper_binomial_error": .02}


def _compact_trace_maxima(trace_json: str, true_change: int, horizon: int, *,
                          planned_looks: Sequence[int] | None = None) -> tuple[float, float, int]:
    """Separate pre/post raw maxima in the runner's zero-based time convention."""
    try:
        trace = json.loads(trace_json)
    except (TypeError, ValueError) as exc:
        raise EvidenceError("secondary analysis requires valid raw look traces") from exc
    if not isinstance(trace, list):
        raise EvidenceError("raw look trace must be a list")
    before = after = -np.inf
    days, unavailable = [], 0
    for look in trace:
        if not isinstance(look, dict) or look.get("status") not in {"available", "unavailable"}:
            raise EvidenceError("trace look must explicitly distinguish unavailable from missing computation")
        day = look.get("day")
        if not isinstance(day, (int, float)) or not np.isfinite(day) or int(day) != day or not 0 <= day < horizon:
            raise EvidenceError("raw look day is missing or outside its episode")
        days.append(int(day))
        if look["status"] == "unavailable":
            unavailable += 1
            continue
        score = look.get("raw_statistic")
        if not isinstance(score, (int, float)) or not np.isfinite(score):
            raise EvidenceError("available raw look statistic is missing or nonfinite")
        if day < true_change:
            before = max(before, float(score))
        else:
            after = max(after, float(score))
    if days != sorted(set(days)):
        raise EvidenceError("raw look days must be unique and in chronological order")
    if planned_looks is not None and days != list(planned_looks):
        raise EvidenceError("secondary analysis requires every planned look, including unavailable looks")
    return before, after, unavailable


def _compact_standardized_power(output: Path, evaluation: pd.DataFrame,
        evaluation_manifest: Mapping[str, Any], audit_manifest: Mapping[str, Any]
        ) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Stream the amendment's raw traces only after the primary stage validation."""
    methods = COMPACT_METHOD_ORDER[1:]
    source = output / "evaluation" / RAW_FILE
    header = set(pd.read_csv(source, nrows=0).columns)
    metadata: dict[str, Any] = {
        "designation": "secondary analysis specified after null audit and before alternative-power inspection",
        "estimand": "P(A <= c < B), c = population-null CDF inverse at 0.95",
        "methods": list(methods), "scope": "all 16 on-model alternative cells; four raw-score procedures",
        "null_source": "independent audit only; no training sample pooling",
        "float_representation": "CSV round_trip; JSON Python binary64; no jitter or rounding",
        "uncertainty": "99% quantile bracket + two 98% one-sided Clopper-Pearson envelope bounds",
        "error_allocation": {"quantile": .01, "lower_envelope": .02, "upper_envelope": .02},
        "coverage": "at least 95% pointwise Monte Carlo; not simultaneous or conditional on amendment selection",
        "ties": "literal <= and >; population strict-crossing null error at most 0.05",
        "operational_thresholds_changed": False, "bootstrap": False,
        "audit_raw_sha256": _manifest_file_hash(audit_manifest, RAW_FILE),
        "evaluation_raw_sha256": _manifest_file_hash(evaluation_manifest, RAW_FILE)}
    # Frozen provenance identifiers; no private runtime dependency.
    metadata["amendment_documents"] = {'ANALYSIS_AMENDMENT_01.md': 'b217cc934a738693da58a75abbe6b2265b33c49bfc67d5ff25fc0681ed91be28', 'SIZE_STANDARDIZATION_DESIGN.md': 'c31f94d0f9460c1c4918694b05eaaf4b5379d65d0fab3954945ed1e49fbb2550'}
    if "trace_json" not in header and evaluation_manifest.get("synthetic_fixture", False):
        metadata.update(status="skipped_synthetic_fixture_without_traces", rows=0)
        return None, metadata
    required = {"scenario", "family", "method", "replicate", "split", "misspecification",
                "status", "true_change", "horizon", "trace_json", "episode_statistic"}
    if not required <= header:
        raise EvidenceError(f"secondary raw evidence lacks {sorted(required-header)}")
    selected = evaluation[evaluation.alternative & evaluation.misspecification.eq("on_model") &
                          evaluation.method.isin(methods)]
    expected = {(str(s), str(m)): set(group.replicate.astype(int)) for (s, m), group in
                selected.groupby(["scenario", "method"])}
    if not evaluation_manifest.get("synthetic_fixture", False) and len(expected) != 64:
        raise EvidenceError("secondary analysis requires all 16 alternative cells and four raw procedures")
    null = pd.read_csv(output/"audit"/RAW_FILE, float_precision="round_trip",
        usecols=["family", "method", "replicate", "status", "episode_statistic"])
    quantiles = {}
    for (family, method), group in null.groupby(["family", "method"]):
        if method not in methods or not group.status.eq("success").all():
            raise EvidenceError("audit raw quantiles require complete successful score episodes")
        quantiles[(family, method)] = compact_null_quantile_bracket(group.episode_statistic)
        if not audit_manifest.get("synthetic_fixture", False) and len(group) != 5000:
            raise EvidenceError("secondary null quantiles require all 5000 audit episodes")
    records: dict[tuple[str, str], list[tuple[int, float, float, int]]] = {key: [] for key in expected}
    for chunk in pd.read_csv(source, usecols=lambda column: column in required,
                             chunksize=1024, float_precision="round_trip"):
        chunk = chunk[chunk.method.isin(methods) & chunk.split.eq("alternative") &
                      chunk.misspecification.eq("on_model")]
        for row in chunk.itertuples(index=False):
            key = (row.scenario, row.method)
            if key not in records or row.status != "success":
                raise EvidenceError("secondary trace does not belong to a completed allocated episode")
            planned = None
            if not evaluation_manifest.get("synthetic_fixture", False):
                settings = evaluation_manifest["scientific_config"]["methods"]
                day_labels = list(range(settings["reference_end"]+settings["look_every"],
                                        int(row.horizon)+1, settings["look_every"]))
                if not day_labels or day_labels[-1] != row.horizon:
                    day_labels.append(int(row.horizon))
                planned = [day-1 for day in day_labels]
            a, b, unavailable = _compact_trace_maxima(row.trace_json, int(row.true_change),
                                                     int(row.horizon), planned_looks=planned)
            if max(a, b) != row.episode_statistic:
                raise EvidenceError("raw episode maximum and per-look raw statistics disagree")
            records[key].append((int(row.replicate), a, b, unavailable))
    rows = []
    for (scenario, method), episodes in records.items():
        if len(episodes) != len(expected[(scenario, method)]) or {r[0] for r in episodes} != expected[(scenario, method)]:
            raise EvidenceError("secondary traces omit or duplicate allocated alternative episodes")
        group = selected[(selected.scenario == scenario) & (selected.method == method)]
        family = group.family.iloc[0]
        if (family, method) not in quantiles:
            raise EvidenceError("secondary alternative lacks its independent audit score distribution")
        quantile = quantiles[(family, method)]
        result = compact_standardized_power_envelope([r[1] for r in episodes], [r[2] for r in episodes],
            cutoff=quantile["cutoff"], cutoff_low=quantile["cutoff_low"], cutoff_high=quantile["cutoff_high"])
        rows.append({"scenario": scenario, "family": family, "method": method,
            "risk_ratio": group.risk_ratio.iloc[0], "change_day": group.change_day.iloc[0],
            "horizon": group.horizon.iloc[0], **quantile, **result,
            "unavailable_looks": sum(r[3] for r in episodes),
            "empty_before_n": sum(np.isneginf(r[1]) for r in episodes),
            "empty_after_n": sum(np.isneginf(r[2]) for r in episodes)})
    result = pd.DataFrame(rows).sort_values(["family", "scenario", "method"], ignore_index=True)
    metadata.update(status="complete", rows=len(result),
                    null_n=sorted({int(q["null_n"]) for q in quantiles.values()}),
                    alternative_n=sorted(result.alternative_n.astype(int).unique().tolist()))
    return result, metadata


def compact_summaries(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All-replicate episode metrics, unconditional delay curves, paired contrasts."""
    summaries, curves, contrasts, failures = [], [], [], []
    metadata = ["stage", "scenario", "family", "risk_ratio", "change_day", "horizon",
                "misspecification", "split"]
    for (_, _, _), group in frame.groupby(["stage", "scenario", "method"], sort=True):
        common = {column: group[column].iloc[0] for column in metadata}
        for column in ("true_delay_mean_days", "assumed_delay_mean_days", "phi"):
            if column in group:
                common[column] = group[column].iloc[0]
        common["method"] = group.method.iloc[0]
        counts = {"failed_n": int(group.failed.sum()),
                  "absent_alarm_n": int((~group.any_alarm).sum()),
                  "premature_alarm_n": int(group.premature.sum())}
        for metric, column in (("alarm_probability", "any_alarm"),
                               ("useful_power", "useful"),
                               ("premature_probability", "premature")):
            if metric != "alarm_probability" and not group.alternative.iloc[0]:
                continue
            summaries.append({**common, "metric": metric,
                              **_compact_probability(group[column]), **counts})
        if group.alternative.iloc[0]:
            for day in range(int(group.horizon.iloc[0]-group.change_day.iloc[0])+1):
                curves.append({**common, "delay_day": day,
                    **_compact_probability(group.useful & (group.delay <= day))})
        for status, subset in group.groupby("status", dropna=False):
            failures.append({**common, "status": status, "rows": len(subset),
                             "failed": status != "success", "requested_n": len(group),
                             "invalid_looks": int(pd.to_numeric(subset.get("unavailable_looks",
                                 pd.Series(0, index=subset.index)), errors="coerce").fillna(0).sum())})
    for (stage, scenario), group in frame.groupby(["stage", "scenario"], sort=True):
        common = {column: group[column].iloc[0] for column in metadata}
        metrics = [("alarm_probability", "any_alarm"), ("useful_power", "useful")]
        if ORACLE_OCFR in set(group.method) and group.alternative.iloc[0]:
            group = group.assign(timely28=group.useful & group.delay.le(28))
            metrics.append(("useful_detection_within_28", "timely28"))
            if group.misspecification.iloc[0] == "on_model" and "tau_hat_at_alarm" in group:
                cohort_error = (pd.to_numeric(group.tau_hat_at_alarm, errors="coerce") - group.true_change).abs()
                group = group.assign(joint7=group.useful & cohort_error.le(7))
                metrics.append(("useful_detection_localized_within_7", "joint7"))
        for metric, column in metrics:
            if metric == "useful_power" and not group.alternative.iloc[0]:
                continue
            pivot = group.pivot(index="replicate", columns="method", values=column)
            order = COMPACT_PAIRED_METHODS if "OCFR" in pivot else COMPACT_METHOD_ORDER
            methods = [method for method in order if method in pivot]
            for i, first in enumerate(methods):
                for second in methods[i+1:]:
                    difference = pivot[first].astype(float)-pivot[second].astype(float)
                    n = len(difference)
                    contrasts.append({**common, "metric": metric, "method_a": first,
                        "method_b": second, "n": n, "difference": difference.mean(),
                        "paired_mcse": difference.std(ddof=1)/np.sqrt(n) if n > 1 else np.nan,
                        "a_only_n": int((difference == 1).sum()),
                        "b_only_n": int((difference == -1).sum())})
    return {"compact_summary": pd.DataFrame(summaries),
            "compact_cumulative_detection": pd.DataFrame(curves),
            "compact_paired_contrasts": pd.DataFrame(contrasts),
            "compact_failures": pd.DataFrame(failures)}


def compact_localization_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    """Localization at useful first alarms, conditional on each method's detections."""
    if "tau_hat_at_alarm" not in frame:
        raise EvidenceError("v2 localization requires stored first-alarm cohort dates")
    selected = frame[frame.stage.eq("evaluation") & frame.alternative &
                     frame.misspecification.eq("on_model")]
    paired = selected.method.eq(ORACLE_OCFR).any()
    rows = []
    for (scenario, method), group in selected.groupby(["scenario", "method"], sort=True):
        useful = group[group.useful]
        tau = pd.to_numeric(useful.tau_hat_at_alarm, errors="coerce")
        if (~np.isfinite(tau)).any():
            raise EvidenceError(f"{scenario}/{method}: useful alarm lacks cohort-date estimate")
        error = (tau - useful.true_change).abs()
        n = len(error)
        mean = float(error.mean()) if n else np.nan
        mcse = float(error.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        half = float(student_t.ppf(.975, n-1) * mcse) if n > 1 else np.nan
        row = dict(scenario=scenario, method=method, family=group.family.iloc[0],
            risk_ratio=float(group.risk_ratio.iloc[0]), change_day=int(group.change_day.iloc[0]),
            n_requested=len(group), n_useful=n, mean_absolute_error_days=mean,
            mcse=mcse, mc_ci_low=max(0., mean-half) if n > 1 else np.nan,
            mc_ci_high=mean+half if n > 1 else np.nan,
            location_convention=("selected cohort change date" if method in ("OCFR", ORACLE_OCFR) else
                "prespecified median-delay cohort proxy"),
            denominator="method-specific useful first alarms")
        if paired:
            normal_half = 1.96 * mcse
            joint = _compact_probability(group.useful & (
                (pd.to_numeric(group.tau_hat_at_alarm, errors="coerce") - group.true_change).abs() <= 7))
            row.update(normal_ci_low=max(0., mean-normal_half) if n > 1 else np.nan,
                       normal_ci_high=mean+normal_half, joint_events=joint["events"],
                       joint_estimate=joint["estimate"], joint_ci_low=joint["ci_low"],
                       joint_ci_high=joint["ci_high"], joint_denominator="all requested episodes")
        rows.append(row)
    return pd.DataFrame(rows)


def _compact_localization_figure(table: pd.DataFrame, directory: Path) -> str:
    if table.method.eq(ORACLE_OCFR).any():
        return _compact_paired_localization_figure(table, directory)
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.4), sharex=True)
    methods = ("OCFR", "NB-GLR", "Page-CUSUM")
    ratios = np.array([2/3, 1.25, 1.5, 2.])
    for ax, family, letter in zip(axes, ("steady", "sparse", "wave"), "ABC"):
        part = table[table.family.eq(family) & table.change_day.eq(92)]
        for offset, method in zip((-.10, 0., .10), methods):
            values = []
            for ratio in ratios:
                rows = part[part.method.eq(method) & np.isclose(part.risk_ratio, ratio)]
                if len(rows) != 1:
                    raise EvidenceError(f"localization requires one row for {family}/{method}/RR={ratio}")
                values.append(rows.iloc[0])
            data = pd.DataFrame(values)
            x = np.arange(4) + offset
            mean = data.mean_absolute_error_days.to_numpy(float)
            valid = np.isfinite(mean)
            interval = valid & np.isfinite(data.mc_ci_low) & np.isfinite(data.mc_ci_high)
            ax.plot(x[valid], mean[valid], marker=COMPACT_MARKERS[method], linestyle="none",
                color=COMPACT_COLORS[method], label=method)
            ax.errorbar(x[interval], mean[interval],
                yerr=np.vstack([mean[interval]-data.mc_ci_low.to_numpy()[interval],
                    data.mc_ci_high.to_numpy()[interval]-mean[interval]]), fmt="none",
                color=COMPACT_COLORS[method], capsize=2)
        ax.set_title(family.capitalize())
        ax.set_ylabel("Mean absolute cohort-\ndate error (days)")
        ax.set_ylim(bottom=0)
        ax.set_xticks(np.arange(4), ["2/3", "1.25", "1.5", "2"])
        ax.tick_params(labelbottom=True)
        _compact_letter(ax, letter)
    axes[-1].set_xlabel("CFR risk ratio")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(.53, .995), fontsize=10.5)
    fig.subplots_adjust(left=.19, right=.97, bottom=.08, top=.91, hspace=.48)
    _compact_save(fig, directory, "CompactFig4_localization")
    return ("Mean absolute cohort-date error at useful first alarms for changes on day 92. "
        "OCFR uses its selected cohort date; NB-GLR and Page-CUSUM use the prespecified median-delay "
        "cohort proxy. Each method conditions on its own useful detections, so conditional samples differ. "
        "Bars are approximate 95% Monte Carlo Student-t intervals for the mean absolute error when "
        "at least two detections are available; singleton estimates have no bar and zero-detection "
        "cells have no point. Requested-episode and useful-detection counts accompany the numerical summaries. "
        "This conditional localization comparison is separate from useful detection probability.")


def _compact_paired_localization_figure(table: pd.DataFrame, directory: Path) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 5), sharex=True, sharey="row")
    ratios = np.array([2/3, 1.25, 1.5, 2.])
    methods = COMPACT_PAIRED_METHODS
    for column, family in enumerate(("steady", "sparse", "wave")):
        family_rows = table[table.family.eq(family) & table.change_day.eq(92)]
        for index, method in enumerate(methods):
            part = family_rows[family_rows.method.eq(method)].sort_values("risk_ratio")
            if len(part) != len(ratios) or not np.allclose(part.risk_ratio, ratios):
                raise EvidenceError(f"localization requires the day-92 risk-ratio grid for {family}/{method}")
            x = np.arange(len(ratios)) + (index - (len(methods) - 1)/2) * .12
            for ax, fields in zip(axes[:, column], (
                    ("mean_absolute_error_days", "normal_ci_low", "normal_ci_high"),
                    ("joint_estimate", "joint_ci_low", "joint_ci_high"))):
                mean, lo, hi = part[list(fields)].to_numpy(float).T
                color, marker = COMPACT_COLORS[method], COMPACT_MARKERS[method]
                ax.plot(x, mean, color=color, marker=marker, markersize=3.8,
                        linestyle=COMPACT_LINESTYLES[method])
                valid = np.isfinite(lo) & np.isfinite(hi)
                ax.errorbar(x[valid], mean[valid],
                    yerr=np.vstack([mean[valid] - lo[valid], hi[valid] - mean[valid]]),
                    fmt="none", color=color, capsize=2, elinewidth=.95)
        axes[0, column].set_title(family.capitalize())
        axes[1, column].set(ylim=(0, 1.02), xlabel="CFR risk ratio")
        axes[1, column].set_xticks(np.arange(4), ["2/3", "1.25", "1.5", "2"])
    axes[0, 0].set_ylabel("Conditional mean absolute\ncohort-date error (days)")
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylabel("Useful alarm and localization\nwithin 7 days (probability)")
    for letter, ax in zip("ABCDEF", axes.flat):
        _compact_letter(ax, letter)
    fig.legend(handles=_compact_method_handles(methods), ncol=4, loc="upper center",
               bbox_to_anchor=(.54, .996), fontsize=9.5, frameon=False)
    fig.subplots_adjust(left=.155, right=.98, bottom=.12, top=.87, hspace=.34, wspace=.31)
    _compact_save(fig, directory, "CompactFig4_localization")
    return ("Columns show steady, sparse and wave incidence with changes on day 92. "
        "A-C: mean absolute cohort-date error among each method's useful first alarms, "
        "with approximate 95% normal Monte Carlo intervals. D-F: probability of a useful "
        "first alarm localized within seven days, with exact 95% binomial Monte Carlo "
        "intervals and all requested episodes in the denominator. Both OCFR rules use "
        "their selected cohort date; report-day benchmarks use the median-delay cohort proxy.")


ESTIMATION_TARGETS = ("pi_left", "pi_right", "risk_ratio")


def compact_estimation_summaries(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """First-alarm summaries; no-alarm episodes never become fictitious errors."""
    summaries, locations = [], []
    for scenario, group in frame.groupby("scenario", sort=True):
        first = group.iloc[0]
        alarm = _as_binary(group.alarm, "alarm").astype(bool)
        points = group[[f"{t}_point_valid" for t in ESTIMATION_TARGETS]].apply(
            lambda x: _as_binary(x, x.name)).all(axis=1) & alarm
        n, a, p = len(group), int(alarm.sum()), int(points.sum())
        location = group.loc[alarm, "tau_hat_at_alarm"] - first.true_change
        if not np.isfinite(location).all():
            raise EvidenceError(f"{scenario}: missing stored first-alarm location")
        common = dict(scenario=scenario, family=first.family, method=first.method,
            risk_ratio=float(first.risk_ratio), change_day=int(first.true_change)+1,
            N_requested=n, n_alarm=a, n_no_alarm=n-a,
            n_premature=int((alarm & (group.alarm_day < first.true_change)).sum()),
            n_point=p, n_point_failed=a-p,
            n_fit_failed=int((alarm & ~_as_binary(group.fit_success, "fit_success").astype(bool)).sum()))
        locations.append({**common, "n_location": len(location),
            "location_mae_days": float(location.abs().mean()),
            "location_bias_days": float(location.mean())})
        for target in ESTIMATION_TARGETS:
            truth = float(first[target])
            valid = _as_binary(group[f"{target}_interval_valid"], target).astype(bool) & alarm
            lo, hi = group[f"{target}_lower"], group[f"{target}_upper"]
            if (valid & (~points | ~np.isfinite(lo) | ~np.isfinite(hi) | (hi < lo))).any():
                raise EvidenceError(f"{scenario}/{target}: invalid declared available interval")
            v = int(valid.sum())
            covered = int((valid & (lo <= truth) & (hi >= truth)).sum())
            q = covered/a if a else np.nan
            ci = compact_binomial_interval(covered, a) if a else (np.nan, np.nan)
            error = group.loc[points, f"{target}_estimate"] - truth
            if not np.isfinite(error).all():
                raise EvidenceError(f"{scenario}/{target}: nonfinite available point")
            width = hi[valid]-lo[valid]
            scale = 1 if target == "risk_ratio" else 100
            summaries.append({**common, "target": target, "truth": truth,
                "point_availability": p/a if a else np.nan,
                "relative_bias": float(error.mean()/truth),
                "relative_rmse": float(np.sqrt(np.mean((error/truth)**2))) if p else np.nan,
                "bias": float(error.mean()*scale),
                "rmse": float(np.sqrt(np.mean(error**2))*scale) if p else np.nan,
                "error_units": "RR" if scale == 1 else "CFR percentage points",
                "n_interval": v, "n_interval_unavailable": a-v,
                "interval_availability": v/a if a else np.nan, "n_contained": covered,
                "all_alarm_containment": q, "containment_ci_low": ci[0], "containment_ci_high": ci[1],
                "valid_only_containment": covered/v if v else np.nan,
                "missing_interval_sensitivity_upper": (covered+a-v)/a if a else np.nan,
                "median_relative_width": float(width.median()/truth),
                "width_median": float(width.median()), "width_q25": float(width.quantile(.25)),
                "width_q75": float(width.quantile(.75)), "width_mean": float(width.mean()),
                "width_max": float(width.max()), "width_units": "RR" if scale == 1 else "CFR probability"})
    return {"compact_estimation_summary": pd.DataFrame(summaries),
            "compact_estimation_location": pd.DataFrame(locations)}


def _compact_load_estimation(output: Path, evaluation: pd.DataFrame,
                             original: Mapping[str, Any]) -> tuple[dict, dict]:
    directory = output / "estimation"
    if not directory.exists():
        return {}, {"status": "absent"}
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise EvidenceError("estimation directory lacks a complete manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("stage") != "estimation":
        raise EvidenceError("estimation stage is not complete")
    for key, expected in (("original_manifest_sha256", _sha256(output/"evaluation"/MANIFEST_FILE)),
            ("original_csv_sha256", _manifest_file_hash(original, RAW_FILE)),
            ("original_scientific_digest", original["scientific_digest"]),
            ("scientific_digest", original["scientific_digest"])):
        if manifest.get(key) != expected:
            raise EvidenceError(f"estimation original source/digest mismatch: {key}")
    if manifest.get("scientific_config") != original.get("scientific_config"):
        raise EvidenceError("estimation scientific configuration changed")
    records = manifest.get("postprocessor_code_files", {})
    source_hash = hashlib.sha256(_canonical_json(records).encode("ascii")).hexdigest()
    if not records or manifest.get("postprocessor_code_hash") != source_hash:
        raise EvidenceError("estimation postprocessor source hash mismatch")
    raw = directory / RAW_FILE
    if _sha256(raw) != _manifest_file_hash(manifest, RAW_FILE):
        raise EvidenceError("estimation raw-file hash mismatch")
    frame = pd.read_csv(raw)
    required = {"scenario", "replicate", "method", "family", "true_change", "horizon",
        "alarm", "alarm_day", "tau_hat_at_alarm", "fit_success", "confidence"}
    required |= set(ESTIMATION_TARGETS)
    required |= {f"{t}_{s}" for t in ESTIMATION_TARGETS for s in
                 ("estimate", "lower", "upper", "point_valid", "interval_valid")}
    if required-set(frame) or frame.empty:
        raise EvidenceError(f"estimation missing columns: {sorted(required-set(frame))}")
    selected = evaluation[evaluation.alternative & evaluation.misspecification.eq("on_model") &
                          evaluation.method.eq("OCFR" if original.get("protocol") == "compact-v2" else "OCFR Gaussian")]
    names = set(selected.scenario)
    if len(names) != 16 or set(frame.scenario) != names or set(manifest.get("scenarios", [])) != names:
        raise EvidenceError("estimation requires the complete 16-cell grid")
    mode = manifest.get("mode")
    if mode not in {"demo", "full"}:
        raise EvidenceError("estimation mode must be demo or full")
    count = 10 if mode == "demo" else 1000
    if frame.duplicated(["scenario", "replicate"]).any():
        raise EvidenceError("duplicate estimation episode")
    if not frame.method.eq("OCFR" if original.get("protocol") == "compact-v2" else "OCFR Gaussian").all() or not np.isclose(frame.confidence, .95).all():
        raise EvidenceError("estimation method/confidence violates frozen contract")
    for scenario, group in frame.groupby("scenario"):
        if (manifest.get("replications_by_scenario", {}).get(scenario) != count or
            len(group) != count or set(group.replicate) != set(range(count))):
            raise EvidenceError(f"{scenario}: incomplete estimation episode grid")
        source = selected[selected.scenario.eq(scenario)].set_index("replicate").loc[group.replicate]
        for column in ("family", "true_change", "horizon", "risk_ratio"):
            if not np.array_equal(group[column].to_numpy(), source[column].to_numpy()):
                raise EvidenceError(f"{scenario}: estimation metadata mismatch {column}")
        alarm = _as_binary(group.alarm, "alarm").astype(bool)
        if not np.array_equal(alarm, source.any_alarm.to_numpy()):
            raise EvidenceError("estimation first-alarm cohort changed")
        if not np.allclose(group.alarm_day+1, source.alarm_day.to_numpy(), equal_nan=True):
            raise EvidenceError("estimation first-alarm date changed")
        if "tau_hat_at_alarm" in source and not np.allclose(group.tau_hat_at_alarm,
                source.tau_hat_at_alarm.to_numpy(), equal_nan=True):
            raise EvidenceError("estimation stored first-alarm location changed")
        tau = group.tau_hat_at_alarm
        if (alarm & (~np.isfinite(tau) | (tau < 1) | (tau > group.alarm_day))).any():
            raise EvidenceError("invalid stored first-alarm location")
        for target in ESTIMATION_TARGETS:
            point = _as_binary(group[f"{target}_point_valid"], target).astype(bool)
            interval = _as_binary(group[f"{target}_interval_valid"], target).astype(bool)
            estimate = group[f"{target}_estimate"]
            if ((point | interval) & ~alarm).any() or (interval & ~point).any():
                raise EvidenceError("estimation availability violates alarm/point nesting")
            if group.loc[~alarm, [f"{target}_{s}" for s in ("estimate", "lower", "upper")]].notna().any().any():
                raise EvidenceError("no-alarm episode has fabricated estimation output")
            if (point & (~np.isfinite(estimate) | (estimate <= 0) |
                        ((estimate >= 1) if target != "risk_ratio" else False))).any():
                raise EvidenceError("invalid available estimation point")
        truth = group[[*ESTIMATION_TARGETS]]
        if (truth.nunique() != 1).any() or not np.allclose(group.pi_right/group.pi_left, group.risk_ratio):
            raise EvidenceError("inconsistent estimation generating truth")
        if not np.allclose(group.pi_left, .02):
            raise EvidenceError("estimation generating CFR changed")
    tables = compact_estimation_summaries(frame)
    metadata = {"status": "complete", "protocol_date": "2026-11-06", "mode": mode,
        "source_manifest_sha256": _sha256(path), "source_csv_sha256": _sha256(raw),
        "original_manifest_sha256": manifest["original_manifest_sha256"],
        "original_csv_sha256": manifest["original_csv_sha256"],
        "original_scientific_digest": original["scientific_digest"],
        "postprocessor_code_hash": source_hash, "confidence_level": .95,
        "scope": "first Gaussian alarm including premature alarms; no selection adjustment",
        "point_denominator": "admissible joint point fits among first alarms",
        "containment_denominator": "all first alarms; unavailable intervals count as noncontainment",
        "width_denominator": "target-specific available finite ordered intervals",
        "localization_denominator": "all stored first-alarm locations, regardless of fit availability"}
    digest_payload = {"schema_version": original.get("schema_version"),
                      "scientific": original["scientific_config"]}
    if "code_hash" in original:
        digest_payload["code_hash"] = original["code_hash"]
    metadata["original_scientific_digest_full"] = hashlib.sha256(
        _canonical_json(digest_payload).encode("utf-8")).hexdigest()
    return tables, metadata


def compact_estimation_reference_summaries(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Paired point comparison, retaining finite inadmissible moment estimates."""
    summaries, contrasts = [], []
    for scenario, group in frame.groupby("scenario", sort=True):
        alarm = _as_binary(group.alarm, "alarm").astype(bool)
        first = group.iloc[0]
        common = dict(scenario=scenario, family=first.family,
            risk_ratio=float(first.risk_ratio), change_day=int(first.true_change)+1,
            N_requested=len(group), n_alarm=int(alarm.sum()),
            n_premature=int((alarm & (group.alarm_day < first.true_change)).sum()))
        for target in ESTIMATION_TARGETS:
            truth = float(first[target])
            errors, masks = {}, {}
            for method, prefix in (("OCFR", ""), ("Nishiura extension", "nishiura_")):
                estimate = pd.to_numeric(group[f"{prefix}{target}_estimate"], errors="raise")
                finite = _as_binary(group[f"{prefix}{target}_point_valid"], target).astype(bool) & alarm
                if (finite & ~np.isfinite(estimate)).any():
                    raise EvidenceError("declared available reference estimate is nonfinite")
                admissible = (_as_binary(group[f"nishiura_{target}_admissible"], target).astype(bool)
                              if prefix else finite)
                error = (estimate[finite]-truth)/truth
                squared = error**2
                mse = float(squared.mean()) if len(squared) else np.nan
                rmse = float(np.sqrt(mse))
                mcse = (float(squared.std(ddof=1)/np.sqrt(len(squared))/(2*rmse))
                        if len(squared) > 1 and rmse > 0 else 0. if mse == 0 else np.nan)
                summaries.append({**common, "target": target, "truth": truth, "method": method,
                    "n_point": int(finite.sum()), "n_unavailable": int((alarm & ~finite).sum()),
                    "n_inadmissible": int((finite & ~admissible).sum()),
                    "relative_bias": float(error.mean()), "relative_rmse": rmse,
                    "relative_rmse_mcse": mcse,
                    "relative_rmse_mc_low": max(0., rmse-1.96*mcse),
                    "relative_rmse_mc_high": rmse+1.96*mcse})
                errors[method], masks[method] = (estimate-truth)/truth, finite
            paired = masks["OCFR"] & masks["Nishiura extension"]
            difference = (errors["OCFR"][paired]**2 - errors["Nishiura extension"][paired]**2)
            contrast = {**common, "target": target, "n_paired": int(paired.sum()),
                "mean_squared_relative_error_difference": float(difference.mean()),
                "paired_mcse": float(difference.std(ddof=1)/np.sqrt(len(difference))) if len(difference)>1 else np.nan}
            for method, prefix in (("OCFR", "ocfr"), ("Nishiura extension", "reference")):
                squared = errors[method][paired]**2
                rmse = float(np.sqrt(squared.mean()))
                mcse = (float(squared.std(ddof=1)/np.sqrt(len(squared))/(2*rmse))
                        if len(squared)>1 and rmse>0 else 0. if rmse==0 else np.nan)
                contrast[f"{prefix}_paired_relative_rmse"] = rmse
                contrast[f"{prefix}_paired_relative_rmse_mcse"] = mcse
            contrasts.append(contrast)
    return {"compact_estimation_reference_summary": pd.DataFrame(summaries),
            "compact_estimation_reference_paired": pd.DataFrame(contrasts)}


def _compact_load_estimation_reference(output: Path) -> tuple[dict, dict]:
    directory = output / "estimation-reference"
    if not directory.exists():
        return {}, {"status": "absent"}
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise EvidenceError("reference estimation stage lacks a manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("stage") != "estimation-reference":
        raise EvidenceError("reference estimation stage is incomplete")
    if manifest.get("mode") != "full":
        raise EvidenceError("publication analysis requires the full reference stage, not its demo subset")
    raw = directory / RAW_FILE
    if _sha256(raw) != _manifest_file_hash(manifest, RAW_FILE):
        raise EvidenceError("reference estimation file hash mismatch")
    source = output / "estimation"
    for key, path in (("original_estimation_csv_sha256", source/RAW_FILE),
                      ("original_estimation_manifest_sha256", source/MANIFEST_FILE)):
        if manifest.get(key) != _sha256(path):
            raise EvidenceError(f"reference estimation source mismatch: {key}")
    nb_manifest = json.loads((source/MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("scientific_digest") != nb_manifest.get("scientific_digest"):
        raise EvidenceError("reference scientific digest differs from NB estimation")
    records = manifest.get("postprocessor_code_files", {})
    if not records or manifest.get("postprocessor_code_hash") != hashlib.sha256(
            _canonical_json(records).encode("ascii")).hexdigest():
        raise EvidenceError("reference postprocessor code hash mismatch")
    frame, original = pd.read_csv(raw), pd.read_csv(source/RAW_FILE)
    if frame.duplicated(["scenario", "replicate"]).any() or not set(original).issubset(frame):
        raise EvidenceError("reference estimation episode contract changed")
    try:
        pd.testing.assert_frame_equal(frame[original.columns].reset_index(drop=True),
            original.reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-15)
    except AssertionError as exc:
        raise EvidenceError("reference estimation altered the original NB rows") from exc
    return compact_estimation_reference_summaries(frame), {
        "status": "complete", "source_manifest_sha256": _sha256(manifest_path),
        "source_csv_sha256": _sha256(raw), "mode": manifest.get("mode"),
        "point_scope": "same original OCFR alarms and dates; all finite raw estimates retained",
        "interval_scope": "reference supplies point estimates only"}


def _compact_load_stabilized_estimation(output: Path) -> tuple[dict, dict]:
    directory = output / "estimation-stabilized"
    if not directory.exists():
        return {}, {"status": "absent"}
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise EvidenceError("stabilized estimation lacks a complete manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("stage") != "estimation-stabilized"
            or manifest.get("mode") != "full"):
        raise EvidenceError("publication analysis requires complete full stabilized estimation")
    original_dir = output / "estimation-reference"
    for key, source in (("original_reference_csv_sha256", original_dir/RAW_FILE),
                         ("original_reference_manifest_sha256", original_dir/MANIFEST_FILE)):
        if manifest.get(key) != _sha256(source):
            raise EvidenceError("stabilized estimation source mismatch")
    config = manifest.get("post_alarm_estimation_config", {})
    if config.get("precision") != 1. or config.get("covariance") != "sandwich":
        raise EvidenceError("stabilized reporting settings differ from the manuscript")
    records = manifest.get("postprocessor_code_files", {})
    if not records or manifest.get("postprocessor_code_hash") != hashlib.sha256(
            _canonical_json(records).encode("ascii")).hexdigest():
        raise EvidenceError("stabilized postprocessor code hash mismatch")
    raw = directory/RAW_FILE
    if _sha256(raw) != _manifest_file_hash(manifest, RAW_FILE):
        raise EvidenceError("stabilized estimation raw-file hash mismatch")
    frame, original = pd.read_csv(raw), pd.read_csv(original_dir/RAW_FILE)
    try:
        pd.testing.assert_frame_equal(frame[original.columns].reset_index(drop=True),
            original.reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-15)
    except (KeyError, AssertionError) as exc:
        raise EvidenceError("stabilized stage changed frozen MLE/reference episodes") from exc
    reporting = frame.copy()
    for target in ESTIMATION_TARGETS:
        for suffix in ("estimate", "lower", "upper", "point_valid", "interval_valid"):
            reporting[f"{target}_{suffix}"] = frame[f"regularized_{target}_{suffix}"]
    reporting["method"] = "OCFR stabilized"
    reporting["fit_success"] = reporting[[f"{t}_point_valid" for t in ESTIMATION_TARGETS]].all(axis=1)
    tables = compact_estimation_summaries(reporting)
    tables = {key.replace("compact_estimation_", "compact_stabilized_"): value for key, value in tables.items()}
    ref = compact_estimation_reference_summaries(reporting)
    paired = ref["compact_estimation_reference_paired"]
    # Every displayed point estimator uses the same common-finite episodes.
    for index, record in paired.iterrows():
        target, scenario = record.target, record.scenario
        group = frame[frame.scenario.eq(scenario)]
        common = _as_binary(group.alarm, "alarm").astype(bool)
        for prefix in ("", "regularized_", "nishiura_"):
            common &= _as_binary(group[f"{prefix}{target}_point_valid"], target).astype(bool)
        truth = float(group[target].iloc[0])
        paired.at[index, "n_paired"] = int(common.sum())
        for prefix, name in (("regularized_", "ocfr"), ("nishiura_", "reference"), ("", "mle")):
            squared = (group.loc[common, f"{prefix}{target}_estimate"]/truth-1.)**2
            rmse = float(np.sqrt(squared.mean()))
            mcse = (float(squared.std(ddof=1)/np.sqrt(len(squared))/(2*rmse))
                if len(squared)>1 and rmse>0 else 0. if rmse==0 else np.nan)
            paired.at[index, f"{name}_paired_relative_rmse"] = rmse
            paired.at[index, f"{name}_paired_relative_rmse_mcse"] = mcse
    tables["compact_stabilized_reference_paired"] = paired
    tables["compact_stabilized_reference_summary"] = ref["compact_estimation_reference_summary"]
    return tables, {"status": "complete", "source_manifest_sha256": _sha256(path),
        "source_csv_sha256": _sha256(raw), "precision": 1., "covariance": "penalty-aware sandwich",
        "detection_scope": "original first Gaussian alarms and selected dates retained",
        "interval_scope": "approximate unadjusted post-selection summaries"}


def _compact_estimation_comparison_figure(tables: dict[str, pd.DataFrame], directory: Path,
        *, name: str = "CompactFig5_estimation", poisson: bool = False, ls: bool = False) -> str:
    if poisson and ls:
        raise EvidenceError("choose one main point comparator")
    stabilized = "compact_stabilized_summary" in tables
    if ls and not stabilized:
        raise EvidenceError("LS main Figure 5 requires stabilized OCFR interval summaries")
    prefix = "compact_stabilized_" if stabilized else "compact_estimation_"
    interval = tables[prefix+"summary"]
    reference_method = "Full-delay LS" if ls else "Matched Poisson" if poisson else "Nishiura extension"
    if poisson or ls:
        source = tables["compact_ls_main_pair_summary" if ls else "compact_poisson_summary"]
        point = source[source.bank.eq("Original bank") & source.method.isin(
            ["OCFR", reference_method])].copy()
        paired_rows = []
        for (scenario, target), group in source[source.bank.eq("Original bank")].groupby(
                ["scenario", "target"], sort=True):
            record = {"scenario": scenario, "target": target}
            point_methods = (("OCFR", "ocfr"), (reference_method, "reference")) if ls else (
                ("OCFR", "ocfr"), ("Unpenalized NB", "mle"), (reference_method, "reference"))
            for method, score in point_methods:
                selected = group[group.method.eq(method)]
                if len(selected) != 1:
                    raise EvidenceError("main point comparison requires one summary per method and target")
                row = selected.iloc[0]
                record[f"{score}_paired_relative_rmse"] = row.common_paired_relative_rmse
                record[f"{score}_paired_relative_rmse_mcse"] = row.common_paired_relative_rmse_mcse
            paired_rows.append(record)
        paired = pd.DataFrame(paired_rows)
    else:
        point = tables[prefix+"reference_summary"]
        paired = tables[prefix+"reference_paired"]
    ordered = []
    for family in ("steady", "sparse", "wave"):
        for rr in (2/3, 1.25, 1.5):
            rows = point[point.family.eq(family) & point.change_day.eq(92) &
                         np.isclose(point.risk_ratio, rr)]
            if rows.scenario.nunique() != 1 or len(rows) != 6:
                raise EvidenceError("comparative estimation figure requires all nine cells and both methods")
            ordered.append(rows.scenario.iloc[0])
    fig, axes = plt.subplots(3, 2, figsize=(6.8, 7.8), sharex=True)
    x = np.arange(9)
    colors = {"OCFR": "#185889", reference_method: "#B44C43", "Unpenalized NB": "#737373"}
    methods = (("OCFR", "ocfr", -.23, "o"), ("Unpenalized NB", "mle", 0., "^"),
               (reference_method, "reference", .23, "s")) if stabilized else (
               ("OCFR", "ocfr", -.12, "o"), ("Nishiura extension", "reference", .12, "s"))
    if ls:
        methods = (("Full-delay LS", "reference", -.12, "s"), ("OCFR", "ocfr", .12, "o"))
    for row, (target, label) in enumerate(zip(ESTIMATION_TARGETS,
            ("Pre-change CFR", "Post-change CFR", "Risk ratio"))):
        left, right = axes[row]
        data = paired[paired.target.eq(target)].set_index("scenario").loc[ordered]
        for method, score_prefix, shift, marker in methods:
            value = data[f"{score_prefix}_paired_relative_rmse"]
            mcse = data[f"{score_prefix}_paired_relative_rmse_mcse"]
            left.errorbar(x+shift, value,
                yerr=np.vstack([np.minimum(value, 1.96*mcse), 1.96*mcse]),
                fmt=marker, color=colors[method], capsize=1.7, markersize=4, linewidth=.8)
        data = interval[interval.target.eq(target)].set_index("scenario").loc[ordered]
        value = data.all_alarm_containment
        right.errorbar(x, value, yerr=np.vstack([value-data.containment_ci_low,
            data.containment_ci_high-value]), fmt="o", color=colors["OCFR"], capsize=1.7, markersize=4, linewidth=.8)
        right.axhline(.95, color=".4", linestyle="--", linewidth=.8)
        right.set_ylim(0, 1.04)
        left.set_ylim(bottom=0)
        left.set_ylabel(f"{label}\nRelative RMSE")
        right.set_ylabel("OCFR interval coverage")
        for column, ax in enumerate((left, right)):
            ax.set_xlim(-.5, 8.5)
            ax.axvspan(-.5, 2.5, color="#edf2f6", zorder=-5)
            ax.axvspan(5.5, 8.5, color="#edf2f6", zorder=-5)
            for boundary in (2.5, 5.5):
                ax.axvline(boundary, color=".8", linewidth=.6)
            _compact_letter(ax, "ABCDEF"[2*row+column])
            ax.set_xticks(x, ["2/3", "1.25", "1.5"]*3)
            ax.tick_params(axis="x", labelsize=9.5)
        for pos, family in zip((1, 4, 7), ("Steady", "Sparse", "Wave")):
            for ax in (left, right):
                ax.text(pos, 1.035, family, transform=ax.get_xaxis_transform(), ha="center", fontsize=9.5)
    for ax in axes[-1]:
        ax.set_xlabel("True risk ratio")
    fig.legend(handles=[plt.Line2D([], [], marker=marker, color=colors[method], linestyle="none", label=method)
        for method, _, _, marker in methods],
        loc="upper center", ncol=len(methods), bbox_to_anchor=(.53, .995), columnspacing=1.)
    fig.subplots_adjust(left=.13, right=.98, top=.91, bottom=.09, wspace=.49, hspace=.36)
    _compact_save(fig, directory, name)
    if ls:
        return ("Point estimation and OCFR interval coverage at the original selected first alarms. "
            "Rows: pre-change CFR, post-change CFR, and risk ratio. A,C,E: relative RMSE of OCFR "
            "(blue) and full-delay convolution least squares (red). Both use the same full-delay "
            "mean model, observed prefix, and selected change date. LS has no penalty; OCFR uses "
            "fixed log-RR penalty precision 1. Point comparisons use target-specific common available "
            "LS and OCFR episodes, without conditioning on unpenalized NB, Poisson, Nishiura, or "
            "interval availability. Bars are approximate "
            "95% Monte Carlo intervals from the analytic delta method for RMSE. B,D,F: unchanged "
            "OCFR-only unadjusted Wald interval coverage among all first alarms, with exact 95% "
            "binomial Monte Carlo bars; .95 marks nominal coverage. These intervals are not "
            "selection-adjusted. LS supplies point estimates only. Each of the nine predeclared "
            "day-92 conditions contains 1000 episodes; premature alarms are retained. All-available "
            "errors, counts, and all 16 conditions are retained in the LS source tables.")
    if poisson:
        return ("Point estimation and OCFR interval coverage at the original selected first alarms. "
            "Rows: pre-change CFR, post-change CFR, and risk ratio. A,C,E: relative RMSE of "
            "unpenalized NB, OCFR, and matched full-delay joint Poisson fits. OCFR and Poisson "
            "use the same fixed log-RR penalty of 1. Point comparisons use the target-specific "
            "common available episodes for these three methods, without conditioning on Nishiura "
            "availability. Bars are approximate 95% Monte Carlo intervals from the delta method. "
            "B,D,F: unchanged OCFR-only unadjusted Wald interval coverage among all first alarms, "
            "with exact 95% binomial Monte Carlo bars; .95 marks nominal coverage. No Poisson "
            "intervals are supplied. The nine day-92 cells include premature alarms. All-available "
            "errors and availability counts are retained in the source tables.")
    return (("OCFR uses a fixed log-risk-ratio penalty; the unpenalized NB fit is retained as a reference. "
        if stabilized else "") + "Point estimation and OCFR interval coverage at the same selected first alarms. "
        "Rows: pre-change CFR, post-change CFR, and risk ratio. A,C,E: relative RMSE for the joint NB fit "
        "and a two-stage extension of Nishiura delay adjustment; bars are approximate 95% Monte Carlo "
        "intervals from the delta method for RMSE. Point comparisons use the same target-specific episodes "
        "with finite estimates from both methods, retaining any finite inadmissible outputs. "
        "All-available errors and unavailability counts are also retained. "
        "B,D,F: OCFR-only Wald interval coverage with exact 95% binomial "
        "Monte Carlo bars; .95 marks nominal coverage. The reference supplies point estimates only. "
        "The nine day-92 cells and all first alarms, including premature alarms, are retained. "
        "Availability, paired errors, interval widths, and location errors are retained in the source summaries.")


POISSON_ESTIMATION_METHODS = (
    ("Unpenalized NB", "", "NB", 0.),
    ("OCFR", "regularized_", "NB", 1.),
    ("Unpenalized Poisson", "poisson_unpenalized_", "Poisson", 0.),
    ("Matched Poisson", "poisson_regularized_", "Poisson", 1.),
)
POISSON_MAIN_METHODS = ("Unpenalized NB", "OCFR", "Matched Poisson")


def _relative_error_scores(squared: pd.Series) -> dict[str, float]:
    """Analytic Monte Carlo uncertainty, retaining extreme finite errors."""
    mse = float(squared.mean()) if len(squared) else np.nan
    rmse = float(np.sqrt(mse))
    se = (float(squared.std(ddof=1)/np.sqrt(len(squared))/(2*rmse))
          if len(squared) > 1 and rmse > 0 else 0. if rmse == 0 else np.nan)
    return {"relative_mse": mse, "relative_rmse": rmse,
            "relative_rmse_mcse": se}


def compact_poisson_summaries(banks: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Point-only 2-by-2 likelihood/penalty comparison on frozen alarm episodes.

    Availability always divides by all alarms. Main paired errors require the
    three displayed estimators, separately for each target. The four-way cohort
    is also exported for the complete factorial comparison. Pairwise contrasts
    use their own explicitly reported intersection and analytic paired MCSE.
    """
    summaries, contrasts, aggregates = [], [], []
    for bank, frame in banks.items():
        for scenario, group in frame.groupby("scenario", sort=True):
            first = group.iloc[0]
            alarm = _as_binary(group.alarm, "alarm").astype(bool)
            common_meta = dict(bank=bank, scenario=scenario, family=first.family,
                risk_ratio=float(first.risk_ratio), change_day=int(first.true_change)+1,
                N_requested=len(group), n_alarm=int(alarm.sum()),
                n_no_alarm=int((~alarm).sum()),
                n_premature=int((alarm & (group.alarm_day < group.true_change)).sum()))
            for target in ESTIMATION_TARGETS:
                truth = pd.to_numeric(group[target], errors="raise")
                if (not np.isfinite(truth).all() or (truth <= 0).any()
                        or truth.nunique() != 1):
                    raise EvidenceError(f"{bank}/{scenario}/{target}: inconsistent generating truth")
                masks, errors = {}, {}
                for method, prefix, _, _ in POISSON_ESTIMATION_METHODS:
                    point = _as_binary(group[f"{prefix}{target}_point_valid"], target).astype(bool)
                    estimate = pd.to_numeric(group[f"{prefix}{target}_estimate"], errors="raise")
                    if (point & ~alarm).any():
                        raise EvidenceError("Poisson comparison point availability outside alarms")
                    if (point & (~np.isfinite(estimate) | (estimate <= 0) |
                            ((estimate >= 1) if target != "risk_ratio" else False))).any():
                        raise EvidenceError("Poisson comparison has an invalid declared available estimate")
                    if estimate[~alarm].notna().any():
                        raise EvidenceError("no-alarm episode has fabricated Poisson comparison output")
                    masks[method] = alarm & point
                    errors[method] = estimate/truth-1.
                core = alarm.copy()
                four = alarm.copy()
                for method, mask in masks.items():
                    four &= mask
                    if method in POISSON_MAIN_METHODS:
                        core &= mask
                for method, _, likelihood, precision in POISSON_ESTIMATION_METHODS:
                    valid = masks[method]
                    shared = core & valid
                    record = {**common_meta, "target": target, "truth": float(truth.iloc[0]),
                        "method": method, "likelihood": likelihood, "penalty_precision": precision,
                        "n_point": int(valid.sum()), "n_unavailable": int((alarm & ~valid).sum()),
                        "point_availability": float(valid.sum()/alarm.sum()) if alarm.any() else np.nan,
                        "relative_bias": float(errors[method][valid].mean()),
                        "n_common_main": int(core.sum()), "n_common_paired": int(shared.sum()),
                        "n_common_four": int(four.sum()),
                        **_relative_error_scores(errors[method][valid]**2)}
                    for label, mask in (("common_paired", shared), ("four_way", four)):
                        record.update({f"{label}_{key}": value for key, value in
                                       _relative_error_scores(errors[method][mask]**2).items()})
                    summaries.append(record)
                methods = [item[0] for item in POISSON_ESTIMATION_METHODS]
                for index, first_method in enumerate(methods):
                    for second_method in methods[index+1:]:
                        paired = masks[first_method] & masks[second_method]
                        for cohort, mask in (("pairwise_available", paired),
                                             ("main_common", core & paired),
                                             ("four_way_common", four)):
                            difference = (errors[first_method][mask]**2-errors[second_method][mask]**2)
                            contrasts.append({**common_meta, "target": target, "method_a": first_method,
                                "method_b": second_method, "cohort": cohort, "n_paired": int(mask.sum()),
                                "mean_squared_relative_error_difference": float(difference.mean()),
                                "paired_mcse": float(difference.std(ddof=1)/np.sqrt(len(difference)))
                                    if len(difference) > 1 else np.nan})
    summary, contrast = pd.DataFrame(summaries), pd.DataFrame(contrasts)
    for (bank, method, target), group in summary.groupby(["bank", "method", "target"], sort=True):
        record = dict(bank=bank, method=method, target=target, n_cells=len(group),
            n_alarm=int(group.n_alarm.sum()), n_point=int(group.n_point.sum()),
            n_unavailable=int(group.n_unavailable.sum()),
            n_common_paired=int(group.n_common_paired.sum()), n_common_four=int(group.n_common_four.sum()))
        record["point_availability"] = record["n_point"]/record["n_alarm"] if record["n_alarm"] else np.nan
        for label, column in (("all_available", "relative_mse"),
                              ("common_paired", "common_paired_relative_mse"),
                              ("four_way", "four_way_relative_mse")):
            # Missing cells are not silently dropped from an equal-cell estimand.
            record[f"{label}_relative_mse"] = float(group[column].mean()) if group[column].notna().all() else np.nan
            record[f"{label}_relative_rmse"] = float(np.sqrt(record[f"{label}_relative_mse"]))
        comparisons = contrast[(contrast.bank == bank) & (contrast.target == target) &
            contrast.cohort.eq("pairwise_available") &
            ((contrast.method_a == method) | (contrast.method_b == method))]
        for opponent in [item[0] for item in POISSON_ESTIMATION_METHODS if item[0] != method]:
            rows = comparisons[(comparisons.method_a == opponent) | (comparisons.method_b == opponent)]
            delta = rows.mean_squared_relative_error_difference * np.where(rows.method_a.eq(method), 1., -1.)
            key = opponent.lower().replace(" ", "_")
            record[f"mse_wins_vs_{key}"] = int((delta < 0).sum())
            record[f"mse_ties_vs_{key}"] = int((delta == 0).sum())
            record[f"mse_cells_compared_vs_{key}"] = int(delta.notna().sum())
        aggregates.append(record)
    return {"compact_poisson_summary": summary, "compact_poisson_contrasts": contrast,
            "compact_poisson_aggregate": pd.DataFrame(aggregates)}


def _compact_load_poisson_estimation(output: Path) -> tuple[dict, dict]:
    """Bind comparator artifacts to their historical producers and frozen NB rows."""
    banks, metadata = {}, {}
    stages = (("Original bank", "estimation-poisson-original", "estimation-stabilized"),
              ("Independent bank", "estimation-poisson-validation", "estimation-validation"))
    if not any((output/stage).exists() for _, stage, _ in stages):
        return {}, {"status": "absent"}
    for bank, stage, source_stage in stages:
        directory, source = output/stage, output/source_stage
        path, source_path = directory/MANIFEST_FILE, source/MANIFEST_FILE
        if not path.is_file() or not source_path.is_file():
            raise EvidenceError("Poisson comparison requires both complete banks and frozen sources")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
        if (manifest.get("stage") != stage or manifest.get("status") != "complete"
                or manifest.get("mode") not in {"demo", "full"}):
            raise EvidenceError(f"{stage}: incomplete Poisson comparison manifest")
        if (source_manifest.get("status") != "complete" or source_manifest.get("stage") != source_stage
                or source_manifest.get("mode") != "full"):
            raise EvidenceError("Poisson comparator source stage is incomplete")
        raw, source_raw = directory/RAW_FILE, source/RAW_FILE
        for filename, declared in ((raw, _manifest_file_hash(manifest, RAW_FILE)),
                                   (source_raw, _manifest_file_hash(source_manifest, RAW_FILE))):
            if _sha256(filename) != declared:
                raise EvidenceError("Poisson comparison artifact hash mismatch")
        for key, value in (("source_manifest_sha256", _sha256(source_path)),
                           ("source_csv_sha256", _sha256(source_raw))):
            if manifest.get(key) != value:
                raise EvidenceError(f"Poisson comparison frozen source mismatch: {key}")
        expected_bank = "original" if source_stage == "estimation-stabilized" else "validation"
        if (manifest.get("source_bank") != expected_bank or
                manifest.get("source_scientific_digest") != source_manifest.get("scientific_digest") or
                manifest.get("scientific_digest") != source_manifest.get("scientific_digest") or
                manifest.get("scientific_config") != source_manifest.get("scientific_config")):
            raise EvidenceError("Poisson comparison scientific source/configuration mismatch")
        settings = manifest.get("poisson_estimation_config", {})
        if settings != {"primary_precision": 1., "control_precision": 0.,
                        "likelihood": "exact_poisson", "interval_scope": "point_only"}:
            raise EvidenceError("Poisson comparison likelihood, penalties, or interval scope changed")
        records = manifest.get("postprocessor_code_files", {})
        if not records or manifest.get("postprocessor_code_hash") != hashlib.sha256(
                _canonical_json(records).encode("ascii")).hexdigest():
            raise EvidenceError("Poisson postprocessor code hash mismatch")
        provenance_keys = ("source_manifest_sha256", "source_csv_sha256", "source_bank",
            "source_scientific_digest", "scientific_config", "postprocessor_code_hash",
            "postprocessor_code_files", "postprocessor_native_files", "poisson_estimation_config")
        provenance = {key: manifest.get(key) for key in provenance_keys}
        if manifest.get("cache_key") != hashlib.sha256(_canonical_json(provenance).encode("ascii")).hexdigest()[:20]:
            raise EvidenceError("Poisson comparison cache provenance mismatch")
        # These recorded hashes attest the producing version, not today's files.
        frame, original = pd.read_csv(raw), pd.read_csv(source_raw)
        if any(str(column).startswith("poisson_") for column in original):
            raise EvidenceError("Poisson output namespace collides with frozen source")
        if frame.duplicated(["scenario", "replicate"]).any() or not set(original).issubset(frame):
            raise EvidenceError("Poisson comparison episode/column contract changed")
        selected = original if manifest["mode"] == "full" else original[original.replicate < 10]
        try:
            pd.testing.assert_frame_equal(frame[original.columns].reset_index(drop=True),
                selected.reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-15)
        except (KeyError, AssertionError) as exc:
            raise EvidenceError("Poisson stage changed frozen NB/reference observations, alarms, or truth") from exc
        count = 1000 if manifest["mode"] == "full" else 10
        names = set(source_manifest.get("scenarios", []))
        if len(names) != 16 or set(frame.scenario) != names or set(manifest.get("scenarios", [])) != names:
            raise EvidenceError("Poisson comparison requires the complete 16-cell grid")
        required = {f"{prefix}{target}_{suffix}" for _, prefix, _, _ in POISSON_ESTIMATION_METHODS
                    for target in ESTIMATION_TARGETS for suffix in ("estimate", "point_valid")}
        required |= {f"{prefix}{suffix}" for prefix in ("poisson_regularized_", "poisson_unpenalized_")
                     for suffix in ("status", "seconds", "loglik", "iterations")}
        if required-set(frame):
            raise EvidenceError(f"Poisson comparison missing columns: {sorted(required-set(frame))}")
        for scenario, group in frame.groupby("scenario", sort=True):
            if (len(group) != count or set(group.replicate) != set(range(count)) or
                    manifest.get("replications_by_scenario", {}).get(scenario) != count):
                raise EvidenceError(f"{scenario}: incomplete Poisson comparison episode grid")
        if not np.allclose(frame.pi_right/frame.pi_left, frame.risk_ratio):
            raise EvidenceError("Poisson comparison generating risk-ratio truth is inconsistent")
        banks[bank] = frame
        metadata[stage] = {"source_manifest_sha256": _sha256(path), "source_csv_sha256": _sha256(raw),
            "frozen_source_manifest_sha256": _sha256(source_path), "mode": manifest["mode"],
            "postprocessor_code_hash": manifest["postprocessor_code_hash"]}
    return compact_poisson_summaries(banks), {"status": "complete", "banks": metadata,
        "availability_denominator": "all first alarms, including premature alarms",
        "main_paired_cohort": "target-specific common available NB unpenalized, NB penalty 1, Poisson penalty 1",
        "contrast_uncertainty": "analytic sample SD of paired squared relative-error differences / sqrt(n)",
        "interval_scope": "Poisson point estimates only; existing OCFR intervals unchanged"}


def _compact_poisson_comparison_figure(tables: dict, directory: Path) -> str:
    """Equal-cell errors on zero-based axes, except the extreme independent RR."""
    data = tables["compact_poisson_aggregate"]
    fig, axes = plt.subplots(3, 2, figsize=(6.8, 7.4))
    styles = (("Unpenalized NB", "#737373", "^"), ("OCFR", "#185889", "o"),
              ("Unpenalized Poisson", "#BE8A45", "^"), ("Matched Poisson", "#B44C43", "s"))
    for column, bank in enumerate(("Original bank", "Independent bank")):
        for row, (target, label) in enumerate(zip(ESTIMATION_TARGETS,
                ("Pre-change CFR", "Post-change CFR", "Risk ratio"))):
            ax = axes[row, column]
            for x, (method, color, marker) in enumerate(styles):
                selected = data[data.bank.eq(bank) & data.target.eq(target) & data.method.eq(method)]
                if len(selected) != 1:
                    raise EvidenceError("Poisson supplementary figure needs both complete four-method banks")
                value = float(selected.iloc[0].four_way_relative_rmse)
                if not np.isfinite(value) or value <= 0:
                    raise EvidenceError("Poisson log-RMSE figure requires positive available errors in every cell")
                ax.plot(x, value, marker, color=color, markersize=6)
                ax.annotate(f"{value:.3g}", (x, value), xytext=(0, 9),
                            textcoords="offset points", ha="center", fontsize=9.5)
            target_values = data[data.target.eq(target)].four_way_relative_rmse
            if row == 2 and column == 1:
                values = data[data.target.eq(target) & data.bank.eq(bank)].four_way_relative_rmse
                lower, upper = min(.08, float(values.min())*.5), max(6e6, float(values.max())*4)
                ax.set_yscale("log")
                ax.set_ylim(lower, upper)
                low_exp, high_exp = int(np.ceil(np.log10(lower))), int(np.floor(np.log10(upper)))
                exponents = sorted(set([*range(low_exp, high_exp+1, 2), high_exp]))
                ax.set_yticks([10.**exponent for exponent in exponents])
                ylabel = label+"\nRelative RMSE (log scale)"
            else:
                if row < 2:
                    upper = max(.3 if row == 0 else .2, float(target_values.max())*1.2)
                else:
                    values = data[data.target.eq(target) & data.bank.eq(bank)].four_way_relative_rmse
                    upper = float(values.max())*1.2
                ax.set_ylim(0, upper)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=4,
                                                       steps=[1, 2, 2.5, 5, 10]))
                ylabel = label+"\nRelative RMSE"
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: (
                f"{value:g}" if value == 0 or .001 <= value < 1000 else
                f"{value:.0e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-"))))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks(range(4), ["NB\n0", "NB\n1", "Poisson\n0", "Poisson\n1"])
            ax.set_xlim(-.5, 3.5)
            ax.set_ylabel(ylabel)
            _compact_letter(ax, "ABCDEF"[2*row+column])
            if row == 0:
                ax.set_title(bank, pad=12)
            if row == 2:
                ax.set_xlabel("Likelihood / log-RR penalty precision")
    fig.subplots_adjust(left=.15, right=.98, top=.92, bottom=.095, wspace=.52, hspace=.4)
    _compact_save(fig, directory, "CompactFigS3_poisson_comparison")
    return ("Full-delay joint NB and Poisson point fits with log-RR penalty precision 0 or 1. "
        "Rows show pre-change CFR, post-change CFR, and risk ratio; columns show the original and "
        "independent banks. Zero denotes no penalty and 1 denotes the fixed unit-precision log-RR "
        "penalty; NB 1 is OCFR and Poisson 1 is matched Poisson. Symbols are point summaries without "
        "uncertainty bars. Squared relative errors are averaged equally across all 16 conditions "
        "before taking the square root. Every method uses the same target-specific four-way available "
        "episodes within each condition. Panels A-E use linear axes starting at zero, with matching "
        "limits across banks for each CFR target. Panel F uses a logarithmic axis to retain the extreme "
        "finite independent-bank RR errors. Availability, "
        "all-available errors, three-method main comparisons, and analytic paired Monte Carlo contrasts "
        "are supplied in the source tables. These are point comparisons and contain no Poisson intervals.")


def _compact_load_stability_validation(output: Path) -> tuple[dict, dict]:
    path = output / "estimation-validation" / MANIFEST_FILE
    if not path.exists():
        return {}, {"status": "absent"}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("stage") != "estimation-validation" or manifest.get("status") != "complete"
            or manifest.get("mode") != "full"):
        raise EvidenceError("independent estimation validation is not complete and full")
    if manifest.get("validation_config", {}).get("validation_master_seed") != 2026110702:
        raise EvidenceError("independent validation seed differs from the manuscript")
    if manifest.get("scientific_config", {}).get("master_seed") != 2026110702:
        raise EvidenceError("independent validation scientific seed mismatch")
    records = manifest.get("postprocessor_code_files", {})
    if not records or manifest.get("postprocessor_code_hash") != hashlib.sha256(
            _canonical_json(records).encode("ascii")).hexdigest():
        raise EvidenceError("independent validation postprocessor hash mismatch")
    config = manifest.get("post_alarm_estimation_config", {})
    if config.get("precision") != 1 or config.get("covariance") != "sandwich":
        raise EvidenceError("independent validation estimator settings changed")
    raw = path.parent / RAW_FILE
    if _sha256(raw) != _manifest_file_hash(manifest, RAW_FILE):
        raise EvidenceError("independent validation raw-file hash mismatch")
    original = pd.read_csv(output/"estimation-stabilized"/RAW_FILE)
    independent = pd.read_csv(raw)
    if (len(independent) != 16000 or independent.duplicated(["scenario", "replicate"]).any()
            or set(independent.scenario) != set(original.scenario)):
        raise EvidenceError("independent validation episode grid changed")
    joined = independent.merge(original[["scenario", "replicate", "case_hash", "death_hash"]],
        on=["scenario", "replicate"], suffixes=("", "_original"), validate="one_to_one")
    if ((joined.case_hash == joined.case_hash_original).any()
            or (joined.death_hash == joined.death_hash_original).any()):
        raise EvidenceError("independent validation reused original observations")
    rows = []
    for bank, frame in (("Original bank", original), ("Independent bank", independent)):
        if len(frame) != 16000:
            raise EvidenceError("stability validation requires two complete banks")
        for target in ESTIMATION_TARGETS:
            for method, prefix in (("Unpenalized NB", ""), ("OCFR", "regularized_")):
                influences, means = np.zeros(1000), []
                n_alarm = n_point = 0
                for scenario, group in frame.groupby("scenario"):
                    if set(group.replicate) != set(range(1000)):
                        raise EvidenceError("incomplete independent validation cell")
                    alarm = _as_binary(group.alarm, "alarm").astype(bool)
                    good = alarm & _as_binary(group[prefix+target+"_point_valid"], target).astype(bool)
                    n_alarm += int(alarm.sum()); n_point += int(good.sum())
                    squared = (group[prefix+target+"_estimate"]/group[target]-1.)**2
                    mean = float(squared[good].mean()); means.append(mean)
                    if good.sum():
                        influence = np.where(good, 1000/good.sum()*(squared-mean), 0.)
                        influences[group.replicate.to_numpy(int)] += np.nan_to_num(influence)/16
                rmse = float(np.sqrt(np.mean(means)))
                mcse = float(influences.std(ddof=1)/np.sqrt(1000)/(2*rmse)) if rmse>0 else 0.
                rows.append(dict(bank=bank, target=target, method=method,
                    relative_rmse=rmse, monte_carlo_se=mcse, n_alarm=n_alarm, n_point=n_point))
    return {"compact_stability_validation": pd.DataFrame(rows)}, {
        "status": "complete", "source_manifest_sha256": _sha256(path),
        "source_csv_sha256": _sha256(raw), "master_seed": 2026110702,
        "artifact_origin": manifest.get("artifact_origin", "native canonical validation driver"),
        "mc_uncertainty": "delta method; replicate clusters across conditions",
        "scope": "same first Gaussian alarm per estimator within each independent episode"}


def _compact_stability_validation_figure(tables: dict, directory: Path) -> str:
    data = tables["compact_stability_validation"]
    fig, axes = plt.subplots(3, 2, figsize=(6.8, 7.4))
    for column, bank in enumerate(("Original bank", "Independent bank")):
        for row, (target, label) in enumerate(zip(ESTIMATION_TARGETS,
                ("Pre-change CFR", "Post-change CFR", "Risk ratio"))):
            ax = axes[row, column]
            for x, (method, color) in enumerate((("Unpenalized NB", "#737373"), ("OCFR", "#185889"))):
                record = data[data.bank.eq(bank) & data.target.eq(target) & data.method.eq(method)].iloc[0]
                value, se = record.relative_rmse, record.monte_carlo_se
                lower = max(0., value-1.96*se)
                if row == 2 and column == 1:
                    lower = max(lower, value*.02)
                ax.errorbar(x, value, yerr=[[value-lower], [1.96*se]], fmt="o",
                    color=color, capsize=3, markersize=5)
                text = f"{value:.3f}" if value < 10 else f"{value:.2e}"
                ax.annotate(text, (x, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9.5)
            ax.set_xticks([0, 1], ["Unpenalized NB", "OCFR"])
            ax.set_xlim(-.5, 1.5)
            ax.tick_params(axis="x", labelsize=9.5)
            ax.set_ylabel(label+"\nRelative RMSE")
            if row == 2 and column == 1:
                ax.set_yscale("log"); ax.set_ylim(.08, 6e6)
                ax.set_yticks([.1, 10, 1e3, 1e5, 1e6], ["0.1", "10", "1e3", "1e5", "1e6"])
                ax.set_ylabel(label+"\nRelative RMSE (log scale)")
            else:
                ax.set_ylim(bottom=0); ax.margins(y=.18)
            _compact_letter(ax, "ABCDEF"[2*row+column])
            if row == 0:
                ax.set_title(bank, pad=12)
    fig.subplots_adjust(left=.14, right=.98, top=.92, bottom=.075, wspace=.5, hspace=.45)
    _compact_save(fig, directory, "CompactFigS1_estimation_stability")
    return ("All-alarm risk estimation in the original and independent banks. Rows: pre-change CFR, "
        "post-change CFR, and RR. OCFR uses the fixed log-RR penalty; the unpenalized NB fit is retained. "
        "Each bank contains 16 conditions with 1000 episodes each; errors are averaged equally over "
        "conditions before taking the square root. Bars are approximate 95% Monte Carlo intervals "
        "using replicate clusters across conditions. Every first alarm is retained. Panel F is logarithmic "
        "because two premature alarms produced near-zero unpenalized baseline risks and extreme RR estimates.")


def _compact_estimation_figure(tables: dict[str, pd.DataFrame], directory: Path) -> str:
    if "compact_ls_summary" in tables:
        return _compact_estimation_comparison_figure(tables, directory, ls=True)
    if "compact_poisson_summary" in tables:
        return _compact_estimation_comparison_figure(tables, directory, poisson=True)
    if "compact_estimation_reference_summary" in tables:
        return _compact_estimation_comparison_figure(tables, directory)
    summary, location = tables["compact_estimation_summary"], tables["compact_estimation_location"]
    selected = location[location.change_day.eq(92) &
        np.isclose(location.risk_ratio.to_numpy()[:, None], [2/3, 1.25, 1.5]).any(axis=1)]
    ordered = []
    for family in ("steady", "sparse", "wave"):
        for rr in (2/3, 1.25, 1.5):
            row = selected[selected.family.eq(family) & np.isclose(selected.risk_ratio, rr)]
            if len(row) != 1:
                raise EvidenceError("estimation main figure requires nine frozen day-92 cells")
            ordered.append(row.iloc[0])
    fig, axes = plt.subplots(4, 1, figsize=(6.4, 7.8), sharex=True)
    colors, markers = ("#185889", "#B44C43", "#288575"), ("o", "s", "^")
    labels = ("Pre-change CFR", "Post-change CFR", "Risk ratio")
    x = np.arange(9)
    for j, target in enumerate(ESTIMATION_TARGETS):
        data = summary[summary.target.eq(target)].set_index("scenario").loc[[r.scenario for r in ordered]]
        xx = x+(j-1)*.17
        for ax, metric in ((axes[0], "relative_rmse"), (axes[2], "median_relative_width")):
            ax.plot(xx, data[metric], marker=markers[j], color=colors[j], linestyle="none", markersize=4)
        q = data.all_alarm_containment
        axes[1].errorbar(xx, q, yerr=np.vstack([q-data.containment_ci_low,
            data.containment_ci_high-q]), fmt=markers[j], color=colors[j], linestyle="none", capsize=2, markersize=4)
    axes[1].axhline(.95, color=".4", linestyle="--", linewidth=.8)
    axes[1].set_ylim(0, 1.04)
    axes[3].plot(x, [r.location_mae_days for r in ordered], "D", color=".25", markersize=4)
    for i, (ax, label) in enumerate(zip(axes, ("Relative RMSE", "Interval coverage", "Median relative width", "Location MAE (days)"))):
        ax.set_ylabel(label)
        if i != 1:
            ax.set_ylim(bottom=0)
        ax.set_xlim(-.5, 8.5)
        ax.axvspan(-.5, 2.5, color="#edf2f6", zorder=-5)
        ax.axvspan(5.5, 8.5, color="#edf2f6", zorder=-5)
        for boundary in (2.5, 5.5):
            ax.axvline(boundary, color=".8", linewidth=.6)
        _compact_letter(ax, "ABCD"[i])
    for pos, family in zip((1, 4, 7), ("Steady", "Sparse", "Wave")):
        axes[0].text(pos, 1.07, family, transform=axes[0].get_xaxis_transform(), ha="center")
    axes[3].set_xticks(x, [('2/3', '1.25', '1.5')[i%3] for i in range(9)])
    axes[3].set_xlabel("True risk ratio")
    fig.legend(handles=[plt.Line2D([], [], marker=m, color=c, linestyle="none", label=l)
        for m, c, l in zip(markers, colors, labels)], loc="upper center", ncol=3, bbox_to_anchor=(.53, .995),
        columnspacing=1.1, handletextpad=.4)
    fig.subplots_adjust(left=.20, right=.98, top=.88, bottom=.10, hspace=.38)
    _compact_save(fig, directory, "CompactFig5_estimation")
    n = sorted(summary.N_requested.unique())
    alarm_counts = [int(r.n_alarm) for r in ordered]
    return ("First-alarm estimation under Gaussian OCFR in nine day-92 alternatives. "
        f"Each cell requests N={','.join(map(str, n))} independent episodes. "
        "A: relative RMSE among available joint point fits. B: interval availability and containment "
        "among all first alarms, including premature alarms; unavailable intervals count as noncontainment. "
        "Bars are exact 95% binomial Monte Carlo intervals; .95 is a nominal reference. "
        "These unadjusted intervals provide no coverage guarantee at selected alarms. "
        "C: median relative width among available finite intervals. D: mean absolute stored location error. "
        f"First-alarm counts range from {min(alarm_counts)} to {max(alarm_counts)}; "
        "machine-readable summaries retain the exact denominators and point/interval availability. "
        "The figure assesses absolute estimation accuracy and interval performance.")


def _compact_style() -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif"],
        "font.size": 10.5, "axes.labelsize": 10.5, "axes.titlesize": 10.5,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .8,
        "lines.linewidth": 1.4, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300})


def _compact_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-.16, 1.06, letter, transform=ax.transAxes, weight="bold", fontsize=11)


def _compact_probability_panel(ax: plt.Axes, data: pd.DataFrame, xcolumn: str,
                               methods=COMPACT_MAIN_METHODS, xlabels=None, *,
                               connect_categories: bool = False) -> None:
    for index, method in enumerate(methods):
        subset = data[data.method == method].sort_values(xcolumn)
        if subset.empty:
            continue
        branches = ([subset[subset.risk_ratio < 1], subset[subset.risk_ratio == 1],
                     subset[subset.risk_ratio > 1]] if xcolumn == "risk_ratio" else [subset])
        branches = [branch for branch in branches if not branch.empty]
        for branch_index, branch in enumerate(branches):
            x = branch[xcolumn].to_numpy(float)
            if xlabels is not None:
                mapping = {value: i for i, value in enumerate(xlabels)}
                x = np.array([mapping[value] for value in x])+(index-(len(methods)-1)/2)*.11
            ax.errorbar(x, branch.estimate,
                yerr=np.vstack([branch.estimate-branch.ci_low, branch.ci_high-branch.estimate]),
                fmt=COMPACT_MARKERS[method],
                linestyle=(COMPACT_LINESTYLES[method] if (xlabels is None or connect_categories) and
                    (xcolumn != "risk_ratio" or len(branch) > 1) else "none"),
                linewidth=COMPACT_PAIRED_LINEWIDTHS.get(method) if ORACLE_OCFR in methods else None,
                elinewidth=1.1 if ORACLE_OCFR in methods else None,
                markersize=3.5, color=COMPACT_COLORS[method], capsize=2,
                label=method if branch_index == 0 else None)
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)), list(xlabels.values()))
    ax.set_ylim(0, 1.02)


def _compact_save(fig: plt.Figure, directory: Path, name: str) -> None:
    for ax in fig.axes:
        if ax.axison:
            lo, hi = sorted(ax.get_xlim())
            visible_ticks = [tick for tick in ax.get_xticks() if lo <= tick <= hi]
            if len(visible_ticks) < 3:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    # Fixed canvas dimensions keep the physical publication dimensions auditable.
    fig.savefig(directory / f"{name}.pdf", facecolor="white")
    png = directory / f"{name}.png"
    fig.savefig(png, dpi=600, facecolor="white")
    journal_name = name.split("_", 1)[0].removeprefix("Compact")
    with Image.open(png) as raster:
        raster.convert("RGB").save(directory / f"{journal_name}.tif",
                                   compression="tiff_lzw", dpi=(600, 600))
    plt.close(fig)


def _compact_export_metadata(directory: Path, names: Sequence[str], *,
                             minimum_font_points: float = 10.5) -> dict[str, Any]:
    """Inspect actual rasters and record the resolved publication font."""
    from matplotlib.font_manager import FontProperties, findfont
    font_path = findfont(FontProperties(family="serif"))
    exports = {}
    for name in names:
        journal_name = name.split("_", 1)[0].removeprefix("Compact")
        with Image.open(directory/f"{name}.png") as raster:
            png = {"pixels": list(raster.size), "dpi": [float(v) for v in raster.info.get("dpi", ())],
                   "mode": raster.mode}
        with Image.open(directory/f"{journal_name}.tif") as raster:
            tiff = {"pixels": list(raster.size), "dpi": [float(v) for v in raster.info.get("dpi", ())],
                    "mode": raster.mode, "compression": int(raster.tag_v2.get(259, 1))}
        if (png["pixels"] != tiff["pixels"] or tiff["mode"] != "RGB" or
            len(png["dpi"]) != 2 or len(tiff["dpi"]) != 2 or
            any(abs(float(dpi)-600) > .05 for dpi in png["dpi"]+tiff["dpi"]) or
            tiff["compression"] != 5):
            raise EvidenceError(f"publication export metadata is inconsistent for {name}")
        exports[name] = {"pdf": f"{name}.pdf", "png": png, "tiff": tiff,
                         "physical_inches": [value/600 for value in png["pixels"]]}
    return {"font_family": FontProperties(fname=font_path).get_name(),
            "font_file": Path(font_path).name, "minimum_font_points": minimum_font_points,
            "raster_dpi": 600, "pdf_fonttype": 42, "figures": exports}



def _compact_v2_detection_figure(power: pd.DataFrame, curves: pd.DataFrame, directory: Path) -> str:
    """Five data axes, with method colors separate from the OCFR timing styles."""
    if power.method.eq(ORACLE_OCFR).any():
        return _compact_paired_detection_figure(power, curves, directory)
    fig = plt.figure(figsize=(6.4, 5.8))
    grid = fig.add_gridspec(2, 6)
    magnitude = [fig.add_subplot(grid[0, start:start+2]) for start in (0, 2, 4)]
    timing = [fig.add_subplot(grid[1, :3]), fig.add_subplot(grid[1, 3:])]
    methods = ("OCFR", "NB-GLR", "Page-CUSUM")
    for ax, family, letter in zip(magnitude, ("steady", "sparse", "wave"), "ABC"):
        subset = power[power.family.eq(family) & power.change_day.eq(92)]
        _compact_probability_panel(ax, subset, "risk_ratio", methods=methods)
        ax.set(xlabel="CFR risk ratio", ylabel="Detection\nprobability", title=family.capitalize())
        ax.set_xticks([2/3, 1.25, 1.5, 2], ["2/3", "1.25", "1.5", "2"])
        ax.tick_params(axis="x", labelsize=9.5)
        if ax is not magnitude[0]:
            ax.set_ylabel('')
        _compact_letter(ax, letter)
    timing_handles = []
    for day, style in zip((47, 92, 137), ("-", "--", ":")):
        timing_handles.append(plt.Line2D([], [], color=COMPACT_COLORS["OCFR"], linestyle=style,
                                        label=f"Day {day}"))
    for ax, family, letter in zip(timing, ("steady", "wave"), "DE"):
        for day, style in zip((47, 92, 137), ("-", "--", ":")):
            subset = curves[curves.stage.eq("evaluation") & curves.family.eq(family) &
                curves.misspecification.eq("on_model") & curves.change_day.eq(day) &
                np.isclose(curves.risk_ratio, 1.5) & curves.method.eq("OCFR") & (curves.delay_day <= 45)]
            ax.step(subset.delay_day, subset.estimate, where="post", color=COMPACT_COLORS["OCFR"],
                    linestyle=style)
        ax.set(xlim=(0, 45), ylim=(0, 1.02), xlabel="Days after change",
               ylabel="Cumulative detection\nprobability", title=family.capitalize())
        ax.set_xticks([0, 15, 30, 45])
        if ax is not timing[0]:
            ax.set_ylabel('')
        _compact_letter(ax, letter)
    handles = [plt.Line2D([], [], color=COMPACT_COLORS[method], marker=COMPACT_MARKERS[method],
               linestyle=COMPACT_LINESTYLES[method], markersize=4, label=method) for method in methods]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(.53, .995), fontsize=9.5)
    fig.legend(handles=timing_handles, ncol=3, loc="upper center", bbox_to_anchor=(.53, .495),
               title="OCFR change day", fontsize=9.5, title_fontsize=9.5)
    fig.subplots_adjust(left=.12, right=.98, bottom=.11, top=.88, hspace=1.0, wspace=1.25)
    _compact_save(fig, directory, "CompactFig3_detection")
    return ("A-C: useful first-alarm probability across CFR risk ratios with change day 92 in steady, "
        "sparse and wave families; bars are exact 95% binomial Monte Carlo intervals. D-E: OCFR "
        "cumulative useful detection under early (day 47), middle (92) and late (137) changes at "
        "RR=1.5 in steady and wave families, on a common 45-day scale. Timing curves share the OCFR "
        "color; solid, dashed and dotted lines distinguish change days. Every requested replicate "
        "forms each denominator; premature and absent alarms contribute zero useful detections. "
        "The timing curves are unconditional empirical probabilities, not alarm-conditional delay "
        "distributions; pointwise exact intervals and counts are exported. The five panels contain "
        "data; method and timing legends are separate.")


def _compact_paired_detection_figure(power: pd.DataFrame, curves: pd.DataFrame,
                                     directory: Path) -> str:
    """Separate change dates into panels so all curves share one method encoding."""
    methods = COMPACT_PAIRED_METHODS
    fig, axes = plt.subplots(3, 3, figsize=(6.8, 7.4))
    ratios = np.array([2/3, 1.25, 1.5, 2.])
    for ax, family in zip(axes[0], ("steady", "sparse", "wave")):
        subset = power[power.family.eq(family) & power.change_day.eq(92)]
        for method in methods:
            values = np.sort(subset.loc[subset.method.eq(method), "risk_ratio"])
            if len(values) != len(ratios) or not np.allclose(values, ratios):
                raise EvidenceError(f"detection requires the day-92 risk-ratio grid for {family}/{method}")
        _compact_probability_panel(ax, subset, "risk_ratio", methods=methods,
                                   xlabels=dict(zip(sorted(subset.risk_ratio.unique()),
                                                    ("2/3", "1.25", "1.5", "2"))),
                                   connect_categories=True)
        ax.set(xlabel="CFR risk ratio", title=family.capitalize())
    for row, family in enumerate(("steady", "wave"), start=1):
        for ax, day in zip(axes[row], (47, 92, 137)):
            subset = curves[curves.stage.eq("evaluation") & curves.family.eq(family) &
                curves.misspecification.eq("on_model") & curves.change_day.eq(day) &
                np.isclose(curves.risk_ratio, 1.5) & curves.delay_day.le(45)]
            for method in methods:
                series = subset[subset.method.eq(method)].sort_values("delay_day")
                if not np.array_equal(series.delay_day.to_numpy(), np.arange(46)):
                    raise EvidenceError(f"missing 45-day detection curve for {family}/day {day}/{method}")
                ax.step(series.delay_day, series.estimate, where="post",
                        color=COMPACT_COLORS[method], linestyle=COMPACT_LINESTYLES[method],
                        linewidth=COMPACT_PAIRED_LINEWIDTHS.get(method))
            ax.set(xlim=(0, 45), ylim=(0, 1.02), xticks=[0, 15, 30, 45],
                   title=f"{family.capitalize()}, day {day}")
            if row == 2:
                ax.set_xlabel("Days after change")
    axes[0, 0].set_ylabel("Useful detection\nprobability")
    for ax in axes[1:, 0]:
        ax.set_ylabel("Cumulative useful\ndetection probability")
    for letter, ax in zip("ABCDEFGHI", axes.flat):
        ax.set_yticks([0, .25, .5, .75, 1.])
        ax.tick_params(labelleft=ax in axes[:, 0])
        _compact_letter(ax, letter)
    fig.legend(handles=_compact_method_handles(methods), loc="upper center", ncol=4,
               bbox_to_anchor=(.54, .996), fontsize=9.5, frameon=False)
    fig.subplots_adjust(left=.15, right=.98, bottom=.09, top=.885, hspace=.74, wspace=.32)
    _compact_save(fig, directory, "CompactFig3_detection")
    return ("A-C: useful first-alarm probability across CFR risk ratios with change day 92 "
        "in steady, sparse and wave incidence. Lines join the increases. Bars are exact "
        "95% binomial Monte Carlo intervals. D-F and G-I: cumulative useful detection "
        "in steady and wave incidence, respectively, for a change on day 47, 92 or 137 "
        "at RR=1.5, with 45 days of follow-up. Pointwise exact intervals accompany the "
        "curves in the data table. Every denominator includes all requested episodes; "
        "premature and absent first alarms contribute zero.")


def _compact_dispersion_grid(summary, methods):
    expected = {(float(phi), float(rr), method, metric)
        for phi in (5, 10, 20, 40) for rr in (1., 1.5) for method in methods
        for metric in (("alarm_probability",) if rr == 1 else
                       ("alarm_probability", "useful_power", "premature_probability"))}
    actual = list(summary[["phi", "risk_ratio", "method", "metric"]].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected:
        raise EvidenceError("dispersion reporting requires every preselected null and power cell")


def _compact_dispersion_companions(dispersion, directory):
    directory.mkdir(parents=True, exist_ok=True)
    methods = (COMPACT_PAIRED_METHODS if dispersion.method.eq(ORACLE_OCFR).any()
               else _compact_method_contract("compact-v2", "evaluation"))
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    null = dispersion[dispersion.metric.eq("alarm_probability") & np.isclose(dispersion.risk_ratio, 1.)]
    _compact_probability_panel(ax, null, "phi", methods=methods)
    ax.set_xscale("log", base=2)
    ax.set_xticks([5, 10, 20, 40], ["5", "10", "20", "40"])
    ax.set(xlabel=r"NB dispersion $\phi$", ylabel="Null alarm probability",
           ylim=(0, max(.10, 1.08 * null.ci_high.max())))
    ax.axhline(.05, color=".4", linestyle="--", linewidth=.8)
    ax.legend(handles=_compact_method_handles(methods), fontsize=9.5)
    fig.subplots_adjust(left=.18, right=.97, bottom=.22, top=.96)
    name = "CompactFigS2_dispersion_null"
    _compact_save(fig, directory, name)
    (directory / f"{name}_caption.txt").write_text(
        "Reproducibility diagnostic, not a manuscript figure: null alarm audit at every dispersion "
        "with original phi10 cutoffs; dashed line .05. "
        "Bars are exact 95% binomial Monte Carlo intervals; all requested episodes remain in denominators. "
        "Useful power compares complete frozen procedures, not tests adjusted to identical Type I error.\n", encoding="utf-8")
    return [name]


def _compact_sensitivity_figure(evaluation: pd.DataFrame, directory: Path, *,
                                protocol: str = "compact-v2", has_estimation: bool = False,
                                dispersion: pd.DataFrame | None = None) -> tuple[str, str]:
    """Plot validated base or extended sensitivity rows on their actual grids."""
    paired = evaluation.method.eq(ORACLE_OCFR).any()
    main_methods = (COMPACT_PAIRED_METHODS if paired
                    else _compact_method_contract(protocol, "evaluation")[:4])
    handles = _compact_method_handles(main_methods)
    directory.mkdir(parents=True, exist_ok=True)
    _compact_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.8))
    wave = evaluation[evaluation.family == "wave"].copy()
    wave["delay_setting"] = wave["true_delay_mean_days"] if "true_delay_mean_days" in wave else wave.scenario.str.extract(r"delay(\d+)", expand=False).fillna("14").astype(float)
    delay_ticks = sorted(wave.loc[wave.misspecification.isin(("on_model", "delay_error")), "delay_setting"].unique())
    if paired and delay_ticks != list(range(11, 18)):
        raise EvidenceError("paired sensitivity requires the complete seven-cell delay grid")
    # The correct-model control and perturbations share frozen thresholds.
    for ax, metric in zip(axes[0], ("alarm_probability", "useful_power")):
        subset = wave[(wave.misspecification.isin(("on_model", "delay_error"))) &
            (wave.metric == metric) & (wave.change_day == 92) &
            (np.isclose(wave.risk_ratio, 1. if metric == "alarm_probability" else 1.5))]
        if paired:
            expected = {(method, day) for method in main_methods for day in delay_ticks}
            actual = list(subset[["method", "delay_setting"]].itertuples(index=False, name=None))
            if set(actual) != expected or len(actual) != len(expected):
                raise EvidenceError(f"paired sensitivity has an incomplete delay grid for {metric}")
            labels = {day: f"{day:g}" for day in delay_ticks}
            _compact_probability_panel(ax, subset, "delay_setting", methods=main_methods,
                                       xlabels=labels, connect_categories=True)
        else:
            _compact_probability_panel(ax, subset, "delay_setting", methods=main_methods)
            ax.set_xticks(delay_ticks)
        ax.set(xlabel="Underlying Gamma mean (days)", ylabel="False-alarm probability" if metric == "alarm_probability" else "Useful detection\nprobability")
        if metric == "alarm_probability":
            ax.axhline(.05, color=".4", linestyle="--", linewidth=.8)
            if paired:
                ax.set_ylim(0, max(.10, 1.08 * subset.ci_high.max()))
    # Separate the two outcomes so close operating points cannot obscure one
    # another. DGP is a labelled category; method retains the global encoding.
    inner = axes[1, 0].get_subplotspec().subgridspec(1, 2, wspace=1.0)
    axes[1, 0].remove()
    comparison_axes = [fig.add_subplot(inner[0, column]) for column in range(2)]
    axes[1, 0] = comparison_axes[0]
    for ax, metric, ratio, title, ylabel in zip(comparison_axes,
            ("alarm_probability", "useful_power"), (1., 1.5),
            ("False alarms", "Useful detection"),
            ("Null probability", "Power (RR=1.5)")):
        ci_high = []
        for index, method in enumerate(main_methods):
            offset = (index-(len(main_methods)-1)/2)*(.16 if paired else .22)
            for x, setting in enumerate(("on_model", "cohort_binomial")):
                rows = wave[wave.method.eq(method) & wave.misspecification.eq(setting) &
                    wave.change_day.eq(92) & wave.metric.eq(metric) &
                    np.isclose(wave.risk_ratio, ratio)]
                if len(rows) != 1:
                    raise EvidenceError("sensitivity comparison requires one row per DGP, method and outcome")
                row = rows.iloc[0]
                ci_high.append(row.ci_high)
                ax.errorbar(x+offset, row.estimate,
                    yerr=[[row.estimate-row.ci_low], [row.ci_high-row.estimate]],
                    fmt=COMPACT_MARKERS[method], color=COMPACT_COLORS[method],
                    markersize=3.5, capsize=1.5, linewidth=.9)
        ax.set(xlim=(-.35, 1.35), title=title, ylabel=ylabel)
        # An unlabelled central tick preserves categorical labels through the
        # shared export helper's minimum-three-ticks check.
        ax.set_xticks([0, .5, 1], ["NB", "", "Binomial"])
        ax.xaxis.get_major_ticks()[1].tick1line.set_visible(False)
        ax.tick_params(axis="both", labelsize=9, pad=2)
        ax.title.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        if metric == "alarm_probability":
            upper = max(.10, 1.08 * max(ci_high)) if paired else .10
            ax.set_ylim(0, upper)
            if upper == .10:
                ax.set_yticks([0, .05, .10], ["0", ".05", ".10"])
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.axhline(.05, color=".4", linestyle="--", linewidth=.8)
        else:
            ax.set_ylim(0, 1.02)
            ax.set_yticks([0, .5, 1], ["0", ".5", "1"])
    ax = axes[1,1]
    case_grid = ""
    if dispersion is None:
        subset = wave[wave.metric.eq("alarm_probability") &
            (wave.misspecification.str.startswith("exposure_scale_") |
             (wave.misspecification.eq("on_model") & np.isclose(wave.risk_ratio, 1)))].copy()
        subset["scale"] = subset.misspecification.str.replace("exposure_scale_", "", regex=False).replace("on_model", "1").astype(float)
        _compact_probability_panel(ax, subset, "scale", methods=main_methods)
        ax.set(xlabel="Observed case multiplier", ylabel="Alarm probability", xticks=sorted(subset["scale"].unique()))
        case_grid = ", ".join(f"{value:g}" for value in sorted(subset["scale"].unique()))
    else:
        _compact_dispersion_grid(dispersion, main_methods)
        subset = dispersion[dispersion.metric.eq("useful_power") & np.isclose(dispersion.risk_ratio, 1.5)]
        _compact_probability_panel(ax, subset, "phi", methods=main_methods)
        ax.set_xscale("log", base=2)
        ax.set_xticks([5, 10, 20, 40], ["5", "10", "20", "40"])
        ax.set(xlabel=r"NB dispersion $\phi$", ylabel="Useful detection\nprobability", ylim=(0, 1.02))
    for letter, ax in zip("ABCD", axes.flat):
        _compact_letter(ax, letter)
        if letter == "C":
            # Retain the outer-panel letter alignment after dividing its width.
            ax.texts[-1].set_x(-.48)
    fig.legend(handles=handles, ncol=len(main_methods) if protocol == "compact-v2" else 2,
               loc="upper center", bbox_to_anchor=(.53,.998),
               frameon=not paired, fontsize=9.5 if paired else 10.5)
    fig.subplots_adjust(left=.16, right=.97, bottom=.1, top=.91 if protocol == "compact-v2" else .87, hspace=.51, wspace=.57)
    sensitivity_name = "CompactFig6_sensitivity" if protocol == "compact-v2" or has_estimation else "CompactFig5_sensitivity"
    _compact_save(fig, directory, sensitivity_name)
    delay_grid = ", ".join(f"{value:g}" for value in delay_ticks)
    primary = "OCFR" if protocol == "compact-v2" else "Gaussian OCFR"
    caption = (
        "Wave-family sensitivity checks. Benchmark cutoffs remain fixed at reference values; "
        f"{primary} recomputes its boundaries using the same declared rule. "
        f"A-B: null alarm probability and RR=1.5 useful power for true underlying Gamma means {delay_grid} "
        "days, with assumed mean 14 days. C: false-alarm probability (left) and useful detection "
        "probability at RR=1.5 (right); NB denotes negative-binomial reports and Binomial denotes "
        "cohort-binomial outcomes. Methods are "
        "offset within each labelled DGP category; the two outcomes use different vertical scales. "
        "The dashed line marks .05 false-alarm probability. "
        f"D: alarm probability for observed-case multipliers {case_grid} at constant latent CFR. "
        "Bars are exact 95% binomial Monte Carlo intervals. Every requested episode enters "
        "each probability denominator, including premature and absent first alarms. The same "
        "on-model evaluation controls are shared across panels and differ from the larger null audit.")
    if dispersion is not None:
        caption = caption.replace(f"D: alarm probability for observed-case multipliers {case_grid} at constant latent CFR. ",
            "D: RR=1.5 useful power across constant NB dispersions 5, 10, 20, 40 on a log2 axis; "
            "true and assumed delay means are 14 days. Original phi10 benchmark cutoffs are retained. "
            "All four null-error settings are audited in the reproducibility diagnostics. "
            "Power compares complete frozen procedures, not tests adjusted to identical Type I error. ")
    if paired:
        caption = caption.replace(
            f"Benchmark cutoffs remain fixed at reference values; {primary} recomputes its boundaries using the same declared rule.",
            "Oracle OCFR and benchmark cutoffs stay fixed at reference-wave values. "
            "Practical OCFR recomputes its boundaries using the declared spending rule.")
        caption = caption.replace("Original phi10 benchmark cutoffs are retained. ",
            "The original oracle and benchmark cutoffs calibrated at phi=10 are retained. ").replace(
            "Power compares complete frozen procedures, not tests adjusted to identical Type I error.",
            "Power compares procedures at their achieved false-alarm rates.")
    return sensitivity_name, caption


def _compact_comparison_figure(
    audit: pd.DataFrame, curves: pd.DataFrame, directory: Path, *, protocol: str
) -> None:
    """Render the reference audit and detection curves from their summary tables."""
    paired = audit.method.eq(ORACLE_OCFR).any()
    main_methods = (COMPACT_PAIRED_METHODS if paired
                    else _compact_method_contract(protocol, "evaluation")[:4])
    handles = _compact_method_handles(main_methods)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.9))
    ax = axes[0, 0]
    for i, family in enumerate(("steady", "sparse", "wave")):
        for j, method in enumerate(main_methods):
            row = audit[(audit.family == family) & (audit.method == method)]
            if len(row) != 1:
                raise EvidenceError(f"expected one independent audit row for {family}/{method}")
            row = row.iloc[0]
            y = i+(j-(len(main_methods)-1)/2)*.15
            ax.errorbar(row.estimate, y,
                xerr=[[row.estimate-row.ci_low], [row.ci_high-row.estimate]],
                fmt=COMPACT_MARKERS[method], color=COMPACT_COLORS[method], capsize=2,
                markersize=3.5)
    ax.axvline(.05, color=".35", linestyle="--", linewidth=.9)
    ax.set_yticks(range(3), ["Steady", "Sparse", "Wave"])
    ax.set_xlabel("Episode false-alarm probability")
    if paired:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.invert_yaxis()
    for ax, family in zip(axes.flat[1:], ("steady", "sparse", "wave")):
        subset = curves[(curves.stage == "evaluation") & (curves.family == family) &
            (curves.misspecification == "on_model") & (curves.change_day == 92) &
            np.isclose(curves.risk_ratio, 1.5)]
        for method in main_methods:
            row = subset[subset.method == method].sort_values("delay_day")
            ax.step(row.delay_day, row.estimate, where="post", color=COMPACT_COLORS[method],
                    linestyle=COMPACT_LINESTYLES[method],
                    linewidth=COMPACT_PAIRED_LINEWIDTHS.get(method) if paired else None)
            if protocol == "compact-v2":
                day28 = row[row.delay_day.eq(28)]
                if len(day28) != 1:
                    raise EvidenceError("day-28 comparison requires one validated curve value")
                ax.plot(28, float(day28.iloc[0].estimate), linestyle="none",
                        marker=COMPACT_MARKERS[method], color=COMPACT_COLORS[method],
                        markersize=6 if paired and method == "OCFR" else 4, zorder=4,
                        markerfacecolor="none" if paired and method == "OCFR" else COMPACT_COLORS[method])
        if protocol == "compact-v2":
            ax.axvline(28, color=".65", linestyle=":", linewidth=.8, zorder=0)
            ax.set_xticks([0, 28, 56, 84])
        ax.set(xlim=(0,90), ylim=(0,1.02), xlabel="Days after change", ylabel="Cumulative detection\nprobability")
    for letter, ax in zip("ABCD", axes.flat):
        _compact_letter(ax, letter)
    legend_columns = len(handles) if protocol == "compact-v2" else 2
    fig.legend(handles=handles, ncol=legend_columns, loc="upper center",
               bbox_to_anchor=(.53, .998), frameon=not paired,
               fontsize=9.5 if paired else 10.5)
    top = .90 if protocol == "compact-v2" else .87
    fig.subplots_adjust(left=.16, right=.97, bottom=.09, top=top, hspace=.51, wspace=.57)
    _compact_save(fig, directory, "CompactFig2_comparison")


def _compact_figures(tables: dict[str, pd.DataFrame], directory: Path, *, protocol: str = "compact-v1") -> dict[str, str]:
    """Four distinct main figures; values always originate in validated tables."""
    main_methods = _compact_method_contract(protocol, "evaluation")[:4]
    primary = "OCFR" if protocol == "compact-v2" else "OCFR Gaussian"
    directory.mkdir(parents=True, exist_ok=True)
    _compact_style()
    summary, curves, paired = (tables[name] for name in
        ("compact_summary", "compact_cumulative_detection", "compact_paired_contrasts"))
    evaluation = summary[summary.stage.eq("evaluation")]
    on_model = evaluation[evaluation.misspecification.eq("on_model")]
    power = on_model[on_model.metric.eq("useful_power")]
    audit = pd.concat([summary[summary.stage.eq("audit") & summary.metric.eq("alarm_probability")],
        on_model[on_model.metric.eq("alarm_probability") & on_model.method.eq("OCFR Gaussian") &
            np.isclose(on_model.risk_ratio, 1)]], ignore_index=True)
    if protocol == "compact-v2":
        audit = summary[summary.stage.eq("audit") & summary.metric.eq("alarm_probability")].copy()
    handles = [plt.Line2D([], [], color=COMPACT_COLORS[m], marker=COMPACT_MARKERS[m],
                         linestyle=COMPACT_LINESTYLES[m], markersize=4, label=m)
                         for m in main_methods]
    captions = {}
    _compact_comparison_figure(audit, curves, directory, protocol=protocol)
    captions["CompactFig2_comparison"] = (
        "A: independent null audit in the steady, sparse, and wave incidence families; dashed line: "
        "nominal 0.05. B–D: cumulative useful first-alarm probability "
        "F(d)=P(tau <= T <= min(tau+d,H)) for RR=1.5 in steady, sparse, and wave families, "
        "respectively, with change day 92. These curves show speed and endpoint useful power. Every requested "
        "replicate is included; premature and absent first alarms contribute zero useful detections. "
        "Bars in A are exact two-sided 95% binomial intervals; pointwise intervals for curves "
        "are tabulated. Gaussian OCFR uses the formal evaluation "
        "nulls (n=1000 in full mode); empirical methods use the independent precision audit "
        "(n=5000 per family). The count table records the actual n for every mark.")

    if protocol == "compact-v1":
        fig, axes = plt.subplots(3, 2, figsize=(6.4, 7.8))
        for ax, family, letter in zip((axes[0,0], axes[0,1], axes[1,0]),
                                      ("steady", "sparse", "wave"), "ABC"):
            subset = power[(power.family == family) & (power.change_day == 92)]
            _compact_probability_panel(ax, subset, "risk_ratio", methods=main_methods)
            ax.set(xlabel="CFR risk ratio", ylabel="Detection probability")
            ax.set_xticks([2/3, 1.25, 1.5, 2], ["2/3", "1.25", "1.5", "2"])
            _compact_letter(ax, letter)
        axes[1,1].axis("off")
        method_legend = axes[1,1].legend(handles=handles, loc="upper left", bbox_to_anchor=(0,1.05))
        axes[1,1].add_artist(method_legend)
        timing_colors = ("#0072B2", "#D55E00", "#009E73")
        timing_styles = ("-", "--", ":")
        for ax, family, letter in zip(axes[2], ("steady", "wave"), "DE"):
            for day, color, style in zip((47, 92, 137), timing_colors, timing_styles):
                subset = curves[(curves.stage == "evaluation") & (curves.family == family) &
                    (curves.misspecification == "on_model") & (curves.change_day == day) &
                    np.isclose(curves.risk_ratio, 1.5) & (curves.method == primary) &
                    (curves.delay_day <= 45)]
                ax.step(subset.delay_day, subset.estimate, where="post", color=color,
                        linestyle=style, label=f"Day {day}")
            ax.set(xlim=(0, 45), ylim=(0, 1.02), xlabel="Days after change", ylabel="Cumulative detection\nprobability")
            _compact_letter(ax, letter)
        timing_handles, timing_labels = axes[2,1].get_legend_handles_labels()
        axes[1,1].legend(timing_handles, timing_labels, loc="lower left", bbox_to_anchor=(0,-.13),
                         title="OCFR timing" if protocol == "compact-v2" else "Gaussian OCFR timing", title_fontsize=10.5)
        fig.subplots_adjust(left=.15, right=.97, bottom=.07, top=.96, hspace=.59, wspace=.57)
        _compact_save(fig, directory, "CompactFig3_detection")
        captions["CompactFig3_detection"] = (
            "A–C: useful first-alarm probability P(tau <= T <= H) across CFR risk ratios "
            "with change day 92 in steady, sparse, and wave families; bars are exact 95% binomial "
            "intervals. D–E: Gaussian OCFR "
            "under early (day 47), middle (92), and late (137) changes in steady and wave families, "
            "on a common 45-day scale. All requested replicates form every denominator; premature "
            "first alarms and absent alarms contribute zero. Curves are empirical probabilities, "
            "not alarm-conditional delay distributions; pointwise exact intervals are in the data table.")

    else:
        captions["CompactFig3_detection"] = _compact_v2_detection_figure(power, curves, directory)

    if protocol == "compact-v1":
        fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.8))
        ablation = paired[(paired.method_a == "OCFR empirical") &
            (paired.method_b == "Report-day score") & (paired.metric == "useful_power") &
            (paired.stage == "evaluation") & (paired.misspecification == "on_model")]
        for family, color, marker in zip(("steady", "sparse", "wave"),
            ("#0072B2", "#D55E00", "#009E73"), ("o", "s", "^")):
            subset = ablation[(ablation.family == family) & (ablation.change_day == 92)].sort_values("risk_ratio")
            branches = [branch for branch in (subset[subset.risk_ratio < 1],
                subset[subset.risk_ratio == 1], subset[subset.risk_ratio > 1]) if not branch.empty]
            for branch_index, branch in enumerate(branches):
                axes[0,0].errorbar(branch.risk_ratio, branch.difference,
                    yerr=1.96*branch.paired_mcse, fmt=marker+("-" if len(branch) > 1 else ""),
                    color=color, capsize=2, label=family.capitalize() if branch_index == 0 else None)
        axes[0,0].set(xlabel="CFR risk ratio", ylabel="Paired useful-power difference")
        axes[0,0].set_xticks([2/3, 1.25, 1.5, 2], ["2/3", "1.25", "1.5", "2"])
        axes[0,0].legend(loc="best")
        for ax, family in zip((axes[0,1], axes[1,0]), ("steady", "wave")):
            subset = ablation[(ablation.family == family) & np.isclose(ablation.risk_ratio, 1.5)].sort_values("change_day")
            ax.errorbar(subset.change_day, subset.difference, yerr=1.96*subset.paired_mcse,
                        fmt="o-", color=COMPACT_COLORS["OCFR empirical"], capsize=2)
            ax.set(xlabel="Change day", ylabel="Paired useful-power difference", xticks=[47, 92, 137])
        for ax in (axes[0,0], axes[0,1], axes[1,0]):
            ax.axhline(0, color=".4", linestyle="--", linewidth=.8)
        ablation_audit = audit[audit.method.isin(("OCFR empirical", "Report-day score"))].copy()
        ablation_audit["family_index"] = ablation_audit.family.map({"steady":0, "sparse":1, "wave":2})
        _compact_probability_panel(axes[1,1], ablation_audit, "family_index",
            methods=("OCFR empirical", "Report-day score"), xlabels={0:"Steady", 1:"Sparse", 2:"Wave"})
        axes[1,1].axhline(.05, color=".4", linestyle="--", linewidth=.8)
        axes[1,1].set(ylabel="Episode false-alarm probability", ylim=(0, max(.10, 1.05*ablation_audit.ci_high.max())))
        for letter, ax in zip("ABCD", axes.flat):
            _compact_letter(ax, letter)
        axes[1,1].legend(handles=[plt.Line2D([], [], color=COMPACT_COLORS[m],
            marker=COMPACT_MARKERS[m], linestyle=COMPACT_LINESTYLES[m], label=m)
            for m in ("OCFR empirical", "Report-day score")], loc="upper center")
        fig.subplots_adjust(left=.16, right=.97, bottom=.09, top=.96, hspace=.48, wspace=.57)
        _compact_save(fig, directory, "CompactFig4_ablation")
        captions["CompactFig4_ablation"] = (
            "Delay alignment ablation with separately calibrated empirical episode thresholds. "
            "A: useful-power difference (empirical OCFR minus report-day score) across RR at change "
            "day 92 in all three families. B–C: the same paired difference across change days in "
            "steady and wave families at RR=1.5. Error bars in A–C are ±1.96 paired Monte Carlo "
            "standard errors, not binomial confidence intervals. D: independent-null audit for the "
            "two empirical methods, with exact 95% binomial intervals and nominal 0.05 reference. "
            "All requested replicates are retained. Counts and discordant pairs are tabulated.")

    sensitivity_name, sensitivity_caption = _compact_sensitivity_figure(
        evaluation, directory, protocol=protocol,
        has_estimation="compact_estimation_summary" in tables)
    captions[sensitivity_name] = sensitivity_caption
    if "compact_estimation_summary" in tables and protocol == "compact-v1":
        captions["CompactFig5_estimation"] = _compact_estimation_figure(tables, directory)
    if "compact_stability_validation" in tables:
        captions["CompactFigS1_estimation_stability"] = _compact_stability_validation_figure(tables, directory)
    if "compact_poisson_summary" in tables:
        captions["CompactFigS2_estimation_reference"] = _compact_estimation_comparison_figure(
            tables, directory, name="CompactFigS2_estimation_reference")
        captions["CompactFigS3_poisson_comparison"] = _compact_poisson_comparison_figure(tables, directory)
    evaluation_n = ", ".join(str(int(value)) for value in sorted(evaluation.n.unique()))
    audit_n = ", ".join(str(int(value)) for value in sorted(audit[audit.method.ne("OCFR Gaussian")].n.unique()))
    if protocol == "compact-v2" and "compact_v2_interval_summary" in tables:
        captions["CompactFig5_estimation"] = _compact_v2_estimation_figure(tables, directory)
        captions["CompactFigS5_all_conditions"] = _compact_v2_estimation_figure(tables, directory, all_cells=True)
    if protocol == "compact-v2":
        captions["CompactFig4_localization"] = _compact_localization_figure(tables["compact_localization"], directory)
        captions["CompactFig2_comparison"] = (
            "A: independent null audit for OCFR, NB-GLR and Page-CUSUM in each incidence family. "
            "All methods use the audit denominators in the count table; evaluation nulls are not appended. "
            "Bars are exact 95% binomial intervals; dashed line: nominal 0.05. B-D: cumulative useful "
            "first-alarm probabilities for RR=1.5 and change day 92, retaining every requested episode. "
            "Vertical dotted guides and method markers identify detection within 28 days.")
        captions["CompactFig3_detection"] = captions["CompactFig3_detection"].replace("Gaussian OCFR", "OCFR")
        captions[sensitivity_name] = captions[sensitivity_name].replace("Gaussian OCFR", "OCFR").replace(
            "Empirical cutoffs remain fixed at reference values; ", "Benchmark cutoffs remain fixed at reference values; ")
    for name in captions:
        captions[name] += f" Evaluation uses n={evaluation_n} independent episodes per cell."
        if name in {"CompactFig2_comparison", "CompactFig4_ablation"}:
            captions[name] += f" Independent precision-audit points use n={audit_n} per family."
        (directory / f"{name}_caption.txt").write_text(captions[name]+"\n", encoding="utf-8")
    return captions


def analyze_compact_simulation(output: Path, figure_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Analyze completed compact stages without changing the legacy evidence contract.

    ``output`` is a compact root (or its evaluation directory). Tables are written
    under ``output/analysis``. A complete calibration and independent audit are
    required; neither calibration failures nor missing requested episodes vanish.
    Synthetic manifests support metric tests but never export manuscript figures.
    """
    output = Path(output).resolve()
    if output.name == "evaluation" and (output / MANIFEST_FILE).exists():
        output = output.parent
    destination = output / "analysis"
    report_path = destination / "compact_analysis_manifest.json"
    if report_path.exists():
        report_path.unlink()  # Validation failure must not leave a stale completion marker.
    stages, manifests = [], {}
    for stage in ("evaluation", "audit"):
        frame, manifest = _compact_load_stage(output / stage, stage)
        frame["stage"] = stage
        stages.append(frame)
        manifests[stage] = manifest
    if manifests["evaluation"]["scientific_digest"] != manifests["audit"]["scientific_digest"]:
        raise EvidenceError("evaluation and audit scientific digests disagree")
    protocol = manifests["evaluation"]["protocol"]
    if manifests["audit"]["protocol"] != protocol:
        raise EvidenceError("evaluation and audit protocols disagree")
    calibration_dir = output / "calibration"
    calibration = json.loads((calibration_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    if (calibration.get("status") != "complete" or calibration.get("stage") != "calibration"
        or calibration.get("protocol") != protocol
        or calibration.get("scientific_digest") != manifests["evaluation"]["scientific_digest"]):
        raise EvidenceError("a complete independent compact calibration is required")
    threshold_file = calibration_dir / "thresholds.csv"
    if _sha256(threshold_file) != _manifest_file_hash(calibration, "thresholds.csv"):
        raise EvidenceError("calibration threshold hash mismatch")
    thresholds = pd.read_csv(threshold_file)
    if "n_calibration" not in thresholds or (thresholds.n_calibration < 1).any():
        raise EvidenceError("calibration episode counts are missing")
    _validate_scientific_digest(calibration, thresholds)
    calibration_raw = calibration_dir / RAW_FILE
    if _sha256(calibration_raw) != _manifest_file_hash(calibration, RAW_FILE):
        raise EvidenceError("calibration raw-file hash mismatch")
    if not manifests["evaluation"].get("synthetic_fixture", False):
        expected_pairs = {(family, method) for family in ("steady", "sparse", "wave")
                          for method in _compact_method_contract(protocol, "calibration")}
        if (set(zip(thresholds.family, thresholds.method)) != expected_pairs or
            len(thresholds) != len(expected_pairs) or
            (thresholds.n_calibration != _compact_declared_n(calibration, "calibration")).any()):
            raise EvidenceError("calibration thresholds violate the declared episode/method contract")
    if protocol == "compact-v2" and not manifests["evaluation"].get("synthetic_fixture", False):
        if calibration.get("scientific_config") != manifests["evaluation"].get("scientific_config"):
            raise EvidenceError("calibration scientific configuration differs from evaluation")
        expected_count = _compact_declared_n(calibration, "calibration")
        alpha = float(calibration["scientific_config"]["methods"]["alpha"])
        if ("order_rank" not in thresholds or
                not thresholds.order_rank.eq(int(np.ceil((expected_count+1)*(1-alpha)))).all()):
            raise EvidenceError("v2 calibration order rank disagrees with the declared rule")
        raw_calibration = _compact_read_summary_csv(calibration_raw,
            {"scenario", "method", "replicate", "status", "split"})
        definitions = calibration["scientific_config"]["scenarios"]
        names = {d["name"] for d in definitions if d["split"] == "calibration"}
        if set(raw_calibration.scenario) != names or not raw_calibration.split.eq("calibration").all():
            raise EvidenceError("v2 calibration scenario grid mismatch")
        if not raw_calibration.status.eq("success").all() or raw_calibration.duplicated(
                ["scenario", "method", "replicate"]).any():
            raise EvidenceError("v2 calibration has failed or duplicate episodes")
        for scenario in names:
            group = raw_calibration[raw_calibration.scenario.eq(scenario)]
            if set(group.method) != set(_compact_method_contract(protocol, "calibration")):
                raise EvidenceError("v2 calibration method grid mismatch")
            for method, episodes in group.groupby("method"):
                if len(episodes) != expected_count or set(episodes.replicate) != set(range(expected_count)):
                    raise EvidenceError(f"{scenario}/{method}: incomplete calibration grid")
    for stage, manifest in manifests.items():
        declared = (manifest.get("threshold_artifact") or {}).get("sha256", manifest.get(
            "calibration_threshold_sha256", manifest.get("thresholds_sha256")))
        if declared != _sha256(threshold_file):
            raise EvidenceError(f"{stage} used a different calibration threshold file")
    tables = compact_summaries(pd.concat(stages, ignore_index=True))
    tables["compact_calibration"] = thresholds
    if protocol == "compact-v2":
        tables["compact_localization"] = compact_localization_summaries(stages[0])
    estimation_tables, estimation_metadata = _compact_load_estimation(output, stages[0], manifests["evaluation"])
    tables.update(estimation_tables)
    if protocol == "compact-v1":
        reference_tables, reference_metadata = _compact_load_estimation_reference(output)
        tables.update(reference_tables)
        stabilized_tables, stabilized_metadata = _compact_load_stabilized_estimation(output)
        tables.update(stabilized_tables)
        validation_tables, validation_metadata = _compact_load_stability_validation(output)
        tables.update(validation_tables)
        poisson_tables, poisson_metadata = _compact_load_poisson_estimation(output)
        tables.update(poisson_tables)
        ls_tables, ls_metadata = _compact_load_ls_estimation(output, include_main_pair=True)
        tables.update(ls_tables)
        standardized, secondary_metadata = _compact_standardized_power(output, stages[0],
            manifests["evaluation"], manifests["audit"])
        if standardized is not None:
            tables["compact_standardized_power"] = standardized
    else:
        v2_tables, v2_postalarm_metadata = _compact_load_v2_postalarm(output)
        tables.update(v2_tables)
        stabilized_metadata = validation_metadata = ls_metadata = v2_postalarm_metadata
        reference_metadata = poisson_metadata = {"status": "not_primary", "reason": "v2 main point comparator is full-delay LS"}
        secondary_metadata = {"status": "not_applicable", "reason": "constant raw-score normalization does not represent the spending detector"}
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        _atomic_csv(table, destination / f"{name}.csv")
    synthetic = bool(manifests["evaluation"].get("synthetic_fixture", False))
    if synthetic and figure_dir is not None:
        raise EvidenceError("synthetic fixtures cannot export publication figures")
    figures = Path(figure_dir) if figure_dir is not None else destination / "figures"
    captions = {} if synthetic else _compact_figures(tables, figures, protocol=protocol)
    report = {"protocol": protocol, "status": "complete", "synthetic_fixture": synthetic,
        "scientific_digest": manifests["evaluation"]["scientific_digest"],
        "analysis_source_sha256": _sha256(Path(__file__)), "confidence_level": .95,
        "interval": "two-sided Clopper-Pearson", "paired_uncertainty": "sample SD of paired indicators / sqrt(n)",
        "denominator": "all requested replicates; premature/absent alarms contribute zero useful detections",
        "time_convention": "runner true_change/alarm_day are zero-based; analysis converts both to manuscript day",
        "failed_method_rows": int(sum(frame.failed.sum() for frame in stages)),
        "loading_policy": {"primary_metadata_chunk_rows": 4096,
                           "secondary_trace_chunk_rows": 1024,
                           "primary_trace_json_materialized": False},
        "publication_export": None if synthetic else _compact_export_metadata(figures, list(captions),
            minimum_font_points=9.5 if protocol == "compact-v2" else 10.5),
        "reporting_gaps": (["Figure 5 requires all fresh LS/OCFR selected-fit banks plus frozen independent interval validation."]
            if protocol == "compact-v2" and "compact_v2_interval_summary" not in tables else []),
        "secondary_analyses": {"null_quantile_standardized_useful_power": secondary_metadata,
                               "first_alarm_estimation": estimation_metadata,
                               "first_alarm_estimation_reference": reference_metadata,
                               "stabilized_first_alarm_estimation": stabilized_metadata,
                               "independent_estimation_validation": validation_metadata,
                               "matched_poisson_estimation": poisson_metadata,
                               "full_delay_ls_estimation": ls_metadata},
        "source_manifest_sha256": {stage: _sha256(output/stage/MANIFEST_FILE)
            for stage in ("calibration", "audit", "evaluation")},
        "files": {f"{name}.csv": _sha256(destination/f"{name}.csv") for name in tables},
        "figure_files": {path.name: _sha256(path) for name in captions
            for path in (figures/f"{name}.pdf", figures/f"{name}.png",
                         figures/f"{name.split('_', 1)[0].removeprefix('Compact')}.tif")}}
    _atomic_json(report, report_path)
    return tables


def analyze_compact_poisson_estimation(output: Path, figure_dir: Path | None = None,
                                       *, include_s3: bool = True) -> dict[str, pd.DataFrame]:
    """Refresh only the new comparison tables and Figures 5/S2/(optionally) S3.

    Existing results and their analysis manifest stay intact. New figures use
    ``analysis/poisson_figures`` unless an explicit figure directory is given.
    This report binds the limited refresh to its sources separately.
    Demo packages can produce tables, but never publication figures.
    """
    output = Path(output).resolve()
    if output.name == "evaluation":
        output = output.parent
    poisson, metadata = _compact_load_poisson_estimation(output)
    if not poisson:
        raise EvidenceError("both completed Poisson estimation stages are required")
    tables = dict(poisson)
    full = all(bank["mode"] == "full" for bank in metadata["banks"].values())
    if full:
        stabilized, _ = _compact_load_stabilized_estimation(output)
        if not stabilized:
            raise EvidenceError("frozen stabilized NB summaries are required for unchanged coverage")
        tables.update(stabilized)
    elif figure_dir is not None:
        raise EvidenceError("demo Poisson stages cannot export publication figures")
    destination = output/"analysis"
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in poisson.items():
        _atomic_csv(table, destination/f"{name}.csv")
    captions = {}
    figures = Path(figure_dir) if figure_dir is not None else destination/"poisson_figures"
    if full:
        figures.mkdir(parents=True, exist_ok=True)
        _compact_style()
        captions["CompactFigS2_estimation_reference"] = _compact_estimation_comparison_figure(
            tables, figures, name="CompactFigS2_estimation_reference")
        captions["CompactFig5_estimation"] = _compact_estimation_comparison_figure(tables, figures, poisson=True)
        if include_s3:
            captions["CompactFigS3_poisson_comparison"] = _compact_poisson_comparison_figure(tables, figures)
        for name, caption in captions.items():
            (figures/f"{name}_caption.txt").write_text(caption+"\n", encoding="utf-8")
    report = {"status": "complete", "protocol": "compact-v1", "stage": "poisson-estimation-analysis",
        "analysis_source_sha256": _sha256(Path(__file__)), "source": metadata,
        "figure_directory": str(figures.resolve()) if captions else None,
        "files": {f"{name}.csv": _sha256(destination/f"{name}.csv") for name in poisson},
        "publication_export": _compact_export_metadata(figures, list(captions)) if captions else None,
        "figure_files": {path.name: _sha256(path) for name in captions
            for path in (figures/f"{name}.pdf", figures/f"{name}.png",
                         figures/f"{name.split('_', 1)[0].removeprefix('Compact')}.tif")}}
    _atomic_json(report, destination/"compact_poisson_analysis_manifest.json")
    return poisson


LS_ESTIMATION_METHODS = (("Full-delay LS", "ls_"), ("Unpenalized NB", ""),
                         ("OCFR", "regularized_"))


def compact_ls_summaries(banks: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Point-only full-delay LS/NB comparisons at the same frozen first alarms.

    Common cohorts are target-specific, with no filtering by old moment-fit
    availability, interval availability, global fit status, or effect size.
    """
    return _compact_ls_summaries(banks, LS_ESTIMATION_METHODS)


def compact_ls_main_pair_summary(banks: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Main LS/OCFR point cohort, independent of unpenalized NB availability."""
    methods = (("Full-delay LS", "ls_"), ("OCFR", "regularized_"))
    return _compact_ls_summaries(banks, methods)["compact_ls_summary"]


def _compact_ls_summaries(banks: Mapping[str, pd.DataFrame],
                          methods: Sequence[tuple[str, str]]) -> dict[str, pd.DataFrame]:
    summaries, contrasts, aggregates = [], [], []
    for bank, frame in banks.items():
        for scenario, group in frame.groupby("scenario", sort=True):
            first = group.iloc[0]
            alarm = _as_binary(group.alarm, "alarm").astype(bool)
            meta = dict(bank=bank, scenario=scenario, family=first.family,
                risk_ratio=float(first.risk_ratio), change_day=int(first.true_change)+1,
                N_requested=len(group), n_alarm=int(alarm.sum()), n_no_alarm=int((~alarm).sum()),
                n_premature=int((alarm & (group.alarm_day < group.true_change)).sum()))
            for target in ESTIMATION_TARGETS:
                truth = pd.to_numeric(group[target], errors="raise")
                if not np.isfinite(truth).all() or (truth <= 0).any() or truth.nunique() != 1:
                    raise EvidenceError(f"{bank}/{scenario}/{target}: inconsistent generating truth")
                masks, errors = {}, {}
                common = alarm.copy()
                for method, prefix in methods:
                    valid = _as_binary(group[f"{prefix}{target}_point_valid"], target).astype(bool)
                    estimate = pd.to_numeric(group[f"{prefix}{target}_estimate"], errors="raise")
                    if (valid & ~alarm).any() or estimate[~alarm].notna().any():
                        raise EvidenceError("LS comparison output outside first alarms")
                    if (valid & (~np.isfinite(estimate) | (estimate <= 0) |
                            ((estimate >= 1) if target != "risk_ratio" else False))).any():
                        raise EvidenceError("LS comparison invalid declared available estimate")
                    masks[method] = alarm & valid
                    errors[method] = estimate/truth-1.
                    common &= masks[method]
                for method, prefix in methods:
                    valid = masks[method]
                    record = {**meta, "target": target, "truth": float(truth.iloc[0]),
                        "method": method, "objective": "daily RSS" if prefix == "ls_" else "NB likelihood",
                        "penalty_precision": 1. if prefix == "regularized_" else 0.,
                        "n_point": int(valid.sum()), "n_unavailable": int((alarm & ~valid).sum()),
                        "point_availability": float(valid.sum()/alarm.sum()) if alarm.any() else np.nan,
                        "relative_bias": float(errors[method][valid].mean()),
                        "n_common_paired": int(common.sum()),
                        "common_paired_relative_bias": float(errors[method][common].mean()),
                        **_relative_error_scores(errors[method][valid]**2)}
                    record.update({f"common_paired_{key}": value for key, value in
                        _relative_error_scores(errors[method][common]**2).items()})
                    summaries.append(record)
                for index, (method_a, _) in enumerate(methods):
                    for method_b, _ in methods[index+1:]:
                        for cohort, mask in (("pairwise_available", masks[method_a] & masks[method_b]),
                                             ("three_way_common" if len(methods) == 3 else "two_way_common", common)):
                            difference = errors[method_a][mask]**2-errors[method_b][mask]**2
                            contrasts.append({**meta, "target": target, "method_a": method_a,
                                "method_b": method_b, "cohort": cohort, "n_paired": int(mask.sum()),
                                "mean_squared_relative_error_difference": float(difference.mean()),
                                "paired_mcse": float(difference.std(ddof=1)/np.sqrt(len(difference)))
                                    if len(difference) > 1 else np.nan})
    summary, contrast = pd.DataFrame(summaries), pd.DataFrame(contrasts)
    for (bank, method, target), group in summary.groupby(["bank", "method", "target"], sort=True):
        record = dict(bank=bank, method=method, target=target, n_cells=len(group),
            n_alarm=int(group.n_alarm.sum()), n_point=int(group.n_point.sum()),
            n_unavailable=int(group.n_unavailable.sum()), n_common_paired=int(group.n_common_paired.sum()))
        record["point_availability"] = record["n_point"]/record["n_alarm"] if record["n_alarm"] else np.nan
        for label, column in (("all_available", "relative_mse"),
                              ("common_paired", "common_paired_relative_mse")):
            record[f"{label}_relative_mse"] = float(group[column].mean()) if group[column].notna().all() else np.nan
            record[f"{label}_relative_rmse"] = float(np.sqrt(record[f"{label}_relative_mse"]))
        for label, column in (("all_available", "relative_bias"),
                              ("common_paired", "common_paired_relative_bias")):
            record[f"{label}_relative_bias"] = float(group[column].mean()) if group[column].notna().all() else np.nan
        comparisons = contrast[contrast.bank.eq(bank) & contrast.target.eq(target) &
            contrast.cohort.eq("pairwise_available")]
        for opponent, _ in methods:
            if opponent == method:
                continue
            rows = comparisons[((comparisons.method_a == method) & (comparisons.method_b == opponent)) |
                               ((comparisons.method_b == method) & (comparisons.method_a == opponent))]
            delta = rows.mean_squared_relative_error_difference*np.where(rows.method_a.eq(method), 1., -1.)
            key = opponent.lower().replace(" ", "_").replace("-", "_")
            record[f"mse_wins_vs_{key}"] = int((delta < 0).sum())
            record[f"mse_ties_vs_{key}"] = int((delta == 0).sum())
            record[f"mse_cells_compared_vs_{key}"] = int(delta.notna().sum())
        aggregates.append(record)
    return {"compact_ls_summary": summary, "compact_ls_contrasts": contrast,
            "compact_ls_aggregate": pd.DataFrame(aggregates)}


def _compact_load_ls_estimation(output: Path, *, include_main_pair: bool = False) -> tuple[dict, dict]:
    """Verify producer provenance and preserve all frozen source columns."""
    banks, metadata = {}, {}
    stages = (("Original bank", "estimation-ls-original", "estimation-stabilized"),
              ("Independent bank", "estimation-ls-validation", "estimation-validation"))
    if not any((output/stage).exists() for _, stage, _ in stages):
        return {}, {"status": "absent"}
    for bank, stage, source_stage in stages:
        directory, source = output/stage, output/source_stage
        path, source_path = directory/MANIFEST_FILE, source/MANIFEST_FILE
        if not path.is_file() or not source_path.is_file():
            raise EvidenceError("LS comparison requires both completed banks and frozen sources")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
        if (manifest.get("stage") != stage or manifest.get("status") != "complete" or
                manifest.get("mode") not in {"demo", "full"}):
            raise EvidenceError(f"{stage}: incomplete LS comparison manifest")
        if (source_manifest.get("status") != "complete" or source_manifest.get("stage") != source_stage or
                source_manifest.get("mode") != "full"):
            raise EvidenceError("LS comparison source stage is incomplete")
        raw, source_raw = directory/RAW_FILE, source/RAW_FILE
        for filename, declared in ((raw, _manifest_file_hash(manifest, RAW_FILE)),
                                   (source_raw, _manifest_file_hash(source_manifest, RAW_FILE))):
            if _sha256(filename) != declared:
                raise EvidenceError("LS comparison artifact hash mismatch")
        for key, value in (("source_manifest_sha256", _sha256(source_path)),
                           ("source_csv_sha256", _sha256(source_raw))):
            if manifest.get(key) != value:
                raise EvidenceError(f"LS frozen source mismatch: {key}")
        expected_bank = stage.removeprefix("estimation-ls-")
        if (manifest.get("source_bank") != expected_bank or
                manifest.get("source_scientific_digest") != source_manifest.get("scientific_digest") or
                manifest.get("scientific_digest") != source_manifest.get("scientific_digest") or
                manifest.get("scientific_config") != source_manifest.get("scientific_config")):
            raise EvidenceError("LS scientific source/configuration mismatch")
        settings = manifest.get("ls_estimation_config", {})
        # The runner records the exact fixed design; point-only/no-penalty are
        # scientific invariants rather than inferred from fitted values.
        from config import PostAlarmLSConfig, COMPACT_SCENARIOS
        from dataclasses import asdict
        if settings != asdict(PostAlarmLSConfig()):
            raise EvidenceError("LS objective, bounds, penalty, or interval scope changed")
        records = manifest.get("postprocessor_code_files", {})
        if (not records or any(not isinstance(value, str) or len(value) != 64 for value in records.values()) or
                manifest.get("postprocessor_code_hash") != hashlib.sha256(
                    _canonical_json(records).encode("ascii")).hexdigest()):
            raise EvidenceError("LS postprocessor code hash mismatch")
        provenance_keys = ("source_manifest_sha256", "source_csv_sha256", "source_bank",
            "source_scientific_digest", "scientific_config", "postprocessor_code_hash",
            "postprocessor_code_files", "postprocessor_native_files", "ls_estimation_config")
        provenance = {key: manifest.get(key) for key in provenance_keys}
        if manifest.get("cache_key") != hashlib.sha256(_canonical_json(provenance).encode("ascii")).hexdigest()[:20]:
            raise EvidenceError("LS cache provenance mismatch")
        frame, original = pd.read_csv(raw), pd.read_csv(source_raw)
        if any(str(column).startswith("ls_") for column in original):
            raise EvidenceError("LS output namespace collides with frozen source")
        if frame.duplicated(["scenario", "replicate"]).any() or not set(original).issubset(frame):
            raise EvidenceError("LS episode/column contract changed")
        selected = original if manifest["mode"] == "full" else original[original.replicate < 10]
        try:
            pd.testing.assert_frame_equal(frame[original.columns].reset_index(drop=True),
                selected.reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-15)
        except (KeyError, AssertionError) as exc:
            raise EvidenceError("LS stage changed frozen observations, alarms, selected dates, or truth") from exc
        count = 1000 if manifest["mode"] == "full" else 10
        names = set(source_manifest.get("scenarios", []))
        expected_names = {scenario.name for scenario in COMPACT_SCENARIOS
                          if scenario.split == "alternative" and scenario.misspecification == "on_model"}
        if names != expected_names or set(frame.scenario) != names or set(manifest.get("scenarios", [])) != names:
            raise EvidenceError("LS comparison requires the complete 16-cell grid")
        required = {f"{prefix}{target}_{suffix}" for _, prefix in LS_ESTIMATION_METHODS
                    for target in ESTIMATION_TARGETS for suffix in ("estimate", "point_valid")}
        required |= {"ls_status", "ls_seconds", "ls_rss", "ls_active_set", "ls_kkt_residual",
                     "ls_kkt_tolerance", "case_hash", "death_hash", "alarm_day", "tau_hat_at_alarm"}
        if required-set(frame):
            raise EvidenceError(f"LS comparison missing columns: {sorted(required-set(frame))}")
        if any(column.startswith("ls_") and ("ci_" in column or "coverage" in column) for column in frame):
            raise EvidenceError("LS comparison must remain point-only")
        seconds = pd.to_numeric(frame.ls_seconds, errors="raise")
        rss = pd.to_numeric(frame.ls_rss, errors="raise")
        if (frame.ls_status.isna().any() or not np.isfinite(seconds).all() or (seconds < 0).any() or
                (rss.dropna() < 0).any() or not np.isfinite(rss.dropna()).all()):
            raise EvidenceError("LS status, timing, or RSS diagnostics are invalid")
        for scenario, group in frame.groupby("scenario", sort=True):
            if (len(group) != count or set(group.replicate) != set(range(count)) or
                    manifest.get("replications_by_scenario", {}).get(scenario) != count):
                raise EvidenceError(f"{scenario}: incomplete LS episode grid")
        if not np.allclose(frame.pi_right/frame.pi_left, frame.risk_ratio):
            raise EvidenceError("LS generating risk-ratio truth is inconsistent")
        banks[bank] = frame
        metadata[stage] = dict(source_manifest_sha256=_sha256(path), source_csv_sha256=_sha256(raw),
            frozen_source_manifest_sha256=_sha256(source_path), mode=manifest["mode"],
            postprocessor_code_hash=manifest["postprocessor_code_hash"])
    tables = compact_ls_summaries(banks)
    if include_main_pair:
        tables["compact_ls_main_pair_summary"] = compact_ls_main_pair_summary(banks)
    return tables, dict(status="complete", banks=metadata,
        availability_denominator="all first alarms, including premature alarms",
        common_paired_cohort="target-specific common available LS, unpenalized NB, and OCFR",
        aggregation="equal-cell mean squared relative error, then square root; no aggregate error bars",
        contrast_uncertainty="analytic sample SD of paired squared relative-error differences / sqrt(n)",
        interval_scope="point-only LS comparison; no LS confidence intervals")


def _compact_ls_comparison_figure(tables: dict, directory: Path) -> str:
    data = tables["compact_ls_aggregate"]
    fig, axes = plt.subplots(3, 2, figsize=(6.8, 7.4))
    styles = (("Full-delay LS", "#B44C43", "s"), ("Unpenalized NB", "#737373", "^"),
              ("OCFR", "#185889", "o"))
    for column, bank in enumerate(("Original bank", "Independent bank")):
        for row, (target, label) in enumerate(zip(ESTIMATION_TARGETS,
                ("Pre-change CFR", "Post-change CFR", "Risk ratio"))):
            ax = axes[row, column]
            values = []
            for x, (method, color, marker) in enumerate(styles):
                selected = data[data.bank.eq(bank) & data.target.eq(target) & data.method.eq(method)]
                if len(selected) != 1 or selected.iloc[0].n_cells != 16:
                    raise EvidenceError("LS supplementary figure requires both complete 16-cell banks")
                value = float(selected.iloc[0].common_paired_relative_rmse)
                if not np.isfinite(value) or value < 0:
                    raise EvidenceError("LS figure requires available errors in every condition")
                values.append(value)
                ax.plot(x, value, marker, color=color, markersize=6)
                ax.annotate(f"{value:.3g}", (x, value), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=9.5)
            # Fixed before bank outcomes: either RR bank can contain extreme
            # finite boundary fits; keep smaller methods visible in that case.
            logarithmic = row == 2 and min(values) > 0 and max(values)/min(values) > 100
            if logarithmic:
                ax.set_yscale("log")
                ax.set_ylim(min(values)*.5, max(values)*4)
                lo, hi = int(np.ceil(np.log10(min(values)*.5))), int(np.floor(np.log10(max(values)*4)))
                ax.set_yticks([10.**i for i in sorted(set([*range(lo, hi+1, 2), hi]))])
            else:
                target_values = data[data.target.eq(target)].common_paired_relative_rmse if row < 2 else pd.Series(values)
                ax.set_ylim(0, max(float(target_values.max())*1.25, .01))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=4))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: (
                f"{value:.0e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
                if value >= 1000 else f"{value:g}")))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks(range(3), ["LS", "NB 0", "OCFR"])
            ax.set_xlim(-.5, 2.5)
            ax.set_ylabel(label+"\nRelative RMSE"+(" (log scale)" if logarithmic else ""), fontsize=9.5)
            ax.tick_params(labelsize=9.5)
            _compact_letter(ax, "ABCDEF"[2*row+column])
            if row == 0:
                ax.set_title(bank, pad=12, fontsize=10)
    fig.subplots_adjust(left=.15, right=.98, top=.92, bottom=.07, wspace=.52, hspace=.43)
    _compact_save(fig, directory, "CompactFigS4_ls_comparison")
    return ("Full-delay daily least-squares (LS), unpenalized negative-binomial (NB 0), and OCFR point "
        "estimation at identical frozen first alarms and selected change dates. All use the full delay "
        "mean model; LS and NB 0 have no penalty and OCFR has log-RR penalty precision 1. Rows show "
        "pre-change CFR, post-change CFR, and risk ratio; columns show the original and independent "
        "banks. Each bank contains 16,000 episodes (1,000 per condition). Within each condition "
        "errors use the target-specific three-way available cohort. "
        "Squared relative errors are averaged equally across all 16 conditions before taking the square "
        "root. Symbols have no aggregate uncertainty bars; the tables retain every condition, availability, "
        "all-available errors, signed bias, and paired loss contrasts with analytic Monte Carlo standard "
        "errors. CFR axes start at zero. In either bank, the risk-ratio panel (E or F) uses a log scale "
        "when all values are positive and span more than two orders of magnitude; otherwise it uses "
        "a linear axis starting at zero. This scale rule was fixed before computing LS bank results. "
        "Finite extreme errors are retained. "
        "This figure reports point estimates and no LS confidence intervals or selection-adjusted coverage.")


def analyze_compact_ls_estimation(output: Path, figure_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Limited LS supplement refresh; preserve existing figures and manifests."""
    output = Path(output).resolve()
    if output.name == "evaluation":
        output = output.parent
    tables, metadata = _compact_load_ls_estimation(output)
    if not tables:
        raise EvidenceError("both completed LS estimation stages are required")
    full = all(bank["mode"] == "full" for bank in metadata["banks"].values())
    if not full and figure_dir is not None:
        raise EvidenceError("demo LS stages cannot export publication figures")
    destination = output/"analysis"
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        _atomic_csv(table, destination/f"{name}.csv")
    figures = Path(figure_dir) if figure_dir is not None else destination/"ls_figures"
    captions = {}
    if full:
        figures.mkdir(parents=True, exist_ok=True)
        _compact_style()
        captions["CompactFigS4_ls_comparison"] = _compact_ls_comparison_figure(tables, figures)
        for name, caption in captions.items():
            (figures/f"{name}_caption.txt").write_text(caption+"\n", encoding="utf-8")
    report = dict(status="complete", protocol="compact-v1", stage="ls-estimation-analysis",
        analysis_source_sha256=_sha256(Path(__file__)), source=metadata,
        files={f"{name}.csv": _sha256(destination/f"{name}.csv") for name in tables},
        figure_directory=str(figures.resolve()) if captions else None,
        publication_export=_compact_export_metadata(figures, list(captions)) if captions else None,
        figure_files={path.name: _sha256(path) for name in captions for path in
            (figures/f"{name}.pdf", figures/f"{name}.png", figures/"FigS4.tif")})
    _atomic_json(report, destination/"compact_ls_analysis_manifest.json")
    return tables


def analyze_compact_ls_main_figure(output: Path, figure_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Export only the LS main Figure 5 from verified frozen reporting stages.

    Existing tables, figures, and their manifests are left intact. The returned
    summary is in memory only; the separate manifest hashes every input CSV and
    manifest consumed by the LS and unchanged OCFR coverage loaders.
    """
    output = Path(output).resolve()
    if output.name == "evaluation":
        output = output.parent
    tables, ls_metadata = _compact_load_ls_estimation(output, include_main_pair=True)
    if not tables or not all(bank["mode"] == "full" for bank in ls_metadata["banks"].values()):
        raise EvidenceError("LS main Figure 5 requires both complete full LS estimation banks")
    stabilized, coverage_metadata = _compact_load_stabilized_estimation(output)
    if not stabilized:
        raise EvidenceError("LS main Figure 5 requires unchanged stabilized OCFR interval summaries")
    tables.update(stabilized)
    destination = output/"analysis"
    figures = Path(figure_dir) if figure_dir is not None else destination/"ls_main_figures"
    figures.mkdir(parents=True, exist_ok=True)
    _compact_style()
    name = "CompactFig5_estimation"
    caption = _compact_estimation_comparison_figure(tables, figures, ls=True)
    (figures/f"{name}_caption.txt").write_text(caption+"\n", encoding="utf-8")
    sources = {}
    for stage in ("estimation-ls-original", "estimation-ls-validation", "estimation-stabilized",
                  "estimation-validation", "estimation-reference"):
        sources[stage] = {filename: _sha256(output/stage/filename) for filename in (MANIFEST_FILE, RAW_FILE)}
    summary = tables["compact_ls_main_pair_summary"]
    main = summary[summary.bank.eq("Original bank") & summary.change_day.eq(92) &
                   np.isclose(summary.risk_ratio.to_numpy()[:, None], [2/3, 1.25, 1.5]).any(axis=1)]
    export_metadata = _compact_export_metadata(figures, [name])
    # Family headings and x tick labels in this established main layout are
    # explicitly 9.5 pt, below the global 10.5 pt plotting default.
    export_metadata["minimum_font_points"] = 9.5
    report = dict(status="complete", protocol="compact-v1", stage="ls-main-figure-analysis",
        analysis_source_sha256=_sha256(Path(__file__)),
        source={"full_delay_ls": ls_metadata, "unchanged_ocfr_coverage": coverage_metadata},
        source_files_sha256=sources, point_comparator="Full-delay LS",
        point_methods=["Full-delay LS", "OCFR"],
        point_cohort="target-specific common available LS and OCFR only; independent of unpenalized NB availability",
        displayed_conditions=sorted(main.scenario.unique()),
        displayed_counts=main[["scenario", "target", "method", "N_requested", "n_alarm", "n_point",
                               "n_common_paired"]].to_dict("records"),
        coverage="unchanged OCFR all-first-alarm unadjusted Wald containment with exact binomial Monte Carlo bars",
        figure_directory=str(figures.resolve()),
        publication_export=export_metadata,
        figure_files={path.name: _sha256(path) for path in
            (figures/f"{name}.pdf", figures/f"{name}.png", figures/"Fig5.tif", figures/f"{name}_caption.txt")})
    _atomic_json(report, destination/"compact_ls_main_analysis_manifest.json")
    return {"compact_ls_main_summary": main}



# Versioned v2 post-alarm interval contract; source hashes exclude this reporting file.
INTERVAL_SE_NAMES = ('regularized_pi_left_logit_se', 'regularized_pi_right_logit_se',
                     'regularized_log_rr_se')
INTERVAL_CUTOFF_SCHEMA = 'ocfr-new-rule-global-shorter-cutoffs-v1'


def _interval_sources_match(actual: Mapping, expected: Mapping) -> bool:
    """Producer-specific snapshot/reporting hashes may differ; scientific sources may not."""
    keys = ('stage', 'manifest_sha256', 'results_sha256', 'scientific_code_hash')
    return all(actual.get(key) == expected.get(key) and expected.get(key) for key in keys)


def _v2_scientific_code(manifest: Mapping, prefix: str = 'postprocessor') -> str:
    records = manifest.get(prefix+'_code_files', {})
    if (not isinstance(records, dict) or not {'config.py', 'methods.py', 'simulation.py'} <= set(records)
            or any(not isinstance(v, str) or len(v) != 64 for v in records.values())):
        raise EvidenceError('missing recorded scientific source hashes')
    digest = hashlib.sha256(_canonical_json(records).encode('ascii')).hexdigest()
    if manifest.get(prefix+'_code_hash') != digest:
        raise EvidenceError('recorded scientific source digest mismatch')
    return digest


def _v2_selected_source(output: Path, bank: str) -> tuple[pd.DataFrame, Any, dict, dict]:
    """Validate an entire fresh selected-fit bank without comparing to today's reporting file.

    Configuration reconstruction is read-only. No draws, fits or current-code
    digest recomputation occurs here; executable scientific provenance comes
    from the source manifest and its linked detection manifest.
    """
    from dataclasses import asdict
    from config import PostAlarmEstimationConfig, get_post_alarm_validation_config
    from simulation import _estimation_source_config, _planned_looks
    if bank not in {'original', 'validation'}:
        raise ValueError('selected-fit bank must be original or validation')
    stage = 'estimation-stabilized' if bank == 'original' else 'estimation-validation'
    path = Path(output)/stage/MANIFEST_FILE
    raw = path.parent/RAW_FILE
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('stage') != stage or manifest.get('status') != 'complete'
            or manifest.get('mode') != 'full'):
        raise EvidenceError('requires a complete full selected-fit bank: '+stage)
    if _sha256(raw) != _manifest_file_hash(manifest, RAW_FILE):
        raise EvidenceError('selected-fit CSV artifact hash mismatch')
    try:
        config = _estimation_source_config(dict(manifest, stage='evaluation'), 'full')
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError('selected-fit scientific configuration cannot be reconstructed') from exc
    expected_seed = 2026111404 if bank == 'original' else 2026111405
    if config.protocol != 'compact-v2' or config.master_seed != expected_seed:
        raise EvidenceError('selected-fit source must use the fresh v2 training/independent seed contract')
    if manifest.get('post_alarm_estimation_config') != asdict(PostAlarmEstimationConfig()):
        raise EvidenceError('selected-fit reporting settings changed')
    code_hash = _v2_scientific_code(manifest)
    evaluation_path = Path(output)/'evaluation'/MANIFEST_FILE
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
    if (evaluation.get('status') != 'complete' or evaluation.get('stage') != 'evaluation'
            or evaluation.get('protocol') != 'compact-v2' or evaluation.get('mode') != 'full'):
        raise EvidenceError('selected fits require the completed fresh evaluation source')
    evaluation_raw = evaluation_path.parent/RAW_FILE
    if (_sha256(evaluation_path) != manifest.get('original_manifest_sha256') or
            _sha256(evaluation_raw) != manifest.get('original_csv_sha256') or
            _sha256(evaluation_raw) != _manifest_file_hash(evaluation, RAW_FILE) or
            evaluation.get('scientific_digest') != manifest.get('original_scientific_digest')):
        raise EvidenceError('selected-fit original evaluation linkage mismatch')
    _validate_scientific_digest(evaluation, pd.DataFrame())
    train_scientific = evaluation['scientific_config']
    if train_scientific.get('protocol') != 'compact-v2' or train_scientific.get('master_seed') != 2026111404:
        raise EvidenceError('original detection source is not the fresh v2 training study')
    _v2_scientific_code(dict(postprocessor_code_files=evaluation.get('code_files'),
                             postprocessor_code_hash=evaluation.get('code_hash')))
    expected_scientific = dict(train_scientific, master_seed=expected_seed)
    if config.scientific_payload() != expected_scientific:
        raise EvidenceError('selected-fit source differs from the original detection design')
    if bank == 'original':
        original_hash = _v2_scientific_code(manifest, 'original')
        if (original_hash != evaluation.get('code_hash') or
                manifest.get('original_code_files') != evaluation.get('code_files') or
                manifest.get('scientific_digest') != evaluation.get('scientific_digest')):
            raise EvidenceError('training-fit detection source hashes disagree')
    else:
        if manifest.get('validation_config') != asdict(get_post_alarm_validation_config('compact-v2')):
            raise EvidenceError('independent validation configuration changed')
        expected_digest = hashlib.sha256(_canonical_json(config.scientific_payload()).encode('ascii')).hexdigest()[:20]
        if manifest.get('scientific_digest') != expected_digest:
            raise EvidenceError('independent selected-fit scientific digest mismatch')
    selected = [s for s in config.scenarios if s.split == 'alternative' and s.misspecification == 'on_model']
    names = {s.name for s in selected}
    counts = {name: 1000 for name in names}
    if (len(names) != 16 or set(manifest.get('scenarios', [])) != names or
            manifest.get('replications_by_scenario') != counts):
        raise EvidenceError('selected-fit manifest must declare the complete 16 by 1000 grid')
    frame = pd.read_csv(raw)
    required = {'scenario', 'replicate', 'method', 'family', 'horizon', 'true_change', 'alarm',
                'alarm_day', 'tau_hat_at_alarm', *ESTIMATION_TARGETS, *INTERVAL_SE_NAMES}
    for target in ESTIMATION_TARGETS:
        required.update('regularized_'+target+'_'+suffix for suffix in
                        ('estimate', 'point_valid', 'lower', 'upper', 'interval_valid'))
    if required-set(frame):
        raise EvidenceError('selected-fit source lacks direct SEs or required fields: '+str(sorted(required-set(frame))))
    expected_grid = {(name, rep) for name in names for rep in range(1000)}
    if (len(frame) != 16000 or frame.duplicated(['scenario', 'replicate']).any() or
            set(zip(frame.scenario, frame.replicate)) != expected_grid or not frame.method.eq('OCFR').all()):
        raise EvidenceError('selected-fit episode/method grid is incomplete or duplicated')
    alarm = _as_binary(frame.alarm, 'alarm').astype(bool)
    day = pd.to_numeric(frame.alarm_day, errors='coerce')
    tau = pd.to_numeric(frame.tau_hat_at_alarm, errors='coerce')
    if (not np.array_equal(alarm, day.notna()) or
            (alarm & (~np.isfinite(day) | (day != np.floor(day)) | (day < 0) | (day >= frame.horizon)
                      | ~np.isfinite(tau) | (tau != np.floor(tau)) | (tau < 1) | (tau > day))).any()
            or tau[~alarm].notna().any()):
        raise EvidenceError('selected-fit alarm or cohort date is invalid')
    for scenario in selected:
        group = frame[frame.scenario.eq(scenario.name)]
        for column, expected in (('family', scenario.family), ('horizon', scenario.horizon),
                ('true_change', scenario.true_change), ('pi_left', scenario.pi_left),
                ('pi_right', scenario.pi_right), ('risk_ratio', scenario.risk_ratio)):
            matches = group[column].eq(expected).all() if isinstance(expected, str) else np.allclose(
                group[column].to_numpy(float), expected, rtol=1e-12, atol=1e-15)
            if not matches:
                raise EvidenceError(f'{scenario.name}: selected-fit truth/metadata mismatch: {column}')
    for target in ESTIMATION_TARGETS:
        point = _as_binary(frame['regularized_'+target+'_point_valid'], target).astype(bool)
        interval = _as_binary(frame['regularized_'+target+'_interval_valid'], target).astype(bool)
        value = pd.to_numeric(frame['regularized_'+target+'_estimate'], errors='coerce')
        if ((point & (~alarm | ~np.isfinite(value) | (value <= 0) |
                ((value >= 1) if target != 'risk_ratio' else False))).any() or
                (interval & ~point).any()):
            raise EvidenceError('selected-fit point/interval availability is inconsistent')
        cols = ['regularized_'+target+'_'+suffix for suffix in ('estimate', 'lower', 'upper')]
        if frame.loc[~alarm, cols].notna().any().any():
            raise EvidenceError('no-alarm episode has a fabricated estimate or interval')
    if bank == 'original':
        # Source dates are checked against actual new OCFR evaluation selections.
        source = _compact_read_summary_csv(evaluation_raw,
            {'scenario', 'replicate', 'method', 'alarm_day', 'tau_hat_at_alarm'})
        source = source[source.method.eq('OCFR') & source.scenario.isin(names)].set_index(['scenario', 'replicate'])
        if len(source) != 16000 or source.index.duplicated().any():
            raise EvidenceError('training-fit evaluation selections are incomplete')
        aligned = source.loc[pd.MultiIndex.from_frame(frame[['scenario', 'replicate']])]
        for column in ('alarm_day', 'tau_hat_at_alarm'):
            if not np.allclose(frame[column], aligned[column].to_numpy(), equal_nan=True):
                raise EvidenceError('training fits changed the new OCFR first selection')
        ref_path = Path(output)/'estimation-reference'/MANIFEST_FILE
        ref_raw = ref_path.parent/RAW_FILE
        if (_sha256(ref_path) != manifest.get('original_reference_manifest_sha256') or
                _sha256(ref_raw) != manifest.get('original_reference_csv_sha256')):
            raise EvidenceError('stabilized-fit reference source linkage mismatch')
        reference = json.loads(ref_path.read_text(encoding='utf-8'))
        if (reference.get('status') != 'complete' or reference.get('stage') != 'estimation-reference'
                or reference.get('mode') != 'full' or
                _sha256(ref_raw) != _manifest_file_hash(reference, RAW_FILE)):
            raise EvidenceError('stabilized-fit reference source is incomplete')
        original = pd.read_csv(ref_raw)
        try:
            pd.testing.assert_frame_equal(frame[original.columns].reset_index(drop=True),
                original.reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-15)
        except (KeyError, AssertionError) as exc:
            raise EvidenceError('stabilized stage changed its frozen source observations/selections') from exc
    else:
        if 'trace_json' not in frame:
            raise EvidenceError('independent selected fits lack saved first-crossing traces')
        lookup = {s.name: s for s in selected}
        for row in frame[['scenario', 'alarm_day', 'trace_json']].itertuples(index=False):
            try:
                trace = json.loads(row.trace_json)
                for look in trace:
                    if look.get('available') and (not np.isfinite(float(look['statistic'])) or
                            np.isnan(float(look['boundary'])) or np.isneginf(float(look['boundary']))):
                        raise ValueError('unknown or invalid available-look outcome')
                crossing = next((r['day'] for r in trace if r.get('available') and
                                 float(r['statistic']) > float(r['boundary'])), None)
                expected = None if pd.isna(row.alarm_day) else int(row.alarm_day)
                calendar = [prefix-1 for prefix in _planned_looks(lookup[row.scenario], config)]
                calendar = calendar if expected is None else [day for day in calendar if day <= expected]
                if crossing != expected or [r['day'] for r in trace] != calendar:
                    raise ValueError('first crossing or calendar changed')
            except (TypeError, KeyError, ValueError) as exc:
                raise EvidenceError('independent saved first crossing does not replay') from exc
    provenance = dict(stage=stage, manifest_sha256=_sha256(path), results_sha256=_sha256(raw),
                      scientific_code_hash=code_hash)
    return frame, config, provenance, manifest


def _load_frozen_interval_cutoffs(output: Path, expected_hash: str) -> tuple[dict, dict]:
    """This guard must finish before any independent-validation file is opened."""
    path = Path(output)/'interval-calibration/CUTOFFS.json'
    frozen_path = path.parent/'FROZEN.json'
    if (not isinstance(expected_hash, str) or len(expected_hash) != 64 or
            not path.is_file() or _sha256(path) != expected_hash):
        raise EvidenceError('the frozen cutoff SHA-256 is required before validation loading')
    record = json.loads(path.read_text(encoding='utf-8'))
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    if (record.get('schema') != INTERVAL_CUTOFF_SCHEMA or
            frozen.get('cutoffs_sha256') != expected_hash or
            frozen.get('source') != record.get('training_source')):
        raise EvidenceError('interval cutoff schema or freeze record mismatch')
    if set(record.get('targets', {})) != set(ESTIMATION_TARGETS):
        raise EvidenceError('interval cutoff target set changed')
    for cutoff in record['targets'].values():
        if (set(cutoff) != {'lower', 'upper'} or
                not np.isfinite([cutoff['lower'], cutoff['upper']]).all() or cutoff['lower'] > cutoff['upper']):
            raise EvidenceError('invalid global stopped-root cutoff pair')
    for field, file in (('manifest_sha256', Path(output)/'estimation-stabilized'/MANIFEST_FILE),
                        ('results_sha256', Path(output)/'estimation-stabilized'/RAW_FILE)):
        if record.get('training_source', {}).get(field) != _sha256(file):
            raise EvidenceError('cutoffs belong to a different training source')
    training = json.loads((Path(output)/'estimation-stabilized'/MANIFEST_FILE).read_text(encoding='utf-8'))
    scientific = training.get('scientific_config', {})
    if (training.get('stage') != 'estimation-stabilized' or training.get('status') != 'complete'
            or training.get('mode') != 'full' or scientific.get('protocol') != 'compact-v2'
            or scientific.get('master_seed') != 2026111404
            or scientific.get('compact_contract', {}).get('boundary_policy') !=
                'gaussian-first-crossing-linear-support-discard-v1'
            or record.get('training_source', {}).get('stage') != 'estimation-stabilized'
            or record.get('training_source', {}).get('scientific_code_hash') != _v2_scientific_code(training)):
        raise EvidenceError('cutoff training scientific source/policy contract mismatch')
    return record, frozen


def run_interval_calibration(output: Path) -> dict:
    """Fit/freeze cutoffs from training only; never open the validation stage."""
    output = Path(output).resolve()
    frame, _, provenance, _ = _v2_selected_source(output, 'original')
    provenance['analysis_sha256'] = _sha256(Path(__file__))
    return calibrate_interval_cutoffs(frame, provenance, output)


def run_interval_application(output: Path, expected_cutoff_sha256: str) -> dict:
    """Verify freeze first, then assess the independent seed using fixed cutoffs."""
    output = Path(output).resolve()
    record, _ = _load_frozen_interval_cutoffs(output, expected_cutoff_sha256)
    training_frame, _, training_provenance, training = _v2_selected_source(output, 'original')
    if not _interval_sources_match(record['training_source'], training_provenance):
        raise EvidenceError('frozen cutoffs belong to another training source')
    # Replay the declared training-only optimization before opening validation.
    # An existing identical artifact is preserved byte-for-byte, never retuned.
    calibrate_interval_cutoffs(training_frame, training_provenance, output)
    if _sha256(output/'interval-calibration/CUTOFFS.json') != expected_cutoff_sha256:
        raise EvidenceError('interval application changed the frozen cutoff artifact')
    frame, config, provenance, manifest = _v2_selected_source(output, 'validation')
    train_path = output/'estimation-stabilized'/MANIFEST_FILE
    if (_v2_scientific_code(training) != provenance['scientific_code_hash'] or
            not _interval_sources_match(record['training_source'], dict(stage='estimation-stabilized',
                manifest_sha256=_sha256(train_path), results_sha256=_sha256(train_path.parent/RAW_FILE),
                scientific_code_hash=_v2_scientific_code(training))) or
            dict(training['scientific_config'], master_seed=config.master_seed) != manifest['scientific_config']):
        raise EvidenceError('calibration and validation detector/scientific source contracts differ')
    provenance['analysis_sha256'] = _sha256(Path(__file__))
    return apply_interval_cutoffs(frame, record['targets'], provenance, output, expected_cutoff_sha256)


def _v2_interval_reporting(frame: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """All-alarm coverage/availability and widths for both fixed interval rules."""
    rows = []
    for scenario, group in frame.groupby('scenario', sort=True):
        first = group.iloc[0]
        alarm = _as_binary(group.alarm, 'alarm').astype(bool).to_numpy()
        n = int(alarm.sum())
        selected = intervals.loc[group.index]
        for target in ESTIMATION_TARGETS:
            truth = group[target].to_numpy(float)
            for rule, low, high, available in (
                    ('Uncalibrated Wald', group['regularized_'+target+'_lower'].to_numpy(float),
                     group['regularized_'+target+'_upper'].to_numpy(float),
                     _as_binary(group['regularized_'+target+'_interval_valid'], target).to_numpy(bool)),
                    ('Global calibrated', selected[target+'_lower'].to_numpy(float),
                     selected[target+'_upper'].to_numpy(float),
                     _as_binary(selected[target+'_available'], target).to_numpy(bool))):
                if (available & (~alarm | np.isnan(low) | np.isnan(high) | (low > high)
                        | (low < 0) | ((high > 1) if target != 'risk_ratio' else False))).any():
                    raise EvidenceError('declared interval endpoints or alarm nesting are invalid')
                covered = available & (low <= truth) & (truth <= high)
                k = int(covered.sum())
                ci = compact_binomial_interval(k, n) if n else (np.nan, np.nan)
                with np.errstate(invalid='ignore'):
                    width = high-low
                width[available & np.isposinf(high)] = np.inf
                use = alarm & available
                rows.append(dict(scenario=scenario, family=first.family, target=target,
                    risk_ratio=float(first.risk_ratio), change_day=int(first.true_change)+1,
                    interval_rule=rule, n_requested=len(group), n_alarm=n, n_available=int(use.sum()),
                    n_unavailable=int(np.sum(alarm & ~available)), n_covered=k,
                    coverage=k/n if n else np.nan, coverage_ci_low=ci[0], coverage_ci_high=ci[1],
                    availability=float(use.sum()/n) if n else np.nan,
                    n_infinite_width=int(np.sum(use & np.isinf(width))),
                    mean_available_width=float(np.mean(width[use])) if use.any() else np.nan,
                    median_available_width=float(np.median(width[use])) if use.any() else np.nan,
                    denominator='all first alarms including premature alarms; missing intervals are noncoverage'))
    return pd.DataFrame(rows)


def _compact_load_v2_postalarm(output: Path) -> tuple[dict, dict]:
    """Accept public or private-calibrator artifacts only after source-bound replay."""
    output = Path(output)
    stages = ('estimation-stabilized', 'estimation-validation', 'estimation-ls-original',
              'estimation-ls-validation', 'interval-calibration', 'interval-validation')
    missing = [stage for stage in stages if not (output/stage/MANIFEST_FILE).is_file()
               and stage != 'interval-calibration']
    if not (output/'interval-calibration/CUTOFFS.json').is_file():
        missing.append('interval-calibration')
    if missing:
        return {}, dict(status='absent', missing_stages=missing,
            reason='fresh selected fits, both LS banks and frozen independent interval validation are required')
    training, config, training_provenance, training_manifest = _v2_selected_source(output, 'original')
    cutoff_path = output/'interval-calibration/CUTOFFS.json'
    cutoff_hash = _sha256(cutoff_path)
    cutoff_record, _ = _load_frozen_interval_cutoffs(output, cutoff_hash)
    if not _interval_sources_match(cutoff_record['training_source'], training_provenance):
        raise EvidenceError('interval calibration is not bound to these fresh training fits')
    # Replay calibration from training roots. Never retune on validation.
    frozen_before = cutoff_hash
    calibrate_interval_cutoffs(training, training_provenance, output)
    if _sha256(cutoff_path) != frozen_before:
        raise EvidenceError('analysis must not change the frozen cutoff artifact')
    validation, validation_config, validation_provenance, validation_manifest = _v2_selected_source(output, 'validation')
    if (training_provenance['scientific_code_hash'] != validation_provenance['scientific_code_hash'] or
            dict(config.scientific_payload(), master_seed=validation_config.master_seed) != validation_config.scientific_payload()):
        raise EvidenceError('fresh training and validation use different scientific source/configuration contracts')
    interval_path = output/'interval-validation'/MANIFEST_FILE
    interval_manifest = json.loads(interval_path.read_text(encoding='utf-8'))
    interval_csv = interval_path.parent/'intervals.csv'
    if (interval_manifest.get('status') != 'complete' or interval_manifest.get('stage') != 'interval-validation'
            or interval_manifest.get('cutoff_sha256') != cutoff_hash or
            not interval_manifest.get('intervals_truth_free') or
            not _interval_sources_match(interval_manifest.get('source', {}), validation_provenance) or
            _sha256(interval_csv) != _manifest_file_hash(interval_manifest, 'intervals.csv')):
        raise EvidenceError('fixed-cutoff independent interval artifact/source mismatch')
    intervals = pd.read_csv(interval_csv, float_precision='round_trip')
    if (len(intervals) != len(validation) or intervals.duplicated(['scenario', 'replicate']).any()
            or set(zip(intervals.scenario, intervals.replicate)) != set(zip(validation.scenario, validation.replicate))):
        raise EvidenceError('independent interval application grid is incomplete')
    intervals = intervals.set_index(['scenario', 'replicate']).loc[
        pd.MultiIndex.from_frame(validation[['scenario', 'replicate']])].reset_index()
    if not np.array_equal(_as_binary(intervals.alarm, 'interval alarm'), _as_binary(validation.alarm, 'source alarm')):
        raise EvidenceError('interval artifact changed the independent alarm denominator')
    for target in ESTIMATION_TARGETS:
        center, se, available = _interval_transformed(validation, target)
        low, high, valid = truth_free_apply(center, se, available, target, cutoff_record['targets'][target])
        truth = validation[target].to_numpy(float)
        covered = valid & (low <= truth) & (truth <= high)
        with np.errstate(invalid='ignore'):
            width = high-low
        width[valid & np.isposinf(high)] = np.inf
        for suffix, expected in (('lower', low), ('upper', high), ('width', width)):
            if not np.allclose(intervals[target+'_'+suffix], expected, equal_nan=True, rtol=1e-12, atol=1e-15):
                raise EvidenceError('fixed interval endpoints/widths do not replay from direct SEs')
        for suffix, expected in (('available', valid), ('covered', covered)):
            if not np.array_equal(_as_binary(intervals[target+'_'+suffix], suffix).to_numpy(bool), expected):
                raise EvidenceError('fixed interval availability/coverage does not replay')
        # Assessment uses the verified binary64 construction, avoiding a CSV
        # parser's last-bit perturbation of a truth lying exactly on a cutoff.
        intervals[target+'_lower'], intervals[target+'_upper'] = low, high
        intervals[target+'_width'] = width
    coverage = _v2_interval_reporting(validation, intervals)
    for record in coverage[coverage.interval_rule.eq('Global calibrated')].to_dict('records'):
        saved = interval_manifest.get('reports', {}).get(record['target'], {}).get(record['scenario'], {})
        for label, key in (('alarms', 'n_alarm'), ('available', 'n_available'), ('covered', 'n_covered'),
                           ('unavailable_counted_as_noncoverage', 'n_unavailable'), ('infinite_widths', 'n_infinite_width')):
            if saved.get(label) != record[key]:
                raise EvidenceError('interval assessment counts disagree with all-alarm replay')
    # Existing exact LS provenance/cache/source-column checks apply to v2 too.
    ls_tables, ls_metadata = _compact_load_ls_estimation(output, include_main_pair=True)
    for stage in ('estimation-ls-original', 'estimation-ls-validation'):
        manifest = json.loads((output/stage/MANIFEST_FILE).read_text(encoding='utf-8'))
        if manifest.get('mode') != 'full':
            raise EvidenceError('v2 post-alarm figures require both complete full LS banks')
        source = training_manifest if stage.endswith('original') else validation_manifest
        if _v2_scientific_code(manifest) != source['postprocessor_code_hash']:
            raise EvidenceError('LS and selected-fit scientific source contracts differ')
    pair = ls_tables['compact_ls_main_pair_summary'].copy()
    mapping = []
    selected_scenarios = [scenario for scenario in config.scenarios
        if scenario.split == 'alternative' and scenario.misspecification == 'on_model']
    selected_scenarios.sort(key=lambda scenario: (
        ('steady', 'sparse', 'wave').index(scenario.family), scenario.true_change, scenario.risk_ratio))
    for index, scenario in enumerate(selected_scenarios, start=1):
        mapping.append(dict(cell_index=index, scenario=scenario.name, family=scenario.family,
                            risk_ratio=scenario.risk_ratio, change_day=scenario.true_change+1))
    tables = {'compact_v2_ls_summary': pair, 'compact_v2_interval_summary': coverage,
              'compact_v2_estimation_cells': pd.DataFrame(mapping)}
    tables.update(ls_tables)
    return tables, dict(status='complete', cutoff_sha256=cutoff_hash,
        training=training_provenance, validation=validation_provenance, ls=ls_metadata,
        interval_manifest_sha256=_sha256(interval_path), interval_csv_sha256=_sha256(interval_csv),
        coverage_scope='independent seed 2026111405; all first alarms including premature; unavailable is noncoverage',
        calibration_scope='one global asymmetric pair per target; minimum root width with ceil(.95*N alarms) in each of 16 training cells')


def _compact_v2_estimation_figure(tables: dict, directory: Path, *, all_cells: bool = False) -> str:
    """Fresh training LS/OCFR point errors and independent fixed-cutoff coverage."""
    point = tables['compact_v2_ls_summary']
    point = point[point.bank.eq('Original bank')]
    coverage = tables['compact_v2_interval_summary']
    mapping = tables['compact_v2_estimation_cells']
    if not all_cells:
        families = ('steady', 'sparse', 'wave')
        ratios = (2/3, 1.5)
        wanted = []
        for family in families:
            for ratio in ratios:
                cell = mapping[mapping.family.eq(family) & mapping.change_day.eq(92) & np.isclose(mapping.risk_ratio, ratio)]
                if len(cell) != 1:
                    raise EvidenceError('main v2 estimation figure requires both reciprocal risk ratios in each family')
                wanted.append(cell.iloc[0].scenario)
        labels = ['2/3', '1.5'] * len(families)
    else:
        wanted = mapping.sort_values('cell_index').scenario.tolist()
        labels = [str(x) for x in range(1, len(wanted)+1)]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.4), sharex=True)
    x = np.arange(len(wanted))
    for row, (target, title) in enumerate(zip(ESTIMATION_TARGETS,
            ('Pre-change CFR', 'Post-change CFR', 'Risk ratio'))):
        left, right = axes[row]
        for method, shift, marker, color in (('Full-delay LS', -.13, 's', '#B44C43'),
                                            ('OCFR', .13, 'o', '#185889')):
            data = point[point.target.eq(target) & point.method.eq(method)].set_index('scenario').loc[wanted]
            value = data.common_paired_relative_rmse.to_numpy(float)
            se = data.common_paired_relative_rmse_mcse.to_numpy(float)
            left.errorbar(x+shift, value, yerr=np.vstack([np.minimum(value, 1.96*se), 1.96*se]),
                fmt=marker, color=color, capsize=1.5, markersize=3.5, linewidth=.8, label=method)
        for rule, shift, marker, color in (('Uncalibrated Wald', -.13, 'o', '#737373'),
                ('Global calibrated', .13, 's', '#185889')):
            data = coverage[coverage.target.eq(target) & coverage.interval_rule.eq(rule)].set_index('scenario').loc[wanted]
            value = data.coverage.to_numpy(float)
            right.errorbar(x+shift, value,
                yerr=np.vstack([value-data.coverage_ci_low.to_numpy(float), data.coverage_ci_high.to_numpy(float)-value]),
                fmt=marker, color=color, capsize=1.5, markersize=3.5, linewidth=.8, label=rule)
        left.set_ylabel(title+'\nRelative RMSE')
        left.set_ylim(bottom=0)
        right.set_ylabel('Independent interval\ncoverage')
        right.set_ylim(0, 1.04)
        right.axhline(.95, color='.4', linestyle='--', linewidth=.8)
        for column, ax in enumerate((left, right)):
            ax.set_xlim(-.5, len(wanted)-.5)
            ticks = np.arange(len(wanted)) if not all_cells else np.array([0, 3, 7, 11, 15])
            ax.set_xticks(ticks, [labels[index] for index in ticks])
            ax.tick_params(axis='x', labelsize=10.5, labelbottom=True)
            _compact_letter(ax, 'ABCDEF'[2*row+column])
            if not all_cells:
                for index, family in enumerate(families):
                    first = index * len(ratios)
                    end = first + len(ratios)
                    if index % 2 == 0:
                        ax.axvspan(first-.5, end-.5, color='#edf2f6', zorder=-5)
                    ax.text((first+end-1)/2, 1.035, family.capitalize(),
                            transform=ax.get_xaxis_transform(), ha='center', fontsize=10.5)
    fig.text(.32, .995, 'Point estimation (lower is better)',
             ha='center', va='top', fontsize=10.5)
    fig.text(.78, .995, 'Interval coverage (target 95%)',
             ha='center', va='top', fontsize=10.5)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.32, .967), ncol=2, fontsize=10.5)
    fig.legend(*axes[0, 1].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.78, .967), ncol=1, fontsize=10.5)
    for ax in axes[-1]:
        ax.set_xlabel('Declared condition index' if all_cells else 'True risk ratio')
    fig.subplots_adjust(left=.13, right=.985, bottom=.09, top=.88, hspace=.48, wspace=.50)
    name = 'CompactFigS5_all_conditions' if all_cells else 'CompactFig5_estimation'
    _compact_save(fig, directory, name)
    training_n = sorted(set(point.N_requested.astype(int)))
    validation_n = sorted(set(coverage.n_requested.astype(int)))
    paired_counts = point[point.scenario.isin(wanted)].n_common_paired.astype(int)
    alarm_counts = coverage[coverage.scenario.isin(wanted)].n_alarm.astype(int)
    return ('Rows: pre-change CFR, post-change CFR and risk ratio. A,C,E: relative RMSE of full-delay '
        'least squares and precision-one regularized NB estimation (OCFR) at the same new OCFR '
        'selected first alarm and cohort date in the training bank. Target-specific common available '
        'LS/OCFR points form each comparison; interval availability and other estimators do not filter '
        'those points. Bars are approximate 95% delta-method Monte Carlo intervals for RMSE. B,D,F: '
        'uncalibrated regularized Wald and frozen global-shorter interval coverage in the independent '
        'seed-2026111405 bank, with exact 95% binomial Monte Carlo bars and nominal .95 reference. '
        'All first alarms, including premature alarms, form coverage denominators; unavailable '
        'intervals are noncoverage and defined infinite widths are retained. One asymmetric root '
        'pair per target was calibrated using all 16 training conditions before opening validation. '
        'This is finite-design empirical calibration, not a general selection-adjusted coverage guarantee. '
        + ('All 16 conditions appear. Indices 1-6 are steady incidence with (risk ratio, change day) '
           '(1.5,47), (2/3,92), (1.25,92), (1.5,92), (2,92), (1.5,137), respectively. '
           'Indices 7-10 are sparse incidence with risk ratios 2/3, 1.25, 1.5 and 2, respectively, '
           'all at change day 92. Indices 11-16 are wave incidence in the same risk-ratio and '
           'change-day order as indices 1-6. ' if all_cells else
           'The six conditions shown have change day 92 and reciprocal risk ratios 2/3 and 1.5 '
           'within each family, representing equal log-risk-ratio magnitudes in opposite directions. ')
        + f'Training requested n per cell: {training_n}; independent-validation requested n: {validation_n}. '
        + f'Displayed matched-point counts range from {paired_counts.min()} to {paired_counts.max()}; '
        + f'independent first-alarm denominators range from {alarm_counts.min()} to {alarm_counts.max()}. '
        'Numerical summaries report the paired counts, first-alarm denominators, estimation and interval '
        'availability, and interval widths for all 16 conditions.')


def plot_method_overview(output_dir: Path) -> None:
    """Draw the three schematic panels in Figure 1 from illustrative values."""
    from contextlib import contextmanager
    from matplotlib import _mathtext
    from matplotlib.patches import FancyBboxPatch

    output_dir.mkdir(parents=True, exist_ok=True)
    style = {
        "font.family": "serif", "font.size": 10.6, "axes.labelsize": 10.6,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": None,
    }
    BLUE, ORANGE, PURPLE, SKY, BLACK, LIGHT = (
        "#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#222222", "#D9D9D9",
    )

    def panel(ax, label, *, x=0.0):
        ax.text(x, 1.19, f"({label})", transform=ax.transAxes, fontweight="bold",
                ha="right", va="bottom", clip_on=False, fontsize=10.6)

    @contextmanager
    def math_script_scale():
        # The paper enlarges nested math subscripts in this schematic. Keep
        # Matplotlib's default script sizes for every other figure.
        shrink = _mathtext.SHRINK_FACTOR
        _mathtext.SHRINK_FACTOR = .90
        try:
            yield
        finally:
            _mathtext.SHRINK_FACTOR = shrink

    with matplotlib.rc_context({**matplotlib.rcParamsDefault, **style}), math_script_scale():
        fig, axes = plt.subplots(1, 3, figsize=(6.27, 3.05))
        fig.subplots_adjust(left=0.045, right=0.985, bottom=0.22, top=0.78, wspace=0.32)

        # (a) A confirmed-case cohort contributes to daily deaths through the delay law.
        ax = axes[0]
        time = np.arange(10)
        cases = np.array([0.10, 0.22, 0.38, 0.88, 0.54, 0.32, 0.20, 0.12, 0.08, 0.04])
        deaths = np.array([0.00, 0.00, 0.02, 0.04, 0.08, 0.18, 0.28, 0.22, 0.13, 0.07])
        case_colors = [LIGHT] * len(time)
        case_colors[3] = BLUE
        ax.bar(time, cases, width=0.72, color=case_colors, edgecolor="none")
        ax.bar(time, -deaths, width=0.72, color=ORANGE, alpha=0.78, edgecolor="none")
        for target, weight in zip((5, 6, 7), (0.20, 0.45, 0.25), strict=True):
            ax.annotate(
                "",
                xy=(target, -deaths[target] + 0.01),
                xytext=(3, cases[3] + 0.02),
                arrowprops={
                    "arrowstyle": "->",
                    "color": BLUE,
                    "alpha": 0.35 + weight,
                    "linewidth": 0.8 + 1.2 * weight,
                    "connectionstyle": "arc3,rad=-0.18",
                },
            )
        ax.axhline(0, color=BLACK, linewidth=0.7)
        ax.annotate("cohort $u$", xy=(3, 0.9), xytext=(1.2, 1.02), color=BLUE,
                    arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 0.8})
        ax.text(0.01, 0.80, "cohort\ncases $C_u$", color=BLUE, fontsize=9.5,
                ha="left", transform=ax.transAxes)
        ax.text(0.02, 0.31, "daily deaths $D_s$", color=ORANGE, fontsize=9.5,
                ha="left", transform=ax.transAxes)
        ax.text(5.5, 0.60, "delay\n$f_{s-u}$", color=BLUE, fontsize=9.5)
        ax.text(3.5, -0.85, r"$X_s=\sum_{u\leq s-\ell_0}C_u f_{s-u}$", ha="center", fontsize=10.6)
        ax.set_xlim(-2.6, 9.6)
        ax.set_ylim(-1.20, 1.08)
        ax.set_xticks([0, 3, 6, 9], ["earlier", "", "", "later"])
        ax.set_yticks([])
        ax.set_title("Cohort\nalignment", fontsize=9.5, fontweight="bold", pad=14)
        panel(ax, "a")

        # (b) The nuisance-adjusted score is scanned over eligible change cohorts.
        ax = axes[1]
        tau = np.arange(1, 31)
        score = (
            0.35
            + 0.35 * np.sin(tau / 2.7) ** 2
            + 7.0 * np.exp(-0.5 * ((tau - 19.0) / 2.6) ** 2)
        )
        ax.fill_between(tau, 0, score, color=SKY, alpha=0.18)
        ax.plot(tau, score, color=BLUE, linewidth=1.8)
        peak = int(np.argmax(score))
        ax.axvline(tau[peak], color=ORANGE, linestyle="--", linewidth=1.1)
        ax.annotate(
            r"$\widehat\tau_t$",
            xy=(tau[peak], score[peak]),
            xytext=(tau[peak] + 3.0, score[peak] - 1.1),
            color=ORANGE,
            arrowprops={"arrowstyle": "->", "color": ORANGE, "linewidth": 0.9},
        )
        # Explicitly positioned limit avoids a false whole-span collision from
        # the mathematical max operator; the equation is unchanged.
        for x, y, text, size, ha in (
            (0.04, 0.96, r"$M_t=$", 10.6, 'left'),
            (0.43, 0.96, r"$\max$", 10.6, 'center'),
            (0.43, 0.83, r"$\tau\in\mathcal{K}_t$", 9.5, 'center'),
            (0.64, 0.96, r"$Z_t^2(\tau)$", 10.6, 'left'),
        ):
            ax.text(x, y, text, transform=ax.transAxes, ha=ha, va='top', fontsize=size)
        ax.set_xlim(1, 30)
        ax.set_ylim(0, 11.5)
        ax.set_xlabel("eligible cohort\nchange day $\\tau$")
        ax.set_ylabel(r"score $Z_t^2(\tau)$")
        ax.set_xticks([1, 10, 20, 30])
        ax.set_yticks([0, 4, 8])
        ax.set_title("Efficient-score\nscan", fontsize=9.5, fontweight="bold", pad=14)
        panel(ax, "b")

        # (c) Planned looks are compared with conditional Gaussian multiplier boundaries.
        ax = axes[2]
        looks = np.arange(1, 8)
        boundary = np.array([7.8, 8.1, 8.0, 8.3, 8.4, 8.5, 8.5])
        # The sixth look is the first strict exceedance of the boundary.
        statistic = np.array([2.1, 3.0, 3.7, 5.8, 7.4, 9.2, 12.0])
        ax.fill_between(looks, 0, boundary, color=PURPLE, alpha=0.10)
        ax.plot(looks, boundary, color=PURPLE, linestyle="--", linewidth=1.5)
        ax.plot(looks, statistic, color=BLACK, linewidth=1.8)
        crossing = 5
        ax.vlines(
            looks[crossing],
            2.15,
            boundary[crossing],
            color=ORANGE,
            linestyle=":",
            linewidth=1.0,
            zorder=2,
        )
        ax.plot(
            looks[crossing],
            statistic[crossing],
            marker="v",
            color=ORANGE,
            markersize=6,
            linestyle="none",
            zorder=5,
        )
        ax.annotate(
            "first alarm",
            xy=(looks[crossing], statistic[crossing]),
            xytext=(2.25, 11.9),
            color=ORANGE,
            arrowprops={"arrowstyle": "->", "color": ORANGE, "linewidth": 0.9},
        )
        ax.text(1.15, boundary[0] + 1.3, "multiplier\ncutoff $c_j$", color=PURPLE, fontsize=10.6)
        ax.text(6.0, statistic[-1] + 0.8, r"$M_t$", color=BLACK, fontsize=10.6)
        output = FancyBboxPatch(
            (1.15, 0.35),
            5.8,
            1.6,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            linewidth=0.7,
            edgecolor=ORANGE,
            facecolor="#FFF3EA",
            zorder=3,
        )
        ax.add_patch(output)
        ax.text(
            3.98,
            1.15,
            r"$T$  |  $\widehat\tau_T$  |  direction",
            ha="center",
            va="center",
            fontsize=9.5,
            zorder=4,
        )
        ax.set_xlim(1, 7.15)
        ax.set_ylim(0, 14.0)
        ax.set_xlabel("planned look")
        ax.set_ylabel("scan statistic")
        ax.set_xticks(looks)
        ax.set_yticks([0, 4, 8, 12])
        # The long third-panel title otherwise touches the panel tag after
        # reduction to the manuscript's portrait width.
        ax.set_title(
            "Sequential\ncalibration",
            fontsize=9.5,
            fontweight="bold",
            pad=14,
            x=0.54,
        )
        panel(ax, "c", x=-0.035)

        # Keep a fixed canvas to preserve the panel spacing and physical font sizes.
        for suffix, dpi in (("pdf", None), ("png", 600)):
            fig.savefig(output_dir / f"Fig1_method_overview.{suffix}", dpi=dpi)
        plt.close(fig)


def analyze_paper_estimation(output: Path, figure_dir: Path) -> dict[str, pd.DataFrame]:
    """Export the current Figure 5 and its full-design point and interval tables."""
    tables, evidence = _compact_load_v2_postalarm(output)
    if evidence["status"] != "complete":
        raise EvidenceError("Figure 5 requires the complete independent and calibration estimation banks")
    destination = output / "analysis"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _compact_style()
    name = "CompactFig5_estimation"
    caption = _compact_v2_estimation_figure(tables, figure_dir)
    (figure_dir / f"{name}_caption.txt").write_text(caption + "\n", encoding="utf-8")
    for key in ("compact_v2_ls_summary", "compact_v2_interval_summary", "compact_v2_estimation_cells"):
        _atomic_csv(tables[key], destination / f"{key}.csv")
    _atomic_json({"status": "complete", "estimation_evidence": evidence,
                  "figure_sha256": _sha256(figure_dir / f"{name}.pdf"),
                  "analysis_source_sha256": _sha256(Path(__file__))},
                 destination / "paper_estimation_manifest.json")
    return tables


def _interval_atomic_json(path: Path, value: object) -> None:
    """Diagnostic JSON preserves mathematically defined infinite interval widths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=True)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _interval_flag(value):
    if isinstance(value, str):
        if value not in ('True', 'False', 'true', 'false'):
            raise ValueError('invalid Boolean field')
        return value.lower() == 'true'
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in (0, 1):
        return bool(value)
    raise ValueError('missing or malformed Boolean')


def _interval_transformed(frame, target):
    index = ESTIMATION_TARGETS.index(target)
    estimates = pd.to_numeric(frame['regularized_'+target+'_estimate'], errors='coerce').to_numpy(float)
    standard_errors = pd.to_numeric(frame[INTERVAL_SE_NAMES[index]], errors='coerce').to_numpy(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        center = np.log(estimates) if target == 'risk_ratio' else logit(estimates)
    point_valid = np.array([_interval_flag(value) for value in frame['regularized_'+target+'_point_valid']])
    available = point_valid & np.isfinite(center) & np.isfinite(standard_errors) & (standard_errors > 0)
    return center, standard_errors, available


def shortest_empirical_interval(cells, alarm_counts):
    """Existing tail_screen order-statistic optimization, with ALL-alarm ranks.

cells contains available finite roots only. Missing intervals cannot satisfy
the ceil(.95*N_alarms) requirement; calibration is infeasible if too few remain.
The original ascending-candidate/first-minimum tie convention is preserved.
"""
    ordered = [np.sort(np.asarray(cells[name], dtype=float)) for name in sorted(cells)]
    required = [math.ceil(.95*alarm_counts[name]) for name in sorted(cells)]
    if not ordered or any(m < 1 or len(values) < m for values, m in zip(ordered, required)):
        raise RuntimeError('95% per-cell calibration is infeasible with unavailable alarm intervals')
    if not all(np.all(np.isfinite(values)) for values in ordered):
        raise ValueError('calibration roots must be finite')
    max_lower = min(values[len(values)-m] for values, m in zip(ordered, required))
    roots = np.concatenate(ordered)
    lower = np.unique(roots[roots <= max_lower])
    upper = np.full(lower.shape, -np.inf)
    for values, m in zip(ordered, required):
        start = np.searchsorted(values, lower, side='left')
        if not np.all(start+m <= len(values)):
            raise RuntimeError('invalid global interval order rank')
        upper = np.maximum(upper, values[start+m-1])
    best = int(np.argmin(upper-lower))
    result = dict(lower=float(lower[best]), upper=float(upper[best]))
    for values, m in zip(ordered, required):
        if np.sum((values >= result['lower']) & (values <= result['upper'])) < m:
            raise RuntimeError('calibrated interval misses its per-cell target')
    return result


def truth_free_apply(center, se, available, target, cutoff):
    """No truth, scenario identity, or validation outcome enters this function."""
    center, se, available = np.asarray(center, float), np.asarray(se, float), np.asarray(available, bool)
    if (target not in ESTIMATION_TARGETS or center.ndim != 1 or se.shape != center.shape
            or available.shape != center.shape or set(cutoff) != {'lower', 'upper'}
            or not np.isfinite([cutoff['lower'], cutoff['upper']]).all() or cutoff['lower'] > cutoff['upper']
            or (available & (~np.isfinite(center) | ~np.isfinite(se) | (se <= 0))).any()):
        raise ValueError('invalid transformed interval inputs or cutoff pair')
    lower_t = center-cutoff['upper']*se
    upper_t = center-cutoff['lower']*se
    with np.errstate(over='ignore', invalid='ignore'):
        lower = np.exp(lower_t) if target == 'risk_ratio' else expit(lower_t)
        upper = np.exp(upper_t) if target == 'risk_ratio' else expit(upper_t)
    valid = available & ~np.isnan(lower) & ~np.isnan(upper) & (lower <= upper)
    return np.where(valid, lower, np.nan), np.where(valid, upper, np.nan), valid


def calibrate_interval_cutoffs(frame, provenance, output):
    path = output/'interval-calibration/CUTOFFS.json'
    alarm = frame[np.array([_interval_flag(v) for v in frame.alarm])].copy()
    names = sorted(frame.scenario.unique())
    counts = {name: int(np.sum(alarm.scenario == name)) for name in names}
    if any(n == 0 for n in counts.values()):
        raise RuntimeError('a training cell has no alarm interval denominator')
    cutoffs, diagnostics = {}, {}
    for target in ESTIMATION_TARGETS:
        center, se, available = _interval_transformed(alarm, target)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            truth = np.log(alarm[target].to_numpy(float)) if target == 'risk_ratio' else logit(alarm[target].to_numpy(float))
            roots = (center-truth)/se
        available &= np.isfinite(roots)
        cells = {name: roots[(alarm.scenario.to_numpy() == name) & available] for name in names}
        cutoff = shortest_empirical_interval(cells, counts)
        cutoffs[target] = cutoff
        diagnostics[target] = {name: dict(alarms=counts[name], available=len(cells[name]),
            required=math.ceil(.95*counts[name]), covered=int(np.sum((cells[name]>=cutoff['lower']) & (cells[name]<=cutoff['upper'])))) for name in names}
    record = dict(schema='ocfr-new-rule-global-shorter-cutoffs-v1', targets=cutoffs,
        training_source=provenance, training_diagnostics=diagnostics,
        algorithm='tail_screen.py shortest_empirical_interval; ceil(.95*ALL alarms) per cell',
        scope='finite-design empirical selected-alarm calibration; no general coverage guarantee',
        truth_used='training only; validation file has not been opened by this calibration operation')
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if (existing.get('schema') != record['schema'] or existing.get('targets') != cutoffs or
                existing.get('training_diagnostics') != diagnostics or
                existing.get('algorithm') != record['algorithm'] or
                not _interval_sources_match(existing.get('training_source', {}), provenance)):
            raise EvidenceError('refusing to overwrite different frozen interval cutoffs')
        # The private producer includes its snapshot/calibrator hashes. Preserve
        # those original provenance fields and bytes when adopting its artifact.
        record = existing
    else:
        _interval_atomic_json(path, record)
    frozen = dict(cutoffs_sha256=_sha256(path), source=record['training_source'])
    frozen_path = path.parent/'FROZEN.json'
    if frozen_path.exists():
        if json.loads(frozen_path.read_text(encoding='utf-8')) != frozen:
            raise EvidenceError('cutoff freeze record mismatch')
    else:
        _interval_atomic_json(frozen_path, frozen)
    return frozen


def apply_interval_cutoffs(frame, cutoffs, provenance, output, expected_hash):
    result = frame[['scenario', 'replicate', 'alarm']].copy()
    reports = {}
    for target in ESTIMATION_TARGETS:
        center, se, available = _interval_transformed(frame, target)
        low, high, valid = truth_free_apply(center, se, available, target, cutoffs[target])
        result[target+'_lower'], result[target+'_upper'], result[target+'_available'] = low, high, valid
        # Assessment is downstream of the fixed, truth-free interval construction.
        truth = frame[target].to_numpy(float)
        covered = valid & (low <= truth) & (truth <= high)
        with np.errstate(invalid='ignore'):
            width = high-low
        width[valid & np.isposinf(high)] = np.inf
        result[target+'_covered'], result[target+'_width'] = covered, width
        reports[target] = {}
        for name in sorted(frame.scenario.unique()):
            alarm = np.array([_interval_flag(v) for v in frame.alarm]) & (frame.scenario.to_numpy() == name)
            n = int(alarm.sum())
            use = alarm & valid
            reports[target][name] = dict(alarms=n, available=int(use.sum()), covered=int(np.sum(alarm & covered)),
                coverage=float(np.sum(alarm & covered)/n) if n else None,
                unavailable_counted_as_noncoverage=int(np.sum(alarm & ~valid)),
                infinite_widths=int(np.sum(use & np.isinf(width))),
                mean_available_width=float(np.mean(width[use])) if use.any() else None,
                median_available_width=float(np.median(width[use])) if use.any() else None)
    destination = output/'interval-validation'
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination/'intervals.csv.tmp'
    result.to_csv(temporary, index=False)
    os.replace(temporary, destination/'intervals.csv')
    record = dict(status='complete', stage='interval-validation', cutoff_sha256=expected_hash,
        source=provenance, coverage_scope='conditional on any first alarm, including premature alarms',
        intervals_truth_free=True, reports=reports,
        artifacts={'intervals.csv': _sha256(destination/'intervals.csv')})
    _interval_atomic_json(destination/'manifest.json', record)
    return record



def analyze_sensitivity_extension(base_output: Path, extension_output: Path, *,
        figure_dir: Path | None = None, base_source: Path | None = None,
        dispersion_output: Path | None = None) -> dict[str, pd.DataFrame]:
    """Combine independently verified producers for sensitivity reporting only.

    Original episode rows retain their producer digests. The reporting union
    lives under the extension output; it never replaces the parent experiment.
    """
    from simulation import load_sensitivity_extension

    base = Path(base_output).resolve()
    if base.name == "evaluation":
        base = base.parent
    extension = Path(extension_output).resolve()
    if extension.is_relative_to(base) or base.is_relative_to(extension):
        raise EvidenceError("sensitivity analysis output must be disjoint from the parent study")
    destination = extension / "analysis"
    report_path = destination / "sensitivity_analysis_manifest.json"
    report_path.unlink(missing_ok=True)
    # This checks the frozen threshold producer and numerical compatibility,
    # separately from each stage's complete-grid and raw-file checks below.
    _, validated_rows, declared = load_sensitivity_extension(
        extension, base_output=base, base_source=base_source)
    del validated_rows
    original, parent = _compact_load_stage(base / "evaluation", "evaluation")
    extra, manifest = _compact_load_stage(extension / "sensitivity-extension", "sensitivity-extension")
    if manifest != declared or parent["protocol"] != "compact-v2":
        raise EvidenceError("sensitivity producer identity changed during validation")
    if set(original.scenario) & set(extra.scenario):
        raise EvidenceError("sensitivity extension duplicates a parent condition")
    for frame, name, producer in ((original, "parent", parent), (extra, "extension", manifest)):
        frame["evidence_source"] = name
        frame["evidence_stage"] = producer["stage"]
        frame["producer_scientific_digest"] = producer["scientific_digest"]
        frame["stage"] = "evaluation"  # reporting category, not a new producer identity
    union = pd.concat([original, extra], ignore_index=True)
    if dispersion_output is not None:
        union = union[~union.misspecification.str.startswith("exposure_scale_")].copy()
    if union.duplicated(["scenario", "method", "replicate"]).any():
        raise EvidenceError("duplicate sensitivity reporting episodes")
    tables = compact_summaries(union)
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        _atomic_csv(table, destination / f"{name}.csv")
    roster = union.groupby(["evidence_source", "evidence_stage", "producer_scientific_digest",
        "scenario", "method"], as_index=False).agg(requested_rows=("replicate", "size"))
    _atomic_csv(roster, destination / "sensitivity_evidence_sources.csv")
    figures = Path(figure_dir) if figure_dir is not None else destination / "figures"
    dispersion_summary = None
    dispersion_manifest = None
    companion_names = []
    if dispersion_output is not None:
        from simulation import load_dispersion_sensitivity
        _, _, dispersion_manifest = load_dispersion_sensitivity(dispersion_output,
            base_output=base, base_source=base_source)
        dispersion_rows, checked = _compact_load_stage(Path(dispersion_output) / "dispersion-sensitivity", "dispersion-sensitivity")
        if checked != dispersion_manifest:
            raise EvidenceError("dispersion producer changed during validation")
        dispersion_tables = compact_summaries(dispersion_rows)
        dispersion_summary = dispersion_tables["compact_summary"]
        _compact_dispersion_grid(dispersion_summary, _compact_method_contract("compact-v2", "evaluation"))
        for table_name, table in dispersion_tables.items():
            _atomic_csv(table, destination / f"dispersion_{table_name}.csv")
        _atomic_csv(dispersion_summary[dispersion_summary.metric.eq("alarm_probability") &
            np.isclose(dispersion_summary.risk_ratio, 1.)], destination / "dispersion_null_audit.csv")
        companion_names = _compact_dispersion_companions(dispersion_summary, figures)
    name, caption = _compact_sensitivity_figure(tables["compact_summary"], figures,
        protocol="compact-v2", dispersion=dispersion_summary)
    counts = sorted(int(n) for n in union.groupby(["scenario", "method"]).size().unique())
    caption += f" Requested base/extension episodes per condition: {', '.join(map(str, counts))}."
    if dispersion_summary is not None:
        caption += f" Dispersion episodes per condition: {', '.join(str(int(n)) for n in sorted(dispersion_summary.n.unique()))}."
    (figures / f"{name}_caption.txt").write_text(caption + "\n", encoding="utf-8")
    inputs = {
        "parent_evaluation": {"manifest_sha256": _sha256(base / "evaluation" / MANIFEST_FILE),
            "raw_sha256": _sha256(base / "evaluation" / RAW_FILE),
            "scientific_digest": parent["scientific_digest"], "method_rows": len(original)},
        "sensitivity_extension": {
            "manifest_sha256": _sha256(extension / "sensitivity-extension" / MANIFEST_FILE),
            "raw_sha256": _sha256(extension / "sensitivity-extension" / RAW_FILE),
            "scientific_digest": manifest["scientific_digest"], "method_rows": len(extra)},
    }
    report = {"status": "complete", "analysis": "sensitivity-extension", "protocol": "compact-v2",
        "analysis_source_sha256": _sha256(Path(__file__)), "sources": inputs,
        "dispersion_producer": dispersion_manifest,
        "parent_provenance": manifest["parent_provenance"],
        "reporting_union_method_rows": len(union), "reporting_union_conditions": int(union.scenario.nunique()),
        "new_conditions": int(extra.scenario.nunique()), "failed_method_rows": int(union.failed.sum()),
        "producer_digests_preserved": True, "parent_outputs_modified": False,
        "denominator": "all requested episodes; premature and absent alarms are not useful detections",
        "delay_grid": [11, 12, 13, 14, 15, 16, 17],
        "observed_case_grid": [.75, .9, 1., 1.1, 1.25] if dispersion_output is None else [],
        "reported_conditions_including_dispersion": int(union.scenario.nunique()) + (8 if dispersion_output is not None else 0),
        "dispersion_grid": [5, 10, 20, 40] if dispersion_output is not None else [],
        "publication_export": _compact_export_metadata(figures, [name] + companion_names, minimum_font_points=9.5),
        "artifacts": {path.name: _sha256(path) for path in destination.glob("*.csv")},
    }
    _interval_atomic_json(report_path, report)
    return tables


def _load_paired_detection_stage(output: Path, stage: str) -> tuple[pd.DataFrame, dict]:
    from config import PAIRED_DETECTION_VERSION

    directory = output / stage
    manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    if (manifest.get("protocol") != PAIRED_DETECTION_VERSION
            or manifest.get("stage") != stage or manifest.get("status") != "complete"):
        raise EvidenceError(f"{directory}: expected complete paired detection evidence")
    path = directory / RAW_FILE
    if _sha256(path) != _manifest_file_hash(manifest, RAW_FILE):
        raise EvidenceError(f"{directory}: replicate hash mismatch")
    frame = pd.read_csv(path, float_precision="round_trip",
                        dtype={"case_hash": "string", "death_hash": "string"})
    required = {"scenario", "family", "method", "replicate", "status", "split",
                "null_family", "case_hash", "death_hash", "raw_episode_max"}
    if stage != "calibration":
        required |= {"alarm_day", "tau_hat_at_alarm", "true_change", "horizon",
                     "risk_ratio", "misspecification", "critical_value"}
    if not required.issubset(frame.columns) or frame.empty:
        raise EvidenceError(f"{directory}: missing paired evidence fields {sorted(required - set(frame))}")
    if frame.duplicated(["scenario", "replicate", "method"]).any():
        raise EvidenceError(f"{directory}: duplicated episode-method records")
    expected_methods = set(COMPACT_PAIRED_METHODS)
    if stage == "calibration":
        expected_methods.remove("OCFR")
    if set(manifest["methods"]) != expected_methods:
        raise EvidenceError(f"{directory}: paired method contract disagrees with stage")
    counts = manifest["replications_by_scenario"]
    definitions = manifest["scientific_config"]["scenarios"]
    expected_splits = {
        "calibration": {"calibration"}, "audit": {"size_audit"},
        "evaluation": {"holdout", "alternative"},
        "sensitivity-extension": {"holdout", "alternative"},
        "dispersion-sensitivity": {"alternative"},
    }
    expected_names = {s["name"] for s in definitions if s["split"] in expected_splits[stage]}
    if set(frame.scenario) != set(counts) or set(counts) != expected_names:
        raise EvidenceError(f"{directory}: paired scenario roster disagrees with manifest")
    for name, group in frame.groupby("scenario", sort=False):
        n = counts[name]
        split = str(group.split.iloc[0])
        if (type(n) is not int or n != _compact_declared_n(manifest, split)
                or set(group.method) != expected_methods):
            raise EvidenceError(f"{directory}/{name}: invalid replication or method roster")
        for _, episodes in group.groupby("method"):
            if len(episodes) != n or set(episodes.replicate) != set(range(n)):
                raise EvidenceError(f"{directory}/{name}: incomplete episode identifiers")
        for key in ("case_hash", "death_hash"):
            if group[key].isna().any() or group.groupby("replicate")[key].nunique().ne(1).any():
                raise EvidenceError(f"{directory}/{name}: unpaired {key} streams")
        metadata = ["family", "null_family", "split"]
        if stage != "calibration":
            metadata += ["true_change", "horizon", "risk_ratio", "misspecification"]
        for key in metadata:
            if group[key].nunique(dropna=False) != 1:
                raise EvidenceError(f"{directory}/{name}: inconsistent {key}")
    if not frame.status.eq("success").all():
        raise EvidenceError(f"{directory}: incomplete or failed episode computation")
    if stage == "calibration":
        return frame, manifest
    source_maxima = frame[frame.method.isin(("OCFR", ORACLE_OCFR))].pivot(
        index=["scenario", "replicate"], columns="method", values="raw_episode_max")
    practical, oracle = (pd.to_numeric(source_maxima[method], errors="coerce")
                          for method in ("OCFR", ORACLE_OCFR))
    if practical.isna().any() or np.isposinf(practical).any() or not np.array_equal(practical, oracle):
        raise EvidenceError(f"{directory}: paired OCFR procedures do not share stored raw maxima")
    return _compact_episode_metrics(frame), manifest


def _validate_paired_thresholds(thresholds: pd.DataFrame, training: pd.DataFrame, alpha: float) -> dict:
    keys = ["null_family", "method"]
    if thresholds.duplicated(keys).any():
        raise EvidenceError("paired thresholds contain duplicate family-method rows")
    rows = thresholds.set_index(keys, verify_integrity=True)
    if set(rows.index) != set(training.groupby(keys).groups):
        raise EvidenceError("paired thresholds do not cover the calibration family-method grid")
    cutoffs = {}
    for key, group in training.groupby(keys):
        maxima = pd.to_numeric(group.raw_episode_max, errors="coerce").to_numpy(float)
        n = len(maxima)
        rank = int(np.ceil((n + 1) * (1 - alpha)))
        row = rows.loc[key]
        if (np.isnan(maxima).any() or np.isposinf(maxima).any()
                or row.n_calibration != n or row.order_rank != rank or not 1 <= rank <= n):
            raise EvidenceError(f"{key}: invalid calibration maxima or order rank")
        cutoff = np.partition(maxima, rank - 1)[rank - 1]
        if not np.isfinite(cutoff) or cutoff != row.critical_value:
            raise EvidenceError(f"{key}: calibration cutoff does not match the declared order statistic")
        cutoffs[key] = cutoff
    return cutoffs


def analyze_paired_detection(output: Path, figure_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Summarize paired detector rules without consuming post-alarm estimation banks."""
    output = Path(output).resolve()
    destination = output / "analysis"
    report_path = destination / "paired_detection_analysis_manifest.json"
    report_path.unlink(missing_ok=True)
    stages = ("calibration", "audit", "evaluation", "sensitivity-extension", "dispersion-sensitivity")
    frames, manifests = {}, {}
    for stage in stages:
        frame, manifest = _load_paired_detection_stage(output, stage)
        frame["evidence_stage"] = stage
        frame["stage"] = "evaluation" if stage == "sensitivity-extension" else stage
        frames[stage], manifests[stage] = frame, manifest

    threshold_path = output / "calibration" / "thresholds.csv"
    threshold_hash = _sha256(threshold_path)
    calibration = manifests["calibration"]
    if threshold_hash != _manifest_file_hash(calibration, threshold_path.name):
        raise EvidenceError("paired calibration threshold hash mismatch")
    digest = hashlib.sha256(_canonical_json({"protocol": calibration["protocol"],
        "reference": calibration["source"], "scientific": calibration["scientific_config"]
        }).encode("ascii")).hexdigest()[:20]
    for stage, manifest in manifests.items():
        if (manifest.get("scientific_digest") != digest or manifest.get("crossing") != "strict >"
                or manifest.get("threshold_artifact", {}).get("sha256") != threshold_hash):
            raise EvidenceError(f"{stage}: paired study identity or frozen threshold hash mismatch")
    thresholds = pd.read_csv(threshold_path, float_precision="round_trip")
    cutoffs = _validate_paired_thresholds(thresholds, frames["calibration"],
                                        float(calibration["scientific_config"]["methods"]["alpha"]))
    all_rows = pd.concat([frames[stage] for stage in stages[1:]], ignore_index=True)
    if all_rows.duplicated(["stage", "scenario", "replicate", "method"]).any():
        raise EvidenceError("paired stage union contains duplicate episodes")
    for key, group in all_rows.groupby(["null_family", "method"]):
        if not group.critical_value.eq(cutoffs.get(key, 0.)).all():
            raise EvidenceError(f"{key}: evaluated threshold differs from the frozen cutoff")
        if key[1] == ORACLE_OCFR and not group.raw_episode_max.gt(cutoffs[key]).eq(group.any_alarm).all():
            raise EvidenceError(f"{key}: oracle alarms disagree with strict maximum crossing")
    for stage, expected in (("audit", {"size_audit"}),
                            ("evaluation", {"holdout", "alternative"}),
                            ("sensitivity-extension", {"holdout", "alternative"}),
                            ("dispersion-sensitivity", {"holdout", "alternative"})):
        frame = frames[stage]
        if set(frame.split) != expected or not np.array_equal(
                frame.alternative, ~np.isclose(frame.risk_ratio, 1.)):
            raise EvidenceError(f"{stage}: statistical split and risk-ratio truth disagree")
    evaluation = all_rows[all_rows.stage.ne("dispersion-sensitivity")]
    tables = compact_summaries(evaluation)
    tables["compact_calibration"] = thresholds
    tables["compact_localization"] = compact_localization_summaries(evaluation)
    for name, table in compact_summaries(frames["dispersion-sensitivity"]).items():
        tables[f"dispersion_{name}"] = table
    dispersion = tables["dispersion_compact_summary"]
    _compact_dispersion_grid(dispersion, COMPACT_PAIRED_METHODS)
    tables["dispersion_null_audit"] = dispersion[dispersion.metric.eq("alarm_probability") &
                                               np.isclose(dispersion.risk_ratio, 1.)]
    tables["paired_detection_evidence_sources"] = all_rows.groupby(
        ["evidence_stage", "stage", "scenario", "method"], as_index=False).agg(
        requested_rows=("replicate", "size"))

    # The historical runners retain four ascertainment conditions. The paper's
    # scenario allocation and evaluation summary include the other 41 cells.
    paper_cells = all_rows.loc[all_rows.stage.ne("audit")
                               & ~all_rows.misspecification.str.startswith("exposure_scale_")
                               & all_rows.method.eq("OCFR")]
    design_columns = ["scenario", "family", "split", "horizon", "change_day",
                      "risk_ratio", "phi", "true_delay_mean_days", "assumed_delay_mean_days",
                      "misspecification"]
    tables["evaluation_design"] = (paper_cells.groupby(design_columns, as_index=False)
                                   .agg(n=("replicate", "size")).rename(columns={"scenario": "name"}))
    if len(tables["evaluation_design"]) != 41:
        raise EvidenceError("current paper evaluation requires 41 retained conditions")
    paper_summary = pd.concat([tables["compact_summary"], dispersion], ignore_index=True)
    tables["evaluation_summary"] = paper_summary.loc[paper_summary.scenario.isin(paper_cells.scenario.unique())]
    tables["null_audit"] = tables["compact_summary"].loc[
        tables["compact_summary"].stage.eq("audit") & tables["compact_summary"].metric.eq("alarm_probability")]

    figures = Path(figure_dir).resolve() if figure_dir is not None else destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _compact_style()
    summary = tables["compact_summary"]
    audit = summary[summary.stage.eq("audit") & summary.metric.eq("alarm_probability")]
    evaluated = summary[summary.stage.eq("evaluation")]
    power = evaluated[evaluated.misspecification.eq("on_model") & evaluated.metric.eq("useful_power")]
    curves = tables["compact_cumulative_detection"]
    _compact_comparison_figure(audit, curves, figures, protocol="compact-v2")
    captions = {"CompactFig2_comparison": (
        "A: independent null audit in each incidence family. Bars are exact 95% binomial Monte "
        "Carlo intervals; dashed line marks .05. B-D: cumulative useful detection in steady, "
        "sparse and wave incidence at RR=1.5 and change day 92. Markers show cumulative useful "
        "detection within 28 days. Pointwise exact intervals accompany the curves in the data table.")}
    captions["CompactFig3_detection"] = _compact_v2_detection_figure(power, curves, figures)
    captions["CompactFig4_localization"] = _compact_localization_figure(tables["compact_localization"], figures)
    name, caption = _compact_sensitivity_figure(evaluated, figures, dispersion=dispersion)
    captions[name] = caption
    companion_names = _compact_dispersion_companions(dispersion, figures)
    evaluation_n = ", ".join(str(int(n)) for n in sorted(evaluated.n.unique()))
    audit_n = ", ".join(str(int(n)) for n in sorted(audit.n.unique()))
    dispersion_n = ", ".join(str(int(n)) for n in sorted(dispersion.n.unique()))
    for name, caption in captions.items():
        caption += f" Evaluation uses n={evaluation_n} independent episodes per condition."
        if name == "CompactFig2_comparison":
            caption += f" The independent null audit uses n={audit_n} episodes per family."
        if name == "CompactFig6_sensitivity":
            caption += f" The independent dispersion bank uses n={dispersion_n} episodes per condition."
        (figures / f"{name}_caption.txt").write_text(caption + "\n", encoding="utf-8")

    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        _atomic_csv(table, destination / f"{name}.csv")
    _atomic_json({"status": "complete", "analysis": "paired-detection",
        "scientific_digest": digest, "thresholds_sha256": threshold_hash,
        "methods": list(COMPACT_PAIRED_METHODS), "analysis_source_sha256": _sha256(Path(__file__)),
        "confidence_level": .95, "probability_interval": "two-sided Clopper-Pearson",
        "paired_uncertainty": "sample SD of within-episode indicator differences / sqrt(n)",
        "probability_denominator": "all requested episodes",
        "localization_mean_denominator": "each detector's useful first alarms",
        "source_manifest_sha256": {stage: _sha256(output / stage / MANIFEST_FILE) for stage in stages},
        "files": {f"{name}.csv": _sha256(destination / f"{name}.csv") for name in tables},
        "figure_files": {f"{name}.pdf": _sha256(figures / f"{name}.pdf")
                         for name in list(captions) + companion_names},
        "publication_export": _compact_export_metadata(figures, list(captions) + companion_names,
                                                       minimum_font_points=9)}, report_path)
    return tables


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).parent / "outputs"
    parser.add_argument("--input", type=Path, default=base / "simulation")
    parser.add_argument("--output", type=Path, default=base / "analysis")
    parser.add_argument("--timely-window", type=int, default=28)
    parser.add_argument("--protocol", choices=("auto", "legacy", "compact-v1", "compact-v2"), default="auto")
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--paired-detection", action="store_true",
                        help="analyze the complete paired practical and oracle OCFR detection package")
    parser.add_argument("--sensitivity-extension", type=Path,
        help="combine a completed ten-cell sensitivity extension with --input parent study")
    parser.add_argument("--dispersion-study", type=Path, help="completed dispersion sensitivity study; requires --sensitivity-extension")
    parser.add_argument("--base-code", type=Path,
        help="frozen parent core sources when sensitivity producers use different orchestration versions")
    interval = parser.add_mutually_exclusive_group()
    interval.add_argument("--calibrate-intervals", action="store_true",
        help="freeze new-policy global shorter cutoffs from the complete selected training bank only")
    interval.add_argument("--apply-intervals", action="store_true",
        help="apply a verified frozen cutoff artifact to the independent selected-fit bank")
    parser.add_argument("--expected-cutoff-sha256",
        help="required frozen CUTOFFS.json hash; checked before independent validation is opened")

    parser.add_argument("--estimation-only", action="store_true",
                        help="refresh only Poisson comparison tables and Figures 5/S2/S3")
    parser.add_argument("--no-poisson-s3", action="store_true", help="omit optional Poisson supplementary figure")
    parser.add_argument("--ls-estimation-only", action="store_true",
                        help="refresh only LS comparison tables and supplementary Figure S4")
    parser.add_argument("--ls-main-figure", action="store_true",
                        help="export only main Figure 5 with LS points and unchanged OCFR coverage")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.paired_detection:
        if any((args.sensitivity_extension, args.dispersion_study, args.base_code,
                args.calibrate_intervals, args.apply_intervals, args.estimation_only,
                args.ls_estimation_only, args.ls_main_figure)):
            raise EvidenceError("paired detection analysis consumes the paired stage package only")
        derived = analyze_paired_detection(args.input, figure_dir=args.figure_dir)
        for name, table in derived.items():
            print(f"{name}: {len(table)} derived rows")
        print(f"wrote paired detection report to {(args.input / 'analysis').resolve()}")
        return 0
    if args.sensitivity_extension is not None:
        if args.protocol in {"legacy", "compact-v1"} or any((args.calibrate_intervals,
                args.apply_intervals, args.estimation_only, args.ls_estimation_only, args.ls_main_figure)):
            raise EvidenceError("sensitivity extension analysis requires its separate compact-v2 reporting path")
        derived = analyze_sensitivity_extension(args.input, args.sensitivity_extension,
            figure_dir=args.figure_dir, base_source=args.base_code, dispersion_output=args.dispersion_study)
        for name, table in derived.items():
            print(f"{name}: {len(table)} derived rows")
        print(f"wrote source-bound sensitivity report to {(args.sensitivity_extension / 'analysis').resolve()}")
        return 0
    if args.dispersion_study is not None:
        raise EvidenceError("--dispersion-study requires --sensitivity-extension")
    if args.base_code is not None:
        raise EvidenceError("--base-code is only used with --sensitivity-extension")
    if args.calibrate_intervals or args.apply_intervals:
        if args.protocol in {'legacy', 'compact-v1'}:
            raise EvidenceError('interval calibration/application requires compact-v2')
        if args.estimation_only or args.ls_estimation_only or args.ls_main_figure:
            raise EvidenceError("interval calibration/application cannot be combined with legacy estimation refresh")
        root = args.input.parent if args.input.name == "evaluation" else args.input
        record = (run_interval_calibration(root) if args.calibrate_intervals else
                  run_interval_application(root, args.expected_cutoff_sha256))
        print(json.dumps({"status": "complete", "operation": "calibrate" if args.calibrate_intervals else "apply",
            "cutoff_sha256": record.get("cutoffs_sha256", record.get("cutoff_sha256"))}, sort_keys=True))
        return 0
    compact = args.protocol in {"compact-v1", "compact-v2"}
    if args.protocol == "auto":
        manifest_file = args.input / MANIFEST_FILE
        if not manifest_file.exists():
            manifest_file = args.input / "evaluation" / MANIFEST_FILE
        if manifest_file.exists():
            compact = json.loads(manifest_file.read_text(encoding="utf-8")).get("protocol") in {"compact-v1", "compact-v2"}
    if (args.estimation_only or args.ls_estimation_only or args.ls_main_figure) and not compact:
        raise EvidenceError("estimation-only analysis requires compact-v1 evidence")
    if args.estimation_only or args.ls_estimation_only or args.ls_main_figure:
        source_root = args.input.parent if args.input.name == "evaluation" else args.input
        source_protocol = json.loads((source_root / "evaluation" / MANIFEST_FILE).read_text(encoding="utf-8")).get("protocol")
        if source_protocol == "compact-v2":
            raise EvidenceError("v2 estimation refresh requires new source-bound fits and independently calibrated intervals; legacy refresh is unavailable")
    if sum((args.estimation_only, args.ls_estimation_only, args.ls_main_figure)) > 1:
        raise EvidenceError("choose one limited estimation refresh")
    derived = (analyze_compact_ls_main_figure(args.input, figure_dir=args.figure_dir) if args.ls_main_figure else
        analyze_compact_ls_estimation(args.input, figure_dir=args.figure_dir) if args.ls_estimation_only else
        analyze_compact_poisson_estimation(args.input, figure_dir=args.figure_dir,
                include_s3=not args.no_poisson_s3) if args.estimation_only else
        analyze_compact_simulation(args.input, figure_dir=args.figure_dir) if compact
        else analyze(args.input, args.output, timely_window=args.timely_window))
    for name, table in derived.items():
        print(f"{name}: {len(table)} derived rows")
    compact_root = args.input.parent if args.input.name == "evaluation" else args.input
    print(f"wrote validated figures, tables, and manifest to {(compact_root / 'analysis' if compact else args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
