"""Run the frozen OCFR calibration, holdout and alternative experiment.

The runner uses atomic per-replicate caches, strict cache validation and
versioned component streams. ``demo`` and ``full`` differ only in replication
count, so the first ten full-run replicates are exactly the demo replicates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Literal

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd
from scipy.stats import beta

from config import (ExperimentConfig, SimulationScenario, get_experiment_config,
                    get_size_control_experiment_config, get_compact_experiment_config,
                    COMPACT_METHODS, COMPACT_EMPIRICAL_METHODS, compact_stage_methods,
                    get_compact_release_study_config, get_post_alarm_validation_config)
from methods import (gamma_delay_pmf, make_correlated_epidemic_cases, make_sampler,
                     monte_carlo_critical_value, multiplier_statistics_from_draws,
                     published_benchmark_statistics, report_day_score_scan, score_scan,
                     simulate_nb_deaths, GaussianSpendingBoundary, gaussian_multiplier_bank)

SCHEMA_VERSION = 3
METHODS = ("OCFR", "OCFR-alpha-matched", "Report-day score",
           "NB-GLR", "Page-CUSUM", "FOCuS")
ResetScope = Literal["none", "selected", "derived", "all"]

def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _diagnostic_json(value: object) -> str:
    """Serialize trace/payload diagnostics, retaining mathematical infinities.

    Scientific configuration and digest material use the strict canonical
    serializer above. Trace readers already accept Python JSON's Infinity
    representation for unavailable statistics and unresolved boundaries.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=True)

def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = _canonical_json({"dtype": array.dtype.str, "shape": array.shape})
    return _sha256_bytes(header.encode("ascii") + array.tobytes())

def _code_hash() -> tuple[str, dict[str, str]]:
    """Hash executable scientific sources, independent of checkout location."""
    root = Path(__file__).resolve().parent
    candidates = [root / "config.py", root / "simulation.py", root / "methods.py"]
    csrc = root / "csrc"
    if csrc.is_dir():
        candidates.extend(sorted(p for p in csrc.rglob("*") if p.is_file()))
    records = {p.relative_to(root).as_posix(): _file_hash(p) for p in candidates}
    return _sha256_bytes(_canonical_json(records).encode("ascii")), records

def derive_seed(master_seed: int, contract_version: str, split: str,
                stream_group: str, replicate: int, component: str) -> int:
    """Derive a stable 64-bit seed without mode, replication count or jobs."""
    if replicate < 0:
        raise ValueError("replicate must be non-negative")
    if component not in {"case", "death", "multiplier"}:
        raise ValueError("component must be case, death or multiplier")
    payload = {"component": component, "contract": contract_version,
               "group": stream_group, "master_seed": int(master_seed),
               "replicate": int(replicate), "split": split}
    return int.from_bytes(hashlib.sha256(_canonical_json(payload).encode("ascii")).digest()[:8],
                          "little")

def _scientific_digest(config: ExperimentConfig) -> tuple[str, str, dict[str, str]]:
    code_hash, records = _code_hash()
    payload = {"schema_version": SCHEMA_VERSION, "code_hash": code_hash,
               "scientific": config.scientific_payload()}
    return _sha256_bytes(_canonical_json(payload).encode("ascii"))[:20], code_hash, records

def _scenario_digest(scenario: SimulationScenario) -> str:
    from dataclasses import asdict
    return _sha256_bytes(_canonical_json(asdict(scenario)).encode("ascii"))

def _planned_looks(scenario: SimulationScenario,
                   config: ExperimentConfig) -> tuple[int, ...]:
    first = config.methods.reference_end + config.methods.look_every
    looks = tuple(range(first, scenario.horizon + 1, config.methods.look_every))
    return looks if looks and looks[-1] == scenario.horizon else looks + (scenario.horizon,)

def _component_seeds(scenario: SimulationScenario, config: ExperimentConfig,
                     replicate: int) -> dict[str, int]:
    common = (config.master_seed, config.rng_contract_version, scenario.split)
    return {
        "case": derive_seed(*common, scenario.case_stream_group, replicate, "case"),
        "death": derive_seed(*common, scenario.outcome_stream_group, replicate, "death"),
        "multiplier": derive_seed(*common, scenario.name, replicate, "multiplier"),
    }

def _generate_components(scenario: SimulationScenario, config: ExperimentConfig,
                         replicate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                   dict[str, object]]:
    if config.protocol in {"compact-v1", "compact-v2"}:
        return _compact_generate_components(scenario, config, replicate)
    seeds = _component_seeds(scenario, config, replicate)
    case_sampler = make_sampler(seeds["case"], backend=config.methods.backend)
    cases = make_correlated_epidemic_cases(
        scenario.horizon, scenario.profile, scenario.case_level, case_sampler,
        autoregression=config.dgp.autoregression,
        process_sd=config.dgp.process_sd, case_shape=config.dgp.case_shape,
        weekday_amplitude=config.dgp.weekday_amplitude)
    true_delay = gamma_delay_pmf(
        scenario.true_delay_mean_days, shape=scenario.true_delay_shape,
        maximum_lag=config.dgp.maximum_delay)
    assumed_delay = gamma_delay_pmf(
        scenario.assumed_delay_mean_days, shape=scenario.assumed_delay_shape,
        maximum_lag=config.dgp.maximum_delay)
    cfr = np.full(scenario.horizon, scenario.pi_left, dtype=float)
    if scenario.split == "alternative":
        cfr[scenario.true_change:] = scenario.pi_right
    death_sampler = make_sampler(seeds["death"], backend=config.methods.backend)
    deaths = simulate_nb_deaths(cases, true_delay, cfr, scenario.phi, death_sampler)
    stream_manifest = {
        "rng_contract_version": config.rng_contract_version,
        "seeds": seeds,
        "groups": {"case": scenario.case_stream_group,
                   "death": scenario.outcome_stream_group,
                   "multiplier": scenario.name},
        "backends": {"case": case_sampler.backend, "death": death_sampler.backend},
        "hashes": {"cases": _array_hash(cases), "deaths": _array_hash(deaths),
                   "true_delay": _array_hash(true_delay),
                   "assumed_delay": _array_hash(assumed_delay)},
    }
    return cases, deaths, assumed_delay, stream_manifest

def generate_simulation_data(scenario: SimulationScenario, config: ExperimentConfig,
                             replicate: int) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Compatibility API returning cases, deaths, change and case backend."""
    cases, deaths, _, streams = _generate_components(scenario, config, replicate)
    return cases, deaths, scenario.true_change, str(streams["backends"]["case"])

def _evaluate_one(scenario: SimulationScenario, replicate: int,
                  config: ExperimentConfig, digest: str) -> dict[str, object]:
    started = time.perf_counter()
    scenario_hash = _scenario_digest(scenario)
    seeds = _component_seeds(scenario, config, replicate)
    try:
        cases, deaths, delay, stream_manifest = _generate_components(
            scenario, config, replicate)
        looks = _planned_looks(scenario, config)
        multiplier_sampler = make_sampler(seeds["multiplier"],
                                          backend=config.methods.backend)
        multipliers = multiplier_sampler.standard_normal(
            (config.methods.n_multiplier, scenario.horizon))
        stream_manifest["backends"]["multiplier"] = multiplier_sampler.backend
        stream_manifest["hashes"]["multipliers"] = _array_hash(multipliers)
        local_alpha = config.methods.alpha / len(looks)
        traces: dict[str, list[dict[str, object]]] = {name: [] for name in METHODS}
        for prefix in looks:
            day = prefix - 1
            scan = score_scan(cases[:prefix], deaths[:prefix], delay,
                              use_lindeberg_gate=config.methods.use_lindeberg_gate,
                              backend=config.methods.backend)
            raw, boundary, excess = float("nan"), float("inf"), float("nan")
            if scan.candidates.size and np.isfinite(scan.statistic):
                maxima = multiplier_statistics_from_draws(
                    multipliers[:, :prefix], scan.influence,
                    backend=config.methods.backend)
                boundary = monte_carlo_critical_value(maxima, local_alpha)
                raw, excess = float(scan.statistic), float(scan.statistic - boundary)
            base = {"day": day, "raw_statistic": raw, "boundary": boundary,
                    "statistic": excess, "tau_hat": scan.tau_hat,
                    "direction": int(scan.direction),
                    "fit_status": scan.null_fit.status}
            traces["OCFR"].append(dict(base))
            traces["OCFR-alpha-matched"].append(dict(base))
            report_scan = report_day_score_scan(
                scan, deaths[:prefix],
                use_lindeberg_gate=config.methods.use_lindeberg_gate,
            )
            report_raw, report_boundary, report_excess = (
                float("nan"), float("inf"), float("nan")
            )
            if report_scan.candidates.size and np.isfinite(report_scan.statistic):
                report_maxima = multiplier_statistics_from_draws(
                    multipliers[:, :prefix], report_scan.influence,
                    backend=config.methods.backend,
                )
                report_boundary = monte_carlo_critical_value(
                    report_maxima, local_alpha
                )
                report_raw = float(report_scan.statistic)
                report_excess = report_raw - report_boundary
            traces["Report-day score"].append({
                "day": day, "raw_statistic": report_raw,
                "boundary": report_boundary, "statistic": report_excess,
                "tau_hat": report_scan.tau_hat,
                "direction": int(report_scan.direction),
                "fit_status": report_scan.null_fit.status,
            })
            try:
                benchmark = published_benchmark_statistics(
                    cases[:prefix], deaths[:prefix], delay,
                    reference_end=config.methods.reference_end)
                mapped = {"NB-GLR": benchmark["NB-surveillance-GLR"],
                          "Page-CUSUM": benchmark["Page-CUSUM"],
                          "FOCuS": benchmark["FOCuS-working-model"]}
                for name, result in mapped.items():
                    traces[name].append({"day": day,
                        "raw_statistic": float(result.statistic), "boundary": 0.,
                        "statistic": float(result.statistic),
                        "tau_hat": result.estimated_cohort_change_day,
                        "direction": int(result.direction), "fit_status": "success"})
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                for name in METHODS[3:]:
                    traces[name].append({"day": day, "raw_statistic": float("nan"),
                        "boundary": 0., "statistic": float("nan"), "tau_hat": None,
                        "direction": 0, "fit_status": f"failed:{type(exc).__name__}"})
        rows: list[dict[str, object]] = []
        for method, trace in traces.items():
            finite = [x for x in trace if np.isfinite(float(x["statistic"]))]
            best = max(finite, key=lambda x: float(x["statistic"])) if finite else None
            rows.append({
                "scenario": scenario.name, "family": scenario.family,
                "null_family": scenario.null_family, "split": scenario.split,
                "purpose": scenario.purpose, "motivation": scenario.motivation,
                "misspecification": scenario.misspecification, "method": method,
                "score_definition": ("GP-boundary crossing" if method == "OCFR"
                    else "statistic-minus-GP-boundary" if method == "OCFR-alpha-matched"
                    else "episode maximum scan statistic"),
                "replicate": replicate, "case_seed": seeds["case"],
                "death_seed": seeds["death"], "multiplier_seed": seeds["multiplier"],
                "seed": seeds["death"],
                "case_stream_group": scenario.case_stream_group,
                "outcome_stream_group": scenario.outcome_stream_group,
                "sampler_backend": stream_manifest["backends"]["death"],
                "multiplier_backend": stream_manifest["backends"]["multiplier"],
                "horizon": scenario.horizon, "profile": scenario.profile,
                "case_level": scenario.case_level, "pi_left": scenario.pi_left,
                "pi_right": scenario.pi_right, "risk_ratio": scenario.risk_ratio,
                "phi": scenario.phi, "true_change": scenario.true_change,
                "true_change_day": scenario.true_change + 1,
                "true_delay_mean_days": scenario.true_delay_mean_days,
                "assumed_delay_mean_days": scenario.assumed_delay_mean_days,
                "episode_statistic": float(best["statistic"]) if best else float("nan"),
                "max_day": int(best["day"]) if best else None,
                "tau_hat_at_max": best["tau_hat"] if best else None,
                "direction_at_max": int(best["direction"]) if best else 0,
                "status": "success" if best else "failed:no_finite_statistic",
                "trace": trace})
        return {"schema_version": SCHEMA_VERSION, "scientific_digest": digest,
                "scenario_digest": scenario_hash, "scenario": scenario.name,
                "replicate": replicate, "runtime_seconds": time.perf_counter()-started,
                "stream_manifest": stream_manifest, "rows": rows}
    except Exception as exc:
        common = {"scenario": scenario.name, "family": scenario.family,
                  "null_family": scenario.null_family, "split": scenario.split,
                  "purpose": scenario.purpose, "motivation": scenario.motivation,
                  "misspecification": scenario.misspecification,
                  "replicate": replicate, "case_seed": seeds["case"],
                  "death_seed": seeds["death"],
                  "multiplier_seed": seeds["multiplier"],
                  "seed": seeds["death"],
                  "case_stream_group": scenario.case_stream_group,
                  "outcome_stream_group": scenario.outcome_stream_group,
                  "sampler_backend": "unavailable",
                  "multiplier_backend": "unavailable",
                  "horizon": scenario.horizon, "profile": scenario.profile,
                  "case_level": scenario.case_level, "pi_left": scenario.pi_left,
                  "pi_right": scenario.pi_right, "risk_ratio": scenario.risk_ratio,
                  "phi": scenario.phi, "true_change": scenario.true_change,
                  "true_change_day": scenario.true_change + 1,
                  "true_delay_mean_days": scenario.true_delay_mean_days,
                  "assumed_delay_mean_days": scenario.assumed_delay_mean_days}
        return {"schema_version": SCHEMA_VERSION, "scientific_digest": digest,
                "scenario_digest": scenario_hash, "scenario": scenario.name,
                "replicate": replicate, "runtime_seconds": time.perf_counter()-started,
                "stream_manifest": {"seeds": seeds},
                "rows": [{**common, "method": method,
                          "score_definition": "unavailable",
                          "status": f"exception:{type(exc).__name__}:{exc}",
                          "episode_statistic": float("nan"), "trace": []}
                         for method in METHODS]}

def _payload_success(payload: dict[str, object]) -> bool:
    rows = payload.get("rows")
    expected = payload.get("expected_methods", METHODS)
    return (isinstance(rows, list) and len(rows) == len(expected)
            and {x.get("method") for x in rows if isinstance(x, dict)} == set(expected)
            and all(isinstance(x, dict) and x.get("status") == "success"
                    for x in rows))

def _payload_hash(payload: dict[str, object]) -> str:
    material = {key: value for key, value in payload.items() if key != "payload_hash"}
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=True).encode("ascii")
    return _sha256_bytes(encoded)

def _evaluate_with_retry(scenario: SimulationScenario, replicate: int,
                         config: ExperimentConfig, digest: str,
                         attempts: int = 2) -> dict[str, object]:
    """Retry a failed numerical task once; failed caches remain non-reusable."""
    payload: dict[str, object] | None = None
    for _ in range(attempts):
        payload = _evaluate_one(scenario, replicate, config, digest)
        if _payload_success(payload):
            break
    assert payload is not None
    payload["attempts"] = attempts if not _payload_success(payload) else _ + 1
    payload["payload_hash"] = _payload_hash(payload)
    return payload

def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)

def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)

def _cache_path(root: Path, digest: str, scenario: str, replicate: int) -> Path:
    return root / digest / scenario / f"{replicate:05d}.json"

def _valid_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

def _load_cache(path: Path, *, digest: str, scenario: SimulationScenario,
                replicate: int, config: ExperimentConfig) -> dict[str, object] | None:
    """Return only complete successful cache payloads; failures are retried."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows, streams = payload.get("rows"), payload.get("stream_manifest")
    expected_methods = (_compact_method_names(scenario, config) if config.protocol in {"compact-v1", "compact-v2"}
                        else METHODS)
    if (payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("scientific_digest") != digest
            or payload.get("scenario_digest") != _scenario_digest(scenario)
            or payload.get("scenario") != scenario.name
            or payload.get("replicate") != replicate
            or not isinstance(rows, list) or len(rows) != len(expected_methods)
            or {x.get("method") for x in rows if isinstance(x, dict)} != set(expected_methods)
            or not all(isinstance(x, dict) for x in rows)
            or any(x.get("status") != "success" for x in rows if isinstance(x, dict))
            or not isinstance(streams, dict)
            or not _valid_hex(payload.get("payload_hash"))
            or payload.get("payload_hash") != _payload_hash(payload)):
        return None
    hashes = streams.get("hashes")
    if not isinstance(hashes, dict) or set(hashes) != {
            "cases", "deaths", "true_delay", "assumed_delay", "multipliers"}:
        return None
    if not all(_valid_hex(value) for value in hashes.values()):
        return None
    expected = _component_seeds(scenario, config, replicate)
    if streams.get("seeds") != expected:
        return None
    if (streams.get("rng_contract_version") != config.rng_contract_version
            or streams.get("groups") != {"case": scenario.case_stream_group,
                                         "death": scenario.outcome_stream_group,
                                         "multiplier": scenario.name}):
        return None
    return payload

def _flatten(payloads: Iterable[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        for source in payload["rows"]:
            row = dict(source)
            row["runtime_seconds"] = float(payload.get("runtime_seconds", np.nan))
            row["stream_manifest_json"] = json.dumps(payload["stream_manifest"], separators=(",", ":"))
            row["trace_json"] = json.dumps(row.pop("trace"), separators=(",", ":"))
            rows.append(row)
    return pd.DataFrame(rows)

def _calibrate(frame: pd.DataFrame, alpha: float, digest: str) -> pd.DataFrame:
    rows = []
    source = frame[(frame["split"] == "calibration") & (frame["status"] == "success")]
    for (family, method), group in source.groupby(["null_family", "method"], sort=True):
        values = group["episode_statistic"].to_numpy(float)
        values = values[np.isfinite(values)]
        threshold = 0. if method == "OCFR" else (
            monte_carlo_critical_value(values, alpha) if values.size else np.inf)
        version_payload = {"digest": digest, "family": family, "method": method,
                           "replicates": sorted(group["replicate"].astype(int).tolist()),
                           "threshold": (float(threshold) if np.isfinite(threshold)
                                         else "infinity")}
        rows.append({"null_family": family, "family": family, "method": method,
                     "alpha": alpha, "n_calibration": int(values.size),
                     "critical_value": threshold,
                     "threshold_version": _sha256_bytes(_canonical_json(version_payload).encode("ascii"))[:16],
                     "rule": "fixed GP boundary" if method == "OCFR" else "conservative (B+1) empirical order statistic"})
    return pd.DataFrame(rows)

def _apply_thresholds(frame: pd.DataFrame, thresholds: pd.DataFrame,
                      timely_window: int) -> pd.DataFrame:
    merged = frame.merge(thresholds[["null_family", "method", "critical_value",
                                     "threshold_version"]],
                         on=["null_family", "method"], how="left", validate="many_to_one")
    merged["alarm"] = False
    alarm_days, alarm_taus = [], []
    for record in merged.to_dict("records"):
        first = None
        if record["status"] == "success" and np.isfinite(record["critical_value"]):
            first = next((x for x in json.loads(record["trace_json"])
                          if np.isfinite(float(x["statistic"]))
                          and float(x["statistic"]) > float(record["critical_value"])), None)
        alarm_days.append(float(first["day"]) if first else np.nan)
        alarm_taus.append(float(first["tau_hat"]) if first and first["tau_hat"] is not None else np.nan)
    merged["alarm_day"] = alarm_days
    merged["tau_hat_at_alarm"] = alarm_taus
    merged["alarm"] = np.isfinite(merged["alarm_day"])
    alternative = merged["split"] == "alternative"
    post = alternative & merged["alarm"] & (merged["alarm_day"] >= merged["true_change"])
    merged["premature_alarm"] = alternative & merged["alarm"] & ~post
    merged["detected_within_28"] = post & (merged["alarm_day"] <= merged["true_change"] + timely_window)
    merged["localization_abs_error_conditional"] = np.where(
        merged["detected_within_28"] & np.isfinite(merged["tau_hat_at_alarm"]),
        np.abs(merged["tau_hat_at_alarm"] - merged["true_change"]), np.nan)
    return merged

def _exact_binomial_interval(successes: int, trials: int,
                             confidence: float = 0.90) -> tuple[float, float]:
    """Two-sided Clopper--Pearson interval, with explicit endpoint cases."""
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("successes must lie in [0, trials] with trials positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes,
                                                      trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - tail, successes + 1,
                                                            trials - successes))
    return lower, upper

def _size_control_gate(frame: pd.DataFrame, *, lower: float = 0.04,
                       upper: float = 0.06,
                       confidence: float = 0.90) -> pd.DataFrame:
    """Audit independent-null size without feeding results back into tuning.

    The predeclared criterion is intentionally strict: the complete 90% exact
    binomial confidence interval must lie in ``[lower, upper]``.  This reports
    precision and flags weak evidence, but never selects a new threshold or
    aborts alternative runs.
    """
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("gate bounds must satisfy 0 <= lower < upper <= 1")
    source = frame[(frame["split"] == "holdout") & (frame["status"] == "success")]
    rows: list[dict[str, object]] = []
    for (scenario, family, method), group in source.groupby(
            ["scenario", "null_family", "method"], sort=True):
        trials = int(group.shape[0])
        successes = int(group["alarm"].sum())
        interval_lower, interval_upper = _exact_binomial_interval(
            successes, trials, confidence)
        rows.append({
            "scenario": scenario, "null_family": family, "method": method,
            "n_holdout": trials, "alarms": successes,
            "type_i_error": successes / trials,
            "confidence_level": confidence,
            "exact_ci_lower": interval_lower, "exact_ci_upper": interval_upper,
            "gate_lower": lower, "gate_upper": upper,
            "passes_size_control_gate": bool(interval_lower >= lower
                                                and interval_upper <= upper),
        })
    return pd.DataFrame(rows)

def _expand_selection(config: ExperimentConfig,
                      scenario_names: set[str] | None) -> list[SimulationScenario]:
    if scenario_names is None:
        return list(config.scenarios)
    known = {x.name: x for x in config.scenarios}
    unknown = scenario_names - set(known)
    if unknown:
        raise ValueError(f"unknown scenarios: {sorted(unknown)}")
    families = {known[name].null_family for name in scenario_names}
    # A requested family is always scientifically complete; no gate can abort
    # holdout or alternative evaluation.
    return [x for x in config.scenarios if x.null_family in families]

def _reset_outputs(output: Path, cache_root: Path, digest: str,
                   selected: list[SimulationScenario], scope: ResetScope) -> None:
    derived = ("replicate_results.csv", "thresholds.csv", "size_control_gate.csv",
               "manifest.json")
    if scope in {"selected", "derived", "all"}:
        for name in derived:
            target = output / name
            if target.is_file():
                target.unlink()
    if scope == "selected":
        for scenario in selected:
            target = cache_root / digest / scenario.name
            if target.is_dir():
                shutil.rmtree(target)
    elif scope == "all":
        owned = cache_root / digest
        if owned.is_dir():
            shutil.rmtree(owned)

def run_simulations(config: ExperimentConfig, *, output: Path, jobs: int = 1,
                    reset: bool = False, reset_scope: ResetScope = "none",
                    scenario_names: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run/resume a complete protocol. Calibration failure never hides tests."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = _expand_selection(config, scenario_names)
    if not selected:
        raise ValueError("no configured scenarios selected")
    digest, code_hash, code_records = _scientific_digest(config)
    cache_root = output / "cache"
    scope: ResetScope = "all" if reset and reset_scope == "none" else reset_scope
    _reset_outputs(output, cache_root, digest, selected, scope)
    payloads, missing = [], []
    cached_count = 0
    for scenario in selected:
        for replicate in range(config.replications_for(scenario.split)):
            path = _cache_path(cache_root, digest, scenario.name, replicate)
            cached = _load_cache(path, digest=digest, scenario=scenario,
                                 replicate=replicate, config=config)
            if cached is None:
                missing.append((scenario, replicate))
            else:
                payloads.append(cached)
                cached_count += 1
    worker_count = (os.cpu_count() or 1) if jobs == -1 else jobs
    if worker_count < 1:
        raise ValueError("jobs must be positive or -1")
    if worker_count == 1:
        for position, (scenario, replicate) in enumerate(missing, 1):
            payload = _evaluate_with_retry(scenario, replicate, config, digest)
            _atomic_json(payload, _cache_path(cache_root, digest, scenario.name, replicate))
            payloads.append(payload)
            print(f"completed {position}/{len(missing)}: {scenario.name} rep {replicate}")
    elif missing:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(_evaluate_with_retry, s, r, config, digest): (s, r)
                       for s, r in missing}
            for position, future in enumerate(as_completed(futures), 1):
                scenario, replicate = futures[future]
                payload = future.result()
                _atomic_json(payload, _cache_path(cache_root, digest, scenario.name, replicate))
                payloads.append(payload)
                print(f"completed {position}/{len(missing)}: {scenario.name} rep {replicate}")
    frame = _flatten(payloads).sort_values(["scenario", "replicate", "method"], ignore_index=True)
    thresholds = _calibrate(frame, config.methods.alpha, digest)
    frame = _apply_thresholds(frame, thresholds, config.dgp.timely_window)
    gate = _size_control_gate(frame)
    result_path, threshold_path, gate_path = (output / "replicate_results.csv",
                                               output / "thresholds.csv",
                                               output / "size_control_gate.csv")
    _atomic_csv(frame, result_path)
    _atomic_csv(thresholds, threshold_path)
    _atomic_csv(gate, gate_path)
    failures = int((frame["status"] != "success").sum())
    planned_counts = {scenario.name: config.replications_for(scenario.split)
                      for scenario in selected}
    split_counts = {split: config.replications_for(split)
                    for split in ("calibration", "holdout", "alternative")
                    if any(scenario.split == split for scenario in selected)}
    manifest = {
        "schema_version": SCHEMA_VERSION, "mode": config.mode,
        "replications_per_scenario": config.replications,
        "protocol": config.protocol,
        "replications_by_split": split_counts,
        "replications_by_scenario": planned_counts,
        "demo_full_only_difference": "replications (10 versus 1000)",
        "scientific_digest": digest, "code_hash": code_hash,
        "code_files": code_records, "rng_contract_version": config.rng_contract_version,
        "backend_request": config.methods.backend,
        "scenarios": [x.name for x in selected], "methods": list(METHODS),
        "splits_completed": sorted({x.split for x in selected}),
        "cached_tasks_loaded": cached_count, "tasks_computed": len(missing),
        "failed_method_rows": failures,
        "calibration_gate_aborts_alternatives": False,
        "size_control_gate": {
            "criterion": "two-sided 90% exact binomial CI wholly within [0.04, 0.06]",
            "bounds": [0.04, 0.06], "confidence_level": 0.90,
            "is_audit_only": True, "aborts_alternatives": False,
            "all_pairs_pass": bool(not gate.empty and gate["passes_size_control_gate"].all()),
            "passed_pairs": int(gate["passes_size_control_gate"].sum()) if not gate.empty else 0,
            "total_pairs": int(gate.shape[0]),
        },
        "scientific_config": config.scientific_payload(),
        "artifacts": {"replicate_results.csv": _file_hash(result_path),
                      "thresholds.csv": _file_hash(threshold_path),
                      "size_control_gate.csv": _file_hash(gate_path)}}
    _atomic_json(manifest, output / "manifest.json")
    return frame, thresholds


def _primary_ocfr(config):
    return "OCFR" if config.protocol == "compact-v2" else "OCFR Gaussian"


def _compact_method_names(scenario: SimulationScenario, config=None) -> tuple[str, ...]:
    stage = "calibration" if scenario.split == "calibration" else "audit" if scenario.split == "size_audit" else "evaluation"
    return compact_stage_methods(config.protocol if config is not None else "compact-v1", stage)


def _compact_generate_components(scenario: SimulationScenario, config: ExperimentConfig,
                                 replicate: int):
    """Compact DGP, with paired latent histories for sensitivity controls."""
    seeds = _component_seeds(scenario, config, replicate)
    case_rng = make_sampler(seeds["case"], backend=config.methods.backend)
    if scenario.profile != "compact_wave":
        latent_cases = make_correlated_epidemic_cases(
            scenario.horizon, scenario.profile, scenario.case_level, case_rng,
            autoregression=config.dgp.autoregression, process_sd=config.dgp.process_sd,
            case_shape=config.dgp.case_shape, weekday_amplitude=config.dgp.weekday_amplitude)
    else:
        r = np.arange(scenario.horizon, dtype=float)
        q = .25 + 1.5 * np.exp(-.5 * ((r - .55 * scenario.horizon) /
                                       (.18 * scenario.horizon)) ** 2)
        target = scenario.case_level * q / q.mean()
        epsilon = np.empty(scenario.horizon)
        epsilon[0] = case_rng.normal(0., config.dgp.process_sd)
        innovation = config.dgp.process_sd * np.sqrt(1 - config.dgp.autoregression ** 2)
        for day in range(1, scenario.horizon):
            epsilon[day] = config.dgp.autoregression * epsilon[day - 1] + case_rng.normal(0., innovation)
        mean = target * np.exp(epsilon - .5 * config.dgp.process_sd ** 2)
        mean *= 1 + config.dgp.weekday_amplitude * np.sin(2 * np.pi * r / 7)
        latent_cases = case_rng.negative_binomial(
            config.dgp.case_shape, config.dgp.case_shape / (config.dgp.case_shape + mean)).astype(float)
    true_delay = gamma_delay_pmf(scenario.true_delay_mean_days,
        shape=scenario.true_delay_shape, maximum_lag=config.dgp.maximum_delay)
    delay = gamma_delay_pmf(scenario.assumed_delay_mean_days,
        shape=scenario.assumed_delay_shape, maximum_lag=config.dgp.maximum_delay)
    rates = np.full(scenario.horizon, scenario.pi_left)
    if scenario.split == "alternative":
        rates[scenario.true_change:] = scenario.pi_right
    death_rng = make_sampler(seeds["death"], backend=config.methods.backend)
    if scenario.misspecification == "cohort_binomial":
        # Sequential conditional binomials are an exact multinomial allocation.
        remaining = death_rng.binomial(latent_cases.astype(np.int64), rates).astype(np.int64)
        deaths = np.zeros(scenario.horizon, dtype=float)
        probability_remaining = 1.
        for offset, probability in enumerate(true_delay, 1):
            allocation = (remaining.copy() if offset == len(true_delay) else
                death_rng.binomial(remaining, min(1., probability / probability_remaining)))
            if offset < scenario.horizon:
                deaths[offset:] += allocation[:-offset]
            remaining -= allocation
            probability_remaining = max(0., probability_remaining - probability)
    else:
        deaths = simulate_nb_deaths(latent_cases, true_delay, rates, scenario.phi, death_rng)
    cases = latent_cases.copy()
    if scenario.misspecification.startswith("exposure_scale_"):
        cases[scenario.true_change:] *= float(scenario.misspecification.rsplit("_", 1)[1])
    streams = {
        "rng_contract_version": config.rng_contract_version, "seeds": seeds,
        "groups": {"case": scenario.case_stream_group, "death": scenario.outcome_stream_group,
                   "multiplier": scenario.name},
        "backends": {"case": case_rng.backend, "death": death_rng.backend},
        "latent_case_hash": _array_hash(latent_cases),
        "hashes": {"cases": _array_hash(cases), "deaths": _array_hash(deaths),
                   "true_delay": _array_hash(true_delay), "assumed_delay": _array_hash(delay)}}
    return cases, deaths, delay, streams


def _compact_evaluate_one(scenario: SimulationScenario, replicate: int,
                           config: ExperimentConfig, digest: str) -> dict[str, object]:
    """A computational exception propagates; unavailable statistical looks do not alarm."""
    from methods import surveillance_benchmark_traces
    started = time.perf_counter()
    cases, deaths, delay, streams = _compact_generate_components(scenario, config, replicate)
    names = _compact_method_names(scenario, config)
    looks = _planned_looks(scenario, config)
    primary = _primary_ocfr(config)
    gaussian = primary in names
    spending = GaussianSpendingBoundary(config.methods.n_multiplier, len(looks), config.methods.alpha) if config.protocol == "compact-v2" and gaussian else None
    multipliers = np.empty((0, scenario.horizon))
    if gaussian:
        rng = make_sampler(streams["seeds"]["multiplier"], backend=config.methods.backend)
        multipliers = gaussian_multiplier_bank(rng, (config.methods.n_multiplier, scenario.horizon),
            normal_mode="pcg32-numpy-box-muller-v1" if config.protocol == "compact-v2" else "scalar")
        streams["backends"]["multiplier"] = rng.backend
    else:
        streams["backends"]["multiplier"] = "not_generated_training_score_only"
    streams["hashes"]["multipliers"] = _array_hash(multipliers)
    traces = {name: [] for name in names}
    benchmarks = surveillance_benchmark_traces(cases, deaths, delay, looks,
        reference_end=config.methods.reference_end, allowance=config.methods.cusum_allowance,
        backend=config.methods.backend)
    for look_index, prefix in enumerate(looks):
        if gaussian or config.protocol == "compact-v1":
            scan = score_scan(cases[:prefix], deaths[:prefix], delay,
                use_lindeberg_gate=config.methods.use_lindeberg_gate, backend=config.methods.backend)
            available = bool(scan.candidates.size and np.isfinite(scan.statistic) and (config.protocol != "compact-v2" or scan.null_fit.success))
            raw = float(scan.statistic) if available else -float("inf")
            boundary = float("inf")
            boundary_look = None
            if gaussian:
                maxima = multiplier_statistics_from_draws(multipliers[:, :prefix], scan.influence,
                    backend=config.methods.backend) if available else None
                if spending is not None:
                    boundary_look = spending.advance(maxima, available=available)
                    boundary = boundary_look.boundary
                elif available:
                    boundary = monte_carlo_critical_value(maxima, config.methods.alpha / len(looks))
            base = {"day": prefix - 1, "manuscript_day": prefix, "raw_statistic": raw,
                    "boundary": boundary if gaussian else None, "tau_hat": scan.tau_hat,
                    "direction": int(scan.direction), "fit_status": scan.null_fit.status,
                    "status": "available" if available else "unavailable"}
            if boundary_look is not None:
                base.update(asdict(boundary_look))
                base["boundary_policy"] = config.boundary_policy
            if gaussian:
                traces[primary].append({**base, "statistic": raw - boundary})
            if config.protocol == "compact-v1":
                traces["OCFR empirical"].append({**base, "statistic": raw})
                report = report_day_score_scan(scan, deaths[:prefix], initial_candidates=np.arange(1, prefix),
                    use_lindeberg_gate=config.methods.use_lindeberg_gate)
                report_ok = bool(report.candidates.size and np.isfinite(report.statistic))
                report_raw = float(report.statistic) if report_ok else -float("inf")
                traces["Report-day score"].append({**base, "raw_statistic": report_raw,
                    "boundary": None, "statistic": report_raw, "tau_hat": report.tau_hat,
                    "direction": int(report.direction), "fit_status": report.null_fit.status,
                    "status": "available" if report_ok else "unavailable"})
        for name in ("NB-GLR", "Page-CUSUM"):
            result = benchmarks[name][look_index]
            value = float(result.statistic)
            ok = np.isfinite(value)
            traces[name].append({"day": prefix - 1, "manuscript_day": prefix,
                "raw_statistic": value if ok else -float("inf"), "boundary": None,
                "statistic": value if ok else -float("inf"),
                "tau_hat": result.estimated_cohort_change_day, "direction": int(result.direction),
                "fit_status": result.fit_status, "status": "available" if ok else "unavailable"})
    rows = []
    for name, trace in traces.items():
        best = max(trace, key=lambda x: x["statistic"])
        rows.append({"scenario": scenario.name, "family": scenario.family,
            "null_family": scenario.null_family, "split": scenario.split,
            "purpose": "ascertainment" if scenario.misspecification.startswith("exposure_scale_") else scenario.purpose,
            "misspecification": scenario.misspecification, "method": name, "replicate": replicate,
            "horizon": scenario.horizon, "profile": scenario.profile, "case_level": scenario.case_level,
            "pi_left": scenario.pi_left, "pi_right": scenario.pi_right, "risk_ratio": scenario.risk_ratio,
            "phi": scenario.phi, "true_change": scenario.true_change,
            "true_change_day": scenario.true_change + 1,
            "true_delay_mean_days": scenario.true_delay_mean_days,
            "assumed_delay_mean_days": scenario.assumed_delay_mean_days,
            "case_seed": streams["seeds"]["case"], "death_seed": streams["seeds"]["death"],
            "multiplier_seed": streams["seeds"]["multiplier"],
            "score_definition": "raw score minus Gaussian boundary" if name == primary else "raw planned-look statistic",
            "episode_statistic": float(best["statistic"]), "max_day": best["day"],
            "tau_hat_at_max": best["tau_hat"], "direction_at_max": best["direction"],
            "status": "success", "all_looks_unavailable": not any(x["status"] == "available" for x in trace),
            "unavailable_looks": sum(x["status"] != "available" for x in trace), "trace": trace})
    payload = {"schema_version": SCHEMA_VERSION, "scientific_digest": digest,
               "scenario_digest": _scenario_digest(scenario), "scenario": scenario.name,
               "replicate": replicate, "runtime_seconds": time.perf_counter() - started,
               "expected_methods": list(names), "stream_manifest": streams, "rows": rows}
    payload["payload_hash"] = _payload_hash(payload)
    return payload


def _compact_calibrate(frame: pd.DataFrame, alpha: float, digest: str,
                       expected_count: int = 10_000, expected_methods=None) -> pd.DataFrame:
    source = frame[frame["split"] == "calibration"]
    if expected_methods is not None:
        expected = {(family, method) for family in source.null_family.unique() for method in expected_methods}
        if set(zip(source.null_family, source.method)) != expected:
            raise ValueError("calibration method grid mismatch")
    rows = []
    for (family, method), group in source.groupby(["null_family", "method"], sort=True):
        if len(group) != expected_count or group["replicate"].nunique() != expected_count:
            raise ValueError("calibration requires every allocated episode exactly once")
        if not (group["status"] == "success").all():
            raise ValueError("infrastructure failures cannot enter calibration")
        values = group["episode_statistic"].to_numpy(float)
        if np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError("invalid calibration statistic")
        rank = int(np.ceil((expected_count + 1) * (1 - alpha)))
        cutoff = float(np.sort(values)[rank - 1]) if rank <= expected_count else float("inf")
        identity = {"digest": digest, "family": family, "method": method,
                    "n": expected_count, "rank": rank, "cutoff": str(cutoff),
                    "scores_sha256": _array_hash(values)}
        rows.append({"null_family": family, "family": family, "method": method,
                     "alpha": alpha, "n_calibration": expected_count, "order_rank": rank,
                     "critical_value": cutoff, "scientific_digest": digest,
                     "threshold_version": _sha256_bytes(_canonical_json(identity).encode())[:16],
                     "rule": "raw episode maximum; (B+1) order statistic; strict crossing"})
    return pd.DataFrame(rows)


def _compact_apply_thresholds(frame: pd.DataFrame, thresholds: pd.DataFrame,
                               timely_window: int, config=None) -> pd.DataFrame:
    primary = _primary_ocfr(config) if config is not None else "OCFR Gaussian"
    policy = config.boundary_policy if config is not None and config.protocol == "compact-v2" else "fixed-gaussian-alpha-over-J"
    families = sorted(frame["null_family"].unique())
    gaussian = pd.DataFrame([{"null_family": family, "method": primary,
        "critical_value": 0., "threshold_version": policy} for family in families])
    result = _apply_thresholds(frame, pd.concat([thresholds, gaussian], ignore_index=True), timely_window)
    if result["critical_value"].isna().any():
        raise ValueError("frozen thresholds do not cover all selected methods/families")
    # Negative-infinite trained cutoffs are possible for fully unavailable banks.
    for index, record in result.iterrows():
        trace = json.loads(record["trace_json"])
        cutoff = float(record["critical_value"])
        first = next((x for x in trace if x.get("status") == "available"
                      and np.isfinite(x["statistic"]) and x["statistic"] > cutoff), None)
        result.at[index, "alarm_day"] = first["day"] if first else np.nan
        result.at[index, "tau_hat_at_alarm"] = first["tau_hat"] if first and first["tau_hat"] is not None else np.nan
        for look in trace:
            if record["method"] != primary:
                look["gaussian_boundary_diagnostic"] = look.get("boundary")
                look["boundary"] = cutoff
            look["threshold_version"] = record["threshold_version"]
        result.at[index, "trace_json"] = json.dumps(trace, sort_keys=True, separators=(",", ":"))
    result["alarm"] = result["alarm_day"].notna()
    alternative = result["split"].eq("alternative")
    post = alternative & result["alarm"] & result["alarm_day"].ge(result["true_change"])
    result["premature_alarm"] = alternative & result["alarm"] & ~post
    result["postchange_detection"] = post
    result["no_postchange_detection"] = alternative & ~post
    result["no_alarm"] = ~result["alarm"]
    result["detected_within_28"] = post & result["alarm_day"].le(result["true_change"] + timely_window)
    result["localization_abs_error_conditional"] = np.where(
        result["detected_within_28"], abs(result["tau_hat_at_alarm"] - result["true_change"]), np.nan)
    result["restricted_delay"] = np.where(post, result["alarm_day"] - result["true_change"],
                                          result["horizon"] - result["true_change"])
    return result


def _compact_stage_scenarios(config, stage, scenario_names=None):
    if stage not in {"calibration", "audit", "evaluation"}:
        raise ValueError("stage must be calibration, audit or evaluation")
    selected = [s for s in config.scenarios if
        (s.split == "calibration" if stage == "calibration" else
         s.split == "size_audit" if stage == "audit" else s.split in {"holdout", "alternative"})]
    if scenario_names is not None:
        unknown = set(scenario_names) - {s.name for s in selected}
        if unknown:
            raise ValueError(f"scenarios outside selected stage: {sorted(unknown)}")
        selected = [s for s in selected if s.name in scenario_names]
    if not selected:
        raise ValueError("no scenarios selected")
    return selected


def _compact_read_thresholds(path: Path, digest: str, families, config=None) -> pd.DataFrame:
    manifest_path = path.parent / "manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError("run the explicit calibration stage first; frozen threshold artifact required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("scientific_digest") != digest or manifest.get("stage") != "calibration"
            or manifest.get("status") != "complete"
            or manifest.get("artifacts", {}).get(path.name) != _file_hash(path)):
        raise ValueError("threshold manifest/digest/hash mismatch")
    thresholds = pd.read_csv(path)
    methods = config.stage_methods("calibration") if config is not None else COMPACT_EMPIRICAL_METHODS
    count = config.replications_for("calibration") if config is not None else 10_000
    expected = {(family, method) for family in families for method in methods}
    found = set(zip(thresholds["null_family"], thresholds["method"]))
    if not expected.issubset(found) or thresholds.duplicated(["null_family", "method"]).any():
        raise ValueError("threshold artifact has missing or duplicate methods")
    if not thresholds["n_calibration"].eq(count).all():
        raise ValueError("scientific thresholds have incorrect calibration allocation")
    return thresholds


def _compact_scalar_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    """Keep episode metadata in memory while complete traces stay in caches."""
    if not _payload_success(payload):
        raise RuntimeError("incomplete compact episode")
    rows = []
    for source in payload["rows"]:
        if (source["scenario"] != payload["scenario"]
                or source["replicate"] != payload["replicate"]):
            raise RuntimeError("inconsistent compact episode identifiers")
        row = {key: value for key, value in source.items() if key != "trace"}
        row["runtime_seconds"] = float(payload.get("runtime_seconds", np.nan))
        rows.append(row)
    return rows


def _compact_cache_batches(config: ExperimentConfig, selected: list[SimulationScenario],
                            cache_root: Path, digest: str, batch_size: int):
    """Read a complete task grid in final CSV order, never rerunning a task."""
    batch = []
    for scenario in sorted(selected, key=lambda item: item.name):
        for replicate in range(config.replications_for(scenario.split)):
            path = _cache_path(cache_root, digest, scenario.name, replicate)
            payload = _load_cache(path, digest=digest, scenario=scenario,
                                  replicate=replicate, config=config)
            if payload is None:
                raise RuntimeError(f"cache became unavailable during aggregation: {path}")
            batch.append(payload)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _compact_write_results(config: ExperimentConfig, selected: list[SimulationScenario],
                            cache_root: Path, digest: str, stage: str,
                            thresholds: pd.DataFrame | None, scalar_frame: pd.DataFrame,
                            result_path: Path, batch_size: int) -> None:
    """Apply the original transformations in batches with global scalar dtypes."""
    dtypes = scalar_frame.dtypes.to_dict()
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    columns, count = None, 0
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        for payloads in _compact_cache_batches(config, selected, cache_root, digest, batch_size):
            frame = _flatten(payloads).astype(dtypes).sort_values(
                ["scenario", "replicate", "method"], ignore_index=True)
            if stage != "calibration":
                frame = _compact_apply_thresholds(frame, thresholds, config.dgp.timely_window, config)
            for column in ("stream_manifest_json", "trace_json"):
                frame[column] = frame[column].map(lambda value: json.dumps(json.loads(value),
                    sort_keys=True, separators=(",", ":")))
            frame = frame.reindex(sorted(frame.columns), axis=1)
            if columns is None:
                columns = list(frame.columns)
            if list(frame.columns) != columns:
                raise RuntimeError("inconsistent compact result columns")
            frame.to_csv(destination, index=False, header=count == 0)
            count += len(frame)
        if count != len(scalar_frame):
            raise RuntimeError("incomplete compact result aggregation")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, result_path)


def run_compact_simulations(config: ExperimentConfig, *, output: Path,
                            stage: str = "evaluation", thresholds_path: Path | None = None,
                            jobs: int = 1, reset_scope: ResetScope = "none",
                            scenario_names: set[str] | None = None,
                            return_mode: Literal["full", "slim"] = "full",
                            aggregation_batch_size: int = 50):
    """Run independent stages with bounded trace aggregation.

    The default full DataFrame retains the original library API and requires
    memory for that returned table. ``return_mode='slim'`` omits trace_json and
    stream_manifest_json from the return only; the CSV always stores both.
    """
    if config.protocol not in {"compact-v1", "compact-v2"}:
        raise ValueError("compact runner requires compact-v1 or compact-v2 config")
    if jobs == 0 or jobs < -1:
        raise ValueError("jobs must be positive or -1")
    if return_mode not in {"full", "slim"}:
        raise ValueError("return_mode must be full or slim")
    if (not isinstance(aggregation_batch_size, int) or isinstance(aggregation_batch_size, bool)
            or aggregation_batch_size < 1):
        raise ValueError("aggregation_batch_size must be a positive integer")
    output = Path(output).resolve()
    selected = _compact_stage_scenarios(config, stage, scenario_names)
    digest, code_hash, code_records = _scientific_digest(config)
    stage_output = output / stage
    stage_output.mkdir(parents=True, exist_ok=True)
    cache_root = output / "cache"
    thresholds_path = Path(thresholds_path) if thresholds_path else output / "calibration" / "thresholds.csv"
    thresholds = (None if stage == "calibration" else
                  _compact_read_thresholds(thresholds_path, digest, {s.null_family for s in selected}, config))
    # Containment is checked before recursive removal, including symlink resolution.
    if reset_scope in {"selected", "all"}:
        for scenario in selected:
            target = (cache_root / digest / scenario.name).resolve()
            if not target.is_relative_to((cache_root / digest).resolve()):
                raise ValueError("unsafe cache reset target")
            if target.is_dir():
                shutil.rmtree(target)
    if reset_scope != "none":
        for name in ("manifest.json", "replicate_results.csv", "thresholds.csv", "size_control_gate.csv"):
            path = stage_output / name
            if path.is_file():
                path.unlink()
    scalar_rows, missing = [], []
    cached_count = 0
    for scenario in selected:
        for replicate in range(config.replications_for(scenario.split)):
            path = _cache_path(cache_root, digest, scenario.name, replicate)
            cached = _load_cache(path, digest=digest, scenario=scenario, replicate=replicate, config=config)
            if cached is None:
                missing.append((scenario, replicate))
            else:
                scalar_rows.extend(_compact_scalar_rows(cached))
                cached_count += 1
    del cached
    worker_count = (os.cpu_count() or 1) if jobs == -1 else jobs
    manifest = {"schema_version": SCHEMA_VERSION, "protocol": config.protocol, "stage": stage,
        "mode": config.mode, "status": "running", "scientific_digest": digest,
        "code_hash": code_hash, "code_files": code_records,
        "scientific_config": config.scientific_payload(), "backend_request": config.methods.backend,
        "rng_contract_version": config.rng_contract_version, "methods": list(_compact_method_names(selected[0], config)),
        "scenarios": [s.name for s in selected],
        "replications_by_scenario": {s.name: config.replications_for(s.split) for s in selected},
        "cached_tasks_loaded": cached_count, "tasks_computed": 0,
        "threshold_artifact": None if thresholds is None else {"path": str(thresholds_path.resolve()),
             "sha256": _file_hash(thresholds_path)},
        "unavailable_looks": "no alarm; all allocated episodes retained",
        "infrastructure_errors": "incomplete job; propagate and repair", "crossing": "strict >",
        "gaussian_multiplier_draws": config.methods.n_multiplier,
        "empirical_calibration_statistic": "max raw planned-look statistic",
        "aggregation": {"batch_episodes": aggregation_batch_size, "return_mode": return_mode,
                        "trace_columns_in_csv": True, "maximum_pending_tasks": 2 * worker_count}}
    _atomic_json(manifest, stage_output / "manifest.json")
    def store_payload(payload, scenario, replicate):
        if not _payload_success(payload):
            raise RuntimeError("incomplete compact episode")
        _atomic_json(payload, _cache_path(cache_root, digest, scenario.name, replicate))
        scalar_rows.extend(_compact_scalar_rows(payload))
    if worker_count == 1:
        for scenario, replicate in missing:
            store_payload(_compact_evaluate_one(scenario, replicate, config, digest), scenario, replicate)
    elif missing:
        with ProcessPoolExecutor(max_workers=min(worker_count, len(missing))) as pool:
            tasks, pending = iter(missing), {}
            def submit_next():
                task = next(tasks, None)
                if task is None:
                    return False
                scenario, replicate = task
                pending[pool.submit(_compact_evaluate_one, scenario, replicate, config, digest)] = task
                return True
            while len(pending) < 2 * worker_count and submit_next():
                pass
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    store_payload(future.result(), *pending.pop(future))
                completed.clear()
                while len(pending) < 2 * worker_count and submit_next():
                    pass
    scalar_frame = pd.DataFrame(scalar_rows).sort_values(
        ["scenario", "replicate", "method"], ignore_index=True)
    del scalar_rows
    if scalar_frame.duplicated(["scenario", "replicate", "method"]).any():
        raise RuntimeError("duplicate compact episode rows")
    if stage == "calibration":
        thresholds = _compact_calibrate(scalar_frame, config.methods.alpha, digest, config.replications_for("calibration"), config.stage_methods("calibration"))
        _atomic_csv(thresholds, stage_output / "thresholds.csv")
    result_path = stage_output / "replicate_results.csv"
    _compact_write_results(config, selected, cache_root, digest, stage, thresholds,
                           scalar_frame, result_path, aggregation_batch_size)
    dtypes = scalar_frame.dtypes.to_dict()
    unavailable = int(scalar_frame["all_looks_unavailable"].sum())
    del scalar_frame
    excluded = {"trace_json", "stream_manifest_json"} if return_mode == "slim" else set()
    frame = pd.read_csv(result_path, float_precision="round_trip", dtype=dtypes,
                        usecols=lambda name: name not in excluded)
    artifacts = {result_path.name: _file_hash(result_path)}
    if stage == "calibration":
        artifacts["thresholds.csv"] = _file_hash(stage_output / "thresholds.csv")
    if stage == "audit":
        gate_input = frame.assign(split="holdout")
        _atomic_csv(_size_control_gate(gate_input), stage_output / "size_control_gate.csv")
        artifacts["size_control_gate.csv"] = _file_hash(stage_output / "size_control_gate.csv")
    manifest.update(status="complete", tasks_computed=len(missing), artifacts=artifacts,
                    failed_method_rows=0, all_looks_unavailable_rows=unavailable)
    _atomic_json(manifest, stage_output / "manifest.json")
    return frame, thresholds



def _estimation_source_config(manifest: dict, mode: str) -> ExperimentConfig:
    """Deserialize frozen scientific inputs without consulting present defaults."""
    from config import DGPConfig, MethodConfig
    scientific = manifest["scientific_config"]
    if (manifest.get("status") != "complete" or manifest.get("stage") != "evaluation"
            or scientific.get("protocol") not in {"compact-v1", "compact-v2"}):
        raise ValueError("estimation requires a completed compact evaluation manifest")
    plan = scientific.get("split_replications")
    config = ExperimentConfig(mode, 10 if mode == "demo" else 1000,
        scientific["master_seed"], scientific["rng_contract_version"],
        DGPConfig(**scientific["dgp"]), MethodConfig(**scientific["methods"]),
        tuple(SimulationScenario(**x) for x in scientific["scenarios"]),
        scientific["protocol"], None if plan is None else tuple(plan.items()))
    if config.scientific_payload() != scientific:
        raise ValueError("original scientific configuration cannot be reconstructed exactly")
    return config


def _estimation_first_alarm(row: dict, trace: list[dict]) -> tuple[int | None, int | None]:
    first = next((x for x in trace if x.get("status") == "available"
                  and np.isfinite(x["statistic"]) and x["statistic"] > 0.), None)
    def integer(value):
        if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
            return None
        numeric = float(value)
        if not np.isfinite(numeric) or numeric != int(numeric):
            raise ValueError("invalid saved alarm/tau index")
        return int(numeric)
    alarm, tau = integer(row.get("alarm_day")), integer(row.get("tau_hat_at_alarm"))
    expected = (integer(first["day"]), integer(first["tau_hat"])) if first else (None, None)
    if (alarm, tau) != expected:
        raise ValueError("saved first Gaussian alarm does not replay from the original trace")
    return alarm, tau


def _estimation_episode(scenario, replicate, config, source_digest, source_cache, source_row, estimation):
    from methods import (MU_FLOOR, P_BOUNDS, LOG_PHI_BOUNDS, exposure_components, fit_nb_identity,
                         fixed_tau_parameter_intervals)
    payload = _load_cache(_cache_path(source_cache, source_digest, scenario.name, replicate),
        digest=source_digest, scenario=scenario, replicate=replicate, config=config)
    if payload is None:
        raise ValueError(f"missing or corrupt original episode cache: {scenario.name}/{replicate}")
    gaussian = next(x for x in payload["rows"] if x["method"] == estimation.method)
    alarm, tau = _estimation_first_alarm(source_row, gaussian["trace"])
    cases, deaths, delay, streams = _compact_generate_components(scenario, config, replicate)
    for name in ("cases", "deaths", "true_delay", "assumed_delay"):
        if streams["hashes"][name] != payload["stream_manifest"]["hashes"][name]:
            raise ValueError(f"regenerated {name} hash mismatch: {scenario.name}/{replicate}")
    row = {"scenario": scenario.name, "replicate": replicate, "method": estimation.method,
        "family": scenario.family, "horizon": scenario.horizon, "true_change": scenario.true_change,
        "pi_left": scenario.pi_left, "pi_right": scenario.pi_right, "risk_ratio": scenario.risk_ratio,
        "phi": scenario.phi, "alarm_day": alarm, "tau_hat_at_alarm": tau,
        "alarm": alarm is not None, "no_alarm": alarm is None,
        "premature_alarm": alarm is not None and alarm < scenario.true_change,
        "postchange_detection": alarm is not None and alarm >= scenario.true_change,
        "fit_success": False, "fit_status": "no_alarm", "interval_valid": False,
        "point_available": False, "fit_boundary": None,
        "interval_status": "no_alarm", "confidence": estimation.confidence,
        "source_payload_hash": payload["payload_hash"],
        "case_hash": streams["hashes"]["cases"], "death_hash": streams["hashes"]["deaths"],
        "true_delay_hash": streams["hashes"]["true_delay"],
        "assumed_delay_hash": streams["hashes"]["assumed_delay"]}
    for name in ("pi_left", "pi_right", "risk_ratio"):
        for suffix in ("estimate", "lower", "upper"):
            row[f"{name}_{suffix}"] = None
    row["phi_estimate"] = None
    row["log_risk_ratio_lower"] = None
    row["log_risk_ratio_upper"] = None
    for name in ("pi_left", "pi_right", "risk_ratio"):
        row[f"{name}_point_valid"] = False
        row[f"{name}_interval_valid"] = False
        row[f"{name}_interval_status"] = "no_alarm" if alarm is None else "fit_failed"
    if alarm is None:
        return row
    if tau is None or not 1 <= tau < alarm + 1 or not 0 <= alarm < scenario.horizon:
        raise ValueError("invalid selected change index at first alarm")
    total, post = exposure_components(cases[:alarm + 1], delay)
    b = post[:, tau - 1]
    active = total > MU_FLOOR
    design = np.column_stack((total - b, b))[active]
    offset = np.zeros(int(active.sum()))
    fit = fit_nb_identity(deaths[:alarm + 1][active], offset, design)
    row.update(fit_success=bool(fit.success), fit_status=fit.status,
               phi_estimate=float(fit.phi), interval_status="fit_failed",
               fit_boundary=bool(np.any(np.isclose(fit.rates[:, None], np.array(P_BOUNDS),
                    rtol=1e-8, atol=1e-14)) or (fit.phi > 0 and np.isfinite(fit.phi)
                    and np.any(np.isclose(np.log(fit.phi), LOG_PHI_BOUNDS, rtol=0., atol=1e-8)))),
               fit_iterations=int(fit.nit), fit_loglik=float(fit.loglik))
    # Numerical point results and convergence are separate from interval availability.
    row.update(pi_left_estimate=float(fit.rates[0]), pi_right_estimate=float(fit.rates[1]),
               risk_ratio_estimate=float(fit.rates[1] / fit.rates[0]))
    for name in ("pi_left", "pi_right", "risk_ratio"):
        value = row[f"{name}_estimate"]
        row[f"{name}_point_valid"] = bool(fit.success and np.isfinite(value) and value > 0
            and (name == "risk_ratio" or value < 1))
    row["point_available"] = all(row[f"{name}_point_valid"] for name in ("pi_left", "pi_right", "risk_ratio"))
    for name in ("pi_left", "pi_right", "risk_ratio"):
        row[f"{name}_point_valid"] = row["point_available"]
    if not fit.success:
        return row
    if np.any(~np.isfinite(fit.rates)) or np.any((fit.rates <= 0) | (fit.rates >= 1)):
        row["interval_status"] = "rates_outside_probability_interior"
        for name in ("pi_left", "pi_right", "risk_ratio"):
            row[f"{name}_interval_status"] = row["interval_status"]
        return row
    try:
        ci = fixed_tau_parameter_intervals(fit, offset, design, confidence=estimation.confidence)
    except np.linalg.LinAlgError as exc:
        row["interval_status"] = f"singular_information: {exc}"
        for name in ("pi_left", "pi_right", "risk_ratio"):
            row[f"{name}_interval_status"] = row["interval_status"]
        return row
    row.update(pi_left_lower=float(ci.cfr_lower[0]), pi_left_upper=float(ci.cfr_upper[0]),
        pi_right_lower=float(ci.cfr_lower[1]), pi_right_upper=float(ci.cfr_upper[1]),
        risk_ratio_lower=float(ci.risk_ratio_lower), risk_ratio_upper=float(ci.risk_ratio_upper),
        log_risk_ratio_lower=float(ci.log_risk_ratio_lower),
        log_risk_ratio_upper=float(ci.log_risk_ratio_upper))
    for name in ("pi_left", "pi_right", "risk_ratio"):
        row[f"{name}_interval_valid"] = bool(np.isfinite(row[f"{name}_lower"])
            and np.isfinite(row[f"{name}_upper"]) and row[f"{name}_upper"] >= row[f"{name}_lower"])
        row[f"{name}_interval_status"] = "success" if row[f"{name}_interval_valid"] else "nonfinite_or_unordered_interval_endpoint"
    row["interval_valid"] = all(row[f"{name}_interval_valid"] for name in ("pi_left", "pi_right", "risk_ratio"))
    row["interval_status"] = "success" if row["interval_valid"] else "nonfinite_interval_endpoint"
    return row


def run_compact_estimation(*, output: Path, mode: str = "demo", jobs: int = 1,
                           reset_scope: ResetScope = "none", source_manifest: Path | None = None,
                           source_results: Path | None = None):
    """Postprocess frozen episodes; never rerun detection or multiplier draws."""
    import csv
    from datetime import datetime, timezone
    from time import perf_counter
    from dataclasses import asdict
    from config import EstimationConfig
    started = perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    if mode not in {"demo", "full"} or jobs == 0 or jobs < -1:
        raise ValueError("invalid mode or worker count")
    if reset_scope not in {"none", "selected", "derived", "all"}:
        raise ValueError("invalid estimation reset scope")
    output = Path(output).resolve()
    source_manifest = Path(source_manifest or output / "evaluation" / "manifest.json").resolve()
    source_results = Path(source_results or source_manifest.parent / "replicate_results.csv").resolve()
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    config = _estimation_source_config(manifest, mode)
    if manifest.get("mode") != "full":
        raise ValueError("the source evaluation must be the completed full study")
    source_digest = manifest["scientific_digest"]
    original_code_hash = _sha256_bytes(_canonical_json(manifest["code_files"]).encode("ascii"))
    original_digest = _sha256_bytes(_canonical_json({"schema_version": SCHEMA_VERSION,
        "code_hash": original_code_hash, "scientific": config.scientific_payload()}).encode("ascii"))[:20]
    if manifest["code_hash"] != original_code_hash or source_digest != original_digest:
        raise ValueError("original manifest scientific/code digest mismatch")
    if manifest.get("artifacts", {}).get("replicate_results.csv") != _file_hash(source_results):
        raise ValueError("original evaluation CSV hash mismatch")
    selected = [x for x in config.scenarios if x.split == "alternative" and x.misspecification == "on_model"]
    evaluation = [x for x in config.scenarios if x.split in {"holdout", "alternative"}]
    if len(selected) != 16 or set(manifest["scenarios"]) != {x.name for x in evaluation}:
        raise ValueError("original evaluation grid is incomplete")
    if set(manifest["methods"]) != set(config.stage_methods("evaluation")) or any(
            manifest["replications_by_scenario"].get(x.name) != 1000 for x in evaluation):
        raise ValueError("original full study method/replicate allocation is incomplete")
    # Stream the large trace CSV, retaining only a few scalars per target episode.
    expected_names = {x.name for x in evaluation}
    selected_names = {x.name for x in selected}
    seen, sources = set(), {}
    with source_results.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            rep = int(record["replicate"])
            key = (record["scenario"], rep, record["method"])
            if key in seen or key[0] not in expected_names or key[2] not in config.stage_methods("evaluation") or not 0 <= rep < 1000:
                raise ValueError("duplicate or invalid original evaluation row")
            seen.add(key)
            if record["status"] != "success":
                raise ValueError("original evaluation contains unsuccessful episodes")
            if key[0] in selected_names and rep < config.replications and key[2] == _primary_ocfr(config):
                _estimation_first_alarm(record, json.loads(record["trace_json"]))
                sources[(key[0], rep)] = {k: record[k] for k in ("alarm_day", "tau_hat_at_alarm")}
    if len(seen) != len(evaluation) * 1000 * len(config.stage_methods("evaluation")):
        raise ValueError("original evaluation row grid is incomplete")
    estimation = EstimationConfig(method=_primary_ocfr(config))
    post_hash, post_files = _code_hash()
    native_files = {str(p.relative_to(Path(__file__).resolve().parent)): _file_hash(p)
        for pattern in ("*.dll", "*.so", "*.pyd")
        for p in Path(__file__).resolve().parent.rglob(pattern) if p.is_file()}
    provenance = {"original_manifest_sha256": _file_hash(source_manifest),
        "original_csv_sha256": _file_hash(source_results), "original_scientific_digest": source_digest,
        "postprocessor_code_hash": post_hash, "postprocessor_code_files": post_files,
        "postprocessor_native_files": native_files,
        "estimation_config": asdict(estimation)}
    cache_key = _sha256_bytes(_canonical_json(provenance).encode("ascii"))[:20]
    stage = output / "estimation"
    cache_root = stage / "cache"
    target = (cache_root / cache_key).resolve()
    if not target.is_relative_to(stage.resolve()):
        raise ValueError("unsafe estimation cache reset")
    if reset_scope in {"all", "selected"} and target.is_dir():
        shutil.rmtree(target)
    if reset_scope != "none":
        for name in ("manifest.json", "replicate_results.csv"):
            (stage / name).unlink(missing_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    result_manifest = {**provenance, "stage": "estimation", "status": "running", "mode": mode,
        "started_utc": started_utc,
        "scientific_digest": source_digest, "original_code_hash": manifest["code_hash"],
        "original_code_files": manifest["code_files"], "scientific_config": manifest["scientific_config"],
        "scenarios": [x.name for x in selected], "replications_by_scenario": {x.name: config.replications for x in selected},
        "cache_key": cache_key, "selection": "first Gaussian alarm, including premature alarms",
        "interval_scope": "descriptive plug-in fixed-tau Wald; no selection adjustment",
        "source_manifest": str(source_manifest), "source_results": str(source_results)}
    _atomic_json(result_manifest, stage / "manifest.json")
    rows, missing = [], []
    for scenario in selected:
        for rep in range(config.replications):
            path = _cache_path(cache_root, cache_key, scenario.name, rep)
            if path.is_file():
                item = json.loads(path.read_text(encoding="utf-8"))
                if item.get("payload_hash") != _payload_hash(item) or item.get("cache_key") != cache_key:
                    raise ValueError("corrupt estimation cache")
                if item["row"]["scenario"] != scenario.name or item["row"]["replicate"] != rep:
                    raise ValueError("estimation cache identifier mismatch")
                original = _load_cache(_cache_path(source_manifest.parent.parent / "cache", source_digest,
                    scenario.name, rep), digest=source_digest, scenario=scenario, replicate=rep, config=config)
                if original is None or original["payload_hash"] != item["row"]["source_payload_hash"]:
                    raise ValueError("original episode changed since estimation cache was created")
                _estimation_first_alarm(sources[(scenario.name, rep)],
                    next(x for x in original["rows"] if x["method"] == estimation.method)["trace"])
                rows.append(item["row"])
            else:
                missing.append((scenario, rep))
    source_cache = source_manifest.parent.parent / "cache"
    def save(row):
        item = {"cache_key": cache_key, "row": row}
        item["payload_hash"] = _payload_hash(item)
        _atomic_json(item, _cache_path(cache_root, cache_key, row["scenario"], row["replicate"]))
        rows.append(row)
    workers = (os.cpu_count() or 1) if jobs == -1 else jobs
    def arguments(scenario, rep):
        return scenario, rep, config, source_digest, source_cache, sources[(scenario.name, rep)], estimation
    if workers == 1:
        for scenario, rep in missing:
            save(_estimation_episode(*arguments(scenario, rep)))
    elif missing:
        with ProcessPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            tasks, pending = iter(missing), set()
            def submit():
                task = next(tasks, None)
                if task is None:
                    return False
                pending.add(pool.submit(_estimation_episode, *arguments(*task)))
                return True
            while len(pending) < 2 * workers and submit():
                pass
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    save(future.result())
                while len(pending) < 2 * workers and submit():
                    pass
    frame = pd.DataFrame(rows).sort_values(["scenario", "replicate"], ignore_index=True)
    if len(frame) != 16 * config.replications or frame.duplicated(["scenario", "replicate"]).any():
        raise RuntimeError("incomplete estimation episode grid")
    _atomic_csv(frame, stage / "replicate_results.csv")
    result_manifest.update(status="complete", tasks_computed=len(missing), cached_tasks_loaded=len(rows)-len(missing),
        completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=perf_counter()-started,
        artifacts={"replicate_results.csv": _file_hash(stage / "replicate_results.csv")})
    _atomic_json(result_manifest, stage / "manifest.json")
    return frame

def _reference_episode(scenario, replicate, config, source_digest, source_cache, source_row):
    """Replay only original observations and the saved alarm; never scan/refit."""
    from methods import nishiura_two_stage_moments
    payload = _load_cache(_cache_path(source_cache, source_digest, scenario.name, replicate),
        digest=source_digest, scenario=scenario, replicate=replicate, config=config)
    if payload is None or payload['payload_hash'] != source_row['source_payload_hash']:
        raise ValueError('original episode cache does not match saved NB estimation')
    gaussian = next(x for x in payload['rows'] if x['method'] == _primary_ocfr(config))
    alarm, tau = _estimation_first_alarm(source_row, gaussian['trace'])
    cases, deaths, delay, streams = _compact_generate_components(scenario, config, replicate)
    for name, column in (('cases', 'case_hash'), ('deaths', 'death_hash'),
                         ('true_delay', 'true_delay_hash'), ('assumed_delay', 'assumed_delay_hash')):
        if not (streams['hashes'][name] == payload['stream_manifest']['hashes'][name]
                == source_row[column]):
            raise ValueError(f'reference regenerated {name} hash mismatch')
    result = {'scenario': scenario.name, 'replicate': replicate,
              'source_payload_hash': payload['payload_hash'],
              'nishiura_method': 'Nishiura-inspired two-stage moments'}
    if alarm is None:
        values = {'status': 'no_alarm', 'pre_exposure': None, 'post_exposure': None,
                  'pre_contamination': None}
        for target in ('pi_left', 'pi_right', 'risk_ratio'):
            values.update({target + '_estimate': None, target + '_point_valid': False,
                           target + '_admissible': False})
    else:
        values = nishiura_two_stage_moments(cases[:alarm + 1], deaths[:alarm + 1],
                                          delay, tau=tau, alarm=alarm)
    result.update({'nishiura_' + key: value for key, value in values.items()})
    return result


def run_compact_estimation_reference(*, output: Path, mode: str = 'demo', jobs: int = 1,
                                    reset_scope: ResetScope = 'none',
                                    source_manifest: Path | None = None,
                                    source_results: Path | None = None,
                                    estimation_manifest: Path | None = None):
    """Add raw reference points to immutable saved NB first-alarm estimates.

    The completed full NB estimation and original evaluation remain read-only.
    Demo/full use the same seed contract and differ only in replication count.
    Reference caches have their own source-hash and scientific-code digest.
    """
    from datetime import datetime, timezone
    started = time.perf_counter()
    if mode not in {'demo', 'full'} or jobs == 0 or jobs < -1:
        raise ValueError('invalid reference mode or worker count')
    if reset_scope not in {'none', 'selected', 'derived', 'all'}:
        raise ValueError('invalid reference reset scope')
    output = Path(output).resolve()
    source_manifest = Path(source_manifest or output / 'evaluation' / 'manifest.json').resolve()
    source_results = Path(source_results or source_manifest.parent / 'replicate_results.csv').resolve()
    estimation_manifest = Path(estimation_manifest or output / 'estimation' / 'manifest.json').resolve()
    estimation_results = estimation_manifest.parent / 'replicate_results.csv'
    original = json.loads(source_manifest.read_text(encoding='utf-8'))
    nb_manifest = json.loads(estimation_manifest.read_text(encoding='utf-8'))
    config = _estimation_source_config(original, mode)
    source_digest = original['scientific_digest']
    original_code_hash = _sha256_bytes(_canonical_json(original['code_files']).encode('ascii'))
    expected_digest = _sha256_bytes(_canonical_json({'schema_version': SCHEMA_VERSION,
        'code_hash': original_code_hash, 'scientific': config.scientific_payload()}).encode('ascii'))[:20]
    if original['code_hash'] != original_code_hash or expected_digest != source_digest:
        raise ValueError('original evaluation scientific/code digest mismatch')
    if (original.get('mode') != 'full' or nb_manifest.get('mode') != 'full'
            or nb_manifest.get('stage') != 'estimation' or nb_manifest.get('status') != 'complete'):
        raise ValueError('reference requires completed full evaluation and NB estimation')
    hashes = {'original_manifest_sha256': _file_hash(source_manifest),
              'original_csv_sha256': _file_hash(source_results),
              'original_estimation_manifest_sha256': _file_hash(estimation_manifest),
              'original_estimation_csv_sha256': _file_hash(estimation_results)}
    if (original.get('artifacts', {}).get('replicate_results.csv') != hashes['original_csv_sha256']
            or nb_manifest.get('artifacts', {}).get('replicate_results.csv') != hashes['original_estimation_csv_sha256']
            or nb_manifest.get('original_manifest_sha256') != hashes['original_manifest_sha256']
            or nb_manifest.get('original_csv_sha256') != hashes['original_csv_sha256']
            or nb_manifest.get('original_scientific_digest') != source_digest):
        raise ValueError('frozen evaluation/NB estimation provenance mismatch')
    selected = [s for s in config.scenarios if s.split == 'alternative' and s.misspecification == 'on_model']
    base = pd.read_csv(estimation_results)
    expected = {(s.name, rep) for s in selected for rep in range(1000)}
    actual = set(zip(base['scenario'], base['replicate']))
    if len(selected) != 16 or len(base) != len(expected) or actual != expected or base.duplicated(['scenario', 'replicate']).any():
        raise ValueError('frozen NB estimation episode grid is incomplete')
    base = base.loc[base['replicate'] < config.replications].copy()
    sources = {(r['scenario'], r['replicate']): r for r in base.to_dict('records')}
    code_hash, code_files = _code_hash()
    native_files = {p.name: _file_hash(p) for p in Path(__file__).resolve().parent.glob('*')
                    if p.is_file() and p.suffix in {'.dll', '.so', '.dylib', '.pyd'}}
    provenance = {**hashes, 'original_scientific_digest': source_digest,
        'postprocessor_code_hash': code_hash, 'postprocessor_code_files': code_files,
        'postprocessor_native_files': native_files,
        'reference_config': {'method': 'Nishiura-inspired two-stage moments',
            'report_split': 'indices < tau versus >= tau (zero-based OCFR tau)',
            'exposure_column': 'post[:, tau-1]', 'clipping': False, 'intervals': False}}
    cache_key = _sha256_bytes(_canonical_json(provenance).encode('ascii'))[:20]
    stage = output / 'estimation-reference'
    cache_root = stage / 'cache'
    target = (cache_root / cache_key).resolve()
    if not target.is_relative_to(stage.resolve()):
        raise ValueError('unsafe reference cache reset')
    if reset_scope in {'all', 'selected'} and target.is_dir():
        shutil.rmtree(target)
    if reset_scope != 'none':
        for name in ('manifest.json', 'replicate_results.csv'):
            (stage / name).unlink(missing_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    manifest = {**provenance, 'stage': 'estimation-reference', 'status': 'running', 'mode': mode,
        'scientific_digest': source_digest, 'original_code_hash': original['code_hash'],
        'original_code_files': original['code_files'], 'scientific_config': original['scientific_config'],
        'cache_key': cache_key, 'scenarios': [s.name for s in selected],
        'replications_by_scenario': {s.name: config.replications for s in selected},
        'source_manifest': str(source_manifest), 'source_results': str(source_results),
        'source_estimation_manifest': str(estimation_manifest),
        'selection': 'same saved first Gaussian alarm and tau as original NB estimation',
        'point_scope': 'all target-specific finite raw points, including probability-inadmissible values',
        'started_utc': datetime.now(timezone.utc).isoformat()}
    _atomic_json(manifest, stage / 'manifest.json')
    source_cache = source_manifest.parent.parent / 'cache'
    rows, missing = [], []
    for scenario in selected:
        for rep in range(config.replications):
            path = _cache_path(cache_root, cache_key, scenario.name, rep)
            if not path.is_file():
                missing.append((scenario, rep))
                continue
            item = json.loads(path.read_text(encoding='utf-8'))
            if item.get('payload_hash') != _payload_hash(item) or item.get('cache_key') != cache_key:
                raise ValueError('corrupt reference cache')
            row = item['row']
            if row['scenario'] != scenario.name or row['replicate'] != rep:
                raise ValueError('reference cache identifier mismatch')
            source = sources[(scenario.name, rep)]
            payload = _load_cache(_cache_path(source_cache, source_digest, scenario.name, rep),
                digest=source_digest, scenario=scenario, replicate=rep, config=config)
            if payload is None or payload['payload_hash'] != row['source_payload_hash'] or row['source_payload_hash'] != source['source_payload_hash']:
                raise ValueError('original episode changed since reference cache was created')
            _estimation_first_alarm(source, next(r for r in payload['rows'] if r['method'] == _primary_ocfr(config))['trace'])
            for name, column in (('cases', 'case_hash'), ('deaths', 'death_hash'),
                                 ('true_delay', 'true_delay_hash'), ('assumed_delay', 'assumed_delay_hash')):
                if payload['stream_manifest']['hashes'][name] != source[column]:
                    raise ValueError('cached reference source observation hash mismatch')
            rows.append(row)
    def save(row):
        item = {'cache_key': cache_key, 'row': row}
        item['payload_hash'] = _payload_hash(item)
        _atomic_json(item, _cache_path(cache_root, cache_key, row['scenario'], row['replicate']))
        rows.append(row)
    def arguments(task):
        s, rep = task
        return s, rep, config, source_digest, source_cache, sources[(s.name, rep)]
    workers = (os.cpu_count() or 1) if jobs == -1 else jobs
    if workers == 1:
        for task in missing:
            save(_reference_episode(*arguments(task)))
    elif missing:
        with ProcessPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            for future in as_completed([pool.submit(_reference_episode, *arguments(task)) for task in missing]):
                save(future.result())
    reference = pd.DataFrame(rows).drop(columns='source_payload_hash')
    # JSON cache keys are sorted; fix export order so fresh/resumed/parallel
    # runs produce the same CSV bytes and scientific artifact hash.
    reference = reference[['scenario', 'replicate'] + sorted(
        name for name in reference if name.startswith('nishiura_'))]
    frame = base.merge(reference, on=['scenario', 'replicate'], how='left', validate='one_to_one').sort_values(['scenario', 'replicate'], ignore_index=True)
    if len(frame) != 16 * config.replications or frame['nishiura_status'].isna().any():
        raise RuntimeError('incomplete reference episode grid')
    _atomic_csv(frame, stage / 'replicate_results.csv')
    manifest.update(status='complete', tasks_computed=len(missing), cached_tasks_loaded=len(rows)-len(missing),
        elapsed_seconds=time.perf_counter()-started, completed_utc=datetime.now(timezone.utc).isoformat(),
        artifacts={'replicate_results.csv': _file_hash(stage / 'replicate_results.csv')})
    _atomic_json(manifest, stage / 'manifest.json')
    return frame


def _stabilized_observations(scenario, replicate, config, source_digest, source_cache, source_row):
    """Verify the immutable first crossing and regenerate only its observations."""
    payload = _load_cache(_cache_path(source_cache, source_digest, scenario.name, replicate),
        digest=source_digest, scenario=scenario, replicate=replicate, config=config)
    if payload is None or payload['payload_hash'] != source_row['source_payload_hash']:
        raise ValueError('original episode cache does not match stabilized source')
    gaussian = next(r for r in payload['rows'] if r['method'] == _primary_ocfr(config))
    alarm, tau = _estimation_first_alarm(source_row, gaussian['trace'])
    cases, deaths, delay, streams = _compact_generate_components(scenario, config, replicate)
    for name, column in (('cases', 'case_hash'), ('deaths', 'death_hash'),
                         ('true_delay', 'true_delay_hash'), ('assumed_delay', 'assumed_delay_hash')):
        if not (streams['hashes'][name] == payload['stream_manifest']['hashes'][name] == source_row[column]):
            raise ValueError(f'stabilized regenerated {name} hash mismatch')
    if alarm is not None and (tau is None or not 1 <= tau <= alarm or alarm >= len(cases)):
        raise ValueError('invalid saved stabilized alarm/change index')
    return cases, deaths, delay, alarm, tau


def _stabilized_episode(scenario, replicate, config, source_digest, source_cache, source_row, reporting):
    from methods import (MU_FLOOR, NBFit, exposure_components,
                         fit_nb_regularized_identity, regularized_parameter_intervals)
    cases, deaths, delay, alarm, tau = _stabilized_observations(
        scenario, replicate, config, source_digest, source_cache, source_row)
    row = {'scenario': scenario.name, 'replicate': replicate,
           'source_payload_hash': source_row['source_payload_hash'],
           'regularized_status': 'no_alarm', 'regularized_seconds': 0.,
           'regularized_phi': None, 'regularized_loglik': None,
           'regularized_loglik_loss': None, 'regularized_penalized_objective_gain': None}
    for prefix in ('regularized_', 'regularized_unpenalized_'):
        for target in ('pi_left', 'pi_right', 'risk_ratio'):
            for suffix in ('estimate', 'lower', 'upper'):
                row[prefix + target + '_' + suffix] = None
            for suffix in ('point_valid', 'interval_valid'):
                row[prefix + target + '_' + suffix] = False
    for name in ('regularized_pi_left_logit_se', 'regularized_pi_right_logit_se', 'regularized_log_rr_se'):
        row[name] = None
    if alarm is None:
        return row
    total, post = exposure_components(cases[:alarm + 1], delay)
    b = post[:, tau - 1]
    active = total > MU_FLOOR
    design = np.column_stack((total - b, b))[active]
    y, offset = deaths[:alarm + 1][active], np.zeros(int(active.sum()))
    initial = NBFit(np.array([source_row['pi_left_estimate'], source_row['pi_right_estimate']]),
                   float(source_row['phi_estimate']), float(source_row['fit_loglik']),
                   bool(source_row['fit_success']), str(source_row['fit_status']),
                   int(source_row['fit_iterations']))
    started = time.perf_counter()
    fit = fit_nb_regularized_identity(y, offset, design, precision=reporting.precision, initial_fit=initial)
    row.update(regularized_status=fit.status, regularized_seconds=time.perf_counter()-started,
               regularized_phi=float(fit.phi), regularized_loglik=float(fit.loglik),
               regularized_loglik_loss=initial.loglik-fit.loglik)
    penalty = lambda f: -f.loglik + .5 * reporting.precision * np.log(f.rates[1]/f.rates[0])**2
    row['regularized_penalized_objective_gain'] = penalty(initial)-penalty(fit)
    for prefix in ('regularized_', 'regularized_unpenalized_'):
        for target, value in zip(('pi_left', 'pi_right', 'risk_ratio'),
                                 (*fit.rates, fit.rates[1]/fit.rates[0])):
            row[prefix+target+'_estimate'] = float(value)
            row[prefix+target+'_point_valid'] = bool(fit.success and np.isfinite(value))
    if not fit.success:
        return row
    for prefix, covariance in (('regularized_', reporting.covariance),
                                ('regularized_unpenalized_', reporting.diagnostic_covariance)):
        try:
            ci = regularized_parameter_intervals(fit, offset, design, precision=reporting.precision,
                                                confidence=reporting.confidence, covariance=covariance)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            if prefix == 'regularized_':
                row['regularized_status'] += ':interval_unavailable'
            continue
        if prefix == 'regularized_':
            row.update(regularized_pi_left_logit_se=float(ci.cfr_logit_standard_errors[0]),
                       regularized_pi_right_logit_se=float(ci.cfr_logit_standard_errors[1]),
                       regularized_log_rr_se=float(ci.log_risk_ratio_standard_error))
        for target, lower, upper in zip(('pi_left', 'pi_right', 'risk_ratio'),
                (*ci.cfr_lower, ci.risk_ratio_lower), (*ci.cfr_upper, ci.risk_ratio_upper)):
            row[prefix+target+'_lower'], row[prefix+target+'_upper'] = float(lower), float(upper)
            row[prefix+target+'_interval_valid'] = bool(np.isfinite(lower) and np.isfinite(upper) and lower <= upper)
    return row


def run_compact_estimation_stabilized(*, output: Path, mode: str = 'demo', jobs: int = 1,
        reset_scope: ResetScope = 'none', source_manifest: Path | None = None,
        source_results: Path | None = None, reference_manifest: Path | None = None):
    """Report stabilized estimates on frozen alarms, retaining all original columns."""
    from dataclasses import asdict
    from datetime import datetime, timezone
    from config import PostAlarmEstimationConfig
    started = time.perf_counter()
    if mode not in {'demo', 'full'} or jobs == 0 or jobs < -1:
        raise ValueError('invalid stabilized mode or worker count')
    if reset_scope not in {'none', 'selected', 'derived', 'all'}:
        raise ValueError('invalid stabilized reset scope')
    output = Path(output).resolve()
    source_manifest = Path(source_manifest or output/'evaluation'/'manifest.json').resolve()
    source_results = Path(source_results or source_manifest.parent/'replicate_results.csv').resolve()
    reference_manifest = Path(reference_manifest or output/'estimation-reference'/'manifest.json').resolve()
    reference_results = reference_manifest.parent/'replicate_results.csv'
    original = json.loads(source_manifest.read_text(encoding='utf-8'))
    reference = json.loads(reference_manifest.read_text(encoding='utf-8'))
    config = _estimation_source_config(original, mode)
    source_digest = original['scientific_digest']
    code = _sha256_bytes(_canonical_json(original['code_files']).encode('ascii'))
    digest = _sha256_bytes(_canonical_json({'schema_version': SCHEMA_VERSION,
        'code_hash': code, 'scientific': config.scientific_payload()}).encode('ascii'))[:20]
    if original['code_hash'] != code or source_digest != digest:
        raise ValueError('original evaluation scientific/code digest mismatch')
    if (original.get('mode') != 'full' or reference.get('mode') != 'full' or
            reference.get('stage') != 'estimation-reference' or reference.get('status') != 'complete'):
        raise ValueError('stabilized estimation requires completed full evaluation and estimation-reference')
    hashes = {'original_manifest_sha256': _file_hash(source_manifest),
              'original_csv_sha256': _file_hash(source_results),
              'original_reference_manifest_sha256': _file_hash(reference_manifest),
              'original_reference_csv_sha256': _file_hash(reference_results)}
    if (original.get('artifacts', {}).get('replicate_results.csv') != hashes['original_csv_sha256'] or
            reference.get('artifacts', {}).get('replicate_results.csv') != hashes['original_reference_csv_sha256'] or
            reference.get('original_manifest_sha256') != hashes['original_manifest_sha256'] or
            reference.get('original_csv_sha256') != hashes['original_csv_sha256'] or
            reference.get('original_scientific_digest') != source_digest):
        raise ValueError('frozen evaluation/reference provenance mismatch')
    selected = [s for s in config.scenarios if s.split == 'alternative' and s.misspecification == 'on_model']
    base = pd.read_csv(reference_results)
    required = {'scenario', 'replicate', 'alarm_day', 'tau_hat_at_alarm', 'source_payload_hash',
                'case_hash', 'death_hash', 'true_delay_hash', 'assumed_delay_hash',
                'fit_success', 'fit_status', 'fit_iterations', 'fit_loglik', 'phi_estimate', 'nishiura_status'}
    for target in ('pi_left', 'pi_right', 'risk_ratio'):
        required.update(target+'_'+s for s in ('estimate', 'lower', 'upper', 'point_valid', 'interval_valid'))
        required.update('nishiura_'+target+'_'+s for s in ('estimate', 'point_valid', 'admissible'))
    if not required.issubset(base.columns) or any(c.startswith('regularized_') for c in base):
        raise ValueError('reference source columns are incomplete or already stabilized')
    expected = {(s.name, rep) for s in selected for rep in range(1000)}
    if (len(selected) != 16 or len(base) != len(expected) or
            set(zip(base.scenario, base.replicate)) != expected or base.duplicated(['scenario', 'replicate']).any()):
        raise ValueError('frozen reference episode grid is incomplete')
    base = base.loc[base.replicate < config.replications].copy()
    sources = {(r['scenario'], r['replicate']): r for r in base.to_dict('records')}
    reporting = PostAlarmEstimationConfig()
    post_hash, post_files = _code_hash()
    native_files = {p.name: _file_hash(p) for p in Path(__file__).resolve().parent.glob('*')
                    if p.is_file() and p.suffix in {'.dll', '.so', '.dylib', '.pyd'}}
    provenance = {**hashes, 'original_scientific_digest': source_digest,
                  'postprocessor_code_hash': post_hash, 'postprocessor_code_files': post_files,
                  'postprocessor_native_files': native_files, 'post_alarm_estimation_config': asdict(reporting)}
    key = _sha256_bytes(_canonical_json(provenance).encode('ascii'))[:20]
    stage, source_cache = output/'estimation-stabilized', source_manifest.parent.parent/'cache'
    cache_root = stage/'cache'
    target = (cache_root/key).resolve()
    if not target.is_relative_to(stage.resolve()):
        raise ValueError('unsafe stabilized cache reset')
    if reset_scope in {'selected', 'all'} and target.is_dir():
        shutil.rmtree(target)
    if reset_scope != 'none':
        for name in ('manifest.json', 'replicate_results.csv'):
            (stage/name).unlink(missing_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    manifest = {**provenance, 'stage': 'estimation-stabilized', 'status': 'running', 'mode': mode,
        'scientific_digest': source_digest, 'scientific_config': original['scientific_config'],
        'original_code_hash': original['code_hash'], 'original_code_files': original['code_files'],
        'cache_key': key, 'scenarios': [s.name for s in selected],
        'replications_by_scenario': {s.name: config.replications for s in selected},
        'source_manifest': str(source_manifest), 'source_results': str(source_results),
        'source_reference_manifest': str(reference_manifest), 'source_reference_results': str(reference_results),
        'selection': 'same saved first Gaussian alarm and tau as original NB estimation',
        'primary_estimator': reporting.estimator, 'primary_covariance': reporting.covariance,
        'interval_scope': 'approximate finite-penalty sandwich; no selection or stopping adjustment',
        'started_utc': datetime.now(timezone.utc).isoformat()}
    _atomic_json(manifest, stage/'manifest.json')
    rows, missing = [], []
    for scenario in selected:
        for rep in range(config.replications):
            path = _cache_path(cache_root, key, scenario.name, rep)
            if not path.is_file():
                missing.append((scenario, rep))
                continue
            item = json.loads(path.read_text(encoding='utf-8'))
            if item.get('payload_hash') != _payload_hash(item) or item.get('cache_key') != key:
                raise ValueError('corrupt stabilized cache')
            row, source = item['row'], sources[(scenario.name, rep)]
            if (row['scenario'], row['replicate'], row['source_payload_hash']) != (
                    scenario.name, rep, source['source_payload_hash']):
                raise ValueError('stabilized cache source/identifier mismatch')
            _stabilized_observations(scenario, rep, config, source_digest, source_cache, source)
            rows.append(row)
    def save(row):
        item = {'cache_key': key, 'row': row}
        item['payload_hash'] = _payload_hash(item)
        _atomic_json(item, _cache_path(cache_root, key, row['scenario'], row['replicate']))
        rows.append(row)
    def arguments(task):
        s, rep = task
        return s, rep, config, source_digest, source_cache, sources[(s.name, rep)], reporting
    workers = (os.cpu_count() or 1) if jobs == -1 else jobs
    if workers == 1:
        for task in missing:
            save(_stabilized_episode(*arguments(task)))
    elif missing:
        with ProcessPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            for future in as_completed([pool.submit(_stabilized_episode, *arguments(task)) for task in missing]):
                save(future.result())
    extra = pd.DataFrame(rows).drop(columns='source_payload_hash')
    extra = extra[['scenario', 'replicate']+sorted(c for c in extra if c.startswith('regularized_'))]
    frame = base.merge(extra, on=['scenario', 'replicate'], validate='one_to_one').sort_values(
        ['scenario', 'replicate'], ignore_index=True)
    if len(frame) != 16*config.replications or frame.regularized_status.isna().any():
        raise RuntimeError('incomplete stabilized episode grid')
    _atomic_csv(frame, stage/'replicate_results.csv')
    manifest.update(status='complete', tasks_computed=len(missing), cached_tasks_loaded=len(rows)-len(missing),
        completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
        artifacts={'replicate_results.csv': _file_hash(stage/'replicate_results.csv')})
    _atomic_json(manifest, stage/'manifest.json')
    return frame


def _validation_first_alarm(scenario, replicate, config):
    """Independent-seed Gaussian policy, stopping at its first crossing."""
    cases, deaths, delay, streams = _compact_generate_components(scenario, config, replicate)
    bank = gaussian_multiplier_bank(make_sampler(streams['seeds']['multiplier'], backend=config.methods.backend),
        (config.methods.n_multiplier, scenario.horizon), normal_mode="pcg32-numpy-box-muller-v1" if config.protocol == "compact-v2" else "scalar")
    looks = _planned_looks(scenario, config)
    spending = GaussianSpendingBoundary(config.methods.n_multiplier, len(looks), config.methods.alpha) if config.protocol == "compact-v2" else None
    alarm = tau = None
    trace = []
    for prefix in looks:
        scan = score_scan(cases[:prefix], deaths[:prefix], delay,
            use_lindeberg_gate=config.methods.use_lindeberg_gate, backend=config.methods.backend)
        if not scan.candidates.size or not np.isfinite(scan.statistic) or (config.protocol == "compact-v2" and not scan.null_fit.success):
            diagnostic = asdict(spending.advance(available=False)) if spending is not None else {}
            trace.append(dict(day=prefix-1, **dict(diagnostic, available=False)))
            continue
        maxima = multiplier_statistics_from_draws(bank[:, :prefix], scan.influence, backend=config.methods.backend)
        boundary_look = spending.advance(maxima) if spending is not None else None
        boundary = boundary_look.boundary if boundary_look is not None else monte_carlo_critical_value(maxima, config.methods.alpha/len(looks))
        diagnostic = asdict(boundary_look) if boundary_look is not None else {}
        trace.append(dict(day=prefix-1, available=True, statistic=float(scan.statistic), boundary=float(boundary),
                          **{key: value for key, value in diagnostic.items() if key not in {"available", "boundary"}}))
        if scan.statistic > boundary:
            alarm, tau = prefix-1, int(scan.tau_hat)
            break
    return cases, deaths, delay, streams, alarm, tau, trace


def _validation_episode(scenario, replicate, config, reporting):
    """All estimators use the same newly generated first-alarm prefix."""
    from methods import (MU_FLOOR, NBFit, exposure_components, fit_nb_identity,
        fit_nb_regularized_identity, fixed_tau_parameter_intervals,
        regularized_parameter_intervals, nishiura_two_stage_moments)
    started = time.perf_counter()
    cases, deaths, delay, streams, alarm, tau, trace = _validation_first_alarm(scenario, replicate, config)
    row = dict(scenario=scenario.name, replicate=replicate, method=_primary_ocfr(config),
        family=scenario.family, horizon=scenario.horizon, true_change=scenario.true_change,
        pi_left=scenario.pi_left, pi_right=scenario.pi_right, risk_ratio=scenario.risk_ratio, phi=scenario.phi,
        alarm=alarm is not None, no_alarm=alarm is None, alarm_day=alarm, tau_hat_at_alarm=tau,
        premature_alarm=alarm is not None and alarm < scenario.true_change,
        postchange_detection=alarm is not None and alarm >= scenario.true_change,
        fit_success=False, fit_status='no_alarm', fit_iterations=0, fit_loglik=None, phi_estimate=None,
        regularized_status='no_alarm', regularized_seconds=0., regularized_phi=None,
        regularized_loglik=None, regularized_loglik_loss=None, regularized_penalized_objective_gain=None,
        confidence=reporting.confidence, trace_json=_diagnostic_json(trace),
        case_hash=streams['hashes']['cases'], death_hash=streams['hashes']['deaths'],
        true_delay_hash=streams['hashes']['true_delay'], assumed_delay_hash=streams['hashes']['assumed_delay'],
        stream_manifest_json=_canonical_json(streams))
    for name in ('regularized_pi_left_logit_se', 'regularized_pi_right_logit_se', 'regularized_log_rr_se'):
        row[name] = None
    for prefix in ('', 'regularized_', 'regularized_unpenalized_', 'nishiura_'):
        for target in ('pi_left', 'pi_right', 'risk_ratio'):
            row[prefix+target+'_estimate'] = None
            row[prefix+target+'_point_valid'] = False
            if prefix != 'nishiura_':
                row[prefix+target+'_lower'] = row[prefix+target+'_upper'] = None
                row[prefix+target+'_interval_valid'] = False
            else:
                row[prefix+target+'_admissible'] = False
    row['nishiura_method'], row['nishiura_status'] = 'Nishiura-inspired two-stage moments', 'no_alarm'
    if alarm is not None:
        total, post = exposure_components(cases[:alarm+1], delay)
        b = post[:, tau-1]
        active = total > MU_FLOOR
        design = np.column_stack([total-b, b])[active]
        y, offset = deaths[:alarm+1][active], np.zeros(int(active.sum()))
        ordinary = fit_nb_identity(y, offset, design)
        row.update(fit_success=ordinary.success, fit_status=ordinary.status, fit_iterations=ordinary.nit,
                   fit_loglik=ordinary.loglik, phi_estimate=ordinary.phi)
        began = time.perf_counter()
        fit = fit_nb_regularized_identity(y, offset, design, precision=reporting.precision, initial_fit=ordinary)
        row.update(regularized_seconds=time.perf_counter()-began, regularized_status=fit.status,
            regularized_phi=fit.phi, regularized_loglik=fit.loglik,
            regularized_loglik_loss=ordinary.loglik-fit.loglik,
            regularized_penalized_objective_gain=(-ordinary.loglik+.5*reporting.precision*np.log(ordinary.rates[1]/ordinary.rates[0])**2)
                -(-fit.loglik+.5*reporting.precision*np.log(fit.rates[1]/fit.rates[0])**2))
        for prefix, estimate, covariance in (('', ordinary, 'unpenalized'),
                ('regularized_', fit, reporting.covariance),
                ('regularized_unpenalized_', fit, reporting.diagnostic_covariance)):
            if not estimate.success:
                continue
            for target, value in zip(('pi_left', 'pi_right', 'risk_ratio'),
                                     (*estimate.rates, estimate.rates[1]/estimate.rates[0])):
                row[prefix+target+'_estimate'] = float(value)
                row[prefix+target+'_point_valid'] = bool(np.isfinite(value))
            try:
                ci = (fixed_tau_parameter_intervals(estimate, offset, design, confidence=reporting.confidence)
                      if prefix == '' else regularized_parameter_intervals(estimate, offset, design,
                      precision=reporting.precision, confidence=reporting.confidence, covariance=covariance))
                if prefix == 'regularized_':
                    row.update(regularized_pi_left_logit_se=float(ci.cfr_logit_standard_errors[0]),
                               regularized_pi_right_logit_se=float(ci.cfr_logit_standard_errors[1]),
                               regularized_log_rr_se=float(ci.log_risk_ratio_standard_error))
                for target, lower, upper in zip(('pi_left', 'pi_right', 'risk_ratio'),
                        (*ci.cfr_lower, ci.risk_ratio_lower), (*ci.cfr_upper, ci.risk_ratio_upper)):
                    row[prefix+target+'_lower'], row[prefix+target+'_upper'] = float(lower), float(upper)
                    row[prefix+target+'_interval_valid'] = bool(np.isfinite(lower) and np.isfinite(upper) and lower <= upper)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                if prefix == 'regularized_':
                    row['regularized_status'] += ':interval_unavailable'
        reference = nishiura_two_stage_moments(cases[:alarm+1], deaths[:alarm+1], delay, tau=tau)
        row.update({'nishiura_'+name: value for name, value in reference.items()})
    row['elapsed_seconds'] = time.perf_counter()-started
    return row


def run_compact_estimation_validation(*, output: Path, mode: str = 'demo', jobs: int = 1,
        reset_scope: ResetScope = 'none', source_manifest: Path | None = None,
        source_results: Path | None = None):
    """Reproduce independent-seed first-alarm validation without private scripts.

    This stage runs Gaussian detection on new episodes. Full mode is expensive;
    use existing complete, provenance-verified results when available.
    """
    from dataclasses import asdict, replace
    from datetime import datetime, timezone
    from config import PostAlarmEstimationConfig, PostAlarmValidationConfig
    started = time.perf_counter()
    if mode not in {'demo', 'full'} or jobs == 0 or jobs < -1:
        raise ValueError('invalid validation mode or worker count')
    if reset_scope not in {'none', 'selected', 'derived', 'all'}:
        raise ValueError('invalid validation reset scope')
    output = Path(output).resolve()
    source_manifest = Path(source_manifest or output/'evaluation'/'manifest.json').resolve()
    source_results = Path(source_results or source_manifest.parent/'replicate_results.csv').resolve()
    original = json.loads(source_manifest.read_text(encoding='utf-8'))
    original_config = _estimation_source_config(original, mode)
    code = _sha256_bytes(_canonical_json(original['code_files']).encode('ascii'))
    expected_digest = _sha256_bytes(_canonical_json({'schema_version': SCHEMA_VERSION,
        'code_hash': code, 'scientific': original_config.scientific_payload()}).encode('ascii'))[:20]
    if (original.get('mode') != 'full' or original['code_hash'] != code or
            original['scientific_digest'] != expected_digest or
            original.get('artifacts', {}).get('replicate_results.csv') != _file_hash(source_results)):
        raise ValueError('independent validation requires intact completed full evaluation provenance')
    validation, reporting = get_post_alarm_validation_config(original_config.protocol), PostAlarmEstimationConfig()
    config = replace(original_config, master_seed=validation.validation_master_seed)
    selected = [s for s in config.scenarios if s.split == 'alternative' and s.misspecification == 'on_model']
    if len(selected) != 16:
        raise ValueError('independent validation requires all 16 on-model alternative cells')
    post_hash, post_files = _code_hash()
    native_files = {p.name: _file_hash(p) for p in Path(__file__).resolve().parent.glob('*')
                    if p.is_file() and p.suffix in {'.dll', '.so', '.dylib', '.pyd'}}
    provenance = dict(original_manifest_sha256=_file_hash(source_manifest),
        original_csv_sha256=_file_hash(source_results), original_scientific_digest=expected_digest,
        postprocessor_code_hash=post_hash, postprocessor_code_files=post_files,
        postprocessor_native_files=native_files, validation_config=asdict(validation),
        post_alarm_estimation_config=asdict(reporting), scientific_config=config.scientific_payload())
    key = _sha256_bytes(_canonical_json(provenance).encode('ascii'))[:20]
    stage, rows, missing = output/'estimation-validation', [], []
    cache_root = stage/'cache'
    target = (cache_root/key).resolve()
    if not target.is_relative_to(stage.resolve()):
        raise ValueError('unsafe validation cache reset')
    if reset_scope in {'selected', 'all'} and target.is_dir():
        shutil.rmtree(target)
    if reset_scope != 'none':
        for name in ('manifest.json', 'replicate_results.csv'):
            (stage/name).unlink(missing_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    manifest = {**provenance, 'stage': 'estimation-validation', 'status': 'running', 'mode': mode,
        'scientific_digest': _sha256_bytes(_canonical_json(config.scientific_payload()).encode('ascii'))[:20],
        'cache_key': key, 'scenarios': [s.name for s in selected],
        'replications_by_scenario': {s.name: config.replications for s in selected},
        'source_manifest': str(source_manifest), 'source_results': str(source_results),
        'primary_estimator': reporting.estimator, 'primary_covariance': reporting.covariance,
        'selection': 'independent-seed first Gaussian crossing; fixed bank and all planned looks in alpha allocation',
        'interval_scope': 'approximate finite-penalty sandwich; no selection or stopping adjustment',
        'started_utc': datetime.now(timezone.utc).isoformat()}
    _atomic_json(manifest, stage/'manifest.json')
    for scenario in selected:
        for rep in range(config.replications):
            path = _cache_path(cache_root, key, scenario.name, rep)
            if not path.is_file():
                missing.append((scenario, rep))
                continue
            item = json.loads(path.read_text(encoding='utf-8'))
            if item.get('payload_hash') != _payload_hash(item) or item.get('cache_key') != key:
                raise ValueError('corrupt validation cache')
            row = item['row']
            if (row['scenario'], row['replicate']) != (scenario.name, rep):
                raise ValueError('validation cache identifier mismatch')
            rows.append(row)
    def save(row):
        item = {'cache_key': key, 'row': row}
        item['payload_hash'] = _payload_hash(item)
        _atomic_json(item, _cache_path(cache_root, key, row['scenario'], row['replicate']))
        rows.append(row)
    workers = (os.cpu_count() or 1) if jobs == -1 else jobs
    if workers == 1:
        for scenario, rep in missing:
            save(_validation_episode(scenario, rep, config, reporting))
    elif missing:
        with ProcessPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            for future in as_completed([pool.submit(_validation_episode, s, rep, config, reporting) for s, rep in missing]):
                save(future.result())
    frame = pd.DataFrame(rows)
    frame = frame[['scenario', 'replicate']+sorted(c for c in frame if c not in {'scenario', 'replicate'})]
    frame = frame.sort_values(['scenario', 'replicate'], ignore_index=True)
    if len(frame) != 16*config.replications or frame.duplicated(['scenario', 'replicate']).any():
        raise RuntimeError('incomplete independent validation episode grid')
    _atomic_csv(frame, stage/'replicate_results.csv')
    manifest.update(status='complete', tasks_computed=len(missing), cached_tasks_loaded=len(rows)-len(missing),
        completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
        artifacts={'replicate_results.csv': _file_hash(stage/'replicate_results.csv')})
    _atomic_json(manifest, stage/'manifest.json')
    return frame


def _poisson_saved_indices(source_row, horizon):
    def integer(name):
        value = source_row.get(name)
        if value is None or pd.isna(value):
            return None
        if not np.isfinite(float(value)) or float(value) != int(value):
            raise ValueError('invalid saved Poisson alarm/change index')
        return int(value)
    alarm, tau = integer('alarm_day'), integer('tau_hat_at_alarm')
    if ((alarm is None) != (tau is None) or
            alarm is not None and not 1 <= tau <= alarm < horizon):
        raise ValueError('invalid saved Poisson alarm/change index')
    for name, expected in (('alarm', alarm is not None), ('no_alarm', alarm is None)):
        if name in source_row and source_row[name] != expected:
            raise ValueError('saved Poisson alarm flag/index mismatch')
    return alarm, tau


def _poisson_observations(scenario, replicate, config, bank, source_digest,
                          source_cache, source_row):
    """Regenerate observations and verify saved decisions without any detector."""
    saved = _poisson_saved_indices(source_row, scenario.horizon)
    if bank == 'original':
        result = _stabilized_observations(scenario, replicate, config, source_digest,
                                          source_cache, source_row)
        if result[-2:] != saved:
            raise ValueError('saved Poisson original indices do not replay')
        return result
    cases, deaths, delay, streams = _compact_generate_components(scenario, config, replicate)
    original_streams = json.loads(source_row['stream_manifest_json'])
    for name, column in (('cases', 'case_hash'), ('deaths', 'death_hash'),
                         ('true_delay', 'true_delay_hash'), ('assumed_delay', 'assumed_delay_hash')):
        if not (streams['hashes'][name] == source_row[column] == original_streams['hashes'][name]):
            raise ValueError(f'Poisson validation regenerated {name} hash mismatch')
    if streams['seeds'] != original_streams['seeds']:
        raise ValueError('Poisson validation regenerated seed mismatch')
    trace = json.loads(source_row['trace_json'])
    first = next((r for r in trace if r.get('available') and
                  np.isfinite(r['statistic']) and r['statistic'] > r['boundary']), None)
    if (int(first['day']) if first else None) != saved[0]:
        raise ValueError('saved Poisson validation first crossing does not replay')
    return cases, deaths, delay, *saved


def _poisson_episode(scenario, replicate, config, bank, source_digest,
                     source_cache, source_row, reporting):
    from methods import MU_FLOOR, exposure_components, fit_poisson_identity
    cases, deaths, delay, alarm, tau = _poisson_observations(
        scenario, replicate, config, bank, source_digest, source_cache, source_row)
    row = dict(scenario=scenario.name, replicate=replicate)
    for prefix in ('poisson_regularized_', 'poisson_unpenalized_'):
        row.update({prefix+'status': 'no_alarm', prefix+'seconds': 0.,
                    prefix+'loglik': None, prefix+'iterations': 0})
        for target in ('pi_left', 'pi_right', 'risk_ratio'):
            row[prefix+target+'_estimate'] = None
            row[prefix+target+'_point_valid'] = False
    if alarm is None:
        return row
    total, post = exposure_components(cases[:alarm+1], delay)
    b = post[:, tau-1]
    active = total > MU_FLOOR
    design = np.column_stack((total-b, b))[active]
    y, offset = deaths[:alarm+1][active], np.zeros(int(active.sum()))
    for prefix, precision in (('poisson_regularized_', reporting.primary_precision),
                              ('poisson_unpenalized_', reporting.control_precision)):
        started = time.perf_counter()
        fit = fit_poisson_identity(y, offset, design, precision=precision)
        row.update({prefix+'status': fit.status, prefix+'seconds': time.perf_counter()-started,
                    prefix+'loglik': float(fit.loglik) if np.isfinite(fit.loglik) else None,
                    prefix+'iterations': fit.nit})
        for target, value in zip(('pi_left', 'pi_right', 'risk_ratio'),
                                  (*fit.rates, fit.rates[1]/fit.rates[0])):
            valid = bool(fit.success and np.isfinite(value) and value > 0.)
            row[prefix+target+'_estimate'] = float(value) if valid else None
            row[prefix+target+'_point_valid'] = valid
    return row


def _poisson_source(source_manifest, source_results, bank, mode):
    """Validate the complete frozen source bank before selecting demo rows."""
    manifest = json.loads(source_manifest.read_text(encoding='utf-8'))
    expected_stage = 'estimation-stabilized' if bank == 'original' else 'estimation-validation'
    if (manifest.get('stage') != expected_stage or manifest.get('status') != 'complete'
            or manifest.get('mode') != 'full'):
        raise ValueError('Poisson comparison requires a completed full frozen source bank')
    if manifest.get('artifacts', {}).get('replicate_results.csv') != _file_hash(source_results):
        raise ValueError('frozen Poisson source CSV hash mismatch')
    config = _estimation_source_config(dict(manifest, stage='evaluation'), mode)
    from config import PostAlarmEstimationConfig, PostAlarmValidationConfig
    if manifest.get('post_alarm_estimation_config') != __import__('dataclasses').asdict(PostAlarmEstimationConfig()):
        raise ValueError('Poisson comparison requires the fixed precision-1 NB source')
    if bank == 'validation':
        if (config.master_seed != get_post_alarm_validation_config(config.protocol).validation_master_seed or
                manifest.get('validation_config') != __import__('dataclasses').asdict(get_post_alarm_validation_config(config.protocol))):
            raise ValueError('Poisson comparison validation seed/config mismatch')
        expected = _sha256_bytes(_canonical_json(config.scientific_payload()).encode('ascii'))[:20]
    else:
        code = _sha256_bytes(_canonical_json(manifest['original_code_files']).encode('ascii'))
        if code != manifest['original_code_hash']:
            raise ValueError('Poisson original code digest mismatch')
        expected = _sha256_bytes(_canonical_json(dict(schema_version=SCHEMA_VERSION,
            code_hash=code, scientific=config.scientific_payload())).encode('ascii'))[:20]
    if manifest.get('scientific_digest') != expected:
        raise ValueError('Poisson source scientific digest mismatch')
    selected = [s for s in config.scenarios if s.split == 'alternative' and s.misspecification == 'on_model']
    base = pd.read_csv(source_results)
    required = {'scenario', 'replicate', 'alarm_day', 'tau_hat_at_alarm', 'regularized_status',
                'case_hash', 'death_hash', 'true_delay_hash', 'assumed_delay_hash'}
    required.update({'source_payload_hash'} if bank == 'original' else {'trace_json', 'stream_manifest_json'})
    if not required.issubset(base.columns) or any(c.startswith('poisson_') for c in base):
        raise ValueError('Poisson source columns incomplete or already processed')
    expected_grid = {(s.name, rep) for s in selected for rep in range(1000)}
    if (len(selected) != 16 or len(base) != len(expected_grid) or
            set(zip(base.scenario, base.replicate)) != expected_grid or
            base.duplicated(['scenario', 'replicate']).any() or
            manifest.get('replications_by_scenario') != {s.name: 1000 for s in selected}):
        raise ValueError('frozen Poisson source episode grid is incomplete')
    base = base.loc[base.replicate < config.replications].copy()
    return manifest, config, selected, base


def run_compact_estimation_poisson(*, output: Path, mode: str = 'demo', jobs: int = 1,
        reset_scope: ResetScope = 'none', bank: str = 'both',
        source_manifest: Path | None = None, source_results: Path | None = None):
    """Add exact Poisson points to frozen original and/or validation episodes.

    Source overrides identify one frozen reporting bank and require an explicit
    ``bank='original'`` or ``bank='validation'``. No detection or Gaussian draws
    are replayed, and the validation bank is reused rather than newly independent.
    """
    from dataclasses import asdict
    from datetime import datetime, timezone
    from config import PostAlarmPoissonConfig
    if (mode not in {'demo', 'full'} or jobs == 0 or jobs < -1 or
            bank not in {'original', 'validation', 'both'} or
            reset_scope not in {'none', 'selected', 'derived', 'all'}):
        raise ValueError('invalid Poisson stage mode, bank, workers or reset scope')
    if bank == 'both' and (source_manifest is not None or source_results is not None):
        raise ValueError('source overrides are ambiguous with bank=both; select original or validation')
    output = Path(output).resolve()
    if bank == 'both':
        frames = [run_compact_estimation_poisson(output=output, mode=mode, jobs=jobs,
            reset_scope=reset_scope, bank=b).assign(estimation_bank=b) for b in ('original', 'validation')]
        return pd.concat(frames, ignore_index=True)
    started = time.perf_counter()
    name = 'estimation-stabilized' if bank == 'original' else 'estimation-validation'
    source_manifest = Path(source_manifest or output/name/'manifest.json').resolve()
    source_results = Path(source_results or source_manifest.parent/'replicate_results.csv').resolve()
    original, config, selected, base = _poisson_source(source_manifest, source_results, bank, mode)
    reporting = PostAlarmPoissonConfig()
    post_hash, post_files = _code_hash()
    native_files = {p.name: _file_hash(p) for p in Path(__file__).resolve().parent.glob('*')
                    if p.is_file() and p.suffix in {'.dll', '.so', '.dylib', '.pyd'}}
    source_digest = original['scientific_digest']
    # A copied reporting bank can still use the evaluation cache recorded in its
    # manifest; the usual public layout resolves relative to the source bank.
    source_cache = source_manifest.parent.parent/'cache'
    if bank == 'original' and not (source_cache/source_digest).is_dir():
        recorded = Path(original.get('source_manifest', ''))
        if recorded.is_file():
            source_cache = recorded.parent.parent/'cache'
    provenance = dict(source_manifest_sha256=_file_hash(source_manifest),
        source_csv_sha256=_file_hash(source_results), source_bank=bank,
        source_scientific_digest=source_digest, scientific_config=config.scientific_payload(),
        postprocessor_code_hash=post_hash, postprocessor_code_files=post_files,
        postprocessor_native_files=native_files, poisson_estimation_config=asdict(reporting))
    key = _sha256_bytes(_canonical_json(provenance).encode('ascii'))[:20]
    stage = output/('estimation-poisson-'+bank)
    cache_root = stage/'cache'
    target = (cache_root/key).resolve()
    if not target.is_relative_to(stage.resolve()):
        raise ValueError('unsafe Poisson cache reset')
    if reset_scope in {'selected', 'all'} and target.is_dir():
        shutil.rmtree(target)
    if reset_scope != 'none':
        for filename in ('manifest.json', 'replicate_results.csv'):
            (stage/filename).unlink(missing_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    manifest = dict(provenance, stage='estimation-poisson-'+bank, status='running', mode=mode,
        scientific_digest=source_digest, cache_key=key,
        scenarios=[s.name for s in selected],
        replications_by_scenario={s.name: config.replications for s in selected},
        source_manifest=str(source_manifest), source_results=str(source_results),
        selection='same saved first Gaussian alarm and change index as frozen source',
        artifact_origin='point-only reanalysis of existing frozen bank; no new independent episodes',
        primary_estimator='poisson_regularized', control_estimator='poisson_unpenalized',
        interval_scope='point_only; no Poisson confidence intervals',
        started_utc=datetime.now(timezone.utc).isoformat())
    _atomic_json(manifest, stage/'manifest.json')
    sources = {(r['scenario'], r['replicate']): r for r in base.to_dict('records')}
    rows, missing = [], []
    for scenario in selected:
        for rep in range(config.replications):
            source = sources[(scenario.name, rep)]
            path = _cache_path(cache_root, key, scenario.name, rep)
            if not path.is_file():
                missing.append((scenario, rep))
                continue
            item = json.loads(path.read_text(encoding='utf-8'))
            if item.get('payload_hash') != _payload_hash(item) or item.get('cache_key') != key:
                raise ValueError('corrupt Poisson episode cache')
            row = item['row']
            if (row['scenario'], row['replicate']) != (scenario.name, rep):
                raise ValueError('Poisson cache identifier mismatch')
            _poisson_observations(scenario, rep, config, bank, source_digest, source_cache, source)
            rows.append(row)
    def save(row):
        item = dict(cache_key=key, row=row)
        item['payload_hash'] = _payload_hash(item)
        _atomic_json(item, _cache_path(cache_root, key, row['scenario'], row['replicate']))
        rows.append(row)
    def arguments(task):
        scenario, rep = task
        return scenario, rep, config, bank, source_digest, source_cache, sources[(scenario.name, rep)], reporting
    workers = (os.cpu_count() or 1) if jobs == -1 else jobs
    if workers == 1:
        for task in missing:
            save(_poisson_episode(*arguments(task)))
    elif missing:
        with ProcessPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            for future in as_completed([pool.submit(_poisson_episode, *arguments(task)) for task in missing]):
                save(future.result())
    extra = pd.DataFrame(rows)
    extra = extra[['scenario', 'replicate']+sorted(c for c in extra if c.startswith('poisson_'))]
    frame = base.merge(extra, on=['scenario', 'replicate'], validate='one_to_one').sort_values(
        ['scenario', 'replicate'], ignore_index=True)
    if len(frame) != 16*config.replications or frame.poisson_regularized_status.isna().any():
        raise RuntimeError('incomplete Poisson episode grid')
    _atomic_csv(frame, stage/'replicate_results.csv')
    manifest.update(status='complete', tasks_computed=len(missing), cached_tasks_loaded=len(rows)-len(missing),
        completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
        artifacts={'replicate_results.csv': _file_hash(stage/'replicate_results.csv')})
    _atomic_json(manifest, stage/'manifest.json')
    return frame


def _ls_episode(scenario, replicate, config, bank, source_digest,
                source_cache, source_row, reporting):
    """Daily bounded RSS fit at the source bank's saved alarm and change day."""
    from methods import MU_FLOOR, exposure_components, fit_ls_identity
    cases, deaths, delay, alarm, tau = _poisson_observations(
        scenario, replicate, config, bank, source_digest, source_cache, source_row)
    row = dict(scenario=scenario.name, replicate=replicate, ls_status='no_alarm',
        ls_seconds=0., ls_rss=None, ls_active_set='none',
        ls_kkt_residual=None, ls_kkt_tolerance=None)
    for target in ('pi_left', 'pi_right', 'risk_ratio'):
        row['ls_'+target+'_estimate'] = None
        row['ls_'+target+'_point_valid'] = False
    if alarm is None:
        return row
    total, post = exposure_components(cases[:alarm+1], delay)
    b = post[:, tau-1]
    active = total > MU_FLOOR
    design = np.column_stack((total-b, b))[active]
    y, offset = deaths[:alarm+1][active], np.zeros(int(active.sum()))
    started = time.perf_counter()
    fit = fit_ls_identity(y, offset, design)
    row.update(ls_status=fit.status, ls_seconds=time.perf_counter()-started,
        ls_rss=float(fit.rss) if np.isfinite(fit.rss) else None,
        ls_active_set=fit.active_set,
        ls_kkt_residual=float(fit.kkt_residual) if np.isfinite(fit.kkt_residual) else None,
        ls_kkt_tolerance=fit.kkt_tolerance)
    for target, value in zip(('pi_left', 'pi_right', 'risk_ratio'),
                              (*fit.rates, fit.rates[1]/fit.rates[0])):
        valid = bool(fit.success and np.isfinite(value) and value > 0.)
        row['ls_'+target+'_estimate'] = float(value) if valid else None
        row['ls_'+target+'_point_valid'] = valid
    return row


def run_compact_estimation_ls(*, output: Path, mode: str = 'demo', jobs: int = 1,
        reset_scope: ResetScope = 'none', bank: str = 'both',
        source_manifest: Path | None = None, source_results: Path | None = None):
    """Add exact LS points to frozen original and/or validation episodes.

    Source overrides identify one frozen reporting bank and require an explicit
    ``bank='original'`` or ``bank='validation'``. No detection or Gaussian draws
    are replayed, and the validation bank is reused rather than newly independent.
    """
    from dataclasses import asdict
    from datetime import datetime, timezone
    from config import PostAlarmLSConfig
    if (mode not in {'demo', 'full'} or jobs == 0 or jobs < -1 or
            bank not in {'original', 'validation', 'both'} or
            reset_scope not in {'none', 'selected', 'derived', 'all'}):
        raise ValueError('invalid LS stage mode, bank, workers or reset scope')
    if bank == 'both' and (source_manifest is not None or source_results is not None):
        raise ValueError('source overrides are ambiguous with bank=both; select original or validation')
    output = Path(output).resolve()
    if bank == 'both':
        frames = [run_compact_estimation_ls(output=output, mode=mode, jobs=jobs,
            reset_scope=reset_scope, bank=b).assign(estimation_bank=b) for b in ('original', 'validation')]
        return pd.concat(frames, ignore_index=True)
    started = time.perf_counter()
    name = 'estimation-stabilized' if bank == 'original' else 'estimation-validation'
    source_manifest = Path(source_manifest or output/name/'manifest.json').resolve()
    source_results = Path(source_results or source_manifest.parent/'replicate_results.csv').resolve()
    original, config, selected, base = _poisson_source(source_manifest, source_results, bank, mode)
    if any(c.startswith('ls_') for c in base):
        raise ValueError('LS source already contains LS reporting columns')
    reporting = PostAlarmLSConfig()
    post_hash, post_files = _code_hash()
    native_files = {p.name: _file_hash(p) for p in Path(__file__).resolve().parent.glob('*')
                    if p.is_file() and p.suffix in {'.dll', '.so', '.dylib', '.pyd'}}
    source_digest = original['scientific_digest']
    # A copied reporting bank can still use the evaluation cache recorded in its
    # manifest; the usual public layout resolves relative to the source bank.
    source_cache = source_manifest.parent.parent/'cache'
    if bank == 'original' and not (source_cache/source_digest).is_dir():
        recorded = Path(original.get('source_manifest', ''))
        if recorded.is_file():
            source_cache = recorded.parent.parent/'cache'
    provenance = dict(source_manifest_sha256=_file_hash(source_manifest),
        source_csv_sha256=_file_hash(source_results), source_bank=bank,
        source_scientific_digest=source_digest, scientific_config=config.scientific_payload(),
        postprocessor_code_hash=post_hash, postprocessor_code_files=post_files,
        postprocessor_native_files=native_files, ls_estimation_config=asdict(reporting))
    key = _sha256_bytes(_canonical_json(provenance).encode('ascii'))[:20]
    stage = output/('estimation-ls-'+bank)
    cache_root = stage/'cache'
    target = (cache_root/key).resolve()
    if not target.is_relative_to(stage.resolve()):
        raise ValueError('unsafe LS cache reset')
    if reset_scope in {'selected', 'all'} and target.is_dir():
        shutil.rmtree(target)
    if reset_scope != 'none':
        for filename in ('manifest.json', 'replicate_results.csv'):
            (stage/filename).unlink(missing_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    manifest = dict(provenance, stage='estimation-ls-'+bank, status='running', mode=mode,
        scientific_digest=source_digest, cache_key=key,
        scenarios=[s.name for s in selected],
        replications_by_scenario={s.name: config.replications for s in selected},
        source_manifest=str(source_manifest), source_results=str(source_results),
        selection='same saved first Gaussian alarm and change index as frozen source',
        artifact_origin='point-only reanalysis of existing frozen bank; no new independent episodes',
        primary_estimator='ls_unpenalized',
        interval_scope='point_only; no LS confidence intervals',
        started_utc=datetime.now(timezone.utc).isoformat())
    _atomic_json(manifest, stage/'manifest.json')
    sources = {(r['scenario'], r['replicate']): r for r in base.to_dict('records')}
    rows, missing = [], []
    for scenario in selected:
        for rep in range(config.replications):
            source = sources[(scenario.name, rep)]
            path = _cache_path(cache_root, key, scenario.name, rep)
            if not path.is_file():
                missing.append((scenario, rep))
                continue
            item = json.loads(path.read_text(encoding='utf-8'))
            if item.get('payload_hash') != _payload_hash(item) or item.get('cache_key') != key:
                raise ValueError('corrupt LS episode cache')
            row = item['row']
            if (row['scenario'], row['replicate']) != (scenario.name, rep):
                raise ValueError('LS cache identifier mismatch')
            _poisson_observations(scenario, rep, config, bank, source_digest, source_cache, source)
            rows.append(row)
    def save(row):
        item = dict(cache_key=key, row=row)
        item['payload_hash'] = _payload_hash(item)
        _atomic_json(item, _cache_path(cache_root, key, row['scenario'], row['replicate']))
        rows.append(row)
    def arguments(task):
        scenario, rep = task
        return scenario, rep, config, bank, source_digest, source_cache, sources[(scenario.name, rep)], reporting
    workers = (os.cpu_count() or 1) if jobs == -1 else jobs
    if workers == 1:
        for task in missing:
            save(_ls_episode(*arguments(task)))
    elif missing:
        with ProcessPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            for future in as_completed([pool.submit(_ls_episode, *arguments(task)) for task in missing]):
                save(future.result())
    extra = pd.DataFrame(rows)
    extra = extra[['scenario', 'replicate']+sorted(c for c in extra if c.startswith('ls_'))]
    frame = base.merge(extra, on=['scenario', 'replicate'], validate='one_to_one').sort_values(
        ['scenario', 'replicate'], ignore_index=True)
    if len(frame) != 16*config.replications or frame.ls_status.isna().any():
        raise RuntimeError('incomplete LS episode grid')
    _atomic_csv(frame, stage/'replicate_results.csv')
    manifest.update(status='complete', tasks_computed=len(missing), cached_tasks_loaded=len(rows)-len(missing),
        completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
        artifacts={'replicate_results.csv': _file_hash(stage/'replicate_results.csv')})
    _atomic_json(manifest, stage/'manifest.json')
    return frame


def _sensitivity_source_compatibility(source: Path, records: dict[str, str]) -> dict:
    """Verify the parent snapshot and unchanged numerical dependency closure.

    Every scientific top-level node is checked, including constants/imports,
    not just the episode entry point. Only CLI and explicitly enumerated extension
    orchestration definitions may differ across the retained source versions. Numerical methods and all C sources must remain byte identical.
    """
    import ast
    source, current = Path(source).resolve(), Path(__file__).resolve().parent
    if not source.is_dir():
        raise ValueError("a frozen parent core source snapshot is required")
    for name, expected in records.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or _file_hash(path) != expected:
            raise ValueError(f"parent source hash mismatch: {name}")
    _, current_records = _code_hash()
    if records == current_records:
        return {"source_core": str(source), "exact_numerical_files": {
            name: value for name, value in records.items() if name == "methods.py" or name.startswith("csrc/")},
            "ast_checks": {}, "policy": "all scientific source files byte identical"}
    numerical = {name: value for name, value in records.items()
                 if name == "methods.py" or name.startswith("csrc/")}
    current_numerical = {name: value for name, value in current_records.items()
                         if name == "methods.py" or name.startswith("csrc/")}
    if numerical != current_numerical:
        raise ValueError("numerical methods/C source compatibility mismatch")
    additions = {
        "config.py": {"SENSITIVITY_EXTENSION_VERSION", "get_sensitivity_extension_config",
            "DISPERSION_SENSITIVITY_VERSION", "get_dispersion_sensitivity_config"},
        "simulation.py": {"_sensitivity_source_compatibility", "_sensitivity_check_grid",
            "_sensitivity_base_inputs", "run_sensitivity_extension", "load_sensitivity_extension",
            "run_dispersion_sensitivity", "load_dispersion_sensitivity", "_dispersion_write_results"}}
    checks = {}
    for name in ("config.py", "simulation.py"):
        def nodes(path, extension):
            result = []
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                node_name = getattr(node, "name", None)
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    node_name = targets[0].id if len(targets) == 1 and isinstance(targets[0], ast.Name) else None
                if node_name in {"main", "_parser"}:
                    continue
                if node_name in additions[name]:
                    continue
                result.append(ast.dump(node, include_attributes=False))
            return result
        old_nodes, new_nodes = nodes(source/name, False), nodes(current/name, True)
        if old_nodes != new_nodes:
            raise ValueError(f"numerical dependency AST mismatch: {name}")
        checks[name] = {"parent_sha256": records[name], "consumer_sha256": current_records[name],
                        "all_original_non_cli_nodes_identical": True,
                        "excluded_extension_orchestration": sorted(additions[name])}
    return {"source_core": str(source), "exact_numerical_files": numerical,
            "ast_checks": checks, "policy": "all scientific AST nodes excluding enumerated extension orchestration; methods and C byte identical"}


def _sensitivity_check_grid(path: Path, scenarios, methods, counts) -> None:
    """Check the complete Cartesian grid without loading large trace columns."""
    frame = pd.read_csv(path, usecols=["scenario", "replicate", "method", "status"])
    expected = {(s, r, m) for s in scenarios for r in range(counts[s]) for m in methods}
    actual = list(frame[["scenario", "replicate", "method"]].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected or not frame["status"].eq("success").all():
        raise ValueError("sensitivity source/result episode grid incomplete or failed")


def _sensitivity_base_inputs(base_output: Path, *, base_source: Path | None = None):
    """Return verified parent config, original thresholds, and provenance.

    The default snapshot layout is full_sources/<study-id>/core beside
    full_outputs/<study-id>. Public copies can supply ``base_source`` explicitly.
    Neither parent outputs nor the producer threshold artifact are rewritten.
    """
    base = Path(base_output).resolve()
    ep, cp = base/"evaluation"/"manifest.json", base/"calibration"/"manifest.json"
    evaluation = json.loads(ep.read_text(encoding="utf-8"))
    calibration = json.loads(cp.read_text(encoding="utf-8"))
    if evaluation.get("mode") not in {"demo", "full"}:
        raise ValueError("parent must declare demo/full mode")
    cfg = _estimation_source_config(evaluation, evaluation["mode"])
    expected = (get_compact_release_study_config(backend=cfg.methods.backend)
                if cfg.split_replications is not None else
                get_compact_experiment_config(cfg.mode, backend=cfg.methods.backend))
    if (cfg.protocol != "compact-v2" or cfg.scientific_payload() != expected.scientific_payload()
            or cfg.methods.backend not in {"python", "c"}):
        raise ValueError("parent nominal scientific configuration mismatch")
    code_records = evaluation["code_files"]
    parent_code = _sha256_bytes(_canonical_json(code_records).encode("ascii"))
    parent_digest = _sha256_bytes(_canonical_json({"schema_version": SCHEMA_VERSION,
        "code_hash": parent_code, "scientific": cfg.scientific_payload()}).encode("ascii"))[:20]
    if evaluation.get("code_hash") != parent_code or evaluation.get("scientific_digest") != parent_digest:
        raise ValueError("parent code/scientific digest mismatch")
    if (calibration.get("code_files") != code_records or calibration.get("code_hash") != parent_code
            or calibration.get("scientific_config") != cfg.scientific_payload()):
        raise ValueError("parent calibration/evaluation scientific source mismatch")
    verified_hashes = {}
    for manifest, folder in ((evaluation, "evaluation"), (calibration, "calibration")):
        if manifest.get("status") != "complete" or manifest.get("stage") != folder:
            raise ValueError("complete parent calibration/evaluation required")
        for filename, digest in manifest.get("artifacts", {}).items():
            artifact = (base/folder/filename).resolve()
            actual_hash = _file_hash(artifact) if artifact.is_file() else None
            if not artifact.is_relative_to((base/folder).resolve()) or actual_hash != digest:
                raise ValueError(f"parent artifact hash mismatch: {folder}/{filename}")
            verified_hashes[(folder, filename)] = actual_hash
    selected = _compact_stage_scenarios(cfg, "evaluation")
    counts = {s.name: cfg.replications_for(s.split) for s in selected}
    if (evaluation.get("methods") != list(cfg.stage_methods("evaluation"))
            or evaluation.get("scenarios") != [s.name for s in selected]
            or evaluation.get("replications_by_scenario") != counts):
        raise ValueError("parent evaluation allocation/method roster mismatch")
    raw = base/"evaluation"/"replicate_results.csv"
    if not verified_hashes.get(("evaluation", raw.name)):
        raise ValueError("parent evaluation raw hash mismatch")
    _sensitivity_check_grid(raw, counts, cfg.stage_methods("evaluation"), counts)
    threshold_path = base/"calibration"/"thresholds.csv"
    thresholds = _compact_read_thresholds(threshold_path, parent_digest,
                                         {s.null_family for s in cfg.scenarios}, cfg)
    if (not thresholds["scientific_digest"].eq(parent_digest).all()
            or thresholds["threshold_version"].isna().any()):
        raise ValueError("parent threshold producer digest/version mismatch")
    threshold_hash = _file_hash(threshold_path)
    if evaluation.get("threshold_artifact", {}).get("sha256") != threshold_hash:
        raise ValueError("parent threshold consumer hash mismatch")
    calibration_raw = base/"calibration"/"replicate_results.csv"
    if not verified_hashes.get(("calibration", calibration_raw.name)):
        raise ValueError("parent calibration raw hash mismatch")
    calibration_counts = {s.name: cfg.replications_for(s.split) for s in cfg.scenarios if s.split == "calibration"}
    _sensitivity_check_grid(calibration_raw, calibration_counts, cfg.stage_methods("calibration"), calibration_counts)
    source = (Path(base_source) if base_source is not None else
              Path(__file__).resolve().parent if code_records == _code_hash()[1] else
              base.parent.parent/"full_sources"/base.name/"core")
    compatibility = _sensitivity_source_compatibility(source, code_records)
    provenance = {"base_output": str(base), "parent_scientific_digest": parent_digest,
        "parent_code_hash": parent_code, "parent_code_files": code_records,
        "parent_evaluation_manifest_sha256": _file_hash(ep),
        "parent_evaluation_csv_sha256": verified_hashes[("evaluation", raw.name)],
        "parent_calibration_manifest_sha256": _file_hash(cp),
        "parent_calibration_csv_sha256": verified_hashes[("calibration", calibration_raw.name)],
        "threshold_artifact": {"path": str(threshold_path), "sha256": threshold_hash,
            "producer_scientific_digest": calibration["scientific_digest"],
            "producer_code_hash": calibration["code_hash"],
            "producer_manifest_sha256": _file_hash(cp),
            "threshold_versions": sorted(thresholds["threshold_version"].unique().tolist()),
            "import_lineage": calibration.get("import_lineage")},
        "numerical_compatibility": compatibility}
    return cfg, thresholds, provenance


def run_sensitivity_extension(*, base_output: Path, output: Path, mode: str | None = None,
        jobs: int = 1, reset_scope: ResetScope = "none", base_source: Path | None = None,
        return_mode: Literal["full", "slim"] = "slim", aggregation_batch_size: int = 50,
        _dispersion: bool = False):
    """Run only the ten frozen additional sensitivity cells in a new namespace."""
    from dataclasses import replace
    from config import get_sensitivity_extension_config, SENSITIVITY_EXTENSION_VERSION
    if _dispersion:
        from config import get_dispersion_sensitivity_config as get_sensitivity_extension_config
        from config import DISPERSION_SENSITIVITY_VERSION as SENSITIVITY_EXTENSION_VERSION
    if jobs == 0 or jobs < -1 or reset_scope not in {"none", "selected", "derived", "all"}:
        raise ValueError("invalid extension workers/reset scope")
    if return_mode not in {"full", "slim"} or type(aggregation_batch_size) is not int or aggregation_batch_size < 1:
        raise ValueError("invalid extension return mode/batch size")
    parent, thresholds, provenance = _sensitivity_base_inputs(base_output, base_source=base_source)
    mode = parent.mode if mode is None else mode
    if mode not in {"demo", "full"} or (mode == "full" and parent.mode != "full"):
        raise ValueError("full extension requires a full parent")
    config = get_sensitivity_extension_config(replace(parent, mode=mode, replications=10 if mode == "demo" else 1000))
    digest, code_hash, code_records = _scientific_digest(config)
    # Parent provenance enters this separate namespace even when numerical inputs match.
    cache_key = _sha256_bytes(_canonical_json({"extension": SENSITIVITY_EXTENSION_VERSION,
        "scientific_digest": digest, "parent": provenance}).encode("ascii"))[:20]
    stage_name = "dispersion-sensitivity" if _dispersion else "sensitivity-extension"
    stage = Path(output).resolve()/stage_name
    base = Path(base_output).resolve()
    if stage.is_relative_to(base) or base.is_relative_to(stage):
        raise ValueError("extension output must be disjoint from the frozen parent study")
    if not stage.resolve().is_relative_to(Path(output).resolve()):
        raise ValueError("unsafe extension stage path")
    cache_root = stage/"cache"/cache_key
    target = (cache_root/digest).resolve()
    if not target.is_relative_to(stage.resolve()):
        raise ValueError("unsafe extension reset target")
    if reset_scope in {"selected", "all"} and target.is_dir():
        shutil.rmtree(target)
    if reset_scope != "none":
        for name in ("manifest.json", "replicate_results.csv"):
            (stage/name).unlink(missing_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    selected, scalar_rows, missing = list(config.scenarios), [], []
    cached_count = 0
    for scenario in selected:
        for rep in range(config.replications):
            payload = _load_cache(_cache_path(cache_root, digest, scenario.name, rep),
                digest=digest, scenario=scenario, replicate=rep, config=config)
            if payload is None:
                missing.append((scenario, rep))
            else:
                scalar_rows.extend(_compact_scalar_rows(payload))
                cached_count += 1
    workers = (os.cpu_count() or 1) if jobs == -1 else jobs
    manifest = {"schema_version": SCHEMA_VERSION, "extension_version": SENSITIVITY_EXTENSION_VERSION,
        "protocol": config.protocol, "stage": stage_name, "status": "running", "mode": mode,
        "scientific_digest": digest, "cache_key": cache_key, "code_hash": code_hash, "code_files": code_records,
        "scientific_config": config.scientific_payload(), "parent_provenance": provenance,
        "backend_request": config.methods.backend, "methods": list(config.stage_methods("evaluation")),
        "scenarios": [s.name for s in selected], "replications_by_scenario": {s.name: config.replications for s in selected},
        "new_cells": len(selected), "retained_cells_copied": 0, "gaussian_multiplier_draws": config.methods.n_multiplier,
        "planned_looks": len(_planned_looks(selected[0], config)), "crossing": "strict >",
        "threshold_artifact": provenance["threshold_artifact"], "tasks_computed": 0,
        "cached_tasks_loaded": cached_count, "unavailable_looks": "no alarm; all allocated episodes retained",
        "infrastructure_errors": "incomplete job; propagate and repair",
        "aggregation": {"batch_episodes": aggregation_batch_size, "return_mode": return_mode,
                        "trace_columns_in_csv": True, "maximum_pending_tasks": 2 * workers}}
    if _dispersion:
        manifest.update(master_seed=config.master_seed,
            primary_metric="RR1.5 useful power",
            required_audit="RR1 null alarm probability at every phi",
            case_pairing="shared alternative RNG split at both RR levels; RR1 reported as null",
            threshold_policy="original parent phi10 cutoffs; no dispersion-specific recalibration")
    _atomic_json(manifest, stage/"manifest.json")
    def store(payload, scenario, rep):
        if not _payload_success(payload):
            raise RuntimeError("incomplete sensitivity extension episode")
        path = _cache_path(cache_root, digest, scenario.name, rep)
        _atomic_json(payload, path)
        if _load_cache(path, digest=digest, scenario=scenario, replicate=rep, config=config) is None:
            raise RuntimeError("invalid sensitivity extension episode payload")
        scalar_rows.extend(_compact_scalar_rows(payload))
    if workers == 1:
        for scenario, rep in missing:
            store(_compact_evaluate_one(scenario, rep, config, digest), scenario, rep)
    elif missing:
        with ProcessPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            tasks, pending = iter(missing), {}
            def submit_next():
                task = next(tasks, None)
                if task is None:
                    return False
                scenario, rep = task
                pending[pool.submit(_compact_evaluate_one, scenario, rep, config, digest)] = task
                return True
            while len(pending) < 2 * workers and submit_next():
                pass
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    store(future.result(), *pending.pop(future))
                while len(pending) < 2 * workers and submit_next():
                    pass
    scalar = pd.DataFrame(scalar_rows).sort_values(["scenario", "replicate", "method"], ignore_index=True)
    if len(scalar) != len(selected) * config.replications * 3 or scalar.duplicated(["scenario", "replicate", "method"]).any():
        raise RuntimeError("incomplete/duplicate sensitivity extension task grid")
    result = stage/"replicate_results.csv"
    writer = _dispersion_write_results if _dispersion else _compact_write_results
    writer(config, selected, cache_root, digest, "evaluation", thresholds,
           scalar, result, aggregation_batch_size)
    _sensitivity_check_grid(result, manifest["scenarios"], manifest["methods"], manifest["replications_by_scenario"])
    manifest.update(status="complete", tasks_computed=len(missing), failed_method_rows=0,
        all_looks_unavailable_rows=int(scalar["all_looks_unavailable"].sum()),
        artifacts={"replicate_results.csv": _file_hash(result)})
    _atomic_json(manifest, stage/"manifest.json")
    excluded = {"trace_json", "stream_manifest_json"} if return_mode == "slim" else set()
    return pd.read_csv(result, float_precision="round_trip", usecols=lambda name: name not in excluded)


def load_sensitivity_extension(output: Path, *, base_output: Path, base_source: Path | None = None,
        return_mode: Literal["full", "slim"] = "slim", _dispersion: bool = False):
    """Validate an extension and return (config, frame, manifest), slim by default."""
    from dataclasses import replace
    from config import get_sensitivity_extension_config, SENSITIVITY_EXTENSION_VERSION
    if _dispersion:
        from config import get_dispersion_sensitivity_config as get_sensitivity_extension_config
        from config import DISPERSION_SENSITIVITY_VERSION as SENSITIVITY_EXTENSION_VERSION
    if return_mode not in {"full", "slim"}:
        raise ValueError("extension return mode must be full or slim")
    stage_name = "dispersion-sensitivity" if _dispersion else "sensitivity-extension"
    stage = Path(output).resolve()/stage_name
    manifest = json.loads((stage/"manifest.json").read_text(encoding="utf-8"))
    parent, _, provenance = _sensitivity_base_inputs(base_output, base_source=base_source)
    mode = manifest.get("mode")
    if mode not in {"demo", "full"} or (mode == "full" and parent.mode != "full"):
        raise ValueError("extension mode mismatch")
    cfg = get_sensitivity_extension_config(replace(parent, mode=mode, replications=10 if mode == "demo" else 1000))
    scientific, code, records = _scientific_digest(cfg)
    cache_key = _sha256_bytes(_canonical_json({"extension": SENSITIVITY_EXTENSION_VERSION,
        "scientific_digest": scientific, "parent": provenance}).encode("ascii"))[:20]
    counts = {s.name: cfg.replications for s in cfg.scenarios}
    if (manifest.get("stage") != stage_name or manifest.get("status") != "complete"
            or manifest.get("extension_version") != SENSITIVITY_EXTENSION_VERSION
            or manifest.get("scientific_digest") != scientific or manifest.get("cache_key") != cache_key or manifest.get("code_hash") != code
            or manifest.get("code_files") != records or manifest.get("scientific_config") != cfg.scientific_payload()
            or manifest.get("parent_provenance") != provenance or manifest.get("new_cells") != len(cfg.scenarios)
            or manifest.get("scenarios") != [s.name for s in cfg.scenarios]
            or manifest.get("methods") != list(cfg.stage_methods("evaluation"))
            or manifest.get("replications_by_scenario") != counts
            or manifest.get("threshold_artifact") != provenance["threshold_artifact"]):
        raise ValueError("extension manifest/config/source provenance mismatch")
    raw = stage/"replicate_results.csv"
    if manifest.get("artifacts", {}).get(raw.name) != _file_hash(raw):
        raise ValueError("extension raw artifact hash mismatch")
    _sensitivity_check_grid(raw, counts, cfg.stage_methods("evaluation"), counts)
    excluded = {"trace_json", "stream_manifest_json"} if return_mode == "slim" else set()
    return cfg, pd.read_csv(raw, float_precision="round_trip", usecols=lambda name: name not in excluded), manifest


def _dispersion_write_results(config: ExperimentConfig, selected: list[SimulationScenario],
                            cache_root: Path, digest: str, stage: str,
                            thresholds: pd.DataFrame | None, scalar_frame: pd.DataFrame,
                            result_path: Path, batch_size: int) -> None:
    """Apply the original transformations in batches with global scalar dtypes."""
    dtypes = scalar_frame.dtypes.to_dict()
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    columns, count = None, 0
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        for payloads in _compact_cache_batches(config, selected, cache_root, digest, batch_size):
            frame = _flatten(payloads).astype(dtypes).sort_values(
                ["scenario", "replicate", "method"], ignore_index=True)
            # Preserve paired RNG splits in caches; classify statistical nulls in reports.
            frame.loc[np.isclose(frame.risk_ratio, 1.), "split"] = "holdout"
            if stage != "calibration":
                frame = _compact_apply_thresholds(frame, thresholds, config.dgp.timely_window, config)
            for column in ("stream_manifest_json", "trace_json"):
                frame[column] = frame[column].map(lambda value: json.dumps(json.loads(value),
                    sort_keys=True, separators=(",", ":")))
            frame = frame.reindex(sorted(frame.columns), axis=1)
            if columns is None:
                columns = list(frame.columns)
            if list(frame.columns) != columns:
                raise RuntimeError("inconsistent compact result columns")
            frame.to_csv(destination, index=False, header=count == 0)
            count += len(frame)
        if count != len(scalar_frame):
            raise RuntimeError("incomplete compact result aggregation")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, result_path)


def run_dispersion_sensitivity(**kwargs):
    """Run the frozen Figure 6D grid using original parent phi10 cutoffs."""
    return run_sensitivity_extension(**kwargs, _dispersion=True)


def load_dispersion_sensitivity(output: Path, **kwargs):
    """Load only a complete, source-bound eight-cell dispersion study."""
    return load_sensitivity_extension(output, **kwargs, _dispersion=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "full"), default="demo")
    parser.add_argument("--protocol", choices=("standard", "size-control", "compact-v1", "compact-v2", "compact-v2-release"),
                        default="compact-v2",
                        help="size-control: 19,999 calibration, 10,000 holdout and 1,000 alternative replicates")
    parser.add_argument("--jobs", type=int, default=1, help="workers; -1 uses all CPUs")
    parser.add_argument("--backend", choices=("c", "python", "auto"), default=None,
                        help="default is strict C; python/auto are diagnostic overrides")
    parser.add_argument("--stage", choices=("calibration", "audit", "evaluation", "estimation", "estimation-reference", "estimation-stabilized", "estimation-validation", "estimation-poisson", "estimation-ls", "sensitivity-extension", "dispersion-sensitivity"), default="evaluation",
                        help="compact-v1 auxiliary or evaluation stage; no implicit calibration")
    parser.add_argument("--thresholds", type=Path, default=None,
                        help="frozen compact calibration thresholds.csv")
    parser.add_argument("--source-manifest", type=Path, default=None, help="original completed full evaluation manifest for estimation")
    parser.add_argument("--source-results", type=Path, default=None, help="original evaluation CSV for estimation")
    parser.add_argument("--estimation-manifest", type=Path, default=None, help="completed full NB estimation manifest for reference points")
    parser.add_argument("--reference-manifest", type=Path, default=None, help="completed full estimation-reference manifest for stabilized reporting")
    parser.add_argument("--estimation-bank", choices=('original', 'validation', 'both'), default='both',
                        help="Frozen point-comparison reporting bank; source overrides require a single bank")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--base-study", type=Path, default=None, help="completed frozen parent study for sensitivity extension")
    parser.add_argument("--base-source", "--base-code", type=Path, default=None, help="frozen parent core source snapshot (auto-resolved in full_sources layout)")
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--reset", nargs="?", const="selected", default="none",
                        choices=("none", "selected", "derived", "all"),
                        help="selected/all: selected stage and scenarios in current digest; derived: CSV/manifest")
    return parser

def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage in {"sensitivity-extension", "dispersion-sensitivity"}:
        if args.base_study is None or args.output is None:
            raise SystemExit("sensitivity extension requires --base-study and --output")
        if args.backend is not None or args.thresholds is not None or args.scenarios is not None:
            raise SystemExit("sensitivity extension inherits the parent backend/thresholds and runs its complete frozen grid")
        # Omitted --mode inherits the parent's completed allocation.
        explicit_mode = args.mode if any(x == "--mode" or x.startswith("--mode=") for x in (argv if argv is not None else __import__('sys').argv[1:])) else None
        runner = run_dispersion_sensitivity if args.stage == "dispersion-sensitivity" else run_sensitivity_extension
        frame = runner(base_output=args.base_study, output=args.output,
            base_source=args.base_source, mode=explicit_mode, jobs=args.jobs, reset_scope=args.reset)
        print(f"wrote {len(frame)} new method rows to {args.output / args.stage}")
        return 0
    if args.protocol in {"compact-v1", "compact-v2", "compact-v2-release"}:
        if args.protocol == "compact-v2-release" and args.mode != "full":
            raise SystemExit("compact-v2-release requires --mode full")
        config = (get_compact_release_study_config(backend=args.backend) if args.protocol == "compact-v2-release" else
                  get_compact_experiment_config(args.mode, backend=args.backend, protocol=args.protocol))
        output = args.output or Path(__file__).parent / "outputs" / args.protocol.replace("-", "_")
        if args.stage in {"estimation", "estimation-reference", "estimation-stabilized", "estimation-validation", "estimation-poisson", "estimation-ls"}:
            if args.scenarios is not None or args.backend is not None or args.thresholds is not None:
                raise SystemExit("estimation uses all 16 frozen on-model cells and the original backend; only --mode changes sample count")
            runner = {'estimation': run_compact_estimation, 'estimation-reference': run_compact_estimation_reference,
                      'estimation-stabilized': run_compact_estimation_stabilized,
                      'estimation-validation': run_compact_estimation_validation,
                      'estimation-poisson': run_compact_estimation_poisson,
                      'estimation-ls': run_compact_estimation_ls}[args.stage]
            extra = {'estimation_manifest': args.estimation_manifest} if args.stage == 'estimation-reference' else (
                {'reference_manifest': args.reference_manifest} if args.stage == 'estimation-stabilized' else (
                {'bank': args.estimation_bank} if args.stage in {'estimation-poisson', 'estimation-ls'} else {}))
            frame = runner(output=output, mode=args.mode, jobs=args.jobs,
                reset_scope=args.reset, source_manifest=args.source_manifest, source_results=args.source_results, **extra)
            print(f"wrote {len(frame)} episode rows to {output / args.stage}")
            return 0
        frame, thresholds = run_compact_simulations(config, output=output, stage=args.stage,
            thresholds_path=args.thresholds, jobs=args.jobs, reset_scope=args.reset,
            scenario_names=None if args.scenarios is None else set(args.scenarios),
            return_mode="slim")
        print(thresholds.to_string(index=False))
        print(f"wrote {len(frame)} method-level rows to {output / args.stage}")
        return 0
    if args.protocol == "size-control":
        if args.mode != "full":
            raise SystemExit("--protocol size-control requires --mode full")
        config = get_size_control_experiment_config(backend=args.backend)
    else:
        config = get_experiment_config(args.mode, backend=args.backend)
    frame, thresholds = run_simulations(
        config, output=args.output or Path(__file__).parent / "outputs" / "simulation", jobs=args.jobs, reset_scope=args.reset,
        scenario_names=None if args.scenarios is None else set(args.scenarios))
    print(thresholds.to_string(index=False))
    print(f"wrote {len(frame)} method-level rows to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
