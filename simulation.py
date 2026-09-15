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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Literal

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd
from scipy.stats import beta

from config import (ExperimentConfig, SimulationScenario, get_experiment_config,
                    get_size_control_experiment_config)
from methods import (gamma_delay_pmf, make_correlated_epidemic_cases, make_sampler,
                     monte_carlo_critical_value, multiplier_statistics_from_draws,
                     published_benchmark_statistics, report_day_score_scan, score_scan,
                     simulate_nb_deaths)

SCHEMA_VERSION = 3
METHODS = ("OCFR", "OCFR-alpha-matched", "Report-day score",
           "NB-GLR", "Page-CUSUM", "FOCuS")
ResetScope = Literal["none", "selected", "derived", "all"]

def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)

def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def _file_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())

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
    return (isinstance(rows, list) and len(rows) == len(METHODS)
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
    if (payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("scientific_digest") != digest
            or payload.get("scenario_digest") != _scenario_digest(scenario)
            or payload.get("scenario") != scenario.name
            or payload.get("replicate") != replicate
            or not isinstance(rows, list) or len(rows) != len(METHODS)
            or {x.get("method") for x in rows if isinstance(x, dict)} != set(METHODS)
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

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "full"), default="demo")
    parser.add_argument("--protocol", choices=("standard", "size-control"),
                        default="standard",
                        help="size-control: 19,999 calibration, 10,000 holdout and 1,000 alternative replicates")
    parser.add_argument("--jobs", type=int, default=1, help="workers; -1 uses all CPUs")
    parser.add_argument("--backend", choices=("c", "python", "auto"), default=None,
                        help="default is strict C; python/auto are diagnostic overrides")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "outputs" / "simulation")
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--reset", nargs="?", const="selected", default="none",
                        choices=("none", "selected", "derived", "all"),
                        help="selected: selected-family cache; derived: CSV/manifest; all: current digest")
    return parser

def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.protocol == "size-control":
        if args.mode != "full":
            raise SystemExit("--protocol size-control requires --mode full")
        config = get_size_control_experiment_config(backend=args.backend)
    else:
        config = get_experiment_config(args.mode, backend=args.backend)
    frame, thresholds = run_simulations(
        config, output=args.output, jobs=args.jobs, reset_scope=args.reset,
        scenario_names=None if args.scenarios is None else set(args.scenarios))
    print(thresholds.to_string(index=False))
    print(f"wrote {len(frame)} method-level rows to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
