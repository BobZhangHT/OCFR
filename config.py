"""Frozen scientific configuration for the public OCFR simulation runner.

``demo`` and ``full`` are the same experiment; only their replication counts
differ (10 and 1000). Runtime controls never enter the RNG contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

Mode = Literal["demo", "full"]
Split = Literal["calibration", "holdout", "alternative", "size_audit"]
ReplicationPlan = tuple[tuple[Split, int], ...]

# This named plan is deliberately separate from ``full``.  The public
# demo/full contract remains a single B value (10/1000) for every cell, while
# the release audit buys precision where it is needed: calibration and an
# independent size audit, not the alternative power cells.
SIZE_CONTROL_REPLICATION_PLAN: ReplicationPlan = (
    ("calibration", 19_999),
    ("holdout", 10_000),
    ("alternative", 1_000),
)

@dataclass(frozen=True)
class MethodConfig:
    alpha: float = 0.05
    look_every: int = 7
    n_multiplier: int = 10_000
    reference_end: int = 28
    cusum_allowance: float = 0.5
    use_lindeberg_gate: bool = True
    backend: str = "c"  # strict release default

@dataclass(frozen=True)
class EstimationConfig:
    """Descriptive plug-in Wald assessment at the first Gaussian alarm."""
    confidence: float = 0.95
    method: str = "OCFR Gaussian"
    scope: str = "on_model_alternatives_first_alarm"

    def __post_init__(self):
        if self.confidence != 0.95 or self.method not in {"OCFR Gaussian", "OCFR"} or self.scope != "on_model_alternatives_first_alarm":
            raise ValueError("the frozen estimation assessment uses 95% first-Gaussian-alarm Wald intervals")

@dataclass(frozen=True)
class PostAlarmEstimationConfig:
    """Reporting settings, separate from the frozen detection/DGP payload."""
    precision: float = 1.0
    covariance: str = 'sandwich'
    confidence: float = 0.95
    estimator: str = 'regularized_log_rr'
    diagnostic_covariance: str = 'unpenalized'

    def __post_init__(self):
        if (self.precision != 1.0 or self.covariance != 'sandwich' or
                self.confidence != 0.95 or self.estimator != 'regularized_log_rr' or
                self.diagnostic_covariance != 'unpenalized'):
            raise ValueError('post-alarm reporting uses fixed precision 1 and approximate sandwich intervals')


@dataclass(frozen=True)
class PostAlarmValidationConfig:
    """Independent validation seed; detector settings come from the frozen study."""
    validation_master_seed: int = 2026110702
    policy: str = 'first Gaussian crossing with fixed full-episode multiplier bank'


@dataclass(frozen=True)
class PostAlarmPoissonConfig:
    """Point comparison settings, outside the immutable scientific payload."""
    primary_precision: float = 1.0
    control_precision: float = 0.0
    likelihood: str = 'exact_poisson'
    interval_scope: str = 'point_only'

    def __post_init__(self):
        if (self.primary_precision != 1.0 or self.control_precision != 0.0 or
                self.likelihood != 'exact_poisson' or self.interval_scope != 'point_only'):
            raise ValueError('Poisson comparison fixes precisions 1 and 0 and reports points only')


@dataclass(frozen=True)
class PostAlarmLSConfig:
    """Point comparison settings outside the frozen DGP/detection payload."""
    loss: str = 'daily_squared_error'
    penalty: float = 0.0
    interval_scope: str = 'point_only'

    def __post_init__(self):
        if (self.loss != 'daily_squared_error' or self.penalty != 0.0 or
                self.interval_scope != 'point_only'):
            raise ValueError('LS comparison uses unpenalized daily squared error and points only')


@dataclass(frozen=True)
class DGPConfig:
    delay_mean_days: float = 14.0
    delay_shape: float = 4.0
    maximum_delay: int = 45
    autoregression: float = 0.90
    process_sd: float = 0.18
    case_shape: float = 30.0
    weekday_amplitude: float = 0.15
    timely_window: int = 28

@dataclass(frozen=True)
class SimulationScenario:
    """One cell in the independent calibration/holdout/test protocol."""
    name: str
    family: str
    split: Split
    null_family: str
    case_stream_group: str
    outcome_stream_group: str
    horizon: int
    profile: str
    case_level: float
    pi_left: float
    pi_right: float
    phi: float
    true_delay_mean_days: float = 14.0
    true_delay_shape: float = 4.0
    assumed_delay_mean_days: float = 14.0
    assumed_delay_shape: float = 4.0
    change_fraction: float = 0.50
    misspecification: str = "M1_correct_observation_model"
    motivation: str = ""

    @property
    def purpose(self) -> str:
        return "null" if self.split == "holdout" else self.split

    @property
    def risk_ratio(self) -> float:
        return self.pi_right / self.pi_left

    @property
    def true_change(self) -> int:
        """Zero-based cohort index; its manuscript day is one larger."""
        return int(round(self.horizon * self.change_fraction))

@dataclass(frozen=True)
class ExperimentConfig:
    mode: Mode
    replications: int
    master_seed: int
    rng_contract_version: str
    dgp: DGPConfig
    methods: MethodConfig
    scenarios: tuple[SimulationScenario, ...]
    protocol: str = "standard"
    split_replications: ReplicationPlan | None = None

    @property
    def boundary_policy(self) -> str:
        return GAUSSIAN_SPENDING_POLICY if self.protocol == "compact-v2" else LEGACY_GAUSSIAN_POLICY

    def stage_methods(self, stage: str) -> tuple[str, ...]:
        return compact_stage_methods(self.protocol, stage)

    def replications_for(self, split: Split) -> int:
        """Number of replicate tasks for a cell in ``split``."""
        if self.protocol == "compact-v1" and split in {"calibration", "size_audit"}:
            return 10_000 if split == "calibration" else 5_000
        if self.split_replications is None:
            return self.replications
        planned = dict(self.split_replications)
        try:
            value = planned[split]
        except KeyError as exc:  # defensive: a release plan must be complete
            raise ValueError(f"replication plan has no count for {split!r}") from exc
        if value < 1:
            raise ValueError("replication counts must be positive")
        return value

    def scientific_payload(self) -> dict[str, object]:
        payload = {"master_seed": self.master_seed,
                "rng_contract_version": self.rng_contract_version,
                "dgp": asdict(self.dgp), "methods": asdict(self.methods),
                "protocol": self.protocol,
                "split_replications": (dict(self.split_replications)
                                       if self.split_replications is not None else None),
                "scenarios": [asdict(x) for x in self.scenarios]}
        if self.protocol == "compact-v1":
            payload["compact_contract"] = {
                "calibration_episodes_per_family": 10_000,
                "audit_episodes_per_family": 5_000,
                "empirical_statistic": "maximum raw planned-look statistic",
                "calibration_rank": "ceil((B+1)*(1-alpha)); strict crossing",
                "stage_methods": {"calibration": list(COMPACT_EMPIRICAL_METHODS),
                                  "audit": list(COMPACT_EMPIRICAL_METHODS),
                                  "evaluation": list(COMPACT_METHODS)},
                "outcome_model": "NB except explicitly named cohort_binomial",
                "wave_mean": "1000*q/mean(q)",
            }
        if self.protocol == "compact-v2":
            payload["compact_contract"] = {
                "boundary_policy": self.boundary_policy,
                "cumulative_budget": "alpha*j/J; planned look index",
                "rank": "ceil((1-alpha*j/J)*(B+1)); original B",
                "calibration_removal": "eligible maxima >= boundary; all tied supports removed",
                "alarm_rule": "observed raw score > boundary",
                "reference_diagnostics": "strict crossing mask separate from eligibility/support masks",
                "multiplier_bank": "fixed full B by horizon; persistent row identity; prefix columns only",
                "normal_generation": "pcg32-numpy-box-muller-v1 for Python multiplier banks; scalar count streams unchanged",
                "unavailable_or_unresolved": "infinite boundary; advance planned index; masks unchanged",
                "stage_methods": {stage: list(self.stage_methods(stage)) for stage in ("calibration", "audit", "evaluation")},
                "outcome_model": "NB except explicitly named cohort_binomial",
                "wave_mean": "1000*q/mean(q)",
            }
        return payload

@dataclass(frozen=True)
class RealDataConfig:
    covid_url: str
    dengue_url: str
    covid_start: str = "2020-05-15"
    covid_end: str = "2021-01-14"
    covid_country_codes: tuple[str, ...] = ("GB", "JP", "KR")
    dengue_year: int = 2023
    alpha: float = 0.05
    n_multiplier: int = 10_000
    master_seed: int = 2026072602


METHODS = MethodConfig()
DGP = DGPConfig()

def _cell(name: str, family: str, split: Split, profile: str, case_level: float,
          pi_left: float, pi_right: float, phi: float, *,
          change_fraction: float = .50, true_delay: float = 14.,
          assumed_delay: float = 14.,
          misspecification: str = "M1_correct_observation_model",
          case_group: str | None = None, outcome_group: str | None = None,
          motivation: str) -> SimulationScenario:
    return SimulationScenario(
        name, family, split, family, case_group or family, outcome_group or name,
        182, profile, case_level, pi_left, pi_right, phi,
        true_delay_mean_days=true_delay, assumed_delay_mean_days=assumed_delay,
        change_fraction=change_fraction, misspecification=misspecification,
        motivation=motivation)

# The compact release bank retains three scientific regimes and a matched delay
# tolerance bank. Alternative effect cells may share case paths, but each has an
# explicit outcome stream. Split remains part of every derived seed.
SCENARIOS: tuple[SimulationScenario, ...] = (
    _cell("T00_calibration", "typical", "calibration", "constant", 1000, .020, .020, 10, motivation="Train episode thresholds."),
    _cell("T01_null", "typical", "holdout", "constant", 1000, .020, .020, 10, motivation="Independent type-I error validation."),
    _cell("T02_rr067", "typical", "alternative", "constant", 1000, .020, .0134, 10, case_group="typical_mid", motivation="Risk decrease."),
    _cell("T03_rr125", "typical", "alternative", "constant", 1000, .020, .025, 10, case_group="typical_mid", motivation="Modest increase."),
    _cell("T04_rr150", "typical", "alternative", "constant", 1000, .020, .030, 10, case_group="typical_mid", motivation="Primary medium effect."),
    _cell("T05_rr200", "typical", "alternative", "constant", 1000, .020, .040, 10, case_group="typical_mid", motivation="Large-effect control."),
    _cell("T06_early_rr150", "typical", "alternative", "constant", 1000, .020, .030, 10, change_fraction=.25, case_group="typical_early", motivation="Early change."),
    _cell("T07_late_rr150", "typical", "alternative", "constant", 1000, .020, .030, 10, change_fraction=.75, case_group="typical_late", motivation="Late change."),
    _cell("S00_calibration", "sparse", "calibration", "constant", 250, .005, .005, 2.5, motivation="Train sparse thresholds."),
    _cell("S01_null", "sparse", "holdout", "constant", 250, .005, .005, 2.5, motivation="Sparse type-I validation."),
    _cell("S02_rr200", "sparse", "alternative", "constant", 250, .005, .010, 2.5, motivation="Sparse detection."),
    _cell("W00_calibration", "wave", "calibration", "wave", 800, .010, .010, 5, motivation="Train wave thresholds."),
    _cell("W01_null", "wave", "holdout", "wave", 800, .010, .010, 5, motivation="Wave type-I validation."),
    _cell("W02_rising_rr150", "wave", "alternative", "wave", 800, .010, .015, 5, change_fraction=.35, case_group="wave_shared", motivation="Rising-phase detection."),
    _cell("W03_falling_rr150", "wave", "alternative", "wave", 800, .010, .015, 5, change_fraction=.65, case_group="wave_shared", motivation="Falling-phase detection."),
    _cell("D00_calibration", "delay", "calibration", "constant", 1000, .020, .020, 10, motivation="Train correct-delay thresholds."),
    _cell("D01_null_correct", "delay", "holdout", "constant", 1000, .020, .020, 10, case_group="delay_matched", outcome_group="delay_matched", motivation="Correct-delay null control."),
    _cell("D02_null_plus1", "delay", "holdout", "constant", 1000, .020, .020, 10, assumed_delay=15, misspecification="M2_delay_mean_error", case_group="delay_matched", outcome_group="delay_matched", motivation="Matched +1 day null."),
    _cell("D03_null_plus3", "delay", "holdout", "constant", 1000, .020, .020, 10, assumed_delay=17, misspecification="M2_delay_mean_error", case_group="delay_matched", outcome_group="delay_matched", motivation="Matched +3 day null."),
    _cell("D04_rr150_correct", "delay", "alternative", "constant", 1000, .020, .030, 10, case_group="delay_matched", outcome_group="delay_alt_matched", motivation="Correct-delay power."),
    _cell("D05_rr150_plus3", "delay", "alternative", "constant", 1000, .020, .030, 10, assumed_delay=17, misspecification="M2_delay_mean_error", case_group="delay_matched", outcome_group="delay_alt_matched", motivation="Matched +3 day power."),
)

BASE_EXPERIMENT = ExperimentConfig("full", 1000, 2026091401,
                                   "ocfr-pcg32-v2", DGP, METHODS, SCENARIOS)
REAL_DATA = RealDataConfig(
    "https://srhdpeuwpubsa.blob.core.windows.net/whdh/COVID/WHO-COVID-19-global-daily-data.csv",
    "https://dashboard.dghs.gov.bd/pages/heoc_dengue_v1.php")
DEFAULT_OUTPUT = Path("outputs")

def get_experiment_config(mode: Mode, *, backend: str | None = None) -> ExperimentConfig:
    if mode not in {"demo", "full"}:
        raise ValueError("mode must be 'demo' or 'full'")
    methods = METHODS if backend is None else replace(METHODS, backend=backend)
    return replace(BASE_EXPERIMENT, mode=mode,
                   replications=10 if mode == "demo" else 1000,
                   methods=methods)

def get_size_control_experiment_config(*, backend: str | None = None) -> ExperimentConfig:
    """Frozen high-precision protocol for a prospective size-control release.

    It is intentionally an explicit named configuration rather than a change
    to ``full``: full stays suitable for ordinary 1,000-replicate analyses.
    The split-specific counts are scientific inputs and therefore enter the
    digest/cache namespace through :meth:`ExperimentConfig.scientific_payload`.
    """
    full = get_experiment_config("full", backend=backend)
    return replace(full, protocol="size-control-v1",
                   split_replications=SIZE_CONTROL_REPLICATION_PLAN)


COMPACT_METHODS = ("OCFR Gaussian", "OCFR empirical", "NB-GLR", "Page-CUSUM",
                   "Report-day score")
COMPACT_EMPIRICAL_METHODS = COMPACT_METHODS[1:]


def _compact_scenarios() -> tuple[SimulationScenario, ...]:
    """Frozen 27-cell evaluation bank plus explicit auxiliary stages."""
    cells = []
    for family, profile, level in (("steady", "constant", 1000.),
                                    ("sparse", "constant", 250.),
                                    ("wave", "compact_wave", 1000.)):
        def add(suffix, split, rr=1., day=92, intervention="on_model", delay=14.,
                outcome_group=None):
            cells.append(_cell(
                f"{family}_{suffix}", family, split, profile, level,
                .02, .02 * rr, 10., change_fraction=(day - 1) / 182,
                true_delay=delay, case_group=family,
                outcome_group=outcome_group or f"{family}_{suffix}",
                misspecification=intervention, motivation=suffix))
        add("calibration", "calibration")
        add("audit", "size_audit")
        add("null", "holdout")
        for label, rr in (("rr067", 2 / 3), ("rr125", 1.25),
                          ("rr150", 1.5), ("rr200", 2.)):
            add(label, "alternative", rr)
        if family in {"steady", "wave"}:
            for day in (47, 137):
                add(f"rr150_day{day}", "alternative", 1.5, day)
        if family == "wave":
            for delay in (11., 17.):
                for suffix, split, rr in (("null", "holdout", 1.),
                                           ("rr150", "alternative", 1.5)):
                    add(f"delay{int(delay)}_{suffix}", split, rr,
                        intervention="delay_error", delay=delay)
            for suffix, split, rr in (("null", "holdout", 1.),
                                       ("rr150", "alternative", 1.5)):
                add(f"binomial_{suffix}", split, rr,
                    intervention="cohort_binomial")
            for scale in (.75, 1.25):
                add(f"ascertainment{int(scale * 100)}", "holdout",
                    intervention=f"exposure_scale_{scale}",
                    outcome_group="wave_null")
    return tuple(cells)


COMPACT_SCENARIOS = _compact_scenarios()


def get_compact_experiment_config(mode: Mode, *, backend: str | None = None, protocol: str = "compact-v2") -> ExperimentConfig:
    """V2 demo/full vary all allocated stage counts, with fixed numerical settings.

    Explicit v1 reproduction retains its frozen auxiliary allocations.
    """
    if protocol not in {"compact-v1", "compact-v2"}:
        raise ValueError("unknown compact protocol")
    base = get_experiment_config(mode, backend=backend)
    if protocol == "compact-v2":
        base = replace(base, methods=replace(base.methods, n_multiplier=1_999))
    return replace(base, protocol=protocol, master_seed=2026110501 if protocol == "compact-v1" else 2026111404,
                   rng_contract_version="ocfr-compact-pcg32-v1",
                   scenarios=COMPACT_SCENARIOS, split_replications=None)

def validate_mode_contract() -> None:
    demo, full = get_experiment_config("demo"), get_experiment_config("full")
    if (demo.replications, full.replications) != (10, 1000):
        raise AssertionError("demo/full replication contract is broken")
    if demo.scientific_payload() != full.scientific_payload():
        raise AssertionError("demo and full may differ only in replications")

validate_mode_contract()


GAUSSIAN_SPENDING_POLICY = "gaussian-first-crossing-linear-support-discard-v1"
LEGACY_GAUSSIAN_POLICY = "online_gaussian_process_bonferroni_looks"
COMPACT_V2_METHODS = ("OCFR", "NB-GLR", "Page-CUSUM")
COMPACT_V2_CALIBRATION_METHODS = ("NB-GLR", "Page-CUSUM")
COMPACT_RELEASE_REPLICATION_PLAN: ReplicationPlan = (
    ("calibration", 10_000), ("size_audit", 5_000),
    ("holdout", 1_000), ("alternative", 1_000))


def compact_stage_methods(protocol: str, stage: str) -> tuple[str, ...]:
    if stage not in {"calibration", "audit", "evaluation"}:
        raise ValueError("unknown compact stage")
    if protocol == "compact-v2":
        return COMPACT_V2_CALIBRATION_METHODS if stage == "calibration" else COMPACT_V2_METHODS
    if protocol == "compact-v1":
        return COMPACT_EMPIRICAL_METHODS if stage in {"calibration", "audit"} else COMPACT_METHODS
    raise ValueError("unknown compact protocol")


def get_compact_release_study_config(*, backend: str | None = None) -> ExperimentConfig:
    return replace(get_compact_experiment_config("full", backend=backend),
                   split_replications=COMPACT_RELEASE_REPLICATION_PLAN)


def get_boundary_ablation_configs(mode: Mode) -> tuple[ExperimentConfig, ExperimentConfig]:
    """Independent wave banks for the appendix's within-episode comparison."""
    base = get_compact_experiment_config(mode, backend="python")
    scenarios = {scenario.name: scenario for scenario in base.scenarios}
    # The historical null output was labelled wave_null, but its input was
    # wave_audit. Retain the original size_audit split and death stream group.
    audit = replace(base, master_seed=2026111612, scenarios=(scenarios["wave_audit"],),
                    replications=2_000 if mode == "full" else 10)
    alternatives = replace(base, master_seed=2026111613,
                           scenarios=tuple(scenarios[f"wave_{label}"]
                                           for label in ("rr067", "rr125", "rr150")))
    return audit, alternatives


def get_post_alarm_validation_config(protocol: str) -> PostAlarmValidationConfig:
    if protocol == "compact-v2":
        return PostAlarmValidationConfig(2026111405, GAUSSIAN_SPENDING_POLICY)
    if protocol == "compact-v1":
        return PostAlarmValidationConfig()
    raise ValueError("unknown compact protocol")


SENSITIVITY_EXTENSION_VERSION = "figure6-ten-cell-v1"


def get_sensitivity_extension_config(base: ExperimentConfig) -> ExperimentConfig:
    """Ten additional cells only; retain the parent's numerical/RNG settings."""
    if base.protocol != "compact-v2" or base.mode not in {"demo", "full"}:
        raise ValueError("sensitivity extension requires a compact-v2 demo/full parent")
    cells = []
    for delay in (12., 13., 15., 16.):
        for suffix, split, rr in (("null", "holdout", 1.),
                                  ("rr150", "alternative", 1.5)):
            name = f"wave_delay{int(delay)}_{suffix}"
            cells.append(_cell(name, "wave", split, "compact_wave", 1000.,
                .02, .02 * rr, 10., true_delay=delay, case_group="wave",
                outcome_group=name, misspecification="delay_error",
                motivation=f"delay{int(delay)}_{suffix}"))
    for scale in (.9, 1.1):
        name = f"wave_ascertainment{int(scale * 100)}"
        cells.append(_cell(name, "wave", "holdout", "compact_wave", 1000.,
            .02, .02, 10., case_group="wave", outcome_group="wave_null",
            misspecification=f"exposure_scale_{scale}",
            motivation=f"ascertainment{int(scale * 100)}"))
    return replace(base, scenarios=tuple(cells), split_replications=None)


DISPERSION_SENSITIVITY_VERSION = "figure6-dispersion-v1"


def get_dispersion_sensitivity_config(base: ExperimentConfig) -> ExperimentConfig:
    """Frozen eight-cell grid; RR=1 uses the alternative split for case pairing.

    Statistical null/alternative labels are determined from RR when reporting.
    The split is an RNG allocation, not a scientific null declaration here.
    """
    if base.protocol != "compact-v2" or base.mode not in {"demo", "full"}:
        raise ValueError("dispersion sensitivity requires a compact-v2 demo/full parent")
    template = next(s for s in base.scenarios if s.name == "wave_rr150")
    cells = tuple(replace(template, name=f"d_phi{phi}_rr{rr:g}", phi=float(phi),
        pi_right=.02 * rr, case_stream_group="dispersion_shared",
        outcome_stream_group="dispersion_shared", misspecification="dispersion_sensitivity",
        motivation="prespecified dispersion sensitivity under correct mean")
        for phi in (5, 10, 20, 40) for rr in (1., 1.5))
    return replace(base, master_seed=2026111604, scenarios=cells, split_replications=None)


PAIRED_DETECTION_VERSION = "compact-v2-paired-detection-v1"
ORACLE_OCFR = "OCFR-oracle"
PAIRED_DETECTION_METHODS = ("OCFR", ORACLE_OCFR, "NB-GLR", "Page-CUSUM")
