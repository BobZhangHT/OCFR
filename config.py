"""Frozen scientific configuration for the public OCFR simulation runner.

``demo`` and ``full`` are the same experiment; only their replication counts
differ (10 and 1000). Runtime controls never enter the RNG contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

Mode = Literal["demo", "full"]
Split = Literal["calibration", "holdout", "alternative"]
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

    def replications_for(self, split: Split) -> int:
        """Number of replicate tasks for a cell in ``split``."""
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
        return {"master_seed": self.master_seed,
                "rng_contract_version": self.rng_contract_version,
                "dgp": asdict(self.dgp), "methods": asdict(self.methods),
                "protocol": self.protocol,
                "split_replications": (dict(self.split_replications)
                                       if self.split_replications is not None else None),
                "scenarios": [asdict(x) for x in self.scenarios]}

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

def validate_mode_contract() -> None:
    demo, full = get_experiment_config("demo"), get_experiment_config("full")
    if (demo.replications, full.replications) != (10, 1000):
        raise AssertionError("demo/full replication contract is broken")
    if demo.scientific_payload() != full.scientific_payload():
        raise AssertionError("demo and full may differ only in replications")

validate_mode_contract()
