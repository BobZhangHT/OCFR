"""Fetch, preprocess, analyze, and plot the OCFR real-data applications."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import gamma

from config import REAL_DATA, RealDataConfig, PostAlarmEstimationConfig
from methods import (
    GaussianSpendingBoundary, GAUSSIAN_SPENDING_POLICY,
    LEGACY_GAUSSIAN_POLICY, gaussian_multiplier_bank,
    MU_FLOOR, _surveillance_traces_from_baseline,
    binomial_interval,
    exposure_components,
    fit_nb_identity,
    fit_nb_regularized_identity,
    fit_score_null,
    regularized_parameter_intervals,
    fixed_tau_wald_intervals,
    gamma_delay_pmf,
    make_sampler,
    monte_carlo_critical_value,
    multiplier_statistics_from_draws,
    nb_variance,
    published_benchmark_statistics,
    score_scan,
    validate_delay_pmf,
)


class OnlineBenchmarkCalibration:
    """Calibrate NB-GLR and Page CUSUM using only each available data prefix.

    A constant-CFR working model is fitted once to the reference window.
    Conditional on this fit and the case counts, two independent simulated
    banks are extended as new exposures become available. One bank sets
    cumulative first-crossing cutoffs. The other audits them under the same
    working model. Use cumulative case and death prefixes with an unchanged
    history and a monitoring schedule fixed before the first analysis.
    """

    methods = ('NB-GLR', 'Page CUSUM')

    def __init__(self, delay, *, planned_looks, observation_start, reference_end,
                 lag_offset=1, alpha=0.05, n_paths=10000, n_audit=10000, seed=0,
                 workers=1, backend='c'):
        self.looks = tuple(planned_looks)
        if (not self.looks or any(not isinstance(look, (int, np.integer))
                                 for look in self.looks)
                or self.looks != tuple(sorted(set(self.looks)))
                or not 0 <= observation_start < reference_end < self.looks[0]
                or reference_end < 3):
            raise ValueError('Specify increasing looks after a nonempty reference window.')
        if lag_offset not in (0, 1) or backend not in ('c', 'python'):
            raise ValueError('Use lag offset 0 or 1 and backend c or python.')
        if not isinstance(n_audit, int) or n_audit < 0 or not isinstance(workers, int) or workers < 1:
            raise ValueError('Audit size must be nonnegative and worker count positive.')
        self.delay = validate_delay_pmf(delay).copy()
        self.observation_start = observation_start
        self.reference_end = reference_end
        self.lag_offset = lag_offset
        self.seed = seed
        self.workers = workers
        self.backend = backend
        self.n_paths = n_paths
        self.boundaries = [GaussianSpendingBoundary(n_paths, len(self.looks), alpha)
                           for _ in self.methods]
        self.audit_crossed = np.zeros((n_audit, len(self.methods)), dtype=bool)
        self._draws = np.empty((n_paths + n_audit, self.looks[-1] - reference_end))
        self._history = (np.empty(0), np.empty(0))
        self._mean = np.empty(0)
        self.reference_fit = None

    def advance(self, cases_prefix, deaths_prefix):
        """Use exactly the next scheduled prefix and return both look results."""
        look = self.boundaries[0].look
        if look == len(self.looks):
            raise ValueError('All planned analyses have been processed.')
        prefix = self.looks[look]
        cases, deaths = (np.asarray(x, dtype=float) for x in (cases_prefix, deaths_prefix))
        for current, prior in zip((cases, deaths), self._history):
            if current.shape != (prefix,) or np.any(~np.isfinite(current)) or np.any(current < 0):
                raise ValueError('Supply finite, nonnegative counts through the next planned look.')
            if not np.array_equal(current[:prior.size], prior):
                raise ValueError('Counts in already processed history must be preserved.')
        start = self.reference_end
        fit = self.reference_fit
        if fit is None:
            fit = fit_score_null(cases[:start], deaths[:start], self.delay,
                                 observation_start=self.observation_start, lag_offset=self.lag_offset)
            if not fit.success:
                raise RuntimeError(f'Benchmark reference fit failed: {fit.status}')
        exposure, _ = exposure_components(cases, self.delay, lag_offset=self.lag_offset)
        mean = np.maximum(fit.pi * exposure[start:], MU_FLOOR)
        mean[:self._mean.size] = self._mean
        sd = np.sqrt(nb_variance(mean, fit.phi))
        simulated = self._draws[:, :mean.size]
        for day in range(self._mean.size, mean.size):
            # Day and bank determine the stream, independently of later inputs and worker count.
            for bank, rows in enumerate((slice(None, self.n_paths), slice(self.n_paths, None))):
                count = simulated[rows].shape[0]
                if not count:
                    continue
                sampler = make_sampler(self.seed, stream=2 * (start + day) + bank,
                                       backend=self.backend)
                if fit.phi >= 1e7:
                    simulated[rows, day] = sampler.poisson(mean[day], size=count)
                else:
                    probability = fit.phi / (fit.phi + mean[day])
                    simulated[rows, day] = sampler.negative_binomial(fit.phi, probability, size=count)

        def scores(counts):
            residual = (counts - mean) / sd
            return _surveillance_traces_from_baseline(counts, mean, residual, fit.phi,
                                                      0, (mean.size,), 0.5, self.backend)

        statistic, location, direction = scores(deaths[start:])

        def score_batch(batch):
            return np.array([scores(path)[0][0] for path in batch])

        if self.workers == 1:
            reference = score_batch(simulated)
        else:
            batches = np.array_split(simulated, min(self.workers, len(simulated)))
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                reference = np.concatenate(list(pool.map(score_batch, batches)))
        if not np.isfinite(reference).all() or not np.isfinite(statistic).all():
            raise RuntimeError('Nonfinite benchmark statistic in the observed or simulated prefix.')
        results = []
        for column, method in enumerate(self.methods):
            step = self.boundaries[column].advance(reference[:self.n_paths, column])
            self.audit_crossed[:, column] |= reference[self.n_paths:, column] > step.boundary
            results.append(dict(method=method, look=step.look, prefix=prefix,
                                statistic=float(statistic[0, column]), cutoff=step.boundary,
                                eligible_before=step.eligible_before,
                                reference_crossings=step.strict_crossed_total,
                                audit_crossings=int(self.audit_crossed[:, column].sum()),
                                alarm=bool(statistic[0, column] > step.boundary),
                                direction=int(direction[0, column]),
                                outcome_index=start + int(location[0, column])))
        self._history = (cases.copy(), deaths.copy())
        self._mean = mean
        self.reference_fit = fit
        return results


def _processed_dengue_path(data_dir: Path) -> Path:
    """Return the canonical file, accepting the legacy repository filename."""

    canonical = data_dir / 'processed' / f'dghs_dengue_{REAL_DATA.dengue_year}_processed.csv'
    legacy = (
        data_dir / 'processed'
        / f'dghs_bangladesh_dengue_{REAL_DATA.dengue_year}_weekly_processed.csv'
    )
    if canonical.exists():
        return canonical
    if legacy.exists():
        return legacy
    return canonical


def _has_date(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


USER_AGENT = 'OCFR-core/1.0 (aggregate public-health data)'


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _download(url: str, *, data: bytes | None = None) -> bytes:
    headers = {'User-Agent': USER_AGENT}
    if data is not None:
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def _cached_download(
    url: str,
    path: Path,
    *,
    data: bytes | None = None,
    refresh: bool = False,
) -> bytes:
    if path.exists() and not refresh:
        return path.read_bytes()
    content = _download(url, data=data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return content


def reconcile_reported_increments(values: np.ndarray) -> np.ndarray:
    """Move negative revisions backward while preserving the final total."""

    output = np.asarray(values, dtype=float).copy()
    if output.ndim != 1 or not np.all(np.isfinite(output)):
        raise ValueError('increments must be a finite vector')
    for index in range(output.size):
        if output[index] >= 0.0:
            continue
        debt = -float(output[index])
        output[index] = 0.0
        prior = index - 1
        while debt > 1e-12 and prior >= 0:
            payment = min(float(output[prior]), debt)
            output[prior] -= payment
            debt -= payment
            prior -= 1
        if debt > 1e-8:
            raise ValueError('negative revision exceeds all earlier reports')
    return output


def prepare_covid(data_dir: Path, config: RealDataConfig, *, refresh: bool,
                  strict: bool = False) -> dict[str, Any]:
    raw_path = data_dir / 'raw' / 'who_covid.csv'
    raw = _cached_download(config.covid_url, raw_path, refresh=refresh)
    frame = pd.read_csv(
        io.BytesIO(raw),
        usecols=['Date_reported', 'Country_code', 'Country', 'New_cases', 'New_deaths'],
    )
    frame['Date_reported'] = pd.to_datetime(frame['Date_reported'])
    frame = frame[
        frame['Country_code'].isin(config.covid_country_codes)
        & frame['Date_reported'].between(config.covid_start, config.covid_end)
    ].copy()
    if strict and (
        frame[['New_cases', 'New_deaths']].isna().any().any()
        or frame[['New_cases', 'New_deaths']].lt(0).any().any()
        or frame.duplicated(['Country_code', 'Date_reported']).any()
    ):
        raise ValueError('The WHO manuscript extract requires complete, nonnegative daily counts.')
    frame[['New_cases', 'New_deaths']] = frame[['New_cases', 'New_deaths']].fillna(0.0)
    cleaned: list[pd.DataFrame] = []
    for country, group in frame.groupby('Country', sort=True):
        group = group.sort_values('Date_reported').copy()
        group['cases'] = reconcile_reported_increments(group['New_cases'].to_numpy(float))
        group['deaths'] = reconcile_reported_increments(group['New_deaths'].to_numpy(float))
        group['series'] = country
        cleaned.append(group[['series', 'Date_reported', 'cases', 'deaths']])
    processed = pd.concat(cleaned, ignore_index=True).rename(columns={'Date_reported': 'date'})
    output = data_dir / 'processed' / 'who_covid_processed.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output, index=False)
    return {
        'source_url': config.covid_url,
        'raw_sha256': _sha256(raw),
        'processed_file': str(output),
        'rows': int(len(processed)),
        'negative_revision_rule': 'backward redistribution preserving final total',
    }


# Replay specifications record the counting clock, analysis calendar and seeds.
@dataclass(frozen=True)
class SurveillanceReplay:
    name: str
    source_series: str | None
    history_start: str
    outcome_start: str
    first_candidate: str
    first_look: str
    horizon: str
    benchmark_reference_last: str
    lag_offset: int
    seed: int
    audit_reference_last: str | None = None

    def index(self, day: str) -> int:
        return (pd.Timestamp(day) - pd.Timestamp(self.history_start)).days

    @property
    def looks(self) -> tuple[int, ...]:
        calendar = pd.date_range(self.first_look, self.horizon, freq='7D')
        return tuple(self.index(day) + 1 for day in calendar.union([pd.Timestamp(self.horizon)]))

    @property
    def benchmark_looks(self) -> tuple[int, ...]:
        end = self.index(self.benchmark_reference_last) + 1
        return tuple(prefix for prefix in self.looks if prefix > end)


SURVEILLANCE_REPLAYS = {
    key: SurveillanceReplay(
        name, source, '2020-05-15', '2020-06-29', '2020-07-15', '2020-07-28',
        '2021-01-14', '2020-07-14', 1, REAL_DATA.master_seed + index,
    )
    for index, (key, name, source) in enumerate((
        ('japan', 'Japan', 'Japan'),
        ('south_korea', 'South Korea', 'Republic of Korea'),
        ('united_kingdom', 'United Kingdom', 'United Kingdom of Great Britain and Northern Ireland'),
    ))
}
# The sparse January prefix motivates a weeks 1–13 reference for the benchmark
# and case-conditional audit. The OCFR observed replay keeps its January start.
SURVEILLANCE_REPLAYS['brazil'] = SurveillanceReplay(
    'Brazil 2018 seasonal A(H1N1)pdm09', None, '2017-11-16', '2017-12-31', '2018-01-14',
    '2018-01-27', '2018-12-29', '2018-03-31', 0, 2026112101, audit_reference_last='2018-03-31',
)
BRAZIL_DAILY_FILE = 'brazil_2018_influenza_daily.csv'
BRAZIL_SOURCES = {
    'INFLUD17.csv': ('https://d26692udehoye.cloudfront.net/SRAG/2013-2018/INFLUD17.csv',
                     'RES_FLUASU', 'DT_OBITO', '%d/%m/%Y'),
    'INFLUD18.csv': ('https://d26692udehoye.cloudfront.net/SRAG/2013-2018/INFLUD18.csv',
                     'RES_FLUASU', 'DT_OBITO', '%d/%m/%Y'),
    'INFLUD19-23-03-2026.csv': (
        'https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2019/INFLUD19-23-03-2026.csv',
        'PCR_FLUASU', 'DT_EVOLUCA', '%Y-%m-%d',
    ),
}
BRAZIL_DELAY_LAWS = {'ejima': (9.5, 4.7**2), 'nishiura': (9., 39.7)}
# These hashes are for the exact analyzed aggregates with LF line endings.
MANUSCRIPT_INPUT_SHA256 = {
    'who_covid_processed.csv': '4f160c6817d4c15ce61ceb6831c1366277594c2db4d2852ec4be9977801ddd3e',
    BRAZIL_DAILY_FILE: '38f1f97430880902637780b549b5300a9b9af0fcc43e110605b6791e82be7639',
}


def brazil_delay(law: str = 'ejima') -> np.ndarray:
    """Discretize the onset-to-death Gamma law in half-day bins on lags 0–45."""
    return _brazil_gamma_delay(*BRAZIL_DELAY_LAWS[law])


def _brazil_gamma_delay(mean: float, variance: float) -> np.ndarray:
    mass = np.diff(gamma.cdf(np.r_[0, np.arange(46) + .5],
                            a=mean**2 / variance, scale=variance / mean))
    return mass / mass.sum()


def prepare_brazil(data_dir: Path, *, refresh: bool = False) -> dict[str, Any]:
    """Aggregate the 2018 season from the adjacent national SRAG releases."""
    replay = SURVEILLANCE_REPLAYS['brazil']
    dates = pd.date_range(replay.history_start, replay.horizon)
    inputs, eligible = {}, []
    for name, (source, subtype, outcome_date, date_format) in BRAZIL_SOURCES.items():
        date_fields = ('DT_SIN_PRI', 'DT_NOTIFIC', outcome_date)
        eligibility_fields = ['CLASSI_FIN', 'CRITERIO', 'HOSPITAL', subtype]
        fields = (*date_fields, *eligibility_fields, 'EVOLUCAO', 'DT_NASC', 'CS_SEXO')
        raw = _cached_download(source, data_dir / 'raw' / name, refresh=refresh)
        records = pd.read_csv(io.BytesIO(raw), sep=';', encoding='latin1', usecols=fields, dtype=str)
        for column in (*eligibility_fields, 'EVOLUCAO'):
            records[column] = records[column].str.replace(r'\.0$', '', regex=True)
        for column in date_fields:
            records[column] = pd.to_datetime(records[column].str[:10], format=date_format,
                                             errors='coerce')
        if len(records) and records.DT_SIN_PRI.isna().all():
            raise ValueError(f'{name}: could not parse any onset dates.')
        # Code 1 in the four fields selects hospitalized, lab-confirmed A(H1N1)pdm09.
        rows = records.loc[records[eligibility_fields].eq('1').all(axis=1)].rename(
            columns={'DT_SIN_PRI': 'onset', 'DT_NOTIFIC': 'notification'})
        rows['death'] = rows[outcome_date].where(rows.EVOLUCAO.eq('2'))
        relevant = rows.onset.between(dates[0], dates[-1])
        invalid = rows.death.lt(rows.onset)
        selected = rows.loc[relevant & ~invalid, [
            'onset', 'death', 'notification', 'EVOLUCAO', 'DT_NASC', 'CS_SEXO',
        ]].assign(release=name)
        eligible.append(selected)
        inputs[name] = dict(url=source, sha256=_sha256(raw), source_rows=len(records),
                            eligible_all=len(rows), eligible_records=len(selected),
                            missing_onset=int(rows.onset.isna().sum()),
                            negative_date_order_exclusions=int((relevant & invalid).sum()),
                            prehistory_onsets=int(selected.onset.lt(replay.outcome_start).sum()),
                            monitored_onsets=int(selected.onset.ge(replay.outcome_start).sum()),
                            monitored_deaths=int(selected.death.between(replay.outcome_start, replay.horizon).sum()))

    # Separate date axes retain deaths from earlier onset cohorts. Undated
    # outcomes remain in the onset counts and contribute zero dated deaths.
    cohort = pd.concat(eligible, ignore_index=True)
    release_counts = cohort.groupby(['DT_NASC', 'CS_SEXO', 'onset'], dropna=False).release.nunique()
    cross_release_matches = int(release_counts.gt(1).sum())
    if cross_release_matches:
        raise ValueError(f'Inspect {cross_release_matches} cross-release demographic and onset matches.')
    counts = pd.DataFrame({
        'date': dates,
        'cases': cohort.onset.value_counts().reindex(dates, fill_value=0).to_numpy(),
        'deaths': cohort.death.value_counts().reindex(dates, fill_value=0).to_numpy(),
    })
    monitored = cohort.loc[cohort.onset.ge(replay.outcome_start)]
    output = data_dir / 'processed' / BRAZIL_DAILY_FILE
    output.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(output, index=False, date_format='%Y-%m-%d', lineterminator='\n')
    notification_at_planned_looks = {}
    for prefix in replay.looks:
        day = dates[prefix - 1]
        through_look = monitored.loc[monitored.onset.le(day)]
        notification_at_planned_looks[day.date().isoformat()] = {
            'archived_onsets': len(through_look),
            'notified_after_look': int(through_look.notification.gt(day).sum()),
            'missing_notification_date': int(through_look.notification.isna().sum()),
        }
    delay = (monitored.death - monitored.onset).dt.days.dropna()
    return {
        'inputs': inputs, 'processed_file': output.name, 'rows': len(counts),
        'negative_date_order_exclusions': sum(x['negative_date_order_exclusions'] for x in inputs.values()),
        'cross_release_fingerprint_matches': cross_release_matches,
        'deaths_without_date': int((cohort.EVOLUCAO.eq('2') & cohort.death.isna()).sum()),
        'deaths_after_horizon': int(cohort.death.gt(replay.horizon).sum()),
        'prehistory_cases': int(counts.iloc[:replay.index(replay.outcome_start)].cases.sum()),
        'monitored_cohort_cases': len(monitored),
        'monitored_deaths': int(counts.iloc[replay.index(replay.outcome_start):].deaths.sum()),
        'outcomes_for_monitored_cohorts': monitored.EVOLUCAO.fillna('missing').value_counts().to_dict(),
        'median_observed_delay_days': float(delay.median()) if len(delay) else None,
        'clinical_delay_moments': dict(dated_deaths=len(delay),
                                       mean_days=float(delay.mean()) if len(delay) else None,
                                       variance_days_squared=float(delay.var(ddof=1)) if len(delay) > 1 else None),
        'notification_at_planned_looks': notification_at_planned_looks,
    }


def _manuscript_input_hash(path: Path) -> str:
    # Normalize CSV line terminators so a Windows checkout and a Unix replay
    # identify the same sequence of published input rows.
    return _sha256(path.read_bytes().replace(b'\r\n', b'\n'))


def _surveillance_input_hashes(data_dir: Path) -> dict[str, str]:
    return {name: _manuscript_input_hash(data_dir / 'processed' / name)
            for name in MANUSCRIPT_INPUT_SHA256}


def prepare_surveillance(data_dir: Path, *, refresh: bool = False) -> dict[str, Any]:
    """Fetch the current paper's public WHO and Brazilian SRAG downloads."""
    metadata = {
        'covid': prepare_covid(data_dir, REAL_DATA, refresh=refresh, strict=True),
        'brazil': prepare_brazil(data_dir, refresh=refresh),
    }
    processed = data_dir / 'processed'
    metadata['analyzed_sha256'] = _surveillance_input_hashes(data_dir)
    metadata['matches_manuscript_inputs'] = metadata['analyzed_sha256'] == MANUSCRIPT_INPUT_SHA256
    brazil = metadata['brazil']
    outcomes = brazil['outcomes_for_monitored_cohorts']
    record_flow = {
        'history_onsets': brazil['prehistory_cases'],
        'monitored_onsets': brazil['monitored_cohort_cases'],
        'recovered_monitored_cohorts': outcomes.get('1', 0),
        'coded_influenza_deaths_monitored_cohorts': outcomes.get('2', 0),
        'other_deaths_monitored_cohorts': outcomes.get('3', 0),
        'missing_or_unknown_outcomes_monitored_cohorts': outcomes.get('missing', 0) + outcomes.get('9', 0),
        'coded_influenza_deaths_without_death_date_all_cohorts': brazil['deaths_without_date'],
        'coded_influenza_deaths_after_horizon': brazil['deaths_after_horizon'],
        'dated_influenza_deaths_during_monitoring': brazil['monitored_deaths'],
    }
    pd.Series(record_flow, name='count').rename_axis('definition').to_csv(
        processed / 'brazil_record_flow.csv')
    (processed / 'surveillance_provenance.json').write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding='utf-8')
    return metadata


def _highcharts_series(chart: str, name: str) -> list[int]:
    match = re.search(
        rf"name:\s*'{re.escape(name)}'\s*,\s*data:\s*(\[[^\]]+\])",
        chart,
    )
    if match is None:
        raise RuntimeError(f'could not locate {name!r} in the dengue chart')
    values = json.loads(match.group(1))
    if not all(isinstance(value, int) and value >= 0 for value in values):
        raise RuntimeError(f'invalid dengue values for {name!r}')
    return values


def prepare_dengue(data_dir: Path, config: RealDataConfig, *, refresh: bool) -> dict[str, Any]:
    form = urllib.parse.urlencode(
        {'filter_year': str(config.dengue_year), 'search_filter': 'Search'}
    ).encode('ascii')
    raw_path = data_dir / 'raw' / f'dghs_dengue_{config.dengue_year}.html'
    raw = _cached_download(config.dengue_url, raw_path, data=form, refresh=refresh)
    page = raw.decode('utf-8', errors='replace')
    chart_match = re.search(
        r"Highcharts\.chart\('death_case_ration_by_week'.*?\n\s*\}\);",
        page,
        flags=re.DOTALL,
    )
    if chart_match is None:
        raise RuntimeError('could not locate the DGHS weekly dengue chart')
    chart = chart_match.group(0)
    cases = _highcharts_series(chart, 'Affected (Admitted)')
    deaths = _highcharts_series(chart, 'Death')
    if len(cases) < 3 or len(cases) != len(deaths):
        raise RuntimeError('unexpected DGHS weekly series length')
    labels = ['boundary-start', *[f'W{week:02d}' for week in range(1, len(cases) - 1)], 'boundary-end']
    processed = pd.DataFrame({'reporting_week': labels, 'cases': cases, 'deaths': deaths}).iloc[1:-1].copy()
    processed['date'] = [
        pd.Timestamp(f'{config.dengue_year}-01-08') + pd.Timedelta(weeks=week)
        for week in range(len(processed))
    ]
    processed['series'] = 'Bangladesh dengue'
    processed = processed[['series', 'date', 'reporting_week', 'cases', 'deaths']]
    output = data_dir / 'processed' / f'dghs_dengue_{config.dengue_year}_processed.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output, index=False)
    return {
        'source_url': config.dengue_url,
        'raw_sha256': _sha256(raw),
        'processed_file': str(output),
        'rows': int(len(processed)),
        'boundary_rule': 'exclude the two partial boundary bins',
    }


def prepare_all(data_dir: Path, *, refresh: bool = False) -> dict[str, Any]:
    metadata = {
        'covid': prepare_covid(data_dir, REAL_DATA, refresh=refresh),
        'dengue': prepare_dengue(data_dir, REAL_DATA, refresh=refresh),
    }
    path = data_dir / 'processed' / 'provenance.json'
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding='utf-8')
    return metadata


def _dengue_delay(mean_days: float = 3.0, maximum_weeks: int = 3) -> np.ndarray:
    """Weekly lag 0..K under uniform admission time within a report week."""
    if mean_days <= 0 or maximum_weeks < 0:
        raise ValueError('mean_days must be positive and maximum_weeks nonnegative')
    distribution = gamma(a=2.0, scale=mean_days / 2.0)
    values = []
    for week_lag in range(maximum_weeks + 1):
        lower, upper = 7.0 * week_lag, 7.0 * (week_lag + 1)
        mass = quad(
            lambda u: distribution.cdf(max(0.0, upper - u))
            - distribution.cdf(max(0.0, lower - u)),
            0.0,
            7.0,
            epsabs=1e-12,
        )[0] / 7.0
        values.append(mass)
    values = np.asarray(values, dtype=float)
    return values / values.sum()


def _two_rate_intervals(
    cases: np.ndarray,
    deaths: np.ndarray,
    delay: np.ndarray,
    tau: int,
    observation_start: int,
    lag_offset: int = 1,
) -> dict[str, float | str]:
    total, post = exposure_components(cases, delay, lag_offset=lag_offset)
    post_exposure = post[:, tau - 1]
    design = np.column_stack((total - post_exposure, post_exposure))
    active = np.arange(cases.size) >= observation_start
    ordinary = fit_nb_identity(
        deaths[active], np.zeros(int(active.sum()), dtype=float), design[active]
    )
    reporting = PostAlarmEstimationConfig()
    fit = fit_nb_regularized_identity(
        deaths[active], np.zeros(int(active.sum()), dtype=float), design[active],
        precision=reporting.precision, initial_fit=ordinary)
    result: dict[str, float | str] = {
        'pre_cfr': float(fit.rates[0]),
        'post_cfr': float(fit.rates[1]),
        'risk_ratio': float(fit.rates[1] / fit.rates[0]),
        'interval_status': 'not_available',
        'estimator': reporting.estimator,
        'estimator_status': fit.status,
        'estimator_precision': reporting.precision,
        'interval_covariance': reporting.covariance,
        'point_valid': bool(fit.success),
        'mle_pre_cfr': float(ordinary.rates[0]),
        'mle_post_cfr': float(ordinary.rates[1]),
        'mle_risk_ratio': float(ordinary.rates[1] / ordinary.rates[0]),
    }
    if not fit.success:
        result['interval_status'] = 'failed:regularized_fit_nonconvergence'
        return result
    try:
        informative = active & (total > 0)
        covariance_design = design[informative]
        if np.any(deaths[active & ~informative] > 0) or np.any(covariance_design < -MU_FLOOR):
            raise ValueError('Invalid exposure for two-rate covariance.')
        # Structural-zero rows contribute no information to this covariance.
        # Roundoff in subtracting the post-change exposure can create tiny negative values.
        intervals = regularized_parameter_intervals(
            fit, np.zeros(int(informative.sum())), np.maximum(covariance_design, 0.),
            precision=reporting.precision, confidence=reporting.confidence,
            covariance=reporting.covariance,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        result['interval_status'] = f'failed:{type(exc).__name__}'
        return result
    result.update({
        'pre_cfr_lower': float(intervals.cfr_lower[0]),
        'pre_cfr_upper': float(intervals.cfr_upper[0]),
        'post_cfr_lower': float(intervals.cfr_lower[1]),
        'post_cfr_upper': float(intervals.cfr_upper[1]),
        'risk_ratio_lower': float(intervals.risk_ratio_lower),
        'risk_ratio_upper': float(intervals.risk_ratio_upper),
        'interval_status': 'success:regularized_sandwich_approximate',
    })
    return result


def monitor_series(
    frame: pd.DataFrame,
    *,
    series: str,
    delay: np.ndarray,
    observation_start: int,
    candidate_start: int,
    first_look: int,
    look_every: int,
    n_multiplier: int,
    seed: int,
    backend: str = 'c',
    lag_offset: int = 1,
    boundary_policy: str = GAUSSIAN_SPENDING_POLICY,
    include_final_benchmarks: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if boundary_policy not in {GAUSSIAN_SPENDING_POLICY, LEGACY_GAUSSIAN_POLICY}:
        raise ValueError('unknown boundary policy')
    if frame.empty or first_look < 1 or look_every < 1:
        raise ValueError('nonempty series and positive look schedule required')
    frame = frame.sort_values('date').reset_index(drop=True)
    cases = frame['cases'].to_numpy(float)
    deaths = frame['deaths'].to_numpy(float)
    dates = pd.to_datetime(frame['date'])
    looks = list(range(first_look, len(frame) + 1, look_every))
    if not looks or looks[-1] != len(frame):
        looks.append(len(frame))
    normal_mode = 'pcg32-numpy-box-muller-v1' if boundary_policy == GAUSSIAN_SPENDING_POLICY else 'scalar'
    multipliers = gaussian_multiplier_bank(make_sampler(seed, backend=backend),
        (n_multiplier, len(frame)), normal_mode=normal_mode)
    boundary_state = GaussianSpendingBoundary(n_multiplier, len(looks), REAL_DATA.alpha)
    input_payload = frame[['date', 'cases', 'deaths']].to_csv(index=False).encode('utf-8')
    provenance = {
        'boundary_policy': boundary_policy, 'multiplier_seed': int(seed),
        'multiplier_paths': int(n_multiplier), 'multiplier_normal_mode': normal_mode,
        'multiplier_bank_sha256': _sha256(np.ascontiguousarray(multipliers, dtype='<f8').tobytes()),
        'input_series_sha256': _sha256(input_payload),
        'delay_sha256': _sha256(np.ascontiguousarray(delay, dtype='<f8').tobytes()),
        'observation_start': observation_start, 'candidate_start': candidate_start,
        'lag_offset': lag_offset, 'planned_prefixes': looks, 'backend': backend,
        'alpha': REAL_DATA.alpha,
    }
    local_alpha = REAL_DATA.alpha / len(looks)
    trace: list[dict[str, Any]] = []
    alarm = None
    alarm_scan = None
    for look_number, prefix in enumerate(looks, start=1):
        scan = score_scan(
            cases[:prefix], deaths[:prefix], delay,
            observation_start=observation_start,
            candidate_start=candidate_start,
            backend=backend,
            lag_offset=lag_offset,
        )
        threshold = np.inf
        pvalue = np.nan
        maxima = None
        available = bool(scan.candidates.size and np.isfinite(scan.statistic))
        if available:
            maxima = multiplier_statistics_from_draws(
                multipliers[:, :prefix], scan.influence, backend=backend
            )
            if boundary_policy == LEGACY_GAUSSIAN_POLICY:
                threshold = monte_carlo_critical_value(maxima, local_alpha)
            pvalue = (1 + np.count_nonzero(maxima >= scan.statistic)) / (n_multiplier + 1)
        diagnostics = {}
        if boundary_policy == GAUSSIAN_SPENDING_POLICY:
            boundary_look = boundary_state.advance(maxima, available=available)
            threshold = boundary_look.boundary
            diagnostics = {name: getattr(boundary_look, name) for name in boundary_look.__dataclass_fields__}
            diagnostics['numerical_crossing_probability'] = boundary_look.strict_crossed_total / n_multiplier
            diagnostics['reference_diagnostic_scope'] = 'training Gaussian bank; not observed-data size'
        crossed = bool(np.isfinite(scan.statistic) and scan.statistic > threshold)
        row = {
            'series': series,
            'look_number': look_number,
            'prefix': prefix,
            'look_date': dates.iloc[prefix - 1].date().isoformat(),
            'statistic': scan.statistic,
            'critical_value': threshold,
            'multiplier_pvalue': pvalue,
            **({'episode_adjusted_pvalue': min(1.0, len(looks) * pvalue) if np.isfinite(pvalue) else np.nan}
               if boundary_policy == LEGACY_GAUSSIAN_POLICY else {}),
            'boundary_policy': boundary_policy,
            'evidence_margin': float(scan.statistic - threshold),
            'multiplier_pvalue_scope': 'marginal fixed-look diagnostic only',
            **diagnostics,
            'tau_hat': scan.tau_hat,
            'estimated_change_date': (
                dates.iloc[int(scan.tau_hat)].date().isoformat()
                if scan.tau_hat is not None else ''
            ),
            'direction': int(scan.direction),
            'candidate_count': int(scan.candidates.size),
            'crossed': crossed,
            'fit_status': scan.null_fit.status,
            'fitted_risk': scan.null_fit.pi,
            'fitted_shape': scan.null_fit.phi,
            'fit_success': scan.null_fit.success,
        }
        trace.append(row)
        if crossed and alarm is None:
            alarm = row
            alarm_scan = scan

    summary: dict[str, Any] = {
        'series': series,
        'start_date': dates.iloc[0].date().isoformat(),
        'end_date': dates.iloc[-1].date().isoformat(),
        'n_observations': len(frame),
        'n_looks': len(looks),
        'alarm': alarm is not None,
        'alarm_date': '' if alarm is None else alarm['look_date'],
        'estimated_change_date': '' if alarm is None else alarm['estimated_change_date'],
        **({'episode_adjusted_pvalue': np.nan if alarm is None else alarm['episode_adjusted_pvalue']}
           if boundary_policy == LEGACY_GAUSSIAN_POLICY else {}),
        'boundary_policy': boundary_policy,
        'alarm_prefix': np.nan if alarm is None else alarm['prefix'],
        'statistic_at_alarm': np.nan if alarm is None else alarm['statistic'],
        'boundary_at_alarm': np.nan if alarm is None else alarm['critical_value'],
        'cumulative_alpha_at_alarm': np.nan if alarm is None else REAL_DATA.alpha * alarm['look_number'] / len(looks),
        'interval_scope': 'unadjusted descriptive at selected change and random first alarm',
        'provenance': provenance,
    }
    if alarm is not None and alarm_scan is not None:
        summary.update(
            _two_rate_intervals(
                cases[: int(alarm['prefix'])],
                deaths[: int(alarm['prefix'])],
                delay,
                int(alarm_scan.tau_hat),
                observation_start,
                lag_offset=lag_offset,
            )
        )

    if not include_final_benchmarks:
        return summary, pd.DataFrame(trace), pd.DataFrame()

    benchmark = published_benchmark_statistics(
        cases, deaths, delay,
        reference_end=max(observation_start + 14, min(28, len(frame) - 2)),
        observation_start=observation_start,
        lag_offset=lag_offset,
    )
    benchmark_rows = [
        {
            'series': series,
            'method': result.method,
            'final_statistic': result.statistic,
            'estimated_change_date': (
                dates.iloc[int(result.estimated_cohort_change_day)].date().isoformat()
                if result.estimated_cohort_change_day is not None else ''
            ),
            'direction': result.direction,
            'interpretation': 'descriptive statistic; external null calibration required for testing',
        }
        for result in benchmark.values()
    ]
    return summary, pd.DataFrame(trace), pd.DataFrame(benchmark_rows)


def _surveillance_frame(data_dir: Path, replay: SurveillanceReplay) -> pd.DataFrame:
    filename = 'who_covid_processed.csv' if replay.source_series else BRAZIL_DAILY_FILE
    path = data_dir / 'processed' / filename
    if not path.exists():
        raise FileNotFoundError(f'{path} is missing. Run the fetch stage first.')
    frame = pd.read_csv(path, parse_dates=['date'])
    if replay.source_series:
        frame = frame.loc[frame.series.eq(replay.source_series)]
    frame = frame[['date', 'cases', 'deaths']].sort_values('date').reset_index(drop=True)
    dates = pd.date_range(replay.history_start, replay.horizon)
    counts = frame[['cases', 'deaths']].to_numpy(float)
    if (not pd.DatetimeIndex(frame.date).equals(dates) or not np.isfinite(counts).all()
            or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all()):
        raise ValueError(f'{replay.name}: expected one finite nonnegative integer count per planned day.')
    return frame


def _ocfr_replay(frame, replay, delay, backend, boundary_policy, *,
                 first_look=None, candidate_start=None):
    return monitor_series(
        frame, series=replay.name, delay=delay,
        observation_start=replay.index(replay.outcome_start),
        candidate_start=(replay.index(replay.first_candidate) if candidate_start is None
                         else candidate_start),
        first_look=replay.looks[0] if first_look is None else first_look,
        look_every=7, n_multiplier=REAL_DATA.n_multiplier, seed=replay.seed,
        backend=backend, lag_offset=replay.lag_offset, boundary_policy=boundary_policy,
        include_final_benchmarks=False,
    )[:2]


def _primary_delay(replay):
    return gamma_delay_pmf(14., shape=4., maximum_lag=45) if replay.lag_offset else brazil_delay()


def _sensitivity_replays(key, frame, replay, primary, backend, boundary_policy):
    """Reuse the primary runs in the appendix grids and hold the Gaussian seed fixed."""
    summary_rows, trace_rows = [], []

    def retain(label, result, **settings):
        record, trace = result
        first_eligible = trace.loc[trace.candidate_count.gt(0), 'look_date']
        usable = trace.candidate_count.gt(0) & np.isfinite(trace.critical_value)
        first_alarm = trace.loc[trace.crossed].head(1)
        summary_rows.append({
            'series': replay.name, 'sensitivity_id': label, **settings,
            'first_planned': trace.look_date.iloc[0], 'planned_looks': len(trace),
            'first_eligible': first_eligible.iloc[0] if len(first_eligible) else None,
            'usable_looks': int(usable.sum()),
            'alarm_usable_ordinal': int(usable.cumsum().loc[first_alarm.index[0]]) if len(first_alarm) else None,
            **{name: record.get(name) for name in ('alarm_date', 'estimated_change_date',
                   'statistic_at_alarm', 'boundary_at_alarm', 'pre_cfr', 'post_cfr', 'risk_ratio')},
        })
        trace_rows.append(trace.assign(sensitivity_id=label))

    if key == 'brazil':
        observation_start = replay.index(replay.outcome_start)
        # All calendars end on the same date. An unavailable earlier look
        # still consumes its planned index in the first-crossing rule.
        for warmup, grids in ((14, (7,)), (21, (7, 14)), (28, (0, 7, 14))):
            for grid in grids:
                candidate = observation_start + grid
                for law, (mean, variance) in BRAZIL_DELAY_LAWS.items():
                    label = f'brazil-{law}-{warmup}-{grid}'
                    result = primary if (warmup, grid, law) == (28, 14, 'ejima') else _ocfr_replay(
                        frame, replay, brazil_delay(law), backend, boundary_policy,
                        first_look=observation_start + warmup, candidate_start=candidate,
                    )
                    retain(label, result, law=law, mean_delay=mean, shape=mean**2 / variance,
                           first_candidate=frame.date.iloc[candidate].date().isoformat())
        # Compare alarm timing on the calendar available to the fixed-reference
        # charts while retaining OCFR's original likelihood and candidate grid.
        for law, (mean, variance) in BRAZIL_DELAY_LAWS.items():
            result = _ocfr_replay(frame, replay, brazil_delay(law), backend, boundary_policy,
                                  first_look=replay.benchmark_looks[0])
            retain(f'brazil-{law}-common-calendar', result, law=law, mean_delay=mean,
                   shape=mean**2 / variance, first_candidate=replay.first_candidate)
    else:
        specs = [(float(mean), 4., replay.first_candidate) for mean in (11, 13, 14, 15, 17)]
        if key == 'south_korea':
            specs.append((17.2, 1.71, replay.first_candidate))
        if key == 'united_kingdom':
            specs.extend((14., 4., date) for date in ('2020-07-08', '2020-07-01'))
        for mean, shape, candidate in specs:
            result = primary if (mean, shape, candidate) == (14., 4., replay.first_candidate) else _ocfr_replay(
                frame, replay, gamma_delay_pmf(mean, shape=shape, maximum_lag=45),
                backend, boundary_policy, candidate_start=replay.index(candidate),
            )
            retain(f'{key}-gamma-{mean:g}-{candidate}', result, law='gamma', mean_delay=mean,
                   shape=shape, first_candidate=candidate)
    return summary_rows, trace_rows


def run_surveillance(data_dir: Path, output_dir: Path, *, backend: str = 'python',
                     boundary_policy: str = GAUSSIAN_SPENDING_POLICY) -> pd.DataFrame:
    """Generate the four OCFR replays, calendars and delay/grid sensitivity tables."""
    summaries, traces, grids, grid_traces, protocols, residuals, residual_summaries = [], [], [], [], [], [], []
    provenance = {}
    for key, replay in SURVEILLANCE_REPLAYS.items():
        frame = _surveillance_frame(data_dir, replay)
        delay = _primary_delay(replay)
        summary, trace = _ocfr_replay(frame, replay, delay, backend, boundary_policy)
        provenance[key] = summary.pop('provenance')
        summaries.append(summary)
        traces.append(trace)
        residual_summary, residual_frame = _surveillance_residuals(frame, replay, delay)
        residual_summary.update(
            observed_shape_min=float(trace.fitted_shape.min()),
            observed_shape_max=float(trace.fitted_shape.max()),
            observed_shape_upper=int(trace.fitted_shape.ge(1e8 * (1 - 1e-7)).sum()),
            observed_fit_failures=int((~trace.fit_success).sum()),
        )
        residual_summaries.append(residual_summary)
        residuals.append(residual_frame)
        grids_for_series, traces_for_series = _sensitivity_replays(
            key, frame, replay, (summary, trace), backend, boundary_policy,
        )
        grids.extend(grids_for_series)
        grid_traces.extend(traces_for_series)
        protocols.append(dict(
            series=replay.name, history_start=replay.history_start,
            history_end=frame.date.iloc[replay.index(replay.outcome_start) - 1].date().isoformat(),
            outcome_start=replay.outcome_start, first_candidate=replay.first_candidate,
            first_look=replay.first_look, horizon=replay.horizon, n_observations=len(frame),
            n_looks=len(replay.looks), lag_offset=replay.lag_offset,
            benchmark_reference_last=replay.benchmark_reference_last,
            benchmark_first_look=frame.date.iloc[replay.benchmark_looks[0] - 1].date().isoformat(),
            benchmark_looks=len(replay.benchmark_looks),
            audit_reference_last=frame.date.iloc[_audit_reference_end(replay) - 1].date().isoformat(),
        ))
    summary_frame = pd.DataFrame(summaries)
    products = {
        'application_summary.csv': summary_frame,
        'look_trace.csv': pd.concat(traces, ignore_index=True),
        'application_protocol.csv': pd.DataFrame(protocols),
        'sensitivity_summary.csv': pd.DataFrame(grids),
        'sensitivity_trace.csv': pd.concat(grid_traces, ignore_index=True),
        'residual_summary.csv': pd.DataFrame(residual_summaries),
        'residuals.csv': pd.concat(residuals, ignore_index=True),
    }
    _write_surveillance_outputs(data_dir, output_dir, 'manifest.json', products, provenance)
    return summary_frame


def run_brazil_clinical_delay(data_dir: Path, output_dir: Path, *, backend: str = 'python',
                             boundary_policy: str = GAUSSIAN_SPENDING_POLICY) -> pd.DataFrame:
    """Replay an exploratory Gamma law fitted to finalized seasonal death dates."""
    provenance_path = data_dir / 'processed' / 'surveillance_provenance.json'
    if not provenance_path.is_file():
        raise FileNotFoundError('Run the fetch stage to prepare the aggregate clinical delay moments.')
    source = json.loads(provenance_path.read_text(encoding='utf-8'))
    if source['analyzed_sha256'][BRAZIL_DAILY_FILE] != _surveillance_input_hashes(data_dir)[BRAZIL_DAILY_FILE]:
        raise ValueError('The seasonal counts changed after aggregation. Run the fetch stage again.')
    brazil = source['brazil']
    moments = brazil.get('clinical_delay_moments')
    if moments is None:
        raise ValueError('Run the fetch stage to prepare the aggregate clinical delay moments.')
    mean, variance = moments.get('mean_days'), moments.get('variance_days_squared')
    if (mean is None or variance is None or not np.isfinite([mean, variance]).all()
            or min(mean, variance) <= 0):
        raise ValueError('The exploratory Gamma law requires finite positive mean and variance from dated deaths.')
    replay = SURVEILLANCE_REPLAYS['brazil']
    frame = _surveillance_frame(data_dir, replay)
    delay = _brazil_gamma_delay(mean, variance)
    record, trace = _ocfr_replay(frame, replay, delay, backend, boundary_policy)
    settings = dict(
        scope='exploratory delay estimated from finalized seasonal records', moments=moments,
        raw_sha256={name: item['sha256'] for name, item in brazil['inputs'].items()},
        gamma_shape=mean**2 / variance, gamma_scale=variance / mean, delay=delay.tolist(),
        replay=record.pop('provenance'),
    )
    result = pd.DataFrame([record])
    _write_surveillance_outputs(data_dir, output_dir, 'clinical_delay_manifest.json',
                                {'clinical_delay_summary.csv': result, 'clinical_delay_trace.csv': trace},
                                settings)
    return result


def _audit_reference_end(replay):
    if replay.audit_reference_last is None:
        return replay.looks[0] - 1
    return replay.index(replay.audit_reference_last) + 1


def _audit_reference(frame, replay, delay):
    cases, deaths = (frame[field].to_numpy(float) for field in ('cases', 'deaths'))
    start, end = replay.index(replay.outcome_start), _audit_reference_end(replay)
    fit = fit_score_null(cases[:end], deaths[:end], delay,
                         observation_start=start, lag_offset=replay.lag_offset)
    exposure, _ = exposure_components(cases, delay, lag_offset=replay.lag_offset)
    if not fit.success or np.any((exposure[start:] <= MU_FLOOR) & (deaths[start:] > 0)):
        raise RuntimeError(f'{replay.name}: audit requires a successful reference fit and exposure for observed deaths.')
    exposure[exposure <= MU_FLOOR] = 0.
    return fit, fit.pi * exposure, start, end


def _surveillance_residuals(frame, replay, delay):
    fit, mean, start, end = _audit_reference(frame, replay, delay)
    deaths = frame.deaths.to_numpy(float)
    active = mean[start:] > 0
    pearson = np.full(len(mean) - start, np.nan)
    pearson[active] = ((deaths[start:][active] - mean[start:][active])
                       / np.sqrt(nb_variance(mean[start:][active], fit.phi)))
    reference, later = pearson[:end-start], pearson[end-start:]

    def lag_one(values):
        adjacent = np.isfinite(values[:-1]) & np.isfinite(values[1:])
        return float(np.corrcoef(values[:-1][adjacent], values[1:][adjacent])[0, 1])

    summary = dict(series=replay.name, reference_risk=float(fit.pi),
                   reference_shape=float(fit.phi), reference_days=end-start,
                   reference_active_days=int(active[:end-start].sum()),
                   reference_pearson_sd=float(np.nanstd(reference, ddof=1)),
                   reference_acf_lag1=lag_one(reference),
                   all_active_pearson_mean=float(np.nanmean(pearson)),
                   all_active_acf_lag1=lag_one(pearson),
                   later_pearson_mean=float(np.nanmean(later)), later_acf_lag1=lag_one(later))
    residuals = pd.DataFrame(dict(series=replay.name, date=frame.date.iloc[start:],
                                  deaths=deaths[start:], mean_reference_prediction=mean[start:],
                                  pearson_residual=pearson))
    return summary, residuals


def _surveillance_audit_replicate(task):
    key, replay, delay, cases, mean, phi, start, looks, replicate, paths, backend = task
    # Splitting each replay's stream separates the case-conditional NB draws from the
    # Gaussian bank, and the full bank is shared over looks within a replicate.
    if key == 'brazil':
        # The primary and common-calendar audits have disjoint simulation banks.
        root_seed = 2026112118 if looks == replay.benchmark_looks else 2026112117
        seed = np.random.SeedSequence([root_seed, 0, replicate])
    else:
        digest = hashlib.sha256(f'2026111621|{replay.name}|{replicate}'.encode()).digest()
        seed = np.random.SeedSequence(int.from_bytes(digest[:8], 'little'))
    death_seed, bank_seed = seed.spawn(2)
    deaths = np.random.default_rng(death_seed).negative_binomial(phi, phi / (phi + mean))
    bank = np.random.default_rng(bank_seed).standard_normal((paths, len(cases)))
    boundary = GaussianSpendingBoundary(paths, len(looks), REAL_DATA.alpha)
    row = dict(series=replay.name, replicate=replicate, first_alarm_look=None,
               first_alarm_prefix=None, first_alarm_tau=None, unavailable_looks=0,
               fit_failures=0, phi_lower_looks=0, phi_upper_looks=0, risk_boundary_looks=0)
    for number, prefix in enumerate(looks, 1):
        scan = score_scan(cases[:prefix], deaths[:prefix], delay, observation_start=start,
                          candidate_start=replay.index(replay.first_candidate),
                          lag_offset=replay.lag_offset, backend=backend)
        fit = scan.null_fit
        available = bool(fit.success and scan.candidates.size and np.isfinite(scan.statistic))
        row['unavailable_looks'] += int(not available)
        row['fit_failures'] += int(not fit.success)
        row['phi_lower_looks'] += int(fit.phi <= .001 * (1 + 1e-7))
        row['phi_upper_looks'] += int(fit.phi >= 1e8 * (1 - 1e-7))
        row['risk_boundary_looks'] += int(fit.pi <= 1e-10 * (1 + 1e-7)
                                         or fit.pi >= 1 - 1e-10 * (1 + 1e-7))
        # The audit implementation used the same vectorized Python multiplier
        # evaluation for both replay families, even with the native Brazil scan.
        maxima = multiplier_statistics_from_draws(bank[:, :prefix], scan.influence,
                                                  backend='python') if available else None
        step = boundary.advance(maxima, available=available)
        if row['first_alarm_look'] is None and available and scan.statistic > step.boundary:
            row.update(first_alarm_look=number, first_alarm_prefix=prefix,
                       first_alarm_tau=int(scan.tau_hat))
    return row


def run_surveillance_null_audits(data_dir: Path, output_dir: Path, *, replications: int = 1000,
                                 n_paths: int = 10000, workers: int = 1,
                                 backend: str | None = None,
                                 brazil_common_calendar: bool = False) -> pd.DataFrame:
    """Generate independent constant-risk audits conditional on observed case paths."""
    if min(replications, n_paths, workers) < 1:
        raise ValueError('Replication, path and worker counts must be positive.')
    rows, summaries, settings = [], [], {}
    replays = ({'brazil': SURVEILLANCE_REPLAYS['brazil']} if brazil_common_calendar
               else SURVEILLANCE_REPLAYS)
    for key, replay in replays.items():
        frame = _surveillance_frame(data_dir, replay)
        delay = _primary_delay(replay)
        fit, mean, start, end = _audit_reference(frame, replay, delay)
        choices = backend or ('c' if key == 'brazil' else 'python')
        looks = replay.benchmark_looks if brazil_common_calendar else replay.looks
        fixed = (key, replay, delay, frame.cases.to_numpy(float), mean, fit.phi, start, looks)
        tasks = ((*fixed, replicate, n_paths, choices) for replicate in range(replications))
        if workers == 1:
            local = pd.DataFrame(map(_surveillance_audit_replicate, tasks))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                local = pd.DataFrame(pool.map(_surveillance_audit_replicate, tasks, chunksize=4))
        alarms = int(local.first_alarm_look.notna().sum())
        lower, upper = binomial_interval(alarms, replications)
        summaries.append(dict(
            series=replay.name, replications=replications, false_alarms=alarms,
            rate=alarms / replications, exact_ci_low=lower, exact_ci_high=upper,
            planned_looks_evaluated=replications * len(looks),
            **{column: int(local[column].sum()) for column in (
                'unavailable_looks', 'fit_failures', 'phi_lower_looks', 'phi_upper_looks',
                'risk_boundary_looks')},
        ))
        rows.append(local)
        settings[key] = dict(backend=choices, reference_risk=float(fit.pi),
                             reference_shape=float(fit.phi), reference_days=end-start,
                             reference_last=frame.date.iloc[end - 1].date().isoformat(),
                             planned_prefixes=looks)
    result = pd.DataFrame(summaries)
    stem = 'brazil_common_calendar_audit' if brazil_common_calendar else 'null_audit'
    _write_surveillance_outputs(
        data_dir, output_dir, f'{stem}_manifest.json',
        {f'{stem}_summary.csv': result, f'{stem}_replications.csv': pd.concat(rows, ignore_index=True)},
        dict(replications=replications, gaussian_paths_per_replication=n_paths, series=settings),
    )
    return result


def _write_surveillance_outputs(data_dir, output_dir, manifest_name, products, metadata):
    inputs = _surveillance_input_hashes(data_dir)
    # A manifest written last ties a run to its data, source, settings and
    # output bytes. Public downloads that change are visible in the comparison.
    manifest = {
        'analyzed_sha256': inputs, 'matches_manuscript_inputs': inputs == MANUSCRIPT_INPUT_SHA256,
        'source_files': {name: _sha256(Path(__file__).with_name(name).read_bytes())
                         for name in ('real_data.py', 'methods.py', 'config.py')},
        'settings': metadata,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as temporary:
        stage = Path(temporary)
        for name, frame in products.items():
            frame.to_csv(stage / name, index=False, float_format='%.17g')
        manifest['artifacts'] = {name: _sha256((stage / name).read_bytes()) for name in products}
        (stage / manifest_name).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        for name in (*products, manifest_name):
            (stage / name).replace(output_dir / name)


def run_surveillance_benchmarks(data_dir: Path, output_dir: Path, *, n_paths: int = 10000,
                               n_audit: int = 10000, workers: int = 1,
                               backend: str = 'c') -> pd.DataFrame:
    """Replay NB-GLR and Page CUSUM with the manuscript's online calibration."""
    rows, summaries, metadata = [], [], {}
    for index, (key, replay) in enumerate(SURVEILLANCE_REPLAYS.items()):
        frame = _surveillance_frame(data_dir, replay)
        seed = 20261122 + index
        observation = replay.index(replay.outcome_start)
        reference_end = replay.index(replay.benchmark_reference_last) + 1
        delay = _primary_delay(replay)
        chart = OnlineBenchmarkCalibration(
            delay, planned_looks=replay.benchmark_looks, observation_start=observation,
            reference_end=reference_end, lag_offset=replay.lag_offset, n_paths=n_paths,
            n_audit=n_audit, seed=seed, workers=workers, backend=backend,
        )
        dates = frame.date.dt.strftime('%Y-%m-%d').tolist()
        cases, deaths = frame.cases.to_numpy(), frame.deaths.to_numpy()
        local = []
        # The chart receives cumulative counts through each current look.
        # Each new simulated day uses that day's available exposure.
        for prefix in replay.benchmark_looks:
            for result in chart.advance(cases[:prefix], deaths[:prefix]):
                local.append(dict(series=replay.name, series_key=key, **result,
                                  look_date=dates[prefix-1],
                                  estimated_outcome_date=dates[result['outcome_index']]))
        rows.extend(local)
        for method in chart.methods:
            method_rows = [row for row in local if row['method'] == method]
            first = next((row for row in method_rows if row['alarm']), {})
            summaries.append(dict(
                series=replay.name, series_key=key, method=method, alarm=bool(first),
                first_alarm_date=first.get('look_date'),
                estimated_outcome_date=first.get('estimated_outcome_date'),
                direction=first.get('direction'), statistic_at_alarm=first.get('statistic'),
                cutoff_at_alarm=first.get('cutoff'),
                reference_crossings=method_rows[-1]['reference_crossings'],
                audit_crossings=method_rows[-1]['audit_crossings'], audit_replications=n_audit,
                planned_looks=len(method_rows), first_look=method_rows[0]['look_date'],
                reference_start=replay.outcome_start, reference_last=replay.benchmark_reference_last,
                reference_risk=float(chart.reference_fit.pi),
                reference_shape=float(chart.reference_fit.phi),
            ))
        metadata[key] = dict(seed=seed, planned_prefixes=replay.benchmark_looks, reference_end=reference_end,
                             observation_start=observation, delay=delay.tolist())
    result = pd.DataFrame(summaries)
    _write_surveillance_outputs(
        data_dir, output_dir, 'benchmark_manifest.json',
        {'benchmark_summary.csv': result, 'benchmark_trace.csv': pd.DataFrame(rows)},
        dict(backend=backend, calibration_paths=n_paths, audit_paths=n_audit, series=metadata),
    )
    return result


def run_applications(data_dir: Path, output_dir: Path, *, backend: str = 'c',
                     boundary_policy: str = GAUSSIAN_SPENDING_POLICY) -> pd.DataFrame:
    covid = pd.read_csv(data_dir / 'processed' / 'who_covid_processed.csv')
    dengue = pd.read_csv(_processed_dengue_path(data_dir))
    covid_delay = gamma_delay_pmf(14.0, shape=4.0, maximum_lag=45)
    summaries: list[dict[str, Any]] = []
    traces: list[pd.DataFrame] = []
    benchmarks: list[pd.DataFrame] = []
    covid_specs = (
        ('Japan', 'Japan'),
        ('Republic of Korea', 'South Korea'),
        ('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom'),
    )
    for source, display in covid_specs:
        subset = covid[covid['series'] == source].copy()
        dates = pd.to_datetime(subset['date']).reset_index(drop=True)
        observation_start = int(np.flatnonzero(dates >= pd.Timestamp('2020-06-29'))[0])
        candidate_start = int(np.flatnonzero(dates >= pd.Timestamp('2020-07-15'))[0])
        summary, trace, benchmark = monitor_series(
            subset,
            series=display,
            delay=covid_delay,
            observation_start=observation_start,
            candidate_start=candidate_start,
            first_look=candidate_start + 14,
            look_every=7,
            n_multiplier=REAL_DATA.n_multiplier,
            seed=REAL_DATA.master_seed + len(summaries),
            backend=backend,
            boundary_policy=boundary_policy,
        )
        summaries.append(summary)
        traces.append(trace)
        benchmarks.append(benchmark)

    summary, trace, benchmark = monitor_series(
        dengue,
        series='Bangladesh dengue',
        delay=_dengue_delay(),
        observation_start=4,
        candidate_start=12,
        first_look=20,
        look_every=1,
        n_multiplier=REAL_DATA.n_multiplier,
        seed=REAL_DATA.master_seed + len(summaries),
        backend=backend,
        boundary_policy=boundary_policy,
        lag_offset=0,
    )
    summaries.append(summary)
    traces.append(trace)
    benchmarks.append(benchmark)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = [data_dir / 'processed' / 'who_covid_processed.csv', _processed_dengue_path(data_dir)]
    manifest = {
        'status': 'complete', 'boundary_policy': boundary_policy, 'backend': backend,
        'input_files': {path.name: _sha256(path.read_bytes()) for path in input_paths},
        'source_files': {name: _sha256((Path(__file__).parent / name).read_bytes())
                         for name in ('config.py', 'methods.py', 'real_data.py')},
        'series': {row['series']: row.pop('provenance') for row in summaries},
        'interval_scope': 'unadjusted descriptive; no selection-adjusted coverage claim',
    }
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_dir / 'application_summary.csv', index=False)
    pd.concat(traces, ignore_index=True).to_csv(output_dir / 'look_trace.csv', index=False)
    pd.concat(benchmarks, ignore_index=True).to_csv(output_dir / 'benchmark_statistics.csv', index=False)
    manifest['artifacts'] = {name: _sha256((output_dir / name).read_bytes()) for name in
        ('application_summary.csv', 'look_trace.csv', 'benchmark_statistics.csv')}
    temporary = output_dir / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(temporary, output_dir / 'manifest.json')
    return summary_frame


def plot_applications(data_dir: Path, output_dir: Path, summary: pd.DataFrame) -> None:
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'savefig.dpi': 300,
    })
    covid = pd.read_csv(data_dir / 'processed' / 'who_covid_processed.csv')
    dengue = pd.read_csv(_processed_dengue_path(data_dir))
    display = {
        'Japan': 'Japan',
        'Republic of Korea': 'South Korea',
        'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    }
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
    for ax, (source, name) in zip(axes, display.items()):
        part = covid[covid['series'] == source].copy()
        dates = pd.to_datetime(part['date'])
        ax.plot(dates, part['cases'] / max(part['cases'].max(), 1), color='#0072B2', label='Cases (scaled)')
        ax.plot(dates, part['deaths'] / max(part['deaths'].max(), 1), color='#D55E00', label='Deaths (scaled)')
        record = summary[summary['series'] == name].iloc[0]
        if _has_date(record.get('estimated_change_date', '')):
            ax.axvline(pd.Timestamp(record['estimated_change_date']), color='black', linestyle='--', label='Estimated change')
        ax.set_title(name)
        ax.set_ylabel('Scaled count')
        ax.grid(alpha=0.2)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel('Date')
    axes[0].legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / 'Fig_real_data_covid.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'Fig_real_data_covid.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    dates = pd.to_datetime(dengue['date'])
    ax.plot(dates, dengue['cases'] / max(dengue['cases'].max(), 1), color='#0072B2', label='Admissions (scaled)')
    ax.plot(dates, dengue['deaths'] / max(dengue['deaths'].max(), 1), color='#D55E00', label='Deaths (scaled)')
    record = summary[summary['series'] == 'Bangladesh dengue'].iloc[0]
    if _has_date(record.get('estimated_change_date', '')):
        ax.axvline(pd.Timestamp(record['estimated_change_date']), color='black', linestyle='--', label='Estimated change')
    ax.set(xlabel='Week', ylabel='Scaled count', title='Bangladesh dengue, 2023')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / 'Fig_real_data_dengue.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'Fig_real_data_dengue.png', bbox_inches='tight')
    plt.close(fig)


def _plot_covid_surveillance(data_dir, output_dir, summary, trace):
    # Reference-event markers in Figure 7 are descriptive annotations. They
    # play no part in candidate screening, calibration, or interval fitting.
    event_dates = {
        'japan': ('2020-07-22', '2020-08-28'),
        'south_korea': ('2020-08-16', '2020-08-30'),
        'united_kingdom': ('2020-06-16', '2020-07-04'),
    }
    style = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'serif', 'font.size': 13,
             'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.4,
             'figure.dpi': 180}
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.4), sharex='col')
        for column, (key, events) in enumerate(event_dates.items()):
            replay = SURVEILLANCE_REPLAYS[key]
            frame = _surveillance_frame(data_dir, replay)
            for row, (field, color) in enumerate((('cases', '#0072B2'), ('deaths', '#D55E00'))):
                ax = axes[row, column]
                ax.bar(frame.date, frame[field], width=1., color='#D9D9D9', linewidth=0)
                ax.plot(frame.date, frame[field].rolling(7, min_periods=1).mean(), color=color)
                ax.set_ylabel(field.capitalize())
            axes[0, column].set_title(replay.name, fontsize=12, fontweight='bold', y=1.15, pad=8)
            ax = axes[2, column]
            local = trace.loc[trace.series.eq(replay.name) & np.isfinite(trace.evidence_margin)]
            ax.plot(pd.to_datetime(local.look_date), local.evidence_margin, color='#222222')
            ax.axhline(0, color='#D55E00', linestyle='--', linewidth=1)
            record = summary.loc[replay.name]
            if record.alarm:
                alarm = pd.Timestamp(record.alarm_date)
                margin = local.loc[local.look_date.eq(record.alarm_date), 'evidence_margin'].iloc[0]
                for axis in axes[:, column]:
                    axis.axvline(pd.Timestamp(record.estimated_change_date), color='#0072B2',
                                 linestyle='--', linewidth=1)
                ax.plot(alarm, margin, marker='v', color='#D55E00', markersize=5,
                        linestyle='none', zorder=4)
            ax.set_ylabel('Statistic −\nboundary')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
            for row, axis in enumerate(axes[:, column]):
                for event in events:
                    axis.axvline(pd.Timestamp(event), color='#707070', linestyle=':', linewidth=.8)
                label = 'abcdefghi'[3 * row + column]
                axis.text(-.12, 1.04, f'({label})', transform=axis.transAxes,
                          fontweight='bold', fontsize=12)
                axis.tick_params(axis='both', labelsize=11.5, pad=2)
                axis.yaxis.label.set_size(12)
                axis.yaxis.labelpad = 3
                axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7, 10]))
        fig.tight_layout(pad=.8)
        fig.canvas.draw()
        fig.tight_layout(pad=.8)
        for suffix in ('pdf', 'png'):
            fig.savefig(output_dir / f'Fig7_covid_real_data.{suffix}', dpi=600 if suffix == 'png' else None,
                        bbox_inches='tight', facecolor='white')
        plt.close(fig)


def _plot_brazil_surveillance(data_dir, output_dir, trace, sensitivity):
    replay = SURVEILLANCE_REPLAYS['brazil']
    daily = _surveillance_frame(data_dir, replay).set_index('date')
    weekly = daily.loc[replay.outcome_start:replay.horizon].resample('W-SAT').sum()
    primary = trace.loc[trace.series.eq(replay.name)]
    alternative = sensitivity.loc[sensitivity.sensitivity_id.eq('brazil-nishiura-28-14')]
    colors = {'onset': '#0072B2', 'death': '#A7421D', 'alternative': '#087A65'}
    style = {'font.family': 'DejaVu Sans', 'font.size': 9.5, 'pdf.fonttype': 42,
             'axes.spines.top': False, 'axes.spines.right': False}
    with plt.rc_context(style):
        fig, (onsets, deaths, evidence) = plt.subplots(
            3, 1, sharex=True, figsize=(7.2, 6.1), layout='constrained',
            gridspec_kw={'height_ratios': [1, 1, 1.5]},
        )
        fig.suptitle('Brazil seasonal A(H1N1)pdm09, 2018', x=.11, ha='left', fontweight='bold')
        onsets.bar(weekly.index, weekly.cases, width=5, color=colors['onset'])
        onsets.set_ylabel('Weekly cases\nby onset date')
        deaths.bar(weekly.index, weekly.deaths, width=5, color=colors['death'])
        deaths.set_ylabel('Weekly deaths\nby occurrence date')
        alarms = {}
        for local, label, linestyle, color in (
            (primary, 'Ejima delay', '-', colors['onset']),
            (alternative, 'Nishiura delay', '--', colors['alternative']),
        ):
            margin = local.evidence_margin.where(np.isfinite(local.evidence_margin))
            evidence.plot(pd.to_datetime(local.look_date), margin, label=label,
                          linestyle=linestyle, color=color, linewidth=1.65)
            if local.crossed.any():
                row = local.loc[local.crossed].iloc[0]
                alarms[label] = row
                evidence.scatter(pd.Timestamp(row.look_date), row.evidence_margin, color=color, s=17, zorder=6)
        evidence.axhline(0, color='#363636', linewidth=.9, linestyle=(0, (2, 2)))
        evidence.set_ylabel('OCFR statistic\nminus boundary')

        markers = [(pd.Timestamp(primary.look_date.iloc[0]), 'First look', ':', '#676767', 'left')]
        primary_alarm = alarms.get('Ejima delay')
        alternative_alarm = alarms.get('Nishiura delay')
        shared_alarm = (primary_alarm is not None and alternative_alarm is not None
                        and primary_alarm.look_date == alternative_alarm.look_date)
        if primary_alarm is not None:
            onset = pd.Timestamp(primary_alarm.estimated_change_date)
            markers.append((onset, 'Selected onset*' if shared_alarm else 'Primary selected onset',
                            '--', '#753FB1', 'right'))
        if shared_alarm:
            alarm = pd.Timestamp(primary_alarm.look_date)
            markers.append((alarm, 'First alarm', '-', '#262626', 'left'))
            alternative_onset = pd.Timestamp(alternative_alarm.estimated_change_date)
            evidence.set_xlabel(f'*The sensitivity delay selected {alternative_onset.day} {alternative_onset:%b}. '
                                f'Both delays first crossed on {alarm.day} {alarm:%b}.')
        else:
            for label, row in alarms.items():
                markers.append((pd.Timestamp(row.look_date), label.split()[0] + ' first alarm',
                                '-', '#262626', 'left'))
        for day, label, linestyle, color, anchor in markers:
            for ax in fig.axes:
                ax.axvline(day, color=color, linewidth=.95, linestyle=linestyle, zorder=2)
            onsets.annotate(f'{label}\n{day.day} {day:%b}', xy=(day, 1),
                            xycoords=('data', 'axes fraction'),
                            xytext=(-4 if anchor == 'right' else 4, 2), textcoords='offset points',
                            ha=anchor, va='bottom', fontsize=8.8, color=color, annotation_clip=False)
        evidence.legend(loc='upper left', ncols=2, framealpha=1, edgecolor='none')
        evidence.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
        evidence.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        evidence.set_xlim(pd.Timestamp(replay.outcome_start) - pd.Timedelta(days=3),
                          pd.Timestamp(replay.horizon) + pd.Timedelta(days=2))
        fig.savefig(output_dir / 'Fig8_brazil_seasonal_2018.pdf')
        fig.savefig(output_dir / 'Fig8_brazil_seasonal_2018.png', dpi=180)
        plt.close(fig)


def plot_surveillance(data_dir: Path, output_dir: Path) -> None:
    """Reconstruct Figures 7 and 8 from the new run's aggregate rows and traces."""
    manifest_path = output_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'{manifest_path} is missing. Run the manuscript replay first.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name, expected in manifest['analyzed_sha256'].items():
        if _manuscript_input_hash(data_dir / 'processed' / name) != expected:
            raise ValueError(f'{name} changed after the replay. Run the manuscript replay again.')
    for name in ('application_summary.csv', 'look_trace.csv', 'sensitivity_trace.csv'):
        if _sha256((output_dir / name).read_bytes()) != manifest['artifacts'][name]:
            raise ValueError(f'{name} changed after the replay. Run the manuscript replay again.')
    summary = pd.read_csv(output_dir / 'application_summary.csv').set_index('series')
    trace = pd.read_csv(output_dir / 'look_trace.csv', float_precision='round_trip')
    sensitivity = pd.read_csv(output_dir / 'sensitivity_trace.csv', float_precision='round_trip')
    _plot_covid_surveillance(data_dir, output_dir, summary, trace)
    _plot_brazil_surveillance(data_dir, output_dir, trace, sensitivity)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).parent
    parser.add_argument('--stage', choices=('all', 'fetch', 'run', 'plot', 'benchmarks', 'audit', 'clinical-delay'),
                        default='all', help='all prepares missing inputs, runs OCFR and plots')
    parser.add_argument('--application', choices=('manuscript', 'legacy-dengue'), default='manuscript',
                        help='use the manuscript four-series protocol or the older dengue example')
    parser.add_argument('--data', type=Path, default=base / 'data')
    parser.add_argument('--output', type=Path, help='directory for generated replay tables and plots')
    parser.add_argument('--backend', choices=('c', 'python', 'auto'),
                        help='defaults to Python for manuscript replays and native C for online benchmarks')
    parser.add_argument('--boundary-policy', choices=(GAUSSIAN_SPENDING_POLICY, LEGACY_GAUSSIAN_POLICY),
                        default=GAUSSIAN_SPENDING_POLICY)
    parser.add_argument('--refresh', action='store_true', help='download fresh raw inputs for all or fetch')
    parser.add_argument('--require-manuscript-inputs', action='store_true',
                        help='reject processed counts that differ from the manuscript snapshot')
    parser.add_argument('--workers', type=int, default=1,
                        help='parallel workers for the online benchmarks or null audit')
    parser.add_argument('--benchmark-paths', type=int, default=10000)
    parser.add_argument('--benchmark-audit-paths', type=int, default=10000)
    parser.add_argument('--null-replications', type=int, default=1000)
    parser.add_argument('--null-paths', type=int, default=10000)
    parser.add_argument('--brazil-common-calendar', action='store_true',
                        help='use the 39-look benchmark calendar for the Brazil audit only')
    args = parser.parse_args(argv)
    if args.brazil_common_calendar and (args.stage != 'audit' or args.application != 'manuscript'):
        parser.error('--brazil-common-calendar requires the manuscript audit stage')
    data_dir = args.data.resolve()
    default_output = 'real_data' if args.application == 'manuscript' else 'real_data_legacy_dengue'
    output_dir = (args.output or base / 'outputs' / default_output).resolve()
    if args.application == 'manuscript':
        cached = all((data_dir / 'processed' / name).is_file() for name in MANUSCRIPT_INPUT_SHA256)
        if args.stage == 'fetch' or (args.stage == 'all' and (args.refresh or not cached)):
            print(json.dumps(prepare_surveillance(data_dir, refresh=args.refresh), indent=2))
        try:
            input_hashes = _surveillance_input_hashes(data_dir)
        except FileNotFoundError as exc:
            parser.error(f'processed input missing: {exc.filename}. Run the fetch stage first.')
        if input_hashes != MANUSCRIPT_INPUT_SHA256:
            changed = ', '.join(name for name, digest in input_hashes.items()
                                if digest != MANUSCRIPT_INPUT_SHA256[name])
            message = f'Input differs from the manuscript snapshot: {changed}.'
            if args.require_manuscript_inputs:
                parser.error(message)
            print(f'Warning: {message} Results will use the supplied counts.', file=sys.stderr)
        else:
            print('Processed input hashes match the manuscript snapshot.', flush=True)
        if args.stage in {'all', 'run'}:
            result = run_surveillance(data_dir, output_dir, backend=args.backend or 'python',
                                      boundary_policy=args.boundary_policy)
            print(result.to_string(index=False))
        if args.stage in {'all', 'plot'}:
            plot_surveillance(data_dir, output_dir)
        if args.stage == 'benchmarks':
            if args.backend == 'auto':
                parser.error('choose c or python for the online benchmark backend')
            result = run_surveillance_benchmarks(
                data_dir, output_dir, n_paths=args.benchmark_paths, n_audit=args.benchmark_audit_paths,
                workers=args.workers, backend=args.backend or 'c',
            )
            print(result.to_string(index=False))
        if args.stage == 'audit':
            result = run_surveillance_null_audits(
                data_dir, output_dir, replications=args.null_replications,
                n_paths=args.null_paths, workers=args.workers, backend=args.backend,
                brazil_common_calendar=args.brazil_common_calendar,
            )
            print(result.to_string(index=False))
        if args.stage == 'clinical-delay':
            result = run_brazil_clinical_delay(data_dir, output_dir, backend=args.backend or 'python',
                                              boundary_policy=args.boundary_policy)
            print(result.to_string(index=False))
        return 0
    if args.require_manuscript_inputs:
        parser.error('input fingerprint verification uses the manuscript series')
    if args.stage in {'benchmarks', 'audit', 'clinical-delay'}:
        parser.error(f'{args.stage} uses the manuscript series')
    if args.stage in {'all', 'fetch'}:
        print(json.dumps(prepare_all(data_dir, refresh=args.refresh), indent=2))
    summary_path = output_dir / 'application_summary.csv'
    if args.stage in {'all', 'run'}:
        summary = run_applications(data_dir, output_dir, backend=args.backend or 'c',
                                   boundary_policy=args.boundary_policy)
    elif summary_path.exists():
        summary = pd.read_csv(summary_path, keep_default_na=False, na_values=[''])
    else:
        summary = pd.DataFrame()
    if args.stage in {'all', 'plot'}:
        if summary.empty:
            raise FileNotFoundError('run the application stage before plotting')
        plot_applications(data_dir, output_dir, summary)
    if not summary.empty:
        print(summary.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
