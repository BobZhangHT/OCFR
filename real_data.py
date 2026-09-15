"""Fetch, preprocess, analyze, and plot the OCFR real-data applications."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma

from config import REAL_DATA, RealDataConfig
from methods import (
    exposure_components,
    fit_nb_identity,
    fixed_tau_wald_intervals,
    gamma_delay_pmf,
    make_sampler,
    monte_carlo_critical_value,
    multiplier_statistics_from_draws,
    published_benchmark_statistics,
    score_scan,
)


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


def prepare_covid(data_dir: Path, config: RealDataConfig, *, refresh: bool) -> dict[str, Any]:
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
    edges = 7.0 * np.arange(maximum_weeks + 1, dtype=float)
    values = np.diff(gamma.cdf(edges, a=2.0, scale=mean_days / 2.0))
    return values / values.sum()


def _two_rate_intervals(
    cases: np.ndarray,
    deaths: np.ndarray,
    delay: np.ndarray,
    tau: int,
    observation_start: int,
) -> dict[str, float | str]:
    total, post = exposure_components(cases, delay)
    post_exposure = post[:, tau - 1]
    design = np.column_stack((total - post_exposure, post_exposure))
    active = np.arange(cases.size) >= observation_start
    fit = fit_nb_identity(
        deaths[active], np.zeros(int(active.sum()), dtype=float), design[active]
    )
    result: dict[str, float | str] = {
        'pre_cfr': float(fit.rates[0]),
        'post_cfr': float(fit.rates[1]),
        'risk_ratio': float(fit.rates[1] / fit.rates[0]),
        'interval_status': 'not_available',
    }
    try:
        intervals = fixed_tau_wald_intervals(
            fit, np.zeros(int(active.sum()), dtype=float), design[active]
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
        'interval_status': 'success:fixed_tau_descriptive',
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
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    frame = frame.sort_values('date').reset_index(drop=True)
    cases = frame['cases'].to_numpy(float)
    deaths = frame['deaths'].to_numpy(float)
    dates = pd.to_datetime(frame['date'])
    looks = list(range(first_look, len(frame) + 1, look_every))
    if not looks or looks[-1] != len(frame):
        looks.append(len(frame))
    multipliers = make_sampler(seed, backend=backend).standard_normal((n_multiplier, len(frame)))
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
        )
        threshold = np.inf
        pvalue = np.nan
        if scan.candidates.size and np.isfinite(scan.statistic):
            maxima = multiplier_statistics_from_draws(
                multipliers[:, :prefix], scan.influence, backend=backend
            )
            threshold = monte_carlo_critical_value(maxima, local_alpha)
            pvalue = (1 + np.count_nonzero(maxima >= scan.statistic)) / (n_multiplier + 1)
        crossed = bool(np.isfinite(scan.statistic) and scan.statistic > threshold)
        row = {
            'series': series,
            'look_number': look_number,
            'prefix': prefix,
            'look_date': dates.iloc[prefix - 1].date().isoformat(),
            'statistic': scan.statistic,
            'critical_value': threshold,
            'multiplier_pvalue': pvalue,
            'episode_adjusted_pvalue': min(1.0, len(looks) * pvalue) if np.isfinite(pvalue) else np.nan,
            'tau_hat': scan.tau_hat,
            'estimated_change_date': (
                dates.iloc[int(scan.tau_hat)].date().isoformat()
                if scan.tau_hat is not None else ''
            ),
            'direction': int(scan.direction),
            'candidate_count': int(scan.candidates.size),
            'crossed': crossed,
            'fit_status': scan.null_fit.status,
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
        'episode_adjusted_pvalue': np.nan if alarm is None else alarm['episode_adjusted_pvalue'],
    }
    if alarm is not None and alarm_scan is not None:
        summary.update(
            _two_rate_intervals(
                cases[: int(alarm['prefix'])],
                deaths[: int(alarm['prefix'])],
                delay,
                int(alarm_scan.tau_hat),
                observation_start,
            )
        )

    benchmark = published_benchmark_statistics(
        cases, deaths, delay,
        reference_end=max(observation_start + 14, min(28, len(frame) - 2)),
        observation_start=observation_start,
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


def run_applications(data_dir: Path, output_dir: Path, *, backend: str = 'c') -> pd.DataFrame:
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
    )
    summaries.append(summary)
    traces.append(trace)
    benchmarks.append(benchmark)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_dir / 'application_summary.csv', index=False)
    pd.concat(traces, ignore_index=True).to_csv(output_dir / 'look_trace.csv', index=False)
    pd.concat(benchmarks, ignore_index=True).to_csv(output_dir / 'benchmark_statistics.csv', index=False)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).parent
    parser.add_argument('--stage', choices=('all', 'fetch', 'run', 'plot'), default='all')
    parser.add_argument('--data', type=Path, default=base / 'data')
    parser.add_argument('--output', type=Path, default=base / 'outputs' / 'real_data')
    parser.add_argument('--backend', choices=('c', 'python', 'auto'), default='c',
                        help='strict C backend by default; python/auto are portability options')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args(argv)
    data_dir = args.data.resolve()
    output_dir = args.output.resolve()
    if args.stage in {'all', 'fetch'}:
        print(json.dumps(prepare_all(data_dir, refresh=args.refresh), indent=2))
    summary_path = output_dir / 'application_summary.csv'
    if args.stage in {'all', 'run'}:
        summary = run_applications(data_dir, output_dir, backend=args.backend)
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
