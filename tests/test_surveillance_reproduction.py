"""Tests for the released surveillance protocol and prefix-only benchmarks."""

import numpy as np
import pandas as pd
import pytest

import real_data
from methods import fast_available
from real_data import (
    SURVEILLANCE_REPLAYS,
    OnlineBenchmarkCalibration,
    _surveillance_frame,
    brazil_delay,
    prepare_brazil,
)


def test_srag_uses_pinned_parquet_and_separate_onset_and_death_dates(tmp_path, monkeypatch):
    columns = ['CLASSI_FIN', 'CRITERIO', 'HOSPITAL', 'PCR_FLUASU', 'DT_SIN_PRI',
               'DT_EVOLUCA', 'EVOLUCAO', 'DT_NOTIFIC', 'DT_NASC', 'CS_SEXO',
               'SG_UF', 'NU_IDADE_N', 'TP_IDADE']
    def records(rows):
        return pd.DataFrame(rows, columns=columns)
    releases = {
        b'23': records([['1', '1', '1', '1', '2023-06-01', '2023-06-11', '2', '2023-06-03',
                          '1980-01-01', 'F', 'SP', 43, '3']]),
        b'24': records([['1', '1', '1', '1', '2023-12-31', None, '2', '2024-05-28',
                          '1980-01-01', 'F', 'SP', 43, '3'],
                         ['1', '1', '1', '1', '2024-01-02', '2024-01-03', '2', '2024-01-02',
                          '1980-01-01', 'F', 'SP', 43, '3']]),
        b'25': records([['1', '1', '1', '1', '2024-12-28', '2024-12-29', '2', '2024-12-28',
                          '1980-01-01', 'M', 'RS', 70, '3']]),
    }
    tokens = {2023: b'23', 2024: b'24', 2025: b'25'}
    hashes = {year: real_data._sha256(token) for year, token in tokens.items()}
    monkeypatch.setattr(real_data, 'BRAZIL_SOURCES', {year: ('test', digest) for year, digest in hashes.items()})
    monkeypatch.setattr(real_data, '_cached_download', lambda url, path, refresh=False: tokens[int(url.split('/SRAG/')[1][:4])])
    monkeypatch.setattr(real_data.pd, 'read_parquet', lambda source, columns: releases[source.read()])
    audit = prepare_brazil(tmp_path)
    frame = _surveillance_frame(tmp_path, SURVEILLANCE_REPLAYS['brazil']).set_index('date')
    assert len(frame) == 409
    assert tuple(frame.loc['2024-01-02']) == (1, 0)
    assert tuple(frame.loc['2024-01-03']) == (0, 1)
    assert tuple(frame.loc['2024-12-28']) == (1, 0)
    assert frame.cases.sum() == 3
    assert frame.deaths.sum() == 1
    assert audit['negative_date_order_exclusions'] == 0
    assert audit['deaths_without_date'] == 1
    assert audit['notification_at_planned_looks']['2024-04-06']['notified_after_look'] == 1
    assert audit['inputs']['2025']['eligible_records'] == 1
    assert (tmp_path / 'processed' / 'brazil_2023_empirical_delay.csv').is_file()


def test_manuscript_calendars_and_zero_lag_influenza():
    who = SURVEILLANCE_REPLAYS['japan']
    brazil = SURVEILLANCE_REPLAYS['brazil']
    assert who.looks == (*range(75, 244, 7), 245)
    assert brazil.looks == tuple(range(143, 410, 7))
    assert who.benchmark_looks == who.looks
    assert brazil.benchmark_looks == brazil.looks
    assert real_data._audit_reference_end(who) == who.looks[0] - 1
    assert real_data._audit_reference_end(brazil) == 142
    assert (who.index(who.first_candidate), brazil.index(brazil.first_candidate)) == (61, 59)
    for law in ('ejima', 'nishiura'):
        mass = brazil_delay(law)
        assert mass.shape == (46,)
        assert mass[0] > 0
        assert mass.sum() == pytest.approx(1.)


def test_null_audit_allows_structural_zero_exposure():
    replay = SURVEILLANCE_REPLAYS['brazil']
    dates = pd.date_range(replay.history_start, replay.horizon)
    start = replay.index(replay.outcome_start)
    cases = np.r_[np.zeros(start + 10), np.full(len(dates) - start - 10, 12)]
    deaths = np.r_[np.zeros(start + 10), np.tile([1, 2, 3], 119)[:len(dates) - start - 10]]
    frame = pd.DataFrame(dict(date=dates, cases=cases, deaths=deaths))
    fit, mean, _, _ = real_data._audit_reference(frame, replay, [1.])
    assert fit.success
    assert np.all(mean[:start + 10] == 0)
    assert np.all(mean[start + 10:] > 0)
    _, residuals = real_data._surveillance_residuals(frame, replay, [1.])
    assert residuals.pearson_residual.isna().sum() == 10
    frame.loc[start, 'deaths'] = 1
    with pytest.raises(RuntimeError, match='exposure for observed deaths'):
        real_data._audit_reference(frame, replay, [1.])


def test_two_rate_intervals_omit_structural_zero_information():
    cases = np.r_[np.zeros(10), np.full(60, 20.)]
    deaths = np.r_[np.zeros(10), np.tile([1, 2, 3], 10), np.tile([3, 4, 5], 10)]
    result = real_data._two_rate_intervals(cases, deaths, [1.], 40, 0, lag_offset=0)
    assert result['interval_status'] == 'success:regularized_sandwich_approximate'
    assert result['risk_ratio_lower'] < result['risk_ratio'] < result['risk_ratio_upper']


def test_exact_cached_cli_replays_offline_and_rejects_changed_inputs(tmp_path, monkeypatch):
    processed = tmp_path / 'processed'
    processed.mkdir()
    for name in real_data.MANUSCRIPT_INPUT_SHA256:
        (processed / name).write_bytes(b'frozen counts\r\n')
    pd.DataFrame({'day': np.arange(46), 'fatal_records': [1] * 46,
                  'probability': [1 / 46] * 46}).to_csv(
        processed / 'brazil_2023_empirical_delay.csv', index=False)
    monkeypatch.setattr(real_data, 'MANUSCRIPT_INPUT_SHA256',
                        real_data._surveillance_input_hashes(tmp_path))

    def no_download(*args, **kwargs):
        pytest.fail('cached processed inputs should bypass the fetch stage')

    stages = []
    monkeypatch.setattr(real_data, 'prepare_surveillance', no_download)
    monkeypatch.setattr(real_data, 'run_surveillance',
                        lambda *args, **kwargs: stages.append('run') or pd.DataFrame())
    monkeypatch.setattr(real_data, 'plot_surveillance', lambda *args: stages.append('plot'))
    args = ['--data', str(tmp_path), '--output', str(tmp_path / 'out'), '--require-manuscript-inputs']
    assert real_data.main(args) == 0
    assert stages == ['run', 'plot']

    path = processed / real_data.BRAZIL_DAILY_FILE
    path.write_bytes(b'changed counts\n')
    with pytest.raises(SystemExit, match='2'):
        real_data.main(args)
    assert stages == ['run', 'plot']


@pytest.mark.parametrize('backend', ['python', 'c'])
def test_online_benchmarks_use_only_the_current_prefix(backend):
    if backend == 'c' and not fast_available():
        pytest.skip('native kernel is unavailable')
    cases = np.rint(np.linspace(80, 260, 30))
    deaths = np.random.default_rng(87).poisson(0.04 * cases)
    future_cases, future_deaths = cases.copy(), deaths.copy()
    future_cases[22:] += 2500
    future_deaths[22:] += 75
    config = dict(planned_looks=(15, 22, 30), observation_start=3,
                  reference_end=11, lag_offset=0, n_paths=89, n_audit=33, seed=5)
    delay = [0.3, 0.4, 0.3]
    detector = OnlineBenchmarkCalibration(delay, **config, backend=backend)
    reference_detector = OnlineBenchmarkCalibration(delay, **config, backend='python')
    altered = OnlineBenchmarkCalibration(delay, **config, backend=backend, workers=3)

    with pytest.raises(ValueError, match='next planned look'):
        detector.advance(cases, deaths)

    for prefix in config['planned_looks']:
        current = detector.advance(cases[:prefix], deaths[:prefix])
        reference = reference_detector.advance(cases[:prefix], deaths[:prefix])
        changed = altered.advance(future_cases[:prefix], future_deaths[:prefix])
        for actual, expected in zip(current, reference):
            for key in actual:
                if key in ('statistic', 'cutoff'):
                    np.testing.assert_allclose(actual[key], expected[key], atol=2e-8, rtol=2e-9)
                else:
                    assert actual[key] == expected[key]
        assert (current == changed) == (prefix <= 22)
