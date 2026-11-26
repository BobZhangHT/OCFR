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


def test_srag_uses_subtype_and_separate_onset_and_death_dates(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    columns = 'DT_SIN_PRI;DT_OBITO;DT_NOTIFIC;DT_NASC;CS_SEXO;CLASSI_FIN;CRITERIO;HOSPITAL;RES_FLUASU;EVOLUCAO\n'
    (raw / 'INFLUD17.csv').write_text(
        columns + '30/12/2017;02/01/2018;30/12/2017;01/01/1980;F;1;1;1;1;2\n',
        encoding='latin1',
    )
    (raw / 'INFLUD18.csv').write_text(
        columns
        + '31/12/2017;;28/01/2018;;F;1;1;1;1;2\n'
        + '02/01/2018;01/01/2018;02/01/2018;;F;1;1;1;1;2\n'
        + '03/01/2018;03/01/2018;03/01/2018;;F;1;1;1;2;2\n'
        + '04/01/2018;04/01/2018;04/01/2018;;F;1;1;0;1;2\n',
        encoding='latin1',
    )
    (raw / 'INFLUD19-23-03-2026.csv').write_text(
        columns.replace('RES_FLUASU', 'PCR_FLUASU').replace('DT_OBITO', 'DT_EVOLUCA')
        + '2018-12-29;2018-12-29;2018-12-29;;M;1.0;1.0;1.0;1.0;3.0\n'
        + '2018-12-30;2018-12-31;2018-12-30;;M;1.0;1.0;1.0;1.0;2.0\n',
        encoding='latin1',
    )
    audit = prepare_brazil(tmp_path)
    frame = _surveillance_frame(tmp_path, SURVEILLANCE_REPLAYS['brazil']).set_index('date')
    assert len(frame) == 409
    assert tuple(frame.loc['2018-01-02']) == (0, 1)
    assert tuple(frame.loc['2018-12-29']) == (1, 0)
    assert frame.cases.sum() == 3
    assert frame.deaths.sum() == 1
    assert audit['negative_date_order_exclusions'] == 1
    assert audit['deaths_without_date'] == 1
    assert audit['notification_at_planned_looks']['2018-01-27']['notified_after_look'] == 1
    assert audit['inputs']['INFLUD19-23-03-2026.csv']['eligible_records'] == 1


def test_manuscript_calendars_and_zero_lag_influenza():
    who = SURVEILLANCE_REPLAYS['japan']
    brazil = SURVEILLANCE_REPLAYS['brazil']
    assert who.looks == (*range(75, 244, 7), 245)
    assert brazil.looks == tuple(range(73, 410, 7))
    assert who.benchmark_looks == who.looks
    assert brazil.benchmark_looks == tuple(range(143, 410, 7))
    assert real_data._audit_reference_end(who) == who.looks[0] - 1
    assert real_data._audit_reference_end(brazil) == 136
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
