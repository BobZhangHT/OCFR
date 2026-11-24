"""Tests for the released daily surveillance data protocol."""

import pandas as pd
import pytest

import real_data
from real_data import SURVEILLANCE_REPLAYS, _surveillance_frame, brazil_delay, prepare_brazil


def test_sinan_uses_separate_onset_and_death_dates(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    columns = 'DT_SIN_PRI;DT_OBITO;DT_NOTIFIC;DT_ENCERRA;CLASSI_FIN;CRITERIO;HOSPITAL;EVOLUCAO\n'
    (raw / 'INFLUD09.csv').write_text(
        columns
        + '18/07/2009;20/07/2009;18/07/2009;20/07/2009;1;1;1;2\n'
        + '19/07/2009;;16/08/2009;;1;1;1;2\n'
        + '20/07/2009;19/07/2009;20/07/2009;;1;1;1;2\n'
        + '21/07/2009;21/07/2009;21/07/2009;;1;1;0;2\n',
        encoding='latin1',
    )
    (raw / 'INFLUD10.csv').write_text(
        columns + '02/01/2010;02/01/2010;;;1;1;1;3\n', encoding='latin1',
    )
    audit = prepare_brazil(tmp_path)
    frame = _surveillance_frame(tmp_path, SURVEILLANCE_REPLAYS['brazil']).set_index('date')
    assert len(frame) == 213
    assert tuple(frame.loc['2009-07-20']) == (0, 1)
    assert tuple(frame.loc['2010-01-02']) == (1, 0)
    assert audit['negative_date_order_exclusions'] == 1
    assert audit['deaths_without_date'] == 1
    assert audit['notification_at_first_look']['first_notified_after_alarm'] == 1
    assert audit['inputs']['INFLUD10.csv']['eligible_records'] == 1


def test_manuscript_calendars_and_zero_lag_influenza():
    who = SURVEILLANCE_REPLAYS['japan']
    brazil = SURVEILLANCE_REPLAYS['brazil']
    assert who.looks == (*range(75, 244, 7), 245)
    assert brazil.looks == tuple(range(73, 214, 7))
    assert (who.index(who.first_candidate), brazil.index(brazil.first_candidate)) == (61, 59)
    for law in ('ejima', 'nishiura'):
        mass = brazil_delay(law)
        assert mass.shape == (46,)
        assert mass[0] > 0
        assert mass.sum() == pytest.approx(1.)


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

    path = processed / 'brazil_influenza_daily.csv'
    path.write_bytes(b'changed counts\n')
    with pytest.raises(SystemExit, match='2'):
        real_data.main(args)
    assert stages == ['run', 'plot']
