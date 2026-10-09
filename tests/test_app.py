"""End-to-end UI behavior via Streamlit's AppTest (headless, no network)."""
import pytest
from streamlit.testing.v1 import AppTest

from modules.riot_api import RiotInterface

LEGAL_NOTICE = ("NexusNode isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games "
                "or anyone officially involved in producing or managing Riot Games properties. Riot Games, and "
                "all associated properties are trademarks or registered trademarks of Riot Games, Inc.")


def _run_app():
    import streamlit as st
    # AppTest can't drive segmented_control; a radio exposes the same choice.
    st.segmented_control = (lambda label, options, default=None, key=None, label_visibility=None,
                            format_func=str: st.radio(label, options,
                                                      index=options.index(default) if default else None,
                                                      key=key, format_func=format_func))
    exec(open('app.py', encoding='utf-8').read(), {'__name__': '__main__'})


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv('RIOT_KEY', 'test-placeholder-not-a-real-key')
    return AppTest.from_function(_run_app, default_timeout=60).run()


def lock_buttons(at):
    return [b for b in at.button if b.key and b.key.startswith('lock_')]


def test_renders_without_errors(app):
    assert not app.exception
    assert not app.error


def test_shows_exact_legal_notice(app):
    assert any(c.value == LEGAL_NOTICE for c in app.caption)


def test_blind_then_counter_mode(app):
    assert any('Blind pick' in i.value for i in app.info)
    app.selectbox(key='red_BOTTOM').select('Caitlyn').run()
    assert any('Counter pick' in i.value for i in app.info)


def test_lock_in_fills_your_slot(app):
    first = lock_buttons(app)[0]
    champ = first.key.removeprefix('lock_')
    first.click().run()
    assert app.session_state['blue_BOTTOM'] == champ


def test_bans_are_not_recommended(app):
    top_two = [b.key.removeprefix('lock_') for b in lock_buttons(app)[:2]]
    ms = app.multiselect(key='bans')
    for champ in top_two:
        ms = ms.select(champ)
    ms.run()
    shown = {b.key.removeprefix('lock_') for b in lock_buttons(app)}
    assert not shown & set(top_two)


def test_duplicate_and_banned_pick_warnings(app):
    app.selectbox(key='blue_TOP').select('Ornn')
    app.selectbox(key='red_TOP').select('Ornn')
    app.run()
    assert any('only be picked once' in w.value for w in app.warning)


def test_reset_clears_draft(app):
    app.selectbox(key='blue_TOP').select('Ornn').run()
    [b for b in app.button if 'Reset' in b.label][0].click().run()
    assert app.session_state['blue_TOP'] is None


# --- Riot ID import safeguards (Riot API is faked: no network, no key usage) --
def _import(app, riot_id):
    app.text_input(key='riot_id').input(riot_id)
    [b for b in app.button if 'Import' in b.label][0].click().run()
    return [m.value for m in list(app.error) + list(app.warning) + list(app.success)]


def test_invalid_riot_id_rejected_before_any_api_call(app, monkeypatch):
    calls = []
    monkeypatch.setattr(RiotInterface, 'get_puuid', lambda self, n, t: calls.append((n, t)))
    for bad in ['no-hash', 'ab#NA1', 'name#TOOLONGTAG', 'name#n@1']:
        msgs = _import(app, bad)
        assert any('Name#Tag' in m for m in msgs), bad
    assert calls == []


def test_unknown_account_and_cooldown(app, monkeypatch):
    monkeypatch.setattr(RiotInterface, 'get_puuid', lambda self, n, t: None)
    assert any('No account found' in m for m in _import(app, 'Someone#ZZ9Q'))
    assert any('Please wait' in m for m in _import(app, 'Someone#ZZ9Q'))


def test_internal_errors_are_not_shown_to_users(app, monkeypatch):
    def boom(self, n, t):
        raise RuntimeError('secret internal detail https://internal/xyz')
    monkeypatch.setattr(RiotInterface, 'get_puuid', boom)
    msgs = _import(app, 'Another#QQ1Z')
    assert any("Couldn't reach Riot" in m for m in msgs)
    assert not any('secret internal detail' in m for m in msgs)


# --- Desktop mode: League client sync (client is faked) ----------------------
class _FakeClient:
    def __init__(self, session):
        self.session = session

    def champ_select_session(self):
        return self.session


@pytest.fixture
def desktop_app(monkeypatch):
    from modules.lcu import LCUClient
    from tests.test_lcu import SESSION
    monkeypatch.setenv('NEXUSNODE_DESKTOP', '1')
    monkeypatch.setattr(LCUClient, 'from_lockfile', classmethod(lambda cls, path=None: _FakeClient(SESSION)))
    return AppTest.from_function(_run_app, default_timeout=60).run()


def test_league_client_sync_hidden_on_web(app):
    assert not any('Sync with champion select' in t.label for t in app.toggle)


def test_league_client_sync_fills_draft(desktop_app):
    at = desktop_app
    at.toggle(key='lcu_on').set_value(True).run()
    ss = at.session_state
    assert ss['user_role'] == 'BOTTOM'
    allies = (ss['blue_TOP'], ss['blue_JUNGLE'], ss['blue_SUPPORT'], ss['blue_BOTTOM'])
    assert allies == ('Aatrox', 'LeeSin', 'Thresh', 'Ashe')
    assert ss['blue_MIDDLE'] is None
    assert ss['red_TOP'] == 'Malphite' and ss['red_BOTTOM'] == 'Ezreal' and ss['red_SUPPORT'] == 'Lulu'
    assert set(ss['bans']) == {'Ahri', 'MasterYi', 'Zed'}
    assert not at.exception
    assert any('Live' in c.value for c in at.caption)


def test_league_client_not_running(monkeypatch):
    from modules.lcu import LCUClient
    monkeypatch.setenv('NEXUSNODE_DESKTOP', '1')
    monkeypatch.setattr(LCUClient, 'from_lockfile', classmethod(lambda cls, path=None: None))
    at = AppTest.from_function(_run_app, default_timeout=60).run()
    at.toggle(key='lcu_on').set_value(True).run()
    assert any("isn't running" in c.value for c in at.caption)
    assert not at.exception
