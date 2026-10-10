"""Safety and correctness of the read-only League Client reader."""
import ssl

import pytest

from modules import lcu

NAMES = {266: 'Aatrox', 103: 'Ahri', 22: 'Ashe', 412: 'Thresh', 64: 'LeeSin', 157: 'Yasuo',
         81: 'Ezreal', 117: 'Lulu', 238: 'Zed', 54: 'Malphite', 11: 'MasterYi'}

# Shaped like a real LCU /lol-champ-select/v1/session payload, including the
# identity fields the parser must never surface.
SESSION = {
    'localPlayerCellId': 3,
    'myTeam': [
        {'cellId': 0, 'championId': 266, 'assignedPosition': 'top', 'gameName': 'AllyOne', 'tagLine': 'NA1',
         'puuid': 'puuid-ally-0', 'summonerId': 1111},
        {'cellId': 1, 'championId': 64, 'assignedPosition': 'jungle', 'gameName': 'AllyTwo', 'puuid': 'puuid-ally-1'},
        {'cellId': 2, 'championId': 0, 'assignedPosition': 'middle', 'gameName': 'AllyThree', 'puuid': 'puuid-ally-2'},
        {'cellId': 3, 'championId': 22, 'assignedPosition': 'bottom', 'gameName': 'Me', 'puuid': 'puuid-me'},
        {'cellId': 4, 'championId': 412, 'assignedPosition': 'utility', 'gameName': 'AllyFour', 'puuid': 'puuid-ally-4'},
    ],
    'theirTeam': [
        {'cellId': 5, 'championId': 54, 'assignedPosition': '', 'gameName': '', 'puuid': 'puuid-enemy-5'},
        {'cellId': 6, 'championId': 157, 'assignedPosition': '', 'puuid': 'puuid-enemy-6'},
        {'cellId': 7, 'championId': 0, 'assignedPosition': '', 'puuid': 'puuid-enemy-7'},
        {'cellId': 8, 'championId': 81, 'assignedPosition': '', 'puuid': 'puuid-enemy-8'},
        {'cellId': 9, 'championId': 117, 'assignedPosition': '', 'puuid': 'puuid-enemy-9'},
    ],
    'bans': {'myTeamBans': [238], 'theirTeamBans': [11], 'numBans': 10},
    'actions': [[{'type': 'ban', 'championId': 103, 'completed': True, 'actorCellId': 0},
                 {'type': 'ban', 'championId': 157, 'completed': False, 'actorCellId': 5}]],
}


def test_parse_session_extracts_visible_draft():
    state = lcu.parse_session(SESSION, NAMES.get)
    assert state.my_role == 'BOTTOM'
    assert state.my_pick == 'Ashe'
    assert state.allies == {'TOP': 'Aatrox', 'JUNGLE': 'LeeSin', 'SUPPORT': 'Thresh'}  # unpicked mid skipped
    assert state.enemies == ['Malphite', 'Yasuo', 'Ezreal', 'Lulu']
    assert state.bans == ['Ahri', 'MasterYi', 'Zed']  # incomplete ban action ignored


def test_parse_session_never_surfaces_player_identity():
    state = lcu.parse_session(SESSION, NAMES.get)
    flat = repr(state)
    for secret in ['AllyOne', 'AllyTwo', 'Me', 'puuid', '1111', 'NA1']:
        assert secret not in flat, secret


def test_parse_session_handles_empty_or_partial_payloads():
    state = lcu.parse_session({}, NAMES.get)
    assert state.my_role is None and state.allies == {} and state.enemies == [] and state.bans == []


def test_enemy_roles_inferred_from_play_rates():
    role_games = {('Malphite', 'TOP'): 300, ('Yasuo', 'MIDDLE'): 400, ('Yasuo', 'TOP'): 150,
                  ('Ezreal', 'BOTTOM'): 500, ('Lulu', 'SUPPORT'): 300, ('Lulu', 'MIDDLE'): 50}
    assert lcu.assign_enemy_roles(['Malphite', 'Yasuo', 'Ezreal', 'Lulu'], role_games) == {
        'TOP': 'Malphite', 'MIDDLE': 'Yasuo', 'BOTTOM': 'Ezreal', 'SUPPORT': 'Lulu'}
    assert lcu.assign_enemy_roles([], role_games) == {}


def test_client_refuses_any_other_endpoint():
    client = lcu.LCUClient(12345, 'pw')
    for endpoint in ['/lol-champ-select/v1/session/actions/1', '/lol-matchmaking/v1/ready-check/accept',
                     '/lol-summoner/v1/current-summoner']:
        with pytest.raises(lcu.LCUError):
            client._get(endpoint)


def test_client_only_talks_to_localhost():
    assert lcu.LCUClient(12345, 'pw')._base == 'https://127.0.0.1:12345'


def test_tls_trusts_only_riot_ca_and_requires_verification():
    ctx = lcu._pinned_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname  # client cert lists 127.0.0.1 in its SAN
    # Only X509 strict mode is relaxed (Riot's 2013 CA lacks an AKI extension)
    assert not ctx.verify_flags & getattr(ssl, 'VERIFY_X509_STRICT', 0)
    cas = ctx.get_ca_certs()
    assert len(cas) == 1
    subject = dict(x[0] for x in cas[0]['subject'])
    assert subject['organizationName'] == 'Riot Games'


def test_tampered_ca_certificate_is_rejected(tmp_path, monkeypatch):
    original = open(lcu.RIOT_CA_PATH).read()
    fake = tmp_path / 'riotgames.pem'
    lines = original.splitlines()
    lines[5] = lines[5][::-1]  # corrupt one base64 line
    fake.write_text('\n'.join(lines))
    monkeypatch.setattr(lcu, 'RIOT_CA_PATH', str(fake))
    with pytest.raises((lcu.LCUError, ssl.SSLError, ValueError)):
        lcu._pinned_context()


def test_lockfile_parsing(tmp_path):
    good = tmp_path / 'lockfile'
    good.write_text('LeagueClient:1234:53210:s3cr3t:https')
    assert lcu.read_lockfile(str(good)) == (53210, 's3cr3t')
    bad = tmp_path / 'bad'
    bad.write_text('garbage')
    assert lcu.read_lockfile(str(bad)) is None
    assert lcu.read_lockfile(str(tmp_path / 'missing')) is None


def test_blind_mode_session_without_roles():
    """Practice Tool / Blind Pick: no assigned positions, no bans, no enemies."""
    session = {'localPlayerCellId': 0,
               'myTeam': [{'cellId': 0, 'championId': 22, 'assignedPosition': '', 'gameName': 'Me'},
                          {'cellId': 1, 'championId': 412, 'assignedPosition': ''}],
               'theirTeam': [], 'bans': {}, 'actions': []}
    state = lcu.parse_session(session, NAMES.get)
    assert state.my_role is None and state.my_pick == 'Ashe'
    assert state.allies == {} and state.unassigned_allies == ['Thresh']


def test_role_inference_respects_open_roles():
    role_games = {('Thresh', 'SUPPORT'): 400, ('Thresh', 'BOTTOM'): 5}
    assert lcu.assign_enemy_roles(['Thresh'], role_games, roles=['TOP', 'MIDDLE', 'SUPPORT']) == {'SUPPORT': 'Thresh'}
    assert lcu.assign_enemy_roles(['Thresh'], role_games, roles=['TOP', 'MIDDLE']) in ({'TOP': 'Thresh'}, {'MIDDLE': 'Thresh'})
