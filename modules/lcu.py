"""Read-only champion-select reader for the local League Client (desktop companion).

Safety by construction, per Riot's developer policies:
  - Read-only: the client can only GET one allow-listed endpoint. It cannot
    pick, ban, hover, accept, dodge or otherwise act in the client.
  - Visible information only: just champion picks/bans and the local
    player's own assigned position are parsed. Player identity fields in the
    session (names, PUUIDs, summoner IDs) are never read, so players that
    ranked champ select keeps anonymous stay anonymous.
  - Verified TLS: the client's local HTTPS certificate must chain to Riot's
    published "LoL Game Engineering Certificate Authority" (bundled and
    fingerprint-checked), and we only ever connect to 127.0.0.1.

Riot requires League Client API products to list their endpoints on the
Developer Portal and be acknowledged BEFORE release. This module is for
local development until then; the app only enables it with
NEXUSNODE_DESKTOP=1.
"""
import base64
import hashlib
import itertools
import os
import ssl
from dataclasses import dataclass, field

import numpy as np
import requests
from requests.adapters import HTTPAdapter

ALLOWED_ENDPOINT = '/lol-champ-select/v1/session'
LOCKFILE_CANDIDATES = [
    r'C:\Riot Games\League of Legends\lockfile',
    '/Applications/League of Legends.app/Contents/LoL/lockfile',
]
RIOT_CA_PATH = os.path.join(os.path.dirname(__file__), 'certs', 'riotgames.pem')
RIOT_CA_SHA256 = 'ca8c9d325b4cdc464c6c94a585c85e91ec23d40ba5bf3ae2822b951a4a504ea3'
POSITION_TO_ROLE = {'top': 'TOP', 'jungle': 'JUNGLE', 'middle': 'MIDDLE', 'bottom': 'BOTTOM', 'utility': 'SUPPORT'}
ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'SUPPORT']
REQUEST_TIMEOUT = 2


class LCUError(Exception):
    pass


def read_lockfile(path=None):
    """(port, password) from the client's lockfile, or None if the client isn't running.
    Format: name:pid:port:password:protocol"""
    for candidate in ([path] if path else LOCKFILE_CANDIDATES):
        try:
            with open(candidate, 'r', encoding='utf-8') as f:
                parts = f.read().strip().split(':')
        except OSError:
            continue
        if len(parts) == 5 and parts[2].isdigit() and parts[4] == 'https':
            return int(parts[2]), parts[3]
    return None


def _pinned_context():
    """TLS context that only trusts Riot's CA (fingerprint-checked). Hostname
    checking is off because the client's cert names 127.0.0.1 only in its CN,
    which Python no longer matches; the chain to Riot's CA is still required,
    and we only connect to 127.0.0.1."""
    with open(RIOT_CA_PATH, 'r', encoding='ascii') as f:
        pem = f.read()
    if hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest() != RIOT_CA_SHA256:
        raise LCUError('Bundled Riot CA certificate failed its fingerprint check.')
    ctx = ssl.create_default_context(cadata=pem)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


class _PinnedAdapter(HTTPAdapter):
    def __init__(self, ctx, **kwargs):
        self._ctx = ctx
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = self._ctx
        return super().init_poolmanager(*args, **kwargs)


class LCUClient:
    """Minimal read-only client. Exposes exactly one call."""

    def __init__(self, port, password):
        self._base = f'https://127.0.0.1:{int(port)}'
        token = base64.b64encode(f'riot:{password}'.encode()).decode()
        self._session = requests.Session()
        self._session.headers['Authorization'] = f'Basic {token}'
        self._session.mount('https://127.0.0.1', _PinnedAdapter(_pinned_context()))

    @classmethod
    def from_lockfile(cls, path=None):
        creds = read_lockfile(path)
        return cls(*creds) if creds else None

    def _get(self, endpoint):
        if endpoint != ALLOWED_ENDPOINT:
            raise LCUError(f'Endpoint not allowed: {endpoint}')
        return self._session.get(self._base + endpoint, timeout=REQUEST_TIMEOUT)

    def champ_select_session(self):
        """The current champ-select session, or None when not in champ select."""
        response = self._get(ALLOWED_ENDPOINT)
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise LCUError(f'League client returned {response.status_code}.')
        return response.json()


@dataclass
class DraftState:
    my_role: str = None                            # local player's assigned role, if known
    my_pick: str = None
    allies: dict = field(default_factory=dict)     # {role: champion}
    enemies: list = field(default_factory=list)    # champions (enemy roles aren't shown in ranked)
    bans: list = field(default_factory=list)


def parse_session(session, champion_name):
    """Extract ONLY visible draft information from an LCU champ-select session.

    champion_name: callable numeric champion id -> champion name (or None).
    Deliberately never reads identity fields (gameName, puuid, summonerId...).
    """
    def name(champion_id):
        return champion_name(champion_id) if isinstance(champion_id, int) and champion_id > 0 else None

    state = DraftState()
    local_cell = session.get('localPlayerCellId')
    for member in session.get('myTeam', []):
        role = POSITION_TO_ROLE.get(str(member.get('assignedPosition', '')).lower())
        champ = name(member.get('championId'))
        if member.get('cellId') == local_cell:
            state.my_role, state.my_pick = role, champ
        elif champ and role:
            state.allies[role] = champ
    state.enemies = [c for c in (name(m.get('championId')) for m in session.get('theirTeam', [])) if c]

    bans = set()
    ban_lists = session.get('bans', {})
    for key in ('myTeamBans', 'theirTeamBans'):
        bans.update(c for c in (name(cid) for cid in ban_lists.get(key, [])) if c)
    for turn in session.get('actions', []):
        for action in turn:
            if action.get('type') == 'ban' and action.get('completed'):
                c = name(action.get('championId'))
                if c:
                    bans.add(c)
    state.bans = sorted(bans)
    return state


def assign_enemy_roles(enemies, role_games):
    """Ranked champ select doesn't show enemy roles; infer the most likely
    assignment from how often each champion is played in each role.
    role_games: {(champion, role): games}. Returns {role: champion}."""
    enemies = list(enemies)[:len(ROLES)]
    if not enemies:
        return {}
    score = {(c, r): np.log1p(role_games.get((c, r), 0)) for c in enemies for r in ROLES}
    best = max(itertools.permutations(ROLES, len(enemies)),
               key=lambda roles: sum(score[(c, r)] for c, r in zip(enemies, roles)))
    return {r: c for c, r in zip(enemies, best)}
