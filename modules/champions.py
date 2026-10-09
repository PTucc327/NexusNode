import sys
import json
import os

import requests

# Data Dragon is Riot's static asset CDN (champion names, square portraits).
DDRAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMPIONS_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
DDRAGON_ICON_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{ddragon_id}.png"

# Offline fallback so the app still renders names/icons if Data Dragon is down.
SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'static', 'ddragon_champions.json')
REQUEST_TIMEOUT = 5


def _key(name):
    """Normalizes champion identifiers across Riot sources. The match API
    reports e.g. "FiddleSticks" while Data Dragon uses "Fiddlesticks"."""
    return name.lower().replace(" ", "").replace("'", "").replace(".", "")


def fetch_ddragon():
    """Returns {'version': str, 'champions': {ddragon_id: {'name', 'key', 'tags', 'info'}}}
    from the live CDN, falling back to the bundled snapshot. Set
    NEXUSNODE_OFFLINE=1 to always use the snapshot (tests, offline use)."""
    try:
        if os.getenv('NEXUSNODE_OFFLINE'):
            raise RuntimeError('offline mode')
        version = requests.get(DDRAGON_VERSIONS_URL, timeout=REQUEST_TIMEOUT).json()[0]
        data = requests.get(DDRAGON_CHAMPIONS_URL.format(version=version), timeout=REQUEST_TIMEOUT).json()['data']
        return {
            'version': version,
            'champions': {cid: {'name': c['name'], 'key': int(c['key']), 'tags': c.get('tags', []),
                                'info': c.get('info', {})} for cid, c in data.items()},
        }
    except Exception:
        if os.path.exists(SNAPSHOT_PATH):
            with open(SNAPSHOT_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'version': None, 'champions': {}}


def save_snapshot(ddragon):
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    with open(SNAPSHOT_PATH, 'w', encoding='utf-8') as f:
        json.dump(ddragon, f, indent=1, sort_keys=True)


class ChampionCatalog:
    """Maps the dataset's champion names (match API `championName`) to
    display names, portrait URLs and numeric IDs."""

    def __init__(self, ddragon):
        self.version = ddragon.get('version')
        self._by_key = {_key(cid): (cid, c) for cid, c in ddragon.get('champions', {}).items()}
        self._by_numeric_id = {c['key']: cid for cid, c in ddragon.get('champions', {}).items()}

    def _lookup(self, api_name):
        return self._by_key.get(_key(api_name))

    def display_name(self, api_name):
        hit = self._lookup(api_name)
        return hit[1]['name'] if hit else api_name

    def icon_url(self, api_name):
        hit = self._lookup(api_name)
        if not hit or not self.version:
            return None
        return DDRAGON_ICON_URL.format(version=self.version, ddragon_id=hit[0])

    def resolve(self, name, known_names):
        """Matches a Data Dragon id / display name onto the dataset's spelling."""
        lookup = {_key(n): n for n in known_names}
        return lookup.get(_key(name))

    def profile(self, api_name):
        """Class tags (e.g. ['Mage', 'Support']) and Riot's 0-10 attack/magic/
        defense ratings, used for team composition features."""
        hit = self._lookup(api_name)
        if not hit:
            return {'tags': [], 'info': {}}
        return {'tags': hit[1].get('tags', []), 'info': hit[1].get('info', {})}

    def name_from_numeric_id(self, champion_id):
        return self._by_numeric_id.get(champion_id)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # emoji logs on Windows consoles
    # Refresh the bundled offline snapshot.
    dd = fetch_ddragon()
    save_snapshot(dd)
    print(f"Saved {len(dd['champions'])} champions (patch {dd['version']}) to {SNAPSHOT_PATH}")
