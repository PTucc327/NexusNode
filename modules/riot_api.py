from urllib.parse import quote

import requests

from modules.champions import ChampionCatalog, fetch_ddragon

REQUEST_TIMEOUT = 10

# Platform -> regional routing value for account-v1
PLATFORM_ROUTING = {
    'na1': 'americas', 'br1': 'americas', 'la1': 'americas', 'la2': 'americas',
    'euw1': 'europe', 'eun1': 'europe', 'tr1': 'europe', 'ru': 'europe', 'me1': 'europe',
    'kr': 'asia', 'jp1': 'asia',
    'oc1': 'sea', 'sg2': 'sea', 'tw2': 'sea', 'vn2': 'sea',
}


class RiotAPIError(Exception):
    pass


class RiotInterface:
    def __init__(self, api_key, region="na1", catalog=None):
        self.api_key = api_key
        self.region = region
        self.routing = PLATFORM_ROUTING.get(region, 'americas')
        self.headers = {"X-Riot-Token": api_key}
        # Latest Data Dragon champion data (numeric id -> name); the old
        # hardcoded 14.8.1 patch silently dropped every newer champion.
        self.catalog = catalog or ChampionCatalog(fetch_ddragon())

    def _get(self, url):
        response = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 404:
            return None
        if response.status_code in (401, 403):
            raise RiotAPIError("Riot API key is invalid or expired.")
        if response.status_code == 429:
            raise RiotAPIError("Riot API rate limit hit, try again in a minute.")
        raise RiotAPIError(f"Riot API error {response.status_code}.")

    def get_puuid(self, game_name, tag_line):
        url = (f"https://{self.routing}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/"
               f"{quote(game_name)}/{quote(tag_line)}")
        data = self._get(url)
        return data.get('puuid') if data else None

    def get_user_comfort_pool(self, puuid, count=15):
        """Returns Data Dragon ids of the user's top-mastery champions."""
        url = (f"https://{self.region}.api.riotgames.com/lol/champion-mastery/v4/"
               f"champion-masteries/by-puuid/{puuid}/top?count={count}")
        masteries = self._get(url) or []
        names = (self.catalog.name_from_numeric_id(m['championId']) for m in masteries)
        return [n for n in names if n]
