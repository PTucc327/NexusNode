import requests

class RiotInterface:
    def __init__(self, api_key, region="na1"):
        self.api_key = api_key
        self.region = region
        self.headers = {"X-Riot-Token": api_key}
        # Cache for champ map to avoid repeated network calls
        self.champ_map = self._get_champ_name_map()

    def _get_champ_name_map(self):
        """Fetches latest Data Dragon map: ID -> Name"""
        # Note: Using 14.8.1 as a baseline, in production, fetch latest version first
        ddragon_url = "https://ddragon.leagueoflegends.com/cdn/14.8.1/data/en_US/champion.json"
        try:
            data = requests.get(ddragon_url).json()
            return {int(v['key']): k for k, v in data['data'].items()}
        except Exception:
            return {}

    def get_puuid(self, game_name, tag_line):
        url = f"https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json().get('puuid')
        return None

    def get_user_comfort_pool(self, puuid, count=15):
        """Returns a list of champion names the user is proficient with."""
        url = f"https://{self.region}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/top?count={count}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            masteries = response.json()
            return [self.champ_map.get(m['championId']) for m in masteries if m['championId'] in self.champ_map]
        return []