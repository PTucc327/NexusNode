import requests

class RiotInterface:
    def __init__(self, api_key, region="na1"):
        self.api_key = api_key
        self.region = region
        self.headers = {"X-Riot-Token": api_key}

    def get_puuid(self, game_name, tag_line):
        # Riot ID API (v1)
        url = f"https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()['puuid']
        return None

    def get_top_masteries(self, puuid, count=10):
        # Mastery API (v4)
        url = f"https://{self.region}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/top?count={count}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        return []

# Helper to map Champion IDs to Names (Riot uses IDs like 22, 202)
def get_champ_name_map():
    ddragon_url = "https://ddragon.leagueoflegends.com/cdn/14.8.1/data/en_US/champion.json"
    data = requests.get(ddragon_url).json()
    return {int(v['key']): k for k, v in data['data'].items()}