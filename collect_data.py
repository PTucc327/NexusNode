import pandas as pd
from riotwatcher import LolWatcher, ApiError
import time
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv('RIOT_KEY')
PLATFORM_ID = 'na1'        # For Summoner & League data
REGIONAL_ROUTING = 'americas' # For Match-V5 data (Matchlist & Match Details)
QUEUE_TYPE = 'RANKED_SOLO_5x5'

WATCHER = LolWatcher(API_KEY)

def get_high_elo_match_ids(count=100):
    print(f"Fetching Challenger players from {PLATFORM_ID}...")
    match_ids = []
    
    try:
        challenger_league = WATCHER.league.challenger_by_queue(PLATFORM_ID, QUEUE_TYPE)
        
        # Pulling from top 10 players
        for entry in challenger_league['entries'][:10]:
            
            puuid = entry['puuid']
            summoner = WATCHER.summoner.by_puuid(PLATFORM_ID, puuid)
            # CRITICAL: Matchlist uses REGIONAL_ROUTING ('americas')
            player_matches = WATCHER.match.matchlist_by_puuid(REGIONAL_ROUTING, puuid, count=20)
            match_ids.extend(player_matches)
            
            time.sleep(1.2) # Rate limit breathing room
            
    except ApiError as err:
        print(f"API Error in get_ids: {err}")
        
    return list(set(match_ids))

def process_match_data(match_id):
    try:
        # CRITICAL: Match details use REGIONAL_ROUTING ('americas')
        match = WATCHER.match.by_id(REGIONAL_ROUTING, match_id)
        participants = []
        
        for p in match['info']['participants']:
            participants.append({
                'match_id': match_id,
                'champion_name': p['championName'],
                'team_id': p['teamId'],
                'win': p['win'],
                'role': p['teamPosition'],
                'kills': p['kills'],
                'deaths': p['deaths'],
                'assists': p['assists'],
                'damage_to_champs': p['totalDamageDealtToChampions'],
                'gold_earned': p['goldEarned']
            })
        return participants
    except ApiError as err:
        print(f"Error fetching match {match_id}: {err}")
        return None

if __name__ == "__main__":
    if not API_KEY:
        print("❌ Error: RIOT_KEY not found in .env file.")
    else:
        all_match_data = []
        match_ids = get_high_elo_match_ids(count=50)
        
        print(f"Starting processing for {len(match_ids)} matches...")
        
        for i, m_id in enumerate(match_ids):
            print(f"[{i+1}/{len(match_ids)}] Processing: {m_id}")
            data = process_match_data(m_id)
            if data:
                all_match_data.extend(data)
            
            time.sleep(1.2) 

        df = pd.DataFrame(all_match_data)
        df.to_csv('league_match_data.csv', index=False)
        print("✅ Data collection complete! Check 'league_match_data.csv'.")