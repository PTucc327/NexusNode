import pandas as pd
from riotwatcher import LolWatcher, ApiError
import time
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv('RIOT_KEY')
WATCHER = LolWatcher(API_KEY)
QUEUE_TYPE = 'RANKED_SOLO_5x5'

# Define the regions you want to scrape
# Mapping Platform ID to its Regional Routing
REGIONS = {
    'na1': 'americas',
    'euw1': 'europe',
    'kr': 'asia',
    'br1': 'americas' # Adding Brazil for more diversity if you want!
}

def get_massive_match_ids(platform, routing, player_limit=30, matches_per_player=30):
    print(f"🚀 Fetching Challenger data for {platform.upper()}...")
    all_match_ids = set()
    
    try:
        chall_league = WATCHER.league.challenger_by_queue(platform, QUEUE_TYPE)
        players = chall_league.get('entries', [])[:player_limit]
        
        for i, entry in enumerate(players):
            try:
                puuid = entry.get('puuid')
                if not puuid: continue
                
                player_matches = WATCHER.match.matchlist_by_puuid(
                    routing, puuid, count=matches_per_player
                )
                
                if player_matches:
                    all_match_ids.update(player_matches)
                
                time.sleep(1.2) # Rate limit respect
                
            except ApiError as e:
                continue
                
    except ApiError as err:
        print(f"❌ Error in {platform}: {err}")
        
    return list(all_match_ids), routing

def process_match_data(routing, match_id):
    try:
        match = WATCHER.match.by_id(routing, match_id)
        if match['info']['gameDuration'] < 300: return None
            
        participants = []
        for p in match['info']['participants']:
            participants.append({
                'match_id': match_id,
                'region': routing, # Tracking region for future analysis
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
        if err.response.status_code == 429:
            time.sleep(10)
        return None

if __name__ == "__main__":
    if not API_KEY:
        print("❌ RIOT_KEY missing.")
    else:
        full_dataset = []
        
        for platform, routing in REGIONS.items():
            match_ids, region_route = get_massive_match_ids(platform, routing)
            print(f"✅ Found {len(match_ids)} unique matches in {platform}. Processing...")

            for i, m_id in enumerate(match_ids):
                data = process_match_data(region_route, m_id)
                if data:
                    full_dataset.extend(data)
                
                if (i + 1) % 20 == 0:
                    print(f"📦 [{platform.upper()}] Processed {i+1}/{len(match_ids)}...")
                
                time.sleep(1.2)

        # Step 3: Save and Append
        df = pd.DataFrame(full_dataset)
        
        # 'a' mode appends, header=False prevents repeating column names
        file_exists = os.path.isfile('league_match_data.csv')
        df.to_csv('league_match_data.csv', mode='a', index=False, header=not file_exists)
        
        print(f"✨ Success! Dataset now has {len(df)} new entries across {len(REGIONS)} regions.")