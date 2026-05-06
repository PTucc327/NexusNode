import pandas as pd
from riotwatcher import LolWatcher, ApiError
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv('RIOT_KEY')
WATCHER = LolWatcher(API_KEY)
QUEUE_TYPE = 'RANKED_SOLO_5x5'

# Correct relative paths based on your new directory structure
RAW_DATA_PATH = os.path.join('data', 'raw', 'league_match_data.csv')

REGIONS = {
    'na1': 'americas',
    'euw1': 'europe',
    'kr': 'asia',
    'br1': 'americas'
}

def load_processed_ids():
    """Loads existing match IDs from the CSV to avoid double-processing."""
    if os.path.exists(RAW_DATA_PATH):
        try:
            df = pd.read_csv(RAW_DATA_PATH, usecols=['match_id'])
            return set(df['match_id'].unique())
        except Exception:
            return set()
    return set()

def get_massive_match_ids(platform, routing, processed_ids, player_limit=25, matches_per_player=20):
    print(f"🚀 Fetching Challenger data for {platform.upper()}...")
    new_match_ids = set()
    
    try:
        chall_league = WATCHER.league.challenger_by_queue(platform, QUEUE_TYPE)
        players = chall_league.get('entries', [])[:player_limit]
        
        for entry in players:
            try:
                # We need PUUID, which isn't always in the league entry. 
                # Note: In a production pipeline, you'd cache sumonnerID -> PUUID mappings
                summoner_id = entry.get('summonerId')
                summoner = WATCHER.summoner.by_id(platform, summoner_id)
                puuid = summoner.get('puuid')
                
                if not puuid: continue
                
                player_matches = WATCHER.match.matchlist_by_puuid(
                    routing, puuid, count=matches_per_player
                )
                
                for m_id in player_matches:
                    if m_id not in processed_ids:
                        new_match_ids.add(m_id)
                
                time.sleep(1.2) # Rate limit respect
                
            except ApiError:
                continue
                
    except ApiError as err:
        print(f"❌ Error in {platform}: {err}")
        
    return list(new_match_ids)

def process_match_data(routing, match_id):
    try:
        match = WATCHER.match.by_id(routing, match_id)
        # Skip remakes
        if match['info']['gameDuration'] < 300: return None
            
        participants = []
        for p in match['info']['participants']:
            participants.append({
                'match_id': match_id,
                'region': routing,
                'champion_name': p['championName'],
                'team_id': p['teamId'],
                'win': p['win'],
                'role': p['teamPosition'],
                'kills': p['kills'],
                'deaths': p['deaths'],
                'assists': p['assists'],
                'damage_to_champs': p['totalDamageDealtToChampions'],
                'gold_earned': p['goldEarned'],
                'collected_at': datetime.now().strftime("%Y-%m-%d %H:%M")
            })
        return participants
    except ApiError as err:
        if err.response.status_code == 429:
            print("⏳ Rate limited. Sleeping for 20s...")
            time.sleep(20)
        return None

if __name__ == "__main__":
    if not API_KEY:
        print("❌ RIOT_KEY missing in .env")
    else:
        # 1. Ensure directories exist
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        
        # 2. Check for existing work
        processed_ids = load_processed_ids()
        print(f"📂 Loaded {len(processed_ids)} previously processed matches.")

        for platform, routing in REGIONS.items():
            match_ids = get_massive_match_ids(platform, routing, processed_ids)
            print(f"✅ Found {len(match_ids)} NEW matches in {platform}. Processing...")

            batch_data = []
            for i, m_id in enumerate(match_ids):
                data = process_match_data(routing, m_id)
                if data:
                    batch_data.extend(data)
                
                # Check-pointing: Save every 10 matches so we don't lose data on crash
                if len(batch_data) >= 100: # Every 10 matches (10 players each)
                    df_batch = pd.DataFrame(batch_data)
                    file_exists = os.path.isfile(RAW_DATA_PATH)
                    df_batch.to_csv(RAW_DATA_PATH, mode='a', index=False, header=not file_exists)
                    batch_data = [] # Reset batch
                    print(f"💾 Checkpoint reached. Matches saved to {RAW_DATA_PATH}")

                time.sleep(1.2)

            # Final save for the remaining data in the region
            if batch_data:
                df_final = pd.DataFrame(batch_data)
                file_exists = os.path.isfile(RAW_DATA_PATH)
                df_final.to_csv(RAW_DATA_PATH, mode='a', index=False, header=not file_exists)
        
        print(f"✨ Automation Cycle Complete. Data stored in {RAW_DATA_PATH}")