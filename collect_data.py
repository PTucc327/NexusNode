import pandas as pd
from riotwatcher import LolWatcher, ApiError
import time
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv('RIOT_KEY')
PLATFORM_ID = 'na1'        
REGIONAL_ROUTING = 'americas' 
QUEUE_TYPE = 'RANKED_SOLO_5x5'

WATCHER = LolWatcher(API_KEY)

def get_massive_match_ids(player_limit=50, matches_per_player=40):
    print(f"🚀 Fetching Challenger league data...")
    all_match_ids = set()
    
    try:
        chall_league = WATCHER.league.challenger_by_queue(PLATFORM_ID, QUEUE_TYPE)
        players = chall_league.get('entries', [])[:player_limit]
        
        if not players:
            print("❌ No players found in Challenger league.")
            return []

        print(f"🔍 Scanning {len(players)} players directly via PUUID...")
        
        for i, entry in enumerate(players):
            try:
                # GREAT NEWS: puuid is already in the entry!
                puuid = entry.get('puuid')
                
                if not puuid:
                    print(f"⚠️ Skipping player {i}: No PUUID found.")
                    continue
                
                # We go straight to fetching matches!
                player_matches = WATCHER.match.matchlist_by_puuid(
                    REGIONAL_ROUTING, puuid, count=matches_per_player
                )
                
                if player_matches:
                    all_match_ids.update(player_matches)
                
                if (i + 1) % 5 == 0:
                    print(f"  > Progress: {i+1}/{len(players)} players searched. Unique matches: {len(all_match_ids)}")
                
                # Still sleep to respect Match-V5 rate limits
                time.sleep(1.2) 
                
            except ApiError as e:
                print(f"  ! Skipping player {i} due to API Error: {e}")
                continue
                
    except ApiError as err:
        print(f"❌ Critical API Error: {err}")
        
    return list(all_match_ids)


def process_match_data(match_id):
    try:
        match = WATCHER.match.by_id(REGIONAL_ROUTING, match_id)
        # Skip if match is not a standard 5v5 (e.g., remakes)
        if match['info']['gameDuration'] < 300: 
            return None
            
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
        if err.response.status_code == 429:
            print("⚠️ Rate limit hit! Sleeping for 10 seconds...")
            time.sleep(10)
        return None

if __name__ == "__main__":
    if not API_KEY:
        print("❌ Error: RIOT_KEY not found in .env file.")
    else:
        # Step 1: Get a massive list of unique Match IDs
        # 50 players * 40 matches = ~2,000 potential matches (many will be duplicates)
        match_ids = get_massive_match_ids(player_limit=50, matches_per_player=40)
        
        all_match_data = []
        total = len(match_ids)
        print(f"✅ Found {total} unique matches. Starting extraction...")

        # Step 2: Extract details
        for i, m_id in enumerate(match_ids):
            data = process_match_data(m_id)
            if data:
                all_match_data.extend(data)
            
            # Progress bar
            if (i + 1) % 10 == 0:
                print(f"📦 Processed {i+1}/{total} matches...")
            
            time.sleep(1.2) # Essential for Dev Keys

        # Step 3: Save
        df = pd.DataFrame(all_match_data)
        # We append to the existing file or create a new one
        df.to_csv('league_match_data.csv', index=False)
        print(f"✨ Success! Total data points: {len(df)}")