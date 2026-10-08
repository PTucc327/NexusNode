import sys
import pandas as pd
from riotwatcher import LolWatcher, ApiError
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv('RIOT_KEY')
WATCHER = LolWatcher(API_KEY) if API_KEY else None
QUEUE_TYPE = 'RANKED_SOLO_5x5'
RANKED_SOLO_QUEUE_ID = 420  # match-v5 queue id for Ranked Solo/Duo
MAX_RETRIES = 3

# Correct relative paths based on your new directory structure
RAW_DATA_PATH = os.path.join('data', 'raw', 'league_match_data.csv')
RAW_COLUMNS = [
    'match_id', 'region', 'champion_name', 'team_id', 'win', 'role', 'kills', 'deaths',
    'assists', 'damage_to_champs', 'gold_earned', 'collected_at'
]

REGIONS = {
    'na1': 'americas',
    'euw1': 'europe',
    'kr': 'asia',
    'br1': 'americas'
}

def call_with_retry(fn, *args, **kwargs):
    """Calls a Riot endpoint, honoring 429 Retry-After instead of dropping data."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except ApiError as err:
            if err.response is not None and err.response.status_code == 429 and attempt < MAX_RETRIES - 1:
                wait = int(err.response.headers.get('Retry-After', 20))
                print(f"⏳ Rate limited. Sleeping for {wait}s...")
                time.sleep(wait)
                continue
            raise

def load_processed_ids():
    """Loads existing match IDs from the CSV to avoid double-processing."""
    if os.path.exists(RAW_DATA_PATH):
        try:
            df = pd.read_csv(RAW_DATA_PATH, usecols=['match_id'])
            return set(df['match_id'].unique())
        except Exception:
            return set()
    return set()

def ensure_raw_schema():
    """Upgrades an existing raw CSV whose header predates newer columns.
    Appending rows with more fields than the header made eda.py's
    `on_bad_lines='skip'` silently discard every newly scraped row."""
    if not os.path.exists(RAW_DATA_PATH):
        return
    existing_cols = pd.read_csv(RAW_DATA_PATH, nrows=0).columns.tolist()
    if existing_cols != RAW_COLUMNS:
        df = pd.read_csv(RAW_DATA_PATH)
        df.reindex(columns=RAW_COLUMNS).to_csv(RAW_DATA_PATH, index=False)
        print(f"🔧 Migrated {RAW_DATA_PATH} to the current column schema.")

def append_rows(rows):
    df = pd.DataFrame(rows).reindex(columns=RAW_COLUMNS)
    file_exists = os.path.isfile(RAW_DATA_PATH)
    df.to_csv(RAW_DATA_PATH, mode='a', index=False, header=not file_exists)

def get_massive_match_ids(platform, routing, processed_ids, player_limit=25, matches_per_player=20):
    print(f"🚀 Fetching Challenger data for {platform.upper()}...")
    new_match_ids = set()

    try:
        chall_league = call_with_retry(WATCHER.league.challenger_by_queue, platform, QUEUE_TYPE)
        players = chall_league.get('entries', [])[:player_limit]

        for entry in players:
            try:
                # League entries now carry the PUUID directly (Riot removed
                # summonerId from league-v4 in 2025, which made the old
                # summoner.by_id lookup fail for every player).
                puuid = entry.get('puuid')
                if not puuid and entry.get('summonerId'):
                    puuid = call_with_retry(WATCHER.summoner.by_id, platform, entry['summonerId']).get('puuid')

                if not puuid: continue

                player_matches = call_with_retry(
                    WATCHER.match.matchlist_by_puuid,
                    routing, puuid, count=matches_per_player, queue=RANKED_SOLO_QUEUE_ID
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
        match = call_with_retry(WATCHER.match.by_id, routing, match_id)
    except ApiError as err:
        print(f"⚠️ Skipping {match_id}: {err}")
        return None

    info = match['info']
    # Skip remakes and anything that isn't ranked solo (no reliable roles)
    if info['gameDuration'] < 300 or info.get('queueId') != RANKED_SOLO_QUEUE_ID:
        return None

    collected_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    participants = []
    for p in info['participants']:
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
            'collected_at': collected_at
        })
    return participants

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # emoji logs on Windows consoles
    if not API_KEY:
        print("❌ RIOT_KEY missing in .env")
    else:
        # 1. Ensure directories exist and the file header matches what we write
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        ensure_raw_schema()

        # 2. Check for existing work
        processed_ids = load_processed_ids()
        print(f"📂 Loaded {len(processed_ids)} previously processed matches.")

        for platform, routing in REGIONS.items():
            match_ids = get_massive_match_ids(platform, routing, processed_ids)
            print(f"✅ Found {len(match_ids)} NEW matches in {platform}. Processing...")

            batch_data = []
            for i, m_id in enumerate(match_ids):
                data = process_match_data(routing, m_id)
                processed_ids.add(m_id)
                if data:
                    batch_data.extend(data)

                # Check-pointing: Save every 10 matches so we don't lose data on crash
                if len(batch_data) >= 100: # Every 10 matches (10 players each)
                    append_rows(batch_data)
                    batch_data = [] # Reset batch
                    print(f"💾 Checkpoint reached. Matches saved to {RAW_DATA_PATH}")

                time.sleep(1.2)

            # Final save for the remaining data in the region
            if batch_data:
                append_rows(batch_data)

        print(f"✨ Automation Cycle Complete. Data stored in {RAW_DATA_PATH}")
