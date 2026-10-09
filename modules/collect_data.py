import sys
import argparse
import pandas as pd
from riotwatcher import LolWatcher, ApiError
import time
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv('RIOT_KEY')
WATCHER = LolWatcher(API_KEY) if API_KEY else None
QUEUE_TYPE = 'RANKED_SOLO_5x5'
RANKED_SOLO_QUEUE_ID = 420  # match-v5 queue id for Ranked Solo/Duo
MAX_RETRIES = 3

# Scale knobs (env-overridable). Defaults fit a development key's
# 100 requests / 2 minutes; with a production key, lower REQUEST_DELAY.
REQUEST_DELAY = float(os.getenv('RIOT_REQUEST_DELAY', 1.2))
PLAYERS_PER_TIER = int(os.getenv('PLAYERS_PER_TIER', 40))
MATCHES_PER_PLAYER = int(os.getenv('MATCHES_PER_PLAYER', 20))
MAX_NEW_MATCHES_PER_REGION = int(os.getenv('MAX_NEW_MATCHES_PER_REGION', 500))

# Correct relative paths based on your new directory structure
RAW_DATA_PATH = os.path.join('data', 'raw', 'league_match_data.csv')
RAW_COLUMNS = [
    'match_id', 'region', 'champion_name', 'team_id', 'win', 'role', 'kills', 'deaths',
    'assists', 'damage_to_champs', 'gold_earned', 'collected_at',
    'game_version', 'game_start', 'queue_id'
]

REGIONS = {
    'na1': 'americas',
    'euw1': 'europe',
    'kr': 'asia',
    'br1': 'americas'
}

# Read patch strings as text: pandas would parse '16.20' as the float 16.2
TEXT_COLUMNS = {'game_version': str}

# Apex tiers to sample players from (league-v4 endpoint per tier)
TIERS = ['challenger', 'grandmaster']

class RiotAuthError(Exception):
    """Key rejected (401/403). Not an ApiError subclass, so per-player
    `except ApiError: continue` handlers can't swallow it."""

def call_with_retry(fn, *args, **kwargs):
    """Calls a Riot endpoint, honoring 429 Retry-After instead of dropping data."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except ApiError as err:
            if err.response is not None and err.response.status_code in (401, 403):
                raise RiotAuthError(
                    f"Riot API rejected the key ({err.response.status_code}). Development keys expire "
                    "every 24h; set a valid RIOT_KEY (repo secret for the weekly workflow)."
                ) from err
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
        df = pd.read_csv(RAW_DATA_PATH, dtype=TEXT_COLUMNS)
        df.reindex(columns=RAW_COLUMNS).to_csv(RAW_DATA_PATH, index=False)
        print(f"🔧 Migrated {RAW_DATA_PATH} to the current column schema.")

def append_rows(rows):
    df = pd.DataFrame(rows).reindex(columns=RAW_COLUMNS)
    file_exists = os.path.isfile(RAW_DATA_PATH)
    df.to_csv(RAW_DATA_PATH, mode='a', index=False, header=not file_exists)

def get_league_players(platform, tier):
    fetch = getattr(WATCHER.league, f'{tier}_by_queue')
    entries = call_with_retry(fetch, platform, QUEUE_TYPE).get('entries', [])
    # Highest LP first so a partial sample is the strongest players
    entries.sort(key=lambda e: e.get('leaguePoints', 0), reverse=True)
    return entries[:PLAYERS_PER_TIER]

def get_massive_match_ids(platform, routing, processed_ids):
    new_match_ids = set()

    for tier in TIERS:
        print(f"🚀 Fetching {tier.title()} players for {platform.upper()}...")
        try:
            players = get_league_players(platform, tier)
        except ApiError as err:
            print(f"❌ Error fetching {tier} in {platform}: {err}")
            continue

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
                    routing, puuid, count=MATCHES_PER_PLAYER, queue=RANKED_SOLO_QUEUE_ID
                )

                for m_id in player_matches:
                    if m_id not in processed_ids:
                        new_match_ids.add(m_id)

                time.sleep(REQUEST_DELAY) # Rate limit respect

            except ApiError:
                continue

    # Newest first (match ids are sequential per platform), capped per run
    return sorted(new_match_ids, reverse=True)[:MAX_NEW_MATCHES_PER_REGION]

def patch_of(game_version):
    """'16.19.712.4459' -> '16.19'"""
    parts = str(game_version).split('.')
    return '.'.join(parts[:2]) if len(parts) >= 2 else None

def fetch_match(routing, match_id):
    try:
        return call_with_retry(WATCHER.match.by_id, routing, match_id)
    except ApiError as err:
        print(f"⚠️ Skipping {match_id}: {err}")
        return None

def process_match_data(routing, match_id):
    match = fetch_match(routing, match_id)
    if match is None:
        return None

    info = match['info']
    # Skip remakes and anything that isn't ranked solo (no reliable roles)
    if info['gameDuration'] < 300 or info.get('queueId') != RANKED_SOLO_QUEUE_ID:
        return None

    collected_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    game_start = datetime.fromtimestamp(info['gameCreation'] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
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
            'collected_at': collected_at,
            'game_version': patch_of(info.get('gameVersion')),
            'game_start': game_start,
            'queue_id': info.get('queueId'),
        })
    return participants

def backfill_metadata(checkpoint_every=100):
    """One-off: fetches patch/date/queue for rows scraped before those
    columns existed (the old scraper had no queue filter, so queue_id lets
    eda.py drop non-ranked games). Skips matches with no roles at all."""
    df = pd.read_csv(RAW_DATA_PATH, dtype=TEXT_COLUMNS)
    has_role = df.groupby('match_id')['role'].apply(lambda r: r.notna().any())
    todo = [m for m in df.loc[df['game_version'].isna(), 'match_id'].unique() if has_role.get(m, False)]
    routing_of = df.drop_duplicates('match_id').set_index('match_id')['region'].to_dict()
    print(f"🔎 Backfilling patch/date for {len(todo)} matches...")

    def save():
        df.to_csv(RAW_DATA_PATH, index=False)

    for i, m_id in enumerate(todo, 1):
        match = fetch_match(routing_of[m_id], m_id)
        if match is not None:
            info = match['info']
            mask = df['match_id'] == m_id
            df.loc[mask, 'game_version'] = patch_of(info.get('gameVersion'))
            df.loc[mask, 'game_start'] = datetime.fromtimestamp(
                info['gameCreation'] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            df.loc[mask, 'queue_id'] = info.get('queueId')
        if i % checkpoint_every == 0:
            save()
            print(f"💾 {i}/{len(todo)} backfilled")
        time.sleep(REQUEST_DELAY)
    save()
    print("✨ Backfill complete.")

def check_key():
    """Fail fast (non-zero exit) on a rejected key instead of 'succeeding' with no data."""
    try:
        WATCHER.league.challenger_by_queue('na1', QUEUE_TYPE)
    except ApiError as err:
        if err.response is not None and err.response.status_code in (401, 403):
            print(f"❌ Riot API rejected the key ({err.response.status_code}). Development keys expire "
                  "every 24h; set a valid RIOT_KEY (repo secret for the weekly workflow).")
            sys.exit(1)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # emoji logs on Windows consoles
    parser = argparse.ArgumentParser()
    parser.add_argument('--backfill', action='store_true',
                        help='Fill game_version/game_start/queue_id for previously scraped matches, then exit.')
    args = parser.parse_args()

    if not API_KEY:
        print("❌ RIOT_KEY missing in .env")
        sys.exit(1)

    # 1. Ensure directories exist and the file header matches what we write
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    ensure_raw_schema()
    check_key()

    if args.backfill:
        backfill_metadata()
        sys.exit(0)

    # 2. Check for existing work
    processed_ids = load_processed_ids()
    print(f"📂 Loaded {len(processed_ids)} previously processed matches.")
    initial_count = len(processed_ids)

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

            time.sleep(REQUEST_DELAY)

        # Final save for the remaining data in the region
        if batch_data:
            append_rows(batch_data)

    new_count = len(processed_ids) - initial_count
    print(f"✨ Automation Cycle Complete. {new_count} new match IDs processed. Data stored in {RAW_DATA_PATH}")
    if new_count == 0:
        print("⚠️ No new matches collected this run.")
