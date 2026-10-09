import sys
import pandas as pd
import os

ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'SUPPORT']
RANKED_SOLO_QUEUE_ID = 420

def clean_data(input_path='./data/raw/league_match_data.csv',
               output_path='./data/processed/cleaned_league_match_data.csv'):
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found. Ensure collector has run.")
        return

    # 1. Load data (warn rather than silently skip malformed lines)
    # game_version as text: pandas would parse '16.20' as the float 16.2
    df = pd.read_csv(input_path, on_bad_lines='warn', dtype={'game_version': str})
    raw_rows = len(df)

    # 2. Data Cleaning
    # The collector can re-append a match if a run is interrupted mid-batch
    df = df.drop_duplicates(subset=['match_id', 'team_id', 'champion_name'])

    # Ranked Solo only. The original scraper had no queue filter, and other
    # role-assigned queues (e.g. queue 3130) slipped through; matches with no
    # recorded queue can't be verified, so they're dropped too.
    if 'queue_id' in df.columns:
        non_ranked = df.loc[df['queue_id'] != RANKED_SOLO_QUEUE_ID, 'match_id'].nunique()
        df = df[df['queue_id'] == RANKED_SOLO_QUEUE_ID]
        print(f"   - Dropped {non_ranked} matches that aren't (or can't be verified as) Ranked Solo.")

    # Drop rows where role is NaN or empty string
    df = df.dropna(subset=['role'])
    df = df[df['role'] != '']

    # Standardize Role Names (The Support Fix)
    df['role'] = df['role'].replace('UTILITY', 'SUPPORT')
    df = df[df['role'].isin(ROLES)]

    # Keep only complete drafts: 10 players, each team filling all 5 roles
    # once, exactly one winning team. The outcome model trains on whole
    # 5v5 drafts, and the lane matchup table assumes one player per side.
    def is_complete(g):
        if len(g) != 10:
            return False
        per_team = g.groupby('team_id')
        return (
            per_team.ngroups == 2
            and per_team['role'].nunique().eq(5).all()
            and per_team['win'].nunique().eq(1).all()
            and g.groupby('team_id')['win'].first().sum() == 1
        )
    valid_ids = [mid for mid, g in df.groupby('match_id') if is_complete(g)]
    df = df[df['match_id'].isin(valid_ids)]

    # 3. Save Cleaned Data for the GNN Trainer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # NOTE: champion_roles.json is owned by preprocess.py (thresholded role
    # eligibility), so it is intentionally not written here.

    print("✨ Data Science Transformation Complete.")
    print(f"   - Processed {len(df)} valid match-player rows ({len(valid_ids)} complete matches, "
          f"{raw_rows - len(df)} raw rows dropped).")

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding="utf-8")  # emoji logs on Windows consoles
    clean_data()
