import sys
import pandas as pd
import os
import json
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Combat/power stats used as GNN node input features
NODE_FEATURES = ['avg_kills', 'avg_deaths', 'avg_assists', 'avg_damage', 'avg_gold', 'win_rate']

def build_node_features(df):
    """Per-champion averages plus StandardScaled `feat_*` columns. Shared with
    train_gnn.py so the model can rebuild features from its training split
    only (keeps validation matches out of the win_rate feature)."""
    nodes = df.groupby('champion_name').agg({
        'kills': 'mean',
        'deaths': 'mean',
        'assists': 'mean',
        'damage_to_champs': 'mean',
        'gold_earned': 'mean',
        'win': 'mean'
    }).reset_index()
    nodes.columns = ['champion_name'] + NODE_FEATURES
    nodes['games'] = df.groupby('champion_name').size().values

    scaled = StandardScaler().fit_transform(nodes[NODE_FEATURES])
    for i, feat_name in enumerate(NODE_FEATURES):
        nodes[f'feat_{feat_name}'] = scaled[:, i]
    return nodes, scaled

def build_damage_profiles(df):
    """Per-champion damage TYPE from real games: games observed and the share
    of physical vs. magic damage dealt to champions (true damage excluded, as
    it's neither). None if the data predates damage-type tracking."""
    cols = ['physical_damage_to_champs', 'magic_damage_to_champs']
    if not set(cols) <= set(df.columns):
        return None
    d = df.dropna(subset=cols)
    if d.empty:
        return None
    table = d.groupby('champion_name').agg(
        games=('physical_damage_to_champs', 'size'),
        physical=('physical_damage_to_champs', 'sum'),
        magic=('magic_damage_to_champs', 'sum'),
    ).reset_index()
    table['ad_share'] = table['physical'] / (table['physical'] + table['magic']).clip(lower=1)
    return table[['champion_name', 'games', 'ad_share']]


def build_pair_tables(df):
    """Teammate (synergy) and opponent (counter) pair records for every match.

    synergy:  unordered teammate pairs -> games together, wins together
    counters: ordered (champion, opponent) on opposite teams, any roles ->
              games faced, wins for `champion`
    """
    synergy, counters = [], []
    for _, match in df.groupby('match_id'):
        teams = {tid: t for tid, t in match.groupby('team_id')}
        if len(teams) != 2:
            continue
        (tid_a, team_a), (tid_b, team_b) = teams.items()
        for team in (team_a, team_b):
            won = bool(team['win'].iloc[0])
            for a, b in combinations(sorted(team['champion_name']), 2):
                synergy.append((a, b, won))
        a_won = bool(team_a['win'].iloc[0])
        for ca in team_a['champion_name']:
            for cb in team_b['champion_name']:
                counters.append((ca, cb, a_won))
                counters.append((cb, ca, not a_won))

    synergy_df = pd.DataFrame(synergy, columns=['source', 'target', 'win'])
    synergy_table = synergy_df.groupby(['source', 'target']).agg(
        games=('win', 'size'), wins=('win', 'sum'), win_rate=('win', 'mean')
    ).reset_index()

    counters_df = pd.DataFrame(counters, columns=['champion_name', 'opponent_name', 'win'])
    counter_table = counters_df.groupby(['champion_name', 'opponent_name']).agg(
        games=('win', 'size'), wins=('win', 'sum'), win_rate=('win', 'mean')
    ).reset_index()
    return synergy_table, counter_table

def generate_graph_data():
    # HANDOFF POINT: Read from EDA output
    input_path = './data/processed/cleaned_league_match_data.csv'
    output_nodes = './data/processed/champion_nodes.csv'
    output_synergy = './data/processed/champion_synergy_edges.csv'
    output_counters = './data/processed/champion_counter_edges.csv'
    output_roles = './data/processed/champion_roles.json'
    output_matchups = './data/processed/champion_matchups.csv'
    output_damage = './data/processed/champion_damage.csv'

    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)

    print("Building Matchup Table (Lane Counters)...")
    # --- STEP 3b: LANE MATCHUP TABLE ---
    # For each match+role there are exactly 2 rows (one per team) since each
    # role is filled by one player per side. Pair them up and record who won
    # to get a REAL head-to-head win rate per (champion, opponent, role).
    matchup_records = []
    for (match_id, role), group in df.groupby(['match_id', 'role']):
        if len(group) != 2:
            continue  # malformed/incomplete match data, skip
        a, b = group.iloc[0], group.iloc[1]
        if a['team_id'] == b['team_id']:
            continue  # guard against bad data (both rows same team)
        matchup_records.append((a['champion_name'], b['champion_name'], role, a['win']))
        matchup_records.append((b['champion_name'], a['champion_name'], role, b['win']))

    matchups_df = pd.DataFrame(matchup_records, columns=['champion_name', 'opponent_name', 'role', 'win'])
    matchup_table = matchups_df.groupby(['champion_name', 'opponent_name', 'role']).agg(
        games=('win', 'size'),
        win_rate=('win', 'mean')
    ).reset_index()
    # NOTE: sample sizes here are often tiny (median is a single game), so we
    # keep `games` alongside `win_rate` and only surface matchups with enough
    # games as evidence in the app.

    print("Processing nodes (Champion Stats)...")
    # --- STEP 1: CREATE NODES ---
    nodes, scaled_features = build_node_features(df)

    print("Processing edges (Teammate Synergy + Enemy Counters)...")
    # --- STEP 2: CREATE EDGES ---
    # Two relation types: champions on the same team (synergy) and champions
    # on opposite teams (counters). Both are recorded with games and wins so
    # they can serve as graph edges and as human-readable evidence.
    synergy_table, counter_table = build_pair_tables(df)

    print("Cleaning Role Mapping...")
    # --- STEP 3: ROLE MAPPING ---
    # Count games per (champion, role) and total games per champion
    role_counts = df.groupby(['champion_name', 'role']).size().unstack(fill_value=0)
    total_games = role_counts.sum(axis=1)

    # A champ is "eligible" for a role only if:
    #   (a) that role makes up a meaningful share of their games (>15%), AND
    #   (b) there's enough sample size to trust it (>=MIN_ROLE_GAMES in that role)
    # This filters out one-off troll picks / autofills (e.g. Aatrox bot,
    # Ezreal support).
    MIN_ROLE_SHARE = 0.15
    MIN_ROLE_GAMES = 20  # raised from 10 once the dataset passed ~4k matches

    role_mapping = {}
    for role in role_counts.columns:
        share = role_counts[role] / total_games
        mask = (share > MIN_ROLE_SHARE) & (role_counts[role] >= MIN_ROLE_GAMES)
        role_mapping[role] = role_counts.index[mask].tolist()

    # --- STEP 4: PCA FOR VISUALIZATION ONLY (not model input) ---
    pca = PCA(n_components=2)
    nodes_pca = pca.fit_transform(scaled_features)
    nodes['pca_x'] = nodes_pca[:, 0]
    nodes['pca_y'] = nodes_pca[:, 1]

    # --- STEP 5: SAVE FILES ---
    os.makedirs('./data/processed', exist_ok=True)
    nodes.to_csv(output_nodes, index=False)
    synergy_table.to_csv(output_synergy, index=False)
    counter_table.to_csv(output_counters, index=False)
    matchup_table.to_csv(output_matchups, index=False)
    damage_table = build_damage_profiles(df)
    if damage_table is not None:
        damage_table.to_csv(output_damage, index=False)

    with open(output_roles, 'w') as f:
        json.dump(role_mapping, f)

    print(f"✨ Graph Ready! Nodes: {len(nodes)}, Synergy edges: {len(synergy_table)}, "
          f"Counter edges: {len(counter_table)}, Lane matchups: {len(matchup_table)}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # emoji logs on Windows consoles
    generate_graph_data()
