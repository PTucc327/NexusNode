import pandas as pd
import os
import json
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def generate_graph_data():
    # HANDOFF POINT: Read from EDA output
    input_path = './data/processed/cleaned_league_match_data.csv'
    output_nodes = './data/processed/champion_nodes.csv'
    output_edges = './data/processed/champion_edges.csv'
    output_roles = './data/processed/champion_roles.json'

    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    
    print("Processing nodes (Champion Stats)...")
    # --- STEP 1: CREATE NODES ---
    nodes = df.groupby('champion_name').agg({
        'kills': 'mean',
        'deaths': 'mean',
        'assists': 'mean',
        'damage_to_champs': 'mean',
        'gold_earned': 'mean',
        'win': 'mean'
    }).reset_index()
    
    nodes.columns = [
        'champion_name', 'avg_kills', 'avg_deaths', 
        'avg_assists', 'avg_damage', 'avg_gold', 'win_rate'
    ]
    
    print("Processing edges (Champion Synergy)...")
    # --- STEP 2: CREATE EDGES ---
    # We build edges based on winning team compositions
    winning_teams = df[df['win'] == True].groupby(['match_id', 'team_id'])['champion_name'].apply(list)
    
    edge_list = []
    for team in winning_teams:
        # Sort to ensure (A, B) is same as (B, A) for undirected graph
        for pair in combinations(sorted(team), 2):
            edge_list.append(pair)
            
    edges = pd.DataFrame(edge_list, columns=['source', 'target'])
    edge_weights = edges.groupby(['source', 'target']).size().reset_index(name='weight')

    print("Cleaning Role Mapping...")
    # --- STEP 3: ROLE MAPPING ---
    # Count games per (champion, role) and total games per champion
    role_counts = df.groupby(['champion_name', 'role']).size().unstack(fill_value=0)
    total_games = role_counts.sum(axis=1)

    # A champ is "eligible" for a role only if:
    #   (a) that role makes up a meaningful share of their games (>15%), AND
    #   (b) there's enough sample size to trust it (>=10 games in that role)
    # This filters out one-off troll picks / autofills (e.g. Aatrox bot,
    # Ezreal support) that an absolute ">=2 games" threshold let through.
    MIN_ROLE_SHARE = 0.15
    MIN_ROLE_GAMES = 10

    role_mapping = {}
    for role in role_counts.columns:
        share = role_counts[role] / total_games
        mask = (share > MIN_ROLE_SHARE) & (role_counts[role] >= MIN_ROLE_GAMES)
        role_mapping[role] = role_counts.index[mask].tolist()
    
    # --- STEP 4: FEATURE SCALING ---
    # Include win_rate alongside the combat stats -- it's a meaningful signal
    # about a champion's current power level that was previously computed but
    # never actually fed into anything.
    features = ['avg_kills', 'avg_deaths', 'avg_assists', 'avg_damage', 'avg_gold', 'win_rate']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(nodes[features])

    # Save ALL scaled features as the GNN's node input (feat_0..feat_5).
    # Previously these were compressed down to a 2D PCA projection *before*
    # being handed to the GNN, which throws away most of the signal --
    # PCA's job here should only be to produce a 2D projection for plotting,
    # not to be the model's actual input. Let the GCN layers do the
    # dimensionality reduction/embedding themselves from the full feature set.
    for i, feat_name in enumerate(features):
        nodes[f'feat_{feat_name}'] = scaled_features[:, i]

    # --- STEP 4b: PCA FOR VISUALIZATION ONLY (not model input) ---
    pca = PCA(n_components=2)
    nodes_pca = pca.fit_transform(scaled_features)
    nodes['pca_x'] = nodes_pca[:, 0]
    nodes['pca_y'] = nodes_pca[:, 1]
    
    # --- STEP 5: SAVE FILES ---
    os.makedirs('./data/processed', exist_ok=True)
    nodes.to_csv(output_nodes, index=False)
    edge_weights.to_csv(output_edges, index=False)
    
    with open(output_roles, 'w') as f:
        json.dump(role_mapping, f)
        
    print(f"✨ Graph Ready! Nodes: {len(nodes)}, Edges: {len(edge_weights)}")

if __name__ == "__main__":
    generate_graph_data()