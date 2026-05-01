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
    role_counts = df.groupby(['role', 'champion_name']).size().reset_index(name='count')
    # Filter meta: Champ must appear in role at least 2 times
    meta_only = role_counts[role_counts['count'] >= 2]
    role_mapping = meta_only.groupby('role')['champion_name'].apply(list).to_dict()
    
    # --- STEP 4: PCA FOR VISUALIZATION ---
    features = ['avg_kills', 'avg_deaths', 'avg_assists', 'avg_damage', 'avg_gold']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(nodes[features])
    
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