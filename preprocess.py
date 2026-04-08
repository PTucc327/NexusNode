import pandas as pd
from itertools import combinations
import json

# 1. Load the raw data
df = pd.read_csv('league_match_data.csv')

def generate_graph_data(df):
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
    winning_teams = df[df['win'] == True].groupby(['match_id', 'team_id'])['champion_name'].apply(list)
    
    edge_list = []
    for team in winning_teams:
        for pair in combinations(sorted(team), 2):
            edge_list.append(pair)
            
    edges = pd.DataFrame(edge_list, columns=['source', 'target'])
    edge_weights = edges.groupby(['source', 'target']).size().reset_index(name='weight')

    print("Cleaning Role Mapping...")
    # --- STEP 3: ROLE MAPPING (With Threshold) ---
    # We count how many times each champ is played in each role
    role_counts = df.groupby(['role', 'champion_name']).size().reset_index(name='count')
    
    # We only keep a champion for a role if they've been played there at least 2 times 
    # (Adjust this number as your dataset grows)
    meta_only = role_counts[role_counts['count'] >= 2]
    
    # Group them into a dictionary: { "TOP": ["Aatrox", "Camille"], "JUNGLE": [...] }
    role_mapping = meta_only.groupby('role')['champion_name'].apply(list).to_dict()
    
    # --- STEP 4: SAVE FILES ---
    nodes.to_csv('champion_nodes.csv', index=False)
    edge_weights.to_csv('champion_edges.csv', index=False)
    
    # Save the role mapping using the standard json library
    with open('champion_roles.json', 'w') as f:
        json.dump(role_mapping, f)
        
    print(f"Done! Created {len(nodes)} nodes, {len(edge_weights)} edges, and saved role roles.")

if __name__ == "__main__":
    generate_graph_data(df)