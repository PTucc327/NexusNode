import pandas as pd
from itertools import combinations

# 1. Load the raw data you collected
df = pd.read_csv('league_match_data.csv')

def generate_graph_data(df):
    print("Processing nodes (Champion Stats)...")
    # --- STEP 1: CREATE NODES ---
    # We group by champion to get their average performance "fingerprint"
    nodes = df.groupby('champion_name').agg({
        'kills': 'mean',
        'deaths': 'mean',
        'assists': 'mean',
        'damage_to_champs': 'mean',
        'gold_earned': 'mean',
        'win': 'mean'
    }).reset_index()
    
    # Rename columns for clarity
    nodes.columns = [
        'champion_name', 'avg_kills', 'avg_deaths', 
        'avg_assists', 'avg_damage', 'avg_gold', 'win_rate'
    ]
    
    print("Processing edges (Champion Synergy)...")
    # --- STEP 2: CREATE EDGES ---
    # We only look at winning teams to define "Synergy"
    winning_teams = df[df['win'] == True].groupby(['match_id', 'team_id'])['champion_name'].apply(list)
    
    edge_list = []
    for team in winning_teams:
        # We sort the team alphabetically so (Alistar, Yasuo) 
        # is the same edge as (Yasuo, Alistar)
        for pair in combinations(sorted(team), 2):
            edge_list.append(pair)
            
    # Convert list of pairs into a DataFrame and count occurrences
    edges = pd.DataFrame(edge_list, columns=['source', 'target'])
    edge_weights = edges.groupby(['source', 'target']).size().reset_index(name='weight')
    
    # --- STEP 3: SAVE FILES ---
    nodes.to_csv('champion_nodes.csv', index=False)
    edge_weights.to_csv('champion_edges.csv', index=False)
    
    print(f"Done! Created {len(nodes)} nodes and {len(edge_weights)} edges.")

if __name__ == "__main__":
    generate_graph_data(df)