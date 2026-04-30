import pandas as pd
import json

df = pd.read_csv('league_match_data.csv')

# Calculate how often each champion plays each role
role_counts = df.groupby(['champion_name', 'role']).size().unstack(fill_value=0)

# Filter: Only keep a champion in a role if they play it in >15% of their games
primary_roles = {}
for role in role_counts.columns:
    # Get total games per champion to calculate percentage
    total_games = role_counts.sum(axis=1)
    # Filter for the current role
    mask = (role_counts[role] / total_games) > 0.15
    primary_roles[role] = role_counts.index[mask].tolist()

with open('champion_roles.json', 'w') as f:
    json.dump(primary_roles, f)

print("✨ Cleaned champion_roles.json created!")