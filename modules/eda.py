import pandas as pd
import os
import json

def clean_data():
    # Define paths based on your new directory structure
    input_path = './data/raw/league_match_data.csv'
    output_path = './data/processed/cleaned_league_match_data.csv'
    roles_path = './data/processed/champion_roles.json'

    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found. Ensure collector has run.")
        return

    # 1. Load data
    df = pd.read_csv(input_path, on_bad_lines='skip')
    
    # 2. Data Cleaning
    # Drop rows where role is NaN or empty string
    df = df.dropna(subset=['role'])
    df = df[df['role'] != '']
    
    # Standardize Role Names (The Support Fix)
    df['role'] = df['role'].replace('UTILITY', 'SUPPORT')
    
    # 3. Save Cleaned Data for the GNN Trainer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # 4. Update UI Metadata (champion_roles.json)
    # This keeps your Streamlit dropdowns accurate to the current patch
    role_grouping = df.groupby('role')['champion_name'].unique().apply(list).to_dict()
    with open(roles_path, 'w') as f:
        json.dump(role_grouping, f)

    print(f"✨ Data Science Transformation Complete.")
    print(f"   - Processed {len(df)} valid match-player rows.")
    print(f"   - Roles updated: {list(role_grouping.keys())}")

if __name__ == '__main__':
    clean_data()