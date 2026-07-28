import pandas as pd
import os

def clean_data():
    # Define paths based on your new directory structure
    input_path = './data/raw/league_match_data.csv'
    output_path = './data/processed/cleaned_league_match_data.csv'

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

    # NOTE: champion_roles.json is intentionally NOT written here anymore.
    # A prior version wrote an unfiltered role mapping (any champ who ever
    # played a role, even once, qualified) which got overwritten later by
    # preprocess.py's properly-thresholded version -- but only because of
    # pipeline ordering. That made preprocess.py's output fragile: running
    # eda.py on its own (or reordering the pipeline) silently regenerated
    # the broken, unfiltered file. preprocess.py is now the single owner
    # of champion_roles.json.

    print(f"✨ Data Science Transformation Complete.")
    print(f"   - Processed {len(df)} valid match-player rows.")

if __name__ == '__main__':
    clean_data()