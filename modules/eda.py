import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#1. Load data
df = pd.read_csv('./data/league_match_data.csv', on_bad_lines='skip')
#2. Data Cleaning
df = df.dropna(subset=['role'])
df = df[df['role']!= '']

df['role'] = df['role'].replace('UTILITY', 'SUPPORT')

#Save cleaned data
df.to_csv('./data/cleaned_league_match_data.csv', index=False)
df.groupby('role')['champion_name'].unique().to_json('./data/champion_roles.json')