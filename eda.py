import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#1. Load data
df = pd.read_csv('league_match_data.csv')
print(df.head())


#2. data statistics
print('Data Info:')
print(df.info())
print(df.describe())
print('Missing Values:')
print(df.isna().sum()/len(df))
df = df.dropna(subset=['role'])
df = df[df['role']!= '']
print('Missing Values after dropping NA:')

print(df.isna().sum())
#3. EDA
print(df['champion_name'].value_counts())
print(df['role'].value_counts())

df['role'] = df['role'].replace('UTILITY', 'SUPPORT')
print(df['role'].value_counts())

#4. Visualizations
# Role Distribution
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='role', order=df['role'].value_counts().index)
plt.title('Distribution of Roles')
plt.xlabel('Role')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

#Top 20 Most Played Champions
top_20_names = df['champion_name'].value_counts().nlargest(20).index

# 2. Filter original df to only contain rows for these top 20 champions
top_df = df[df['champion_name'].isin(top_20_names)]

# 3. Plot using countplot with the filtered data
plt.figure(figsize=(12, 8)) # Increased size for better visibility
sns.countplot(
    data=top_df, 
    y='champion_name', 
    order=top_20_names, # Ensures consistent ordering
    hue='role',
    palette='viridis',
    dodge=False
)

plt.title('Top 20 Most Played Champions by Role')
plt.xlabel('Number of Games')
plt.ylabel('Champion Name')
plt.tight_layout() # Adjusts layout to fit labels
plt.show()

# Gold Earned Distribution
plt.figure(figsize=(10,6))
sns.histplot(df['gold_earned'], bins=30, kde=True)
plt.title('Distribution of Gold Earned')
plt.xlabel('Gold Earned')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Damage to Champions Distribution
plt.figure(figsize=(10,6))
sns.histplot(df['damage_to_champs'], bins=30, kde=True)
plt.title('Distribution of Damage to Champions')
plt.xlabel('Damage to Champions')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Kills vs Deaths
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='kills', y='deaths', hue='role', alpha=0.6)
plt.title('Kills vs Deaths by Role')
plt.xlabel('Kills')
plt.ylabel('Deaths')
plt.legend(title='Role')
plt.tight_layout()
plt.show()


#5. Correlation Analysis
numeric_cols = ['kills', 'deaths', 'assists', 'damage_to_champs', 'gold_earned']
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.show()



#Save cleaned data
df.to_csv('cleaned_league_match_data.csv', index=False)