import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# 1. LOAD DATA
nodes_df = pd.read_csv('champion_nodes.csv')
edges_df = pd.read_csv('champion_edges.csv')

# 2. CREATE MAPPINGS (Names to IDs)
# We need integers for the GNN to process nodes
champ_to_id = {name: i for i, name in enumerate(nodes_df['champion_name'])}
id_to_champ = {i: name for name, i in champ_to_id.items()}

# 3. PREPARE NODE FEATURES (X)
# We take the stats we calculated (kills, deaths, assists, damage, gold)
# We exclude the name and win_rate to keep features purely performance-based
features = nodes_df[['avg_kills', 'avg_deaths', 'avg_assists', 'avg_damage', 'avg_gold']].values
x = torch.tensor(features, dtype=torch.float)

# 4. PREPARE EDGES (Edge Index)
edge_list = []
for _, row in edges_df.iterrows():
    u = champ_to_id[row['source']]
    v = champ_to_id[row['target']]
    edge_list.append([u, v])
    edge_list.append([v, u]) # Graphs are undirected in LoL synergy

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# 5. DEFINE THE GNN MODEL
class NexusGNN(torch.nn.Module):
    def __init__(self, num_features, embedding_size):
        super(NexusGNN, self).__init__()
        # Layer 1: Aggregates features from immediate teammates
        self.conv1 = GCNConv(num_features, 32)
        # Layer 2: Aggregates features from "teammates of teammates"
        self.conv2 = GCNConv(32, embedding_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 6. RUN THE MODEL TO GET EMBEDDINGS
model = NexusGNN(num_features=5, embedding_size=64)
model.eval()

with torch.no_grad():
    embeddings = model(x, edge_index)

# 7. SAVE THE RESULTS
# Create a dictionary of {ChampionName: Vector}
champion_embeddings = {id_to_champ[i]: embeddings[i].numpy() for i in range(len(nodes_df))}

# Save to a file for the Web App
torch.save(champion_embeddings, 'champion_embeddings.pt')
print("✅ Success! Champion Embeddings generated and saved to 'champion_embeddings.pt'")