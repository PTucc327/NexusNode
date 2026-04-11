import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
nodes_df = pd.read_csv('champion_nodes.csv')
edges_df = pd.read_csv('champion_edges.csv')

# 2. MAP IDs
champ_to_id = {name: i for i, name in enumerate(nodes_df['champion_name'])}
id_to_champ = {i: name for name, i in champ_to_id.items()}

# 3. PREPARE FEATURES (With Scaling)
# Machine learning models hate raw numbers like "15000 gold" mixed with "5 kills"
features = nodes_df[['avg_kills', 'avg_deaths', 'avg_assists', 'avg_damage', 'avg_gold']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
x = torch.tensor(features_scaled, dtype=torch.float)

# 4. PREPARE EDGES
edge_list = []
for _, row in edges_df.iterrows():
    u, v = champ_to_id[row['source']], champ_to_id[row['target']]
    edge_list.append([u, v])
    edge_list.append([v, u])
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# 5. DEFINE TRAINABLE GNN
class NexusGNN(torch.nn.Module):
    def __init__(self, num_features, embedding_size):
        super(NexusGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, embedding_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = NexusGNN(num_features=5, embedding_size=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 6. THE TRAINING LOOP (This is the missing "Learning" part)
# We train the model to make sure teammates have similar vectors
model.train()
for epoch in range(100): # 100 rounds of learning
    optimizer.zero_grad()
    z = model(x, edge_index)
    
    # We want the dot product of connected nodes to be high
    # This is a basic "Link Prediction" loss
    edge_pos_score = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    loss = -torch.log(torch.sigmoid(edge_pos_score)).mean()
    
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# 7. SAVE TRAINED EMBEDDINGS
model.eval()
with torch.no_grad():
    final_embeddings = model(x, edge_index)

champion_embeddings = {id_to_champ[i]: final_embeddings[i].numpy() for i in range(len(nodes_df))}
torch.save(champion_embeddings, 'champion_embeddings.pt')
print("✅ Brain Trained & Saved!")