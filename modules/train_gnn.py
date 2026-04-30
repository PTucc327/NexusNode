import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
nodes_df = pd.read_csv('./data/champion_nodes.csv')
edges_df = pd.read_csv('./data/champion_edges.csv')

# 2. MAP IDs
champ_to_id = {name: i for i, name in enumerate(nodes_df['champion_name'])}
id_to_champ = {i: name for name, i in champ_to_id.items()}

# 3. PREPARE FEATURES (With Scaling)
features = nodes_df[['pca_x', 'pca_y']].values
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

model = NexusGNN(num_features=2, embedding_size=64)

# ADJUSTMENT: Lower learning rate and higher weight_decay to prevent clumping
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# 6. THE TRAINING LOOP
for epoch in range(1001): 
    model.train()
    optimizer.zero_grad()
    z = model(x, edge_index)
    
    # 1. Positive Samples (Actual winning teammates)
    # Binary Cross Entropy with Logits for similarity
    pos_score = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
    
    # 2. Negative Samples (Champions who didn't play together)
    # We generate MORE negative samples than positive ones to force separation
    neg_edge_index = negative_sampling(edge_index, num_nodes=z.size(0), num_neg_samples=edge_index.size(1) * 2)
    neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
    neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()

    # ADJUSTMENT: Increase negative weight to 3.0 to "unfold" the space
    loss = pos_loss + (3.0 * neg_loss)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        # Debugging: Monitor similarity between two random nodes to ensure it's not 1.0
        with torch.no_grad():
            sample_sim = F.cosine_similarity(z[0].unsqueeze(0), z[1].unsqueeze(0)).item()
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | Sample Sim: {sample_sim:.4f}")

# 7. SAVE TRAINED EMBEDDINGS
model.eval()
with torch.no_grad():
    final_embeddings = model(x, edge_index)

# Convert to standard float to avoid Streamlit float32 errors later
champion_embeddings = {id_to_champ[i]: final_embeddings[i].numpy() for i in range(len(nodes_df))}
torch.save(champion_embeddings, './data/champion_embeddings.pt')
print("✅ Brain Trained & Saved with Spread-Optimization!")