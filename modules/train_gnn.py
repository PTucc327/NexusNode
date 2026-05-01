import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.preprocessing import StandardScaler
import os

# 1. LOAD DATA (Updated to use the 'processed' folder)
def train_model():
    nodes_path = './data/processed/champion_nodes.csv'
    edges_path = './data/processed/champion_edges.csv'
    output_path = './champion_embeddings.pt' # Keep in root for app.py to find easily

    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        print("❌ Error: Processed graph data not found. Run preprocess.py first.")
        return

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    # 2. MAP IDs
    champ_to_id = {name: i for i, name in enumerate(nodes_df['champion_name'])}
    id_to_champ = {i: name for name, i in champ_to_id.items()}

    # 3. PREPARE FEATURES
    features = nodes_df[['pca_x', 'pca_y']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    x = torch.tensor(features_scaled, dtype=torch.float)

    # 4. PREPARE EDGES
    edge_list = []
    for _, row in edges_df.iterrows():
        if row['source'] in champ_to_id and row['target'] in champ_to_id:
            u, v = champ_to_id[row['source']], champ_to_id[row['target']]
            edge_list.append([u, v])
            edge_list.append([v, u])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 5. DEFINE GNN
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # 6. TRAINING LOOP
    print("🧠 Training NexusNode Brain...")
    for epoch in range(1001): 
        model.train()
        optimizer.zero_grad()
        z = model(x, edge_index)
        
        pos_score = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
        
        neg_edge_index = negative_sampling(edge_index, num_nodes=z.size(0), num_neg_samples=edge_index.size(1) * 2)
        neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()

        loss = pos_loss + (3.0 * neg_loss)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"   Epoch {epoch:4d} | Loss: {loss.item():.4f}")

    # 7. SAVE EMBEDDINGS
    model.eval()
    with torch.no_grad():
        final_embeddings = model(x, edge_index)

    # Save to a dictionary for easy lookups in app.py
    # We move it to numpy to ensure compatibility across different torch versions
    champion_embeddings = {id_to_champ[i]: final_embeddings[i].numpy() for i in range(len(nodes_df))}
    torch.save(champion_embeddings, output_path)
    print(f"✅ Training Complete. Embeddings saved to {output_path}")

if __name__ == "__main__":
    train_model()