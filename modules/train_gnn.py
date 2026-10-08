"""
NexusNode draft model.

A relational GNN learns champion embeddings from a graph with TWO edge types:
  - synergy edges: champions that appeared on the same team
  - counter edges: champions that appeared on opposite teams
and is trained end-to-end on real match outcomes through an explicit
draft-scoring head. For a draft A vs B (tokens t = champion embedding + role
embedding):

  logit P(A wins) = side_bias
                  + Σ_{i∈A} w·t_i            - Σ_{j∈B} w·t_j            (individual power)
                  + Σ_{i<i'∈A} t_iᵀ S t_i'   - Σ_{j<j'∈B} t_jᵀ S t_j'   (ally synergy, S symmetric)
                  + Σ_{i∈A, j∈B} t_iᵀ K t_j                              (every enemy interaction, K antisymmetric)
                  + Σ_{lanes r}  t_Aᵣᵀ L t_Bᵣ                            (extra lane-opponent term, L antisymmetric)

K and L being antisymmetric makes the model consistent: swapping the teams
exactly flips the prediction. Every term is additive per champion, so the
engine can score a candidate pick against partial drafts and explain each
ally/enemy contribution.
"""
import sys
import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.preprocess import build_node_features  # noqa: E402

ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'SUPPORT']
SEED = 7
# Hyperparameters chosen by 3-fold CV over embed dim {8,16}, weight decay
# {1e-2,5e-2}, dropout {0.3,0.5}. Draft-only signal is weak (~2.5k matches),
# so a small, heavily regularized model generalizes best.
VAL_FRACTION = 0.2
EMBED_DIM = 8
ID_EMBED_DIM = 16
MAX_EPOCHS = 300
PATIENCE = 40
LR = 3e-3
WEIGHT_DECAY = 5e-2
DROPOUT = 0.5
BLUE_TEAM_ID = 100

INPUT_PATH = './data/processed/cleaned_league_match_data.csv'
OUTPUT_PATH = './data/processed/nexus_model.pt'
METRICS_PATH = './data/processed/model_metrics.json'


# --- 1. DATA ---------------------------------------------------------------
def build_match_tensors(df, champ_to_id):
    """Returns (blue_ids [M,5], red_ids [M,5], blue_won [M]) ordered by ROLES."""
    role_idx = {r: i for i, r in enumerate(ROLES)}
    blue, red, y = [], [], []
    for _, match in df.groupby('match_id'):
        sides = {}
        for tid, team in match.groupby('team_id'):
            ids = [0] * len(ROLES)
            for row in team.itertuples():
                ids[role_idx[row.role]] = champ_to_id[row.champion_name]
            sides[tid] = (ids, bool(team['win'].iloc[0]))
        if BLUE_TEAM_ID not in sides or len(sides) != 2:
            continue
        (red_tid,) = [t for t in sides if t != BLUE_TEAM_ID]
        blue.append(sides[BLUE_TEAM_ID][0])
        red.append(sides[red_tid][0])
        y.append(float(sides[BLUE_TEAM_ID][1]))
    return torch.tensor(blue), torch.tensor(red), torch.tensor(y)


def build_graph(blue, red, num_nodes):
    """Edge index/weights for the two relations from a set of matches.
    Weights are log-scaled co-occurrence counts so popular champions
    don't drown out everyone else."""
    def to_edges(pairs):
        counts = {}
        for u, v in pairs:
            counts[(u, v)] = counts.get((u, v), 0) + 1
            counts[(v, u)] = counts.get((v, u), 0) + 1
        if not counts:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0)
        idx = torch.tensor(list(counts.keys()), dtype=torch.long).t().contiguous()
        w = torch.log1p(torch.tensor(list(counts.values()), dtype=torch.float))
        return idx, w / w.max()

    synergy_pairs, counter_pairs = [], []
    for team_a, team_b in zip(blue.tolist(), red.tolist()):
        for team in (team_a, team_b):
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    synergy_pairs.append((team[i], team[j]))
        for a in team_a:
            for b in team_b:
                counter_pairs.append((a, b))
    return to_edges(synergy_pairs), to_edges(counter_pairs)


# --- 2. MODEL --------------------------------------------------------------
class RelationalGCNLayer(nn.Module):
    """One GCN per relation (synergy / counter) plus a self transform."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.synergy = GCNConv(in_dim, out_dim, add_self_loops=False)
        self.counter = GCNConv(in_dim, out_dim, add_self_loops=False)
        self.self_loop = nn.Linear(in_dim, out_dim)

    def forward(self, x, graph):
        (syn_idx, syn_w), (ctr_idx, ctr_w) = graph
        return self.self_loop(x) + self.synergy(x, syn_idx, syn_w) + self.counter(x, ctr_idx, ctr_w)


class NexusDraftModel(nn.Module):
    def __init__(self, num_champs, num_features, embed_dim=EMBED_DIM, use_enemy_terms=True):
        super().__init__()
        self.use_enemy_terms = use_enemy_terms
        self.id_embed = nn.Embedding(num_champs, ID_EMBED_DIM)
        self.conv1 = RelationalGCNLayer(num_features + ID_EMBED_DIM, 2 * embed_dim)
        self.conv2 = RelationalGCNLayer(2 * embed_dim, embed_dim)
        self.role_embed = nn.Parameter(torch.zeros(len(ROLES), embed_dim))
        self.power = nn.Parameter(torch.zeros(embed_dim))
        self.synergy_raw = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        self.counter_raw = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        self.lane_raw = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        self.side_bias = nn.Parameter(torch.zeros(()))

    # Constrained interaction matrices
    def synergy_matrix(self):
        return (self.synergy_raw + self.synergy_raw.T) / 2

    def counter_matrix(self):
        return self.counter_raw - self.counter_raw.T

    def lane_matrix(self):
        return self.lane_raw - self.lane_raw.T

    def embed(self, x, graph):
        ids = torch.arange(x.size(0))
        h = torch.cat([x, self.id_embed(ids)], dim=-1)
        h = F.relu(self.conv1(h, graph))
        h = F.dropout(h, DROPOUT, self.training)
        return self.conv2(h, graph)

    def team_score(self, t):
        """Power + within-team synergy for tokens t [M,5,D]."""
        power = (t @ self.power).sum(dim=1)
        pair = torch.einsum('mid,de,mje->mij', t, self.synergy_matrix(), t)
        synergy = (pair.sum(dim=(1, 2)) - pair.diagonal(dim1=1, dim2=2).sum(dim=1)) / 2
        return power + synergy

    def forward(self, z, blue, red):
        t_blue = F.dropout(z[blue] + self.role_embed, DROPOUT, self.training)
        t_red = F.dropout(z[red] + self.role_embed, DROPOUT, self.training)
        logit = self.side_bias + self.team_score(t_blue) - self.team_score(t_red)
        if self.use_enemy_terms:
            cross = torch.einsum('mid,de,mje->mij', t_blue, self.counter_matrix(), t_red)
            lane = torch.einsum('mid,de,mie->mi', t_blue, self.lane_matrix(), t_red)
            logit = logit + cross.sum(dim=(1, 2)) + lane.sum(dim=1)
        return logit


# --- 3. TRAINING -----------------------------------------------------------
def node_feature_tensor(df, champions):
    nodes, _ = build_node_features(df)
    feats = nodes.set_index('champion_name')[[c for c in nodes.columns if c.startswith('feat_')]]
    feats = feats.reindex(champions).fillna(0.0)  # unseen-in-split champs get neutral features
    return torch.tensor(feats.values, dtype=torch.float)


def evaluate(logits, y):
    p = torch.sigmoid(logits).numpy()
    yt = y.numpy()
    return {
        'log_loss': float(F.binary_cross_entropy_with_logits(logits, y)),
        'accuracy': float(((p > 0.5) == yt).mean()),
        'auc': float(roc_auc_score(yt, p)),
    }


def fit(model, x, graph, blue, red, y, epochs, val=None):
    """Trains full-batch. With `val`, early-stops on validation log loss and
    returns (best_epoch, best_metrics); otherwise trains exactly `epochs`."""
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best = (0, None, float('inf'))
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(model.embed(x, graph), blue, red), y)
        loss.backward()
        optimizer.step()

        if val is not None:
            model.eval()
            with torch.no_grad():
                v_blue, v_red, v_y = val
                metrics = evaluate(model(model.embed(x, graph), v_blue, v_red), v_y)
            if metrics['log_loss'] < best[2]:
                best = (epoch, metrics, metrics['log_loss'])
            elif epoch - best[0] >= PATIENCE:
                break
    return best[0], best[1]


def train_model():
    if not os.path.exists(INPUT_PATH):
        print("❌ Error: Cleaned match data not found. Run eda.py and preprocess.py first.")
        return

    torch.manual_seed(SEED)
    df = pd.read_csv(INPUT_PATH)
    champions = sorted(df['champion_name'].unique())
    champ_to_id = {c: i for i, c in enumerate(champions)}

    blue, red, y = build_match_tensors(df, champ_to_id)
    match_ids = np.array(sorted(df['match_id'].unique()))
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(SEED))
    n_val = int(len(y) * VAL_FRACTION)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"📊 {len(y)} matches · {len(champions)} champions · train {len(train_idx)} / val {len(val_idx)}")

    # Graph + node features from the TRAINING split only (no validation leakage)
    train_match_ids = set(match_ids[train_idx.numpy()])
    train_df = df[df['match_id'].isin(train_match_ids)]
    x_train = node_feature_tensor(train_df, champions)
    graph_train = build_graph(blue[train_idx], red[train_idx], len(champions))
    val = (blue[val_idx], red[val_idx], y[val_idx])

    # --- Validation: full model vs. ablation without enemy terms vs. baselines
    print("🧠 Training NexusNode draft model (validation run)...")
    results = {}
    for name, use_enemy in [('full_model', True), ('allies_only_ablation', False)]:
        torch.manual_seed(SEED)
        m = NexusDraftModel(len(champions), x_train.shape[1], use_enemy_terms=use_enemy)
        best_epoch, metrics = fit(m, x_train, graph_train, blue[train_idx], red[train_idx], y[train_idx],
                                  MAX_EPOCHS, val=val)
        results[name] = {**metrics, 'best_epoch': best_epoch}
        print(f"   {name:22s} | val log loss {metrics['log_loss']:.4f} | acc {metrics['accuracy']:.3f} "
              f"| AUC {metrics['auc']:.3f} | epoch {best_epoch}")

    base_rate = y[train_idx].mean()
    const_logits = torch.full_like(val[2], float(torch.logit(base_rate)))
    results['baseline_side_only'] = evaluate(const_logits, val[2])
    print(f"   {'baseline_side_only':22s} | val log loss {results['baseline_side_only']['log_loss']:.4f} "
          f"| acc {results['baseline_side_only']['accuracy']:.3f}")

    # --- Final fit on ALL matches for the selected number of epochs
    print("🔁 Refitting on all matches...")
    x_all = node_feature_tensor(df, champions)
    graph_all = build_graph(blue, red, len(champions))
    torch.manual_seed(SEED)
    model = NexusDraftModel(len(champions), x_all.shape[1])
    fit(model, x_all, graph_all, blue, red, y, max(results['full_model']['best_epoch'], 1))

    model.eval()
    with torch.no_grad():
        z = model.embed(x_all, graph_all)
        artifact = {
            'format_version': 2,
            'champions': champions,
            'roles': ROLES,
            'embeddings': z.clone(),
            'role_embeddings': model.role_embed.detach().clone(),
            'power': model.power.detach().clone(),
            'synergy': model.synergy_matrix().detach().clone(),
            'counter': model.counter_matrix().detach().clone(),
            'lane': model.lane_matrix().detach().clone(),
        }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(artifact, OUTPUT_PATH)
    metrics_out = {
        'trained_at': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'matches': int(len(y)),
        'champions': len(champions),
        'validation_matches': int(n_val),
        'validation': results,
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"✅ Training Complete. Model saved to {OUTPUT_PATH}, metrics to {METRICS_PATH}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # emoji logs on Windows consoles
    train_model()
