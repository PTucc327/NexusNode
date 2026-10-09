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
from modules.composition import champion_profiles, team_features, NUM_FEATURES  # noqa: E402

ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'SUPPORT']
SEED = 7
# Hyperparameters chosen by 3-fold CV over embed dim {8,16}, weight decay
# {1e-2,5e-2}, dropout {0.3,0.5}. Draft-only signal is weak (~2.5k matches),
# so a small, heavily regularized model generalizes best.
VAL_FRACTION = 0.2
EMBED_DIM = 8
ID_EMBED_DIM = 16
MAX_EPOCHS = 300
# Release gate: the weekly job publishes the model automatically, so refuse
# to save one trained on too little data or that doesn't beat the no-draft
# baseline on the newest held-out matches (keeps the last good model live).
MIN_MATCHES = 1000
PATIENCE = 40
LR = 3e-3
WEIGHT_DECAY = 5e-2
DROPOUT = 0.5
# Prior std (logit) of a champion's strength in a specific role. Acts as
# sample-size shrinkage: a 12-game role win rate barely moves its estimate,
# a 400-game one does. Without it, a champion's results in its main role
# leaked into every other role it is occasionally played in.
ROLE_BIAS_SIGMA = 0.1
LANE_EVIDENCE_SIGMA = 0.1  # prior std (logit) of a lane matchup's effect; CV-chosen
USE_COMPOSITION = True  # time-based CV: better in all 4 seeds, mostly on the newest patch
# Recency weighting: a match N patches older than the newest counts 0.5**(N/half-life).
# None = no weighting. Chosen on a time-based validation split (8 tied 4, beat None and 2).
PATCH_HALF_LIFE = 8
USE_SHARED_POWER = False  # CV: no gain from it, and it leaks strength across roles
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


def match_metadata(df):
    """Per-match (game_start, patch) in the same order as build_match_tensors
    (groupby match_id). Missing columns -> NaN (data predating patch tracking)."""
    cols = [c for c in ('game_start', 'game_version') if c in df.columns]
    meta = df.groupby('match_id')[cols].first() if cols else pd.DataFrame(index=sorted(df['match_id'].unique()))
    return meta.reindex(columns=['game_start', 'game_version'])


PATCHES_PER_SEASON = 24  # LoL ships ~24 patches a year (e.g. 15.24 -> 16.1)


def patch_ages(versions):
    """Patches behind the newest one, by actual patch distance (gaps in the
    data still count), e.g. ['16.8', '16.20', '16.19'] -> [12, 0, 1].
    Unknown patches count as the oldest seen."""
    def ordinal(v):
        major, minor = str(v).split('.')[:2]
        return int(major) * PATCHES_PER_SEASON + int(minor)
    known = [ordinal(v) for v in versions if isinstance(v, str)]
    if not known:
        return np.zeros(len(versions))
    newest, oldest = max(known), min(known)
    return np.array([newest - ordinal(v) if isinstance(v, str) else newest - oldest for v in versions], dtype=float)


def recency_weights(versions, half_life=None):
    if half_life is None:
        return torch.ones(len(versions))
    return torch.tensor(0.5 ** (patch_ages(versions) / half_life), dtype=torch.float)


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
    def __init__(self, num_champs, num_features, embed_dim=EMBED_DIM, use_enemy_terms=True, profiles=None):
        super().__init__()
        # Team composition: champion profiles -> team features (composition.py)
        self.use_composition = USE_COMPOSITION and profiles is not None
        if self.use_composition:
            self.register_buffer('profiles', torch.as_tensor(profiles))
        self.comp_power = nn.Parameter(torch.zeros(NUM_FEATURES))
        self.comp_cross_raw = nn.Parameter(torch.randn(NUM_FEATURES, NUM_FEATURES) * 0.01)
        self.use_enemy_terms = use_enemy_terms
        self.use_shared_power = USE_SHARED_POWER
        # Strength of each (champion, role), shrunk toward 0 by bias_penalty()
        self.role_bias = nn.Embedding(num_champs * len(ROLES), 1)
        nn.init.zeros_(self.role_bias.weight)
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

    def comp_cross_matrix(self):
        return self.comp_cross_raw - self.comp_cross_raw.T

    def embed(self, x, graph):
        ids = torch.arange(x.size(0))
        h = torch.cat([x, self.id_embed(ids)], dim=-1)
        h = F.relu(self.conv1(h, graph))
        h = F.dropout(h, DROPOUT, self.training)
        return self.conv2(h, graph)

    def role_strength(self, ids):
        """Per-(champion, role) strength for ids [M,5] ordered by ROLES."""
        roles = torch.arange(len(ROLES))
        return self.role_bias(ids * len(ROLES) + roles).squeeze(-1).sum(dim=1)

    def bias_penalty(self):
        """Gaussian prior on role strengths (summed negative log prior)."""
        return (self.role_bias.weight ** 2).sum() / (2 * ROLE_BIAS_SIGMA ** 2)

    def team_score(self, t, ids):
        """Power + within-team synergy for tokens t [M,5,D]."""
        power = self.role_strength(ids)
        if self.use_shared_power:
            power = power + (t @ self.power).sum(dim=1)
        pair = torch.einsum('mid,de,mje->mij', t, self.synergy_matrix(), t)
        synergy = (pair.sum(dim=(1, 2)) - pair.diagonal(dim1=1, dim2=2).sum(dim=1)) / 2
        return power + synergy

    def forward(self, z, blue, red):
        t_blue = F.dropout(z[blue] + self.role_embed, DROPOUT, self.training)
        t_red = F.dropout(z[red] + self.role_embed, DROPOUT, self.training)
        logit = self.side_bias + self.team_score(t_blue, blue) - self.team_score(t_red, red)
        if self.use_enemy_terms:
            cross = torch.einsum('mid,de,mje->mij', t_blue, self.counter_matrix(), t_red)
            lane = torch.einsum('mid,de,mie->mi', t_blue, self.lane_matrix(), t_red)
            logit = logit + cross.sum(dim=(1, 2)) + lane.sum(dim=1)
        if self.use_composition:
            g_blue, g_red = team_features(self.profiles[blue]), team_features(self.profiles[red])
            logit = logit + (g_blue - g_red) @ self.comp_power
            if self.use_enemy_terms:
                logit = logit + ((g_blue @ self.comp_cross_matrix()) * g_red).sum(dim=-1)
        return logit


# --- 3. LANE MATCHUP EVIDENCE --------------------------------------------
# With ~2.5k matches the GNN's interaction terms can't separate pairwise
# effects from noise (they shrink to ~0). Observed lane matchups CAN, if
# shrunk by sample size: on held-out matches they improved log loss in every
# CV fold, while observed cross-role counters and teammate synergy made
# predictions worse. So lane matchups are estimated directly as residuals
# against the model, with a Gaussian prior (one Newton step / Laplace):
#     effect(a beats b in role) = Σ(won - p) / (Σ p(1-p) + 1/σ²)
# Stored antisymmetrically: effect(b, a, role) = -effect(a, b, role).
def _lane_pairs(blue_row, red_row):
    for role in range(len(ROLES)):
        a, b = blue_row[role], red_row[role]
        yield ((a, b, role), 1.0) if a <= b else ((b, a, role), -1.0)


def lane_evidence(blue, red, y, base_logits, sigma=LANE_EVIDENCE_SIGMA, weight=None):
    p = torch.sigmoid(base_logits).numpy()
    w = np.ones(len(p)) if weight is None else weight.numpy()
    residual, info = w * (y.numpy() - p), w * p * (1 - p)
    acc = {}
    for m, (b, r) in enumerate(zip(blue.tolist(), red.tolist())):
        for key, sign in _lane_pairs(b, r):
            stats = acc.setdefault(key, [0.0, 0.0])
            stats[0] += sign * residual[m]
            stats[1] += info[m]
    return {k: res / (inf + 1 / sigma ** 2) for k, (res, inf) in acc.items()}


def apply_lane_evidence(blue, red, effects):
    adj = [sum(sign * effects.get(key, 0.0) for key, sign in _lane_pairs(b, r))
           for b, r in zip(blue.tolist(), red.tolist())]
    return torch.tensor(adj, dtype=torch.float)


# --- 4. TRAINING -----------------------------------------------------------
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


def fit(model, x, graph, blue, red, y, epochs, val=None, weight=None):
    """Trains full-batch. With `val`, early-stops on validation log loss and
    returns (best_epoch, best_metrics); otherwise trains exactly `epochs`."""
    torch.manual_seed(SEED)
    # role_bias has its own explicit prior (bias_penalty), so no weight decay on it
    bias_params = list(model.role_bias.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith('role_bias')]
    optimizer = torch.optim.Adam([
        {'params': other_params, 'weight_decay': WEIGHT_DECAY},
        {'params': bias_params, 'weight_decay': 0.0},
    ], lr=LR)
    best = (0, None, float('inf'))
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        w = torch.ones_like(y) if weight is None else weight
        loss = (w * F.binary_cross_entropy_with_logits(model(model.embed(x, graph), blue, red), y, reduction='none')).sum()
        loss = (loss + model.bias_penalty()) / w.sum()
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


def passes_release_gate(n_matches, shipped_log_loss, baseline_log_loss, min_matches=None):
    """True if a newly trained model may replace the published one."""
    min_matches = MIN_MATCHES if min_matches is None else min_matches
    return n_matches >= min_matches and shipped_log_loss < baseline_log_loss


def train_model():
    if not os.path.exists(INPUT_PATH):
        print("❌ Error: Cleaned match data not found. Run eda.py and preprocess.py first.")
        return

    torch.manual_seed(SEED)
    df = pd.read_csv(INPUT_PATH, dtype={'game_version': str})  # '16.20' must not become 16.2
    champions = sorted(df['champion_name'].unique())
    champ_to_id = {c: i for i, c in enumerate(champions)}
    profiles = champion_profiles(champions)

    blue, red, y = build_match_tensors(df, champ_to_id)
    match_ids = np.array(sorted(df['match_id'].unique()))
    meta = match_metadata(df)
    assert len(meta) == len(y)
    n_val = int(len(y) * VAL_FRACTION)
    if meta['game_start'].notna().all():
        # Time-based split: validate on the newest matches ("does it work next week?")
        order = torch.tensor(meta.reset_index().sort_values(['game_start', 'match_id']).index.to_numpy())
        train_idx, val_idx = order[:-n_val], order[-n_val:]
        split = 'newest'
    else:
        perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(SEED))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        split = 'random'
    versions = meta['game_version'].tolist()
    weight_train = recency_weights([versions[i] for i in train_idx.tolist()], PATCH_HALF_LIFE)
    weight_all = recency_weights(versions, PATCH_HALF_LIFE)
    patches = sorted({v for v in versions if isinstance(v, str)}, key=lambda v: tuple(map(int, v.split('.'))))
    print(f"📊 {len(y)} matches · {len(champions)} champions · train {len(train_idx)} / val {len(val_idx)} "
          f"({split} {VAL_FRACTION:.0%}) · patches {patches[0] + '-' + patches[-1] if patches else 'unknown'}")

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
        m = NexusDraftModel(len(champions), x_train.shape[1], use_enemy_terms=use_enemy, profiles=profiles)
        best_epoch, metrics = fit(m, x_train, graph_train, blue[train_idx], red[train_idx], y[train_idx],
                                  MAX_EPOCHS, val=val, weight=weight_train)
        results[name] = {**metrics, 'best_epoch': best_epoch}
        print(f"   {name:22s} | val log loss {metrics['log_loss']:.4f} | acc {metrics['accuracy']:.3f} "
              f"| AUC {metrics['auc']:.3f} | epoch {best_epoch}")

    # Mirror what ships: refit on the training split for the selected number
    # of epochs, then add lane evidence estimated on the training split.
    torch.manual_seed(SEED)
    full_model = NexusDraftModel(len(champions), x_train.shape[1], profiles=profiles)
    fit(full_model, x_train, graph_train, blue[train_idx], red[train_idx], y[train_idx],
        max(results['full_model']['best_epoch'], 1), weight=weight_train)
    full_model.eval()
    with torch.no_grad():
        z_tr = full_model.embed(x_train, graph_train)
        base_tr = full_model(z_tr, blue[train_idx], red[train_idx])
        base_val = full_model(z_tr, *val[:2])
    effects = lane_evidence(blue[train_idx], red[train_idx], y[train_idx], base_tr, weight=weight_train)
    results['full_model_refit'] = evaluate(base_val, val[2])
    results['full_model_plus_lane_evidence'] = evaluate(base_val + apply_lane_evidence(*val[:2], effects), val[2])
    for name in ('full_model_refit', 'full_model_plus_lane_evidence'):
        r = results[name]
        print(f"   {name:22s} | val log loss {r['log_loss']:.4f} | acc {r['accuracy']:.3f} | AUC {r['auc']:.3f}")

    base_rate = y[train_idx].mean()
    const_logits = torch.full_like(val[2], float(torch.logit(base_rate)))
    results['baseline_side_only'] = evaluate(const_logits, val[2])
    print(f"   {'baseline_side_only':22s} | val log loss {results['baseline_side_only']['log_loss']:.4f} "
          f"| acc {results['baseline_side_only']['accuracy']:.3f}")

    # --- Release gate (see MIN_MATCHES)
    shipped, baseline = results['full_model_plus_lane_evidence'], results['baseline_side_only']
    if not passes_release_gate(len(y), shipped['log_loss'], baseline['log_loss']):
        print(f"❌ Release gate failed: {len(y)} matches (min {MIN_MATCHES}), val log loss "
              f"{shipped['log_loss']:.4f} vs no-draft baseline {baseline['log_loss']:.4f}. "
              "Keeping the existing model.")
        sys.exit(1)
    print(f"✅ Release gate passed: beats no-draft baseline by "
          f"{baseline['log_loss'] - shipped['log_loss']:.4f} log loss.")

    # --- Final fit on ALL matches for the selected number of epochs
    print("🔁 Refitting on all matches...")
    x_all = node_feature_tensor(df, champions)
    graph_all = build_graph(blue, red, len(champions))
    torch.manual_seed(SEED)
    model = NexusDraftModel(len(champions), x_all.shape[1], profiles=profiles)
    fit(model, x_all, graph_all, blue, red, y, max(results['full_model']['best_epoch'], 1), weight=weight_all)

    model.eval()
    with torch.no_grad():
        z = model.embed(x_all, graph_all)
        effects = lane_evidence(blue, red, y, model(z, blue, red), weight=weight_all)
        lane_keys = sorted(effects)
        artifact = {
            'format_version': 3,
            'champions': champions,
            'roles': ROLES,
            'embeddings': z.clone(),
            'role_embeddings': model.role_embed.detach().clone(),
            'power': model.power.detach().clone() if model.use_shared_power else torch.zeros_like(model.power),
            'profiles': torch.as_tensor(profiles) if model.use_composition else None,
            'comp_power': model.comp_power.detach().clone(),
            'comp_cross': model.comp_cross_matrix().detach().clone(),
            'role_strength': model.role_bias.weight.detach().view(len(champions), len(ROLES)).clone(),
            'synergy': model.synergy_matrix().detach().clone(),
            'counter': model.counter_matrix().detach().clone(),
            'lane': model.lane_matrix().detach().clone(),
            # Lane matchup evidence: rows (champ_a, champ_b, role) with a<=b
            'lane_evidence_keys': torch.tensor(lane_keys, dtype=torch.long),
            'lane_evidence': torch.tensor([effects[k] for k in lane_keys], dtype=torch.float),
        }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(artifact, OUTPUT_PATH)
    metrics_out = {
        'trained_at': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'matches': int(len(y)),
        'champions': len(champions),
        'validation_matches': int(n_val),
        'validation_split': split,
        'patches': [patches[0], patches[-1]] if patches else None,
        'patch_half_life': PATCH_HALF_LIFE,
        'use_composition': bool(model.use_composition),
        'validation': results,
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"✅ Training Complete. Model saved to {OUTPUT_PATH}, metrics to {METRICS_PATH}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # emoji logs on Windows consoles
    train_model()
