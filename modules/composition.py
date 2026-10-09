"""Team composition features shared by train_gnn.py and engine.py.

Each champion gets a small profile from Data Dragon (class tags and AD/AP
lean). Frontline comes from class tags (Tank, half credit for Fighter), not
Riot's 0-10 "defense" rating, which is unreliable (it rates Taliyah a 7).
A team's features are interpretable summaries of the five profiles: class
mix, how one-sided its damage is, and how much frontline it has. These need only a handful of parameters, so unlike per-pair synergy
they can be learned from a few thousand matches.
"""
import json
import os

import numpy as np
import torch

from modules.champions import ChampionCatalog, SNAPSHOT_PATH

TAGS = ['Fighter', 'Tank', 'Mage', 'Assassin', 'Marksman', 'Support']
PROFILE_DIM = len(TAGS) + 2  # tag weights, AD share, frontline score
FEATURE_NAMES = [f'{t.lower()}_share' for t in TAGS] + ['damage_imbalance', 'max_frontline']
NUM_FEATURES = len(FEATURE_NAMES)


def champion_profiles(champions, catalog=None):
    """[N, PROFILE_DIM] profile matrix in `champions` order."""
    if catalog is None:
        with open(SNAPSHOT_PATH, 'r', encoding='utf-8') as f:
            catalog = ChampionCatalog(json.load(f))
    rows = []
    for champ in champions:
        prof = catalog.profile(champ)
        tag_weights = [0.0] * len(TAGS)
        for rank, tag in enumerate(prof['tags'][:2]):  # primary 1.0, secondary 0.5
            if tag in TAGS:
                tag_weights[TAGS.index(tag)] = 1.0 if rank == 0 else 0.5
        info = prof['info']
        attack, magic = info.get('attack', 5), info.get('magic', 5)
        ad_share = attack / (attack + magic) if attack + magic else 0.5
        frontline = tag_weights[TAGS.index('Tank')] + 0.5 * tag_weights[TAGS.index('Fighter')]
        rows.append(tag_weights + [ad_share, frontline])
    return np.array(rows, dtype=np.float32)


def team_features(profiles):
    """profiles [..., 5, PROFILE_DIM] (torch) -> team features [..., NUM_FEATURES]."""
    tag_share = profiles[..., :len(TAGS)].mean(dim=-2)
    ad_share = profiles[..., len(TAGS)].mean(dim=-1)
    frontline = profiles[..., len(TAGS) + 1]
    imbalance = 4 * (ad_share - 0.5) ** 2  # 0 = even AD/AP split, 1 = all one type
    return torch.cat([
        tag_share,
        imbalance.unsqueeze(-1),
        frontline.max(dim=-1).values.unsqueeze(-1),
    ], dim=-1)


def role_mean_profiles(profiles, role_pick_counts):
    """Pick-rate-weighted average profile per role [5, PROFILE_DIM], used to
    fill not-yet-picked slots so partial teams aren't treated as 'all AD'
    just because only the ADC is locked in."""
    weights = role_pick_counts / np.maximum(role_pick_counts.sum(axis=0, keepdims=True), 1)
    return (weights.T @ profiles).astype(np.float32)


def load_profiles_or_none(champions):
    """Profiles if the champion snapshot exists, else None (feature disabled)."""
    return champion_profiles(champions) if os.path.exists(SNAPSHOT_PATH) else None
