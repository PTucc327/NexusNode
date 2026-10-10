"""Team composition features shared by train_gnn.py and engine.py.

Each champion gets a small profile: class tags (Data Dragon), AD/AP share
measured from the physical vs. magic damage it actually deals in our games,
and frontline from class tags (Tank, half credit for Fighter). Riot's 0-10
"attack/magic/defense" ratings are only a fallback prior: they describe play
style, not damage type (they rate Gragas and Elise ~45% AD; both deal almost
entirely magic damage) and rate Taliyah's defense a 7.
A team's features are interpretable summaries of the five profiles: class
mix, how one-sided its damage is, and how much frontline it has. These need only a handful of parameters, so unlike per-pair synergy
they can be learned from a few thousand matches.
"""
import json
import os

import numpy as np
import torch

from modules.champions import ChampionCatalog, SNAPSHOT_PATH

DAMAGE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'champion_damage.csv')
# Observed AD share is blended with the rating-based prior as if the prior
# were this many games, so champions with little data aren't noisy.
DAMAGE_PRIOR_GAMES = 5

TAGS = ['Fighter', 'Tank', 'Mage', 'Assassin', 'Marksman', 'Support']
PROFILE_DIM = len(TAGS) + 2  # tag weights, AD share, frontline score
# Damage skew is split by direction: with real damage types, AD-heavy teams
# lose noticeably more while AP-heavy teams don't, so one symmetric
# 'imbalance' term can't fit both.
FEATURE_NAMES = [f'{t.lower()}_share' for t in TAGS] + ['ad_heavy', 'ap_heavy', 'max_frontline']
NUM_FEATURES = len(FEATURE_NAMES)


def load_damage_profiles(path=DAMAGE_PATH):
    """{champion: (games, observed_ad_share)} from preprocess.py, or {}."""
    if not os.path.exists(path):
        return {}
    import pandas as pd
    df = pd.read_csv(path)
    return {r.champion_name: (int(r.games), float(r.ad_share)) for r in df.itertuples()}


def champion_profiles(champions, catalog=None, damage=None):
    """[N, PROFILE_DIM] profile matrix in `champions` order."""
    if catalog is None:
        with open(SNAPSHOT_PATH, 'r', encoding='utf-8') as f:
            catalog = ChampionCatalog(json.load(f))
    damage = load_damage_profiles() if damage is None else damage
    rows = []
    for champ in champions:
        prof = catalog.profile(champ)
        tag_weights = [0.0] * len(TAGS)
        for rank, tag in enumerate(prof['tags'][:2]):  # primary 1.0, secondary 0.5
            if tag in TAGS:
                tag_weights[TAGS.index(tag)] = 1.0 if rank == 0 else 0.5
        info = prof['info']
        attack, magic = info.get('attack', 5), info.get('magic', 5)
        prior = attack / (attack + magic) if attack + magic else 0.5
        games, observed = damage.get(champ, (0, prior))
        ad_share = (games * observed + DAMAGE_PRIOR_GAMES * prior) / (games + DAMAGE_PRIOR_GAMES)
        frontline = tag_weights[TAGS.index('Tank')] + 0.5 * tag_weights[TAGS.index('Fighter')]
        rows.append(tag_weights + [ad_share, frontline])
    return np.array(rows, dtype=np.float32)


def team_features(profiles):
    """profiles [..., 5, PROFILE_DIM] (torch) -> team features [..., NUM_FEATURES]."""
    tag_share = profiles[..., :len(TAGS)].mean(dim=-2)
    ad_share = profiles[..., len(TAGS)].mean(dim=-1)
    frontline = profiles[..., len(TAGS) + 1]
    ad_heavy = 4 * torch.clamp(ad_share - 0.5, min=0) ** 2  # 0 at an even split, 1 if all physical
    ap_heavy = 4 * torch.clamp(0.5 - ad_share, min=0) ** 2  # 0 at an even split, 1 if all magic
    return torch.cat([
        tag_share,
        ad_heavy.unsqueeze(-1),
        ap_heavy.unsqueeze(-1),
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
