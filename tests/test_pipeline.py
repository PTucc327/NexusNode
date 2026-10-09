"""Data pipeline and training helpers."""
import numpy as np
import pandas as pd
import pytest
import torch

from modules import train_gnn as T
from modules.collect_data import patch_of
from modules.composition import champion_profiles, team_features, NUM_FEATURES, TAGS
from modules.eda import clean_data


# --- patches & recency ------------------------------------------------------
def test_patch_of():
    assert patch_of('16.20.824.8524') == '16.20'
    assert patch_of(None) is None


def test_patch_ages_use_true_distance_across_gaps():
    # The scraper was down Apr-Oct: 16.8 must be 12 patches old, not "1 behind"
    assert list(T.patch_ages(['16.8', '16.20', '16.19'])) == [12, 0, 1]


def test_patch_ages_cross_season_and_unknown():
    ages = T.patch_ages(['15.24', '16.1', None])
    assert list(ages) == [1, 0, 1]  # unknown counts as the oldest seen


def test_recency_weights():
    w = T.recency_weights(['16.20', '16.12', '16.4'], half_life=8)
    assert w.tolist() == pytest.approx([1.0, 0.5, 0.25])
    assert T.recency_weights(['16.20', '16.1'], None).tolist() == [1.0, 1.0]


# --- release gate -------------------------------------------------------------
def test_release_gate():
    assert T.passes_release_gate(4000, 0.6900, 0.6926)
    assert not T.passes_release_gate(4000, 0.6930, 0.6926)   # worse than no-draft baseline
    assert not T.passes_release_gate(4000, 0.6926, 0.6926)   # not strictly better
    assert not T.passes_release_gate(500, 0.6800, 0.6926)    # too little data


# --- lane evidence ------------------------------------------------------------
def test_lane_evidence_is_antisymmetric_and_shrunk():
    blue = torch.tensor([[0, 1, 2, 3, 4]] * 20)
    red = torch.tensor([[5, 6, 7, 8, 9]] * 20)
    y = torch.ones(20)  # blue always wins
    effects = T.lane_evidence(blue, red, y, torch.zeros(20))
    adj = T.apply_lane_evidence(blue[:1], red[:1], effects)
    adj_swapped = T.apply_lane_evidence(red[:1], blue[:1], effects)
    assert adj.item() > 0
    assert adj.item() == pytest.approx(-adj_swapped.item())
    # 20 straight wins still yields a modest per-lane effect (shrinkage)
    assert all(0 < v < 0.1 for v in effects.values())


# --- composition --------------------------------------------------------------
def test_frontline_comes_from_tags_not_defense_rating():
    prof = champion_profiles(['Ornn', 'Taliyah', 'Fiora', 'Darius'])
    frontline = prof[:, len(TAGS) + 1]
    assert frontline[0] == 1.0      # Tank
    assert frontline[1] == 0.0      # Mage/Support despite Riot's defense rating of 7
    assert frontline[2] == 0.5      # Fighter/Assassin: half credit
    assert frontline[3] == 1.0      # Fighter/Tank juggernaut


def test_team_features_shape_and_damage_balance():
    names = ['Ornn', 'Vi', 'Ahri', 'Jhin', 'Lux']
    feats = team_features(torch.as_tensor(champion_profiles(names)))
    assert feats.shape == (NUM_FEATURES,)
    all_ad = team_features(torch.as_tensor(champion_profiles(['Darius', 'Zed', 'Jhin', 'Draven', 'Pyke'])))
    assert all_ad[len(TAGS)] > feats[len(TAGS)]  # damage imbalance higher for an all-AD team


# --- cleaning -----------------------------------------------------------------
def _team_rows(match_id, champs, team_id, win, queue=420, version='16.20'):
    return [dict(match_id=match_id, region='americas', champion_name=c, team_id=team_id, win=win, role=r,
                 kills=1, deaths=1, assists=1, damage_to_champs=1000, gold_earned=1000, collected_at='x',
                 game_version=version, game_start='2026-10-01', queue_id=queue)
            for c, r in zip(champs, ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY'])]


def test_clean_data_keeps_only_complete_ranked_solo(tmp_path):
    blue = ['Gnar', 'Vi', 'Ahri', 'Jhin', 'Rakan']
    red = ['Jayce', 'Nidalee', 'Syndra', 'Caitlyn', 'Lux']
    rows = (_team_rows('M1', blue, 100, True) + _team_rows('M1', red, 200, False)            # valid
            + _team_rows('M2', blue, 100, True, queue=440) + _team_rows('M2', red, 200, False, queue=440)  # flex
            + _team_rows('M3', blue, 100, True))                                               # incomplete
    rows += _team_rows('M1', blue[:1], 100, True)                                            # duplicate row
    raw, out = tmp_path / 'raw.csv', tmp_path / 'clean.csv'
    pd.DataFrame(rows).to_csv(raw, index=False)
    clean_data(str(raw), str(out))
    df = pd.read_csv(out, dtype={'game_version': str})
    assert set(df.match_id) == {'M1'}
    assert len(df) == 10
    assert set(df.role) == {'TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'SUPPORT'}  # UTILITY renamed
    assert set(df.game_version) == {'16.20'}  # not parsed as 16.2


def test_match_metadata_order_matches_tensors():
    df = pd.DataFrame(_team_rows('B', ['Gnar', 'Vi', 'Ahri', 'Jhin', 'Rakan'], 100, True)
                      + _team_rows('B', ['Jayce', 'Nidalee', 'Syndra', 'Caitlyn', 'Lux'], 200, False)
                      + _team_rows('A', ['Gnar', 'Vi', 'Ahri', 'Jhin', 'Rakan'], 100, False, version='16.19')
                      + _team_rows('A', ['Jayce', 'Nidalee', 'Syndra', 'Caitlyn', 'Lux'], 200, True, version='16.19'))
    df['role'] = df['role'].replace('UTILITY', 'SUPPORT')
    champs = sorted(df.champion_name.unique())
    blue, red, y = T.build_match_tensors(df, {c: i for i, c in enumerate(champs)})
    meta = T.match_metadata(df)
    assert list(meta.index) == ['A', 'B']
    assert y.tolist() == [0.0, 1.0]           # A: blue lost, B: blue won
    assert list(meta.game_version) == ['16.19', '16.20']
    assert np.array_equal(blue[0].numpy(), blue[1].numpy())
