"""Behavioral guarantees of the recommendation engine."""
import numpy as np
import pytest

from modules.engine import DraftingEngine, ROLES
from tests.conftest import ALLIES, ENEMIES


def random_draft(engine, rng):
    """Two full, disjoint 5-champion teams with role-eligible picks."""
    used, teams = set(), ({}, {})
    for team in teams:
        for role in ROLES:
            choices = [c for c in engine.eligible(role) if c not in used]
            pick = choices[rng.integers(len(choices))]
            team[role] = pick
            used.add(pick)
    return teams


def test_swapping_teams_flips_prediction(engine):
    """P(A beats B) + P(B beats A) == 1, including composition terms."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        a, b = random_draft(engine, rng)
        assert engine.win_probability(a, b) + engine.win_probability(b, a) == pytest.approx(1.0, abs=1e-6)


def test_partial_drafts_are_antisymmetric_too(engine):
    a = {'TOP': 'Gnar', 'BOTTOM': 'Jhin'}
    b = {'MIDDLE': 'Syndra'}
    assert engine.win_probability(a, b) + engine.win_probability(b, a) == pytest.approx(1.0, abs=1e-6)


def test_empty_draft_has_no_probability(engine):
    assert engine.win_probability({}, {}) is None


def test_banned_and_picked_champions_are_never_recommended(engine):
    banned = ['Alistar', 'Akali', 'Xerath', 'Ornn']
    recs = engine.recommend('BOTTOM', ALLIES, ENEMIES, banned=banned)
    names = {r.champion for r in recs}
    assert not names & set(banned)
    assert not names & set(ALLIES.values())
    assert not names & set(ENEMIES.values())


@pytest.mark.parametrize('role', ROLES)
def test_only_role_eligible_champions(engine, role):
    recs = engine.recommend(role, {}, {})
    assert recs, f'no recommendations for {role}'
    assert {r.champion for r in recs} <= set(engine.roles_map[role])


def test_recommendations_sorted_by_score(engine):
    recs = engine.recommend('TOP', ALLIES, ENEMIES)
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_blind_mode_when_lane_opponent_unknown(engine):
    enemies = {r: c for r, c in ENEMIES.items() if r != 'SUPPORT'}
    recs = engine.recommend('SUPPORT', ALLIES | {'BOTTOM': 'Jhin'}, enemies)
    assert all(r.blind for r in recs)
    assert all(r.counter_risk <= 0 for r in recs)
    assert all(r.counter_risk_by is not None for r in recs)


def test_counter_mode_when_lane_opponent_known(engine):
    recs = engine.recommend('BOTTOM', ALLIES, ENEMIES)
    assert not any(r.blind for r in recs)
    assert all(r.counter_risk == 0 for r in recs)


def test_comfort_bonus_changes_rank_not_win_probability(engine):
    base = {r.champion: r for r in engine.recommend('BOTTOM', ALLIES, ENEMIES)}
    last = list(base)[-1]
    boosted = engine.recommend('BOTTOM', ALLIES, ENEMIES, comfort_pool=[last], comfort_bonus=0.5)
    assert boosted[0].champion == last
    assert boosted[0].is_comfort
    assert boosted[0].win_prob == pytest.approx(base[last].win_prob)


def test_enemy_weight_zero_ignores_enemies(engine):
    recs = engine.recommend('BOTTOM', ALLIES, ENEMIES, enemy_weight=0.0)
    assert all(abs(v) < 1e-12 for r in recs for _, v in r.counters)


def test_relative_effects_are_centered(engine):
    """Reasons are relative to the average candidate, so each ally's effect averages ~0."""
    recs = engine.recommend('BOTTOM', ALLIES, ENEMIES)
    for ally in ALLIES.values():
        vals = [dict(r.synergy)[ally] for r in recs]
        assert abs(np.mean(vals)) < 1e-3


def test_unknown_and_empty_inputs_are_ignored(engine):
    recs = engine.recommend('TOP', {'JUNGLE': 'NotAChampion', 'MIDDLE': None}, {'TOP': ''})
    assert recs and all(r.blind for r in recs)


def test_reasons_are_short_typed_strings(engine):
    rec = engine.recommend('BOTTOM', ALLIES, ENEMIES)[0]
    for kind, text in DraftingEngine.reasons(rec):
        assert kind in {'meta', 'weak', 'comp', 'synergy', 'strong', 'lane', 'risk', 'safe'}
        assert isinstance(text, str) and 0 < len(text) < 120
