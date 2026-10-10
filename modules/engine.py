import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from modules.composition import team_features, role_mean_profiles, FEATURE_NAMES, TAGS

ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'SUPPORT']

# Below this many head-to-head games, an observed matchup win rate is too
# noisy to quote as evidence (a 1-0 "matchup" is not a counter relationship).
MATCHUP_CONFIDENCE_GAMES = 8
# Contributions smaller than this (in win-probability points) aren't worth
# surfacing as reasons.
REASON_THRESHOLD = 0.002
# Blind pick: score each candidate against this many of the most-played
# opponents in the role. Its counter risk is how much worse it does against
# its worst likely opponent than the other candidates do against that same
# opponent (i.e. a real counter, not just a generally strong champion);
# the ranking subtracts this fraction of it.
BLIND_POOL = 12
BLIND_RISK_WEIGHT = 0.5
# Counter risk smaller than this (win-prob points) is reported as blind-safe
BLIND_SAFE_THRESHOLD = 0.005

# (feature, direction of change) -> reason text. Direction matters: a mage
# replacing the usual marksman LOWERS marksman share, which must not be
# described as "adds a marksman".
COMP_REASONS = {
    ('ad_heavy', -1): 'Balances damage (team is AD-heavy)',
    ('ap_heavy', -1): 'Balances damage (team is AP-heavy)',
    ('max_frontline', 1): 'Adds frontline',
    ('tank_share', 1): 'Adds tankiness',
    ('tank_share', -1): 'Less tank-heavy team',
    ('fighter_share', 1): 'Adds a bruiser',
    ('fighter_share', -1): 'Less bruiser-heavy team',
    ('mage_share', 1): 'Adds magic damage / control',
    ('mage_share', -1): 'Less mage-heavy team',
    ('assassin_share', 1): 'Adds pick potential',
    ('assassin_share', -1): 'Less assassin-heavy team',
    ('marksman_share', 1): 'Adds a marksman',
    ('marksman_share', -1): 'Fewer marksmen (overperforming this patch)',
    ('support_share', 1): 'Adds utility',
    ('support_share', -1): 'Less utility-heavy team',
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Recommendation:
    champion: str
    win_prob: float              # model win probability (blind: average vs. the likely field)
    score: float                 # ranking score (risk-adjusted if blind, + comfort bonus)
    is_comfort: bool
    # [(ally|enemy, Δwin-prob vs. the average candidate)], best first (+ = good for us)
    synergy: list = field(default_factory=list)
    counters: list = field(default_factory=list)
    lane_record: tuple = None                     # (opponent, win_rate, games) observed
    meta: float = 0.0                             # role strength, Δwin-prob vs average candidate
    role_games: int = 0                           # games on this champion in this role
    comp: float = 0.0                             # team-composition effect, Δwin-prob vs average candidate
    comp_reason: tuple = None                     # dominant composition (feature, direction)
    # Blind pick (lane opponent unknown): the likely opponent that counters
    # this pick specifically, and by how much (win-prob points, <= 0)
    blind: bool = False
    counter_risk_by: str = None
    counter_risk: float = 0.0


class DraftingEngine:
    """Scores picks with the NexusNode draft model (see train_gnn.py).

    For a draft A (your team) vs B (enemies), the model's logit is
    role strength + synergy + cross-team counter and lane terms (additive per
    champion/pair) + team composition (from class/damage profiles). Each
    candidate's effect decomposes into those parts, which drive both the
    ranking and the explanations.

    If your lane opponent hasn't picked yet (blind pick), candidates are
    scored against the role's most-played champions instead, with a penalty
    for their worst likely matchup.
    """

    def __init__(self, model_path='./data/processed/nexus_model.pt',
                 roles_path='./data/processed/champion_roles.json',
                 matchups_path='./data/processed/champion_matchups.csv',
                 metrics_path='./data/processed/model_metrics.json'):
        art = torch.load(model_path, weights_only=True)
        self.champions = list(art['champions'])
        self.index = {c: i for i, c in enumerate(self.champions)}
        self.roles = list(art['roles'])
        self.z = art['embeddings'].numpy()
        self.role_z = art['role_embeddings'].numpy()
        self.w = art['power'].numpy()
        # Shrunk per-(champion, role) strength; absent in v2 artifacts
        rs = art.get('role_strength')
        self.role_strength = rs.numpy() if rs is not None else np.zeros((len(self.champions), len(self.roles)))
        self.S = art['synergy'].numpy()
        self.K = art['counter'].numpy()
        self.L = art['lane'].numpy()
        # Shrunk observed lane matchup effects (logit), keyed by (id_a, id_b, role_idx), id_a <= id_b
        keys = art.get('lane_evidence_keys')
        self.lane_evidence = {} if keys is None else {
            tuple(k): float(v) for k, v in zip(keys.tolist(), art['lane_evidence'].tolist())}
        self.roles_map = self._load_json(roles_path)
        self.matchups, self.role_games = self._load_matchups(matchups_path)
        self.metrics = self._load_json(metrics_path)

        # Team composition (absent in older artifacts, or disabled by CV)
        profiles = art.get('profiles')
        self.profiles = profiles.numpy() if profiles is not None else None
        if self.profiles is not None:
            self.comp_u = art['comp_power'].numpy()
            self.comp_Q = art['comp_cross'].numpy()
            counts = np.array([[self.role_games.get((c, r), 0) for r in self.roles] for c in self.champions],
                              dtype=np.float32)
            self.role_profiles = role_mean_profiles(self.profiles, counts)

    @staticmethod
    def _load_json(path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    @staticmethod
    def _load_matchups(path):
        """Returns ({(champion, opponent, role): (win_rate, games)},
        {(champion, role): games}) from observed lane matchups."""
        if not os.path.exists(path):
            return {}, {}
        df = pd.read_csv(path)
        # Every lane game appears once per (champion, opponent), so summing
        # gives games played per (champion, role).
        role_games = df.groupby(['champion_name', 'role'])['games'].sum().to_dict()
        matchups = {
            (row.champion_name, row.opponent_name, row.role): (row.win_rate, row.games)
            for row in df.itertuples()
        }
        return matchups, role_games

    # --- helpers ---------------------------------------------------------
    def _token(self, champ, role):
        return self.z[self.index[champ]] + self.role_z[self.roles.index(role)]

    def _power(self, champ, role, token):
        return self.role_strength[self.index[champ], self.roles.index(role)] + token @ self.w

    def _lane_effect(self, champ, opponent, role):
        """Logit edge of `champ` over `opponent` in `role` from lane matchup evidence."""
        a, b, r = self.index[champ], self.index[opponent], self.roles.index(role)
        if a <= b:
            return self.lane_evidence.get((a, b, r), 0.0)
        return -self.lane_evidence.get((b, a, r), 0.0)

    def _clean(self, picks):
        """{role: champ|None} -> {role: champ} for champs the model knows."""
        return {r: c for r, c in (picks or {}).items() if c and c in self.index}

    def eligible(self, role):
        return [c for c in self.roles_map.get(role, []) if c in self.index]

    def lane_record(self, champ, role, opponent):
        wr, games = self.matchups.get((champ, opponent, role), (None, 0))
        if wr is None or games < MATCHUP_CONFIDENCE_GAMES:
            return None
        return (opponent, float(wr), int(games))

    # --- scoring ---------------------------------------------------------
    def _additive_logit(self, allies, enemies, enemy_weight=1.0):
        """Per-champion and pairwise terms (everything except composition)."""
        ta = {r: self._token(c, r) for r, c in allies.items()}
        tb = {r: self._token(c, r) for r, c in enemies.items()}

        def team(t, picks):
            vecs = list(t.values())
            power = sum(self._power(picks[r], r, v) for r, v in t.items())
            syn = sum(vecs[i] @ self.S @ vecs[j] for i in range(len(vecs)) for j in range(i + 1, len(vecs)))
            return power + syn

        cross = sum(a @ self.K @ b for a in ta.values() for b in tb.values())
        lane = sum(ta[r] @ self.L @ tb[r] + self._lane_effect(allies[r], enemies[r], r) for r in ta if r in tb)
        return team(ta, allies) - team(tb, enemies) + enemy_weight * (cross + lane)

    def _team_comp(self, picks):
        """Composition features for a (possibly partial) team; empty slots use
        the role's average champion profile."""
        rows = [self.profiles[self.index[picks[r]]] if r in picks else self.role_profiles[i]
                for i, r in enumerate(self.roles)]
        return team_features(torch.as_tensor(np.stack(rows))).numpy()

    def _comp_logit(self, allies, enemies, enemy_weight=1.0):
        if self.profiles is None:
            return 0.0
        ga, gb = self._team_comp(allies), self._team_comp(enemies)
        return float((ga - gb) @ self.comp_u + enemy_weight * (ga @ self.comp_Q @ gb))

    def _draft_logit(self, allies, enemies, enemy_weight=1.0):
        """Logit that `allies` beat `enemies` (both {role: champ}, may be partial)."""
        return self._additive_logit(allies, enemies, enemy_weight) + self._comp_logit(allies, enemies, enemy_weight)

    def _comp_explain(self, allies, enemies, role, champ, enemy_weight):
        """Composition effect (logit) of `champ` in `role` vs. an average pick
        there, and the feature that drives it most."""
        if self.profiles is None:
            return 0.0, None
        gb = self._team_comp(enemies)
        diff = self._team_comp({**allies, role: champ}) - self._team_comp(allies)
        # The composition logit is linear in our features given theirs
        contrib = diff * (self.comp_u + enemy_weight * (self.comp_Q @ gb))
        top = int(np.argmax(contrib))
        return float(contrib.sum()), (FEATURE_NAMES[top], 1 if diff[top] > 0 else -1)

    def team_damage(self, picks):
        """(physical share, champions counted) for the picked champions, from
        the damage they actually deal in games; None if nothing is picked."""
        picks = self._clean(picks)
        if self.profiles is None or not picks:
            return None
        shares = [self.profiles[self.index[c], len(TAGS)] for c in picks.values()]
        return float(np.mean(shares)), len(shares)

    def win_probability(self, allies, enemies, enemy_weight=1.0):
        allies, enemies = self._clean(allies), self._clean(enemies)
        if not allies and not enemies:
            return None
        return float(_sigmoid(self._draft_logit(allies, enemies, enemy_weight)))

    def recommend(self, user_role, allies, enemies, comfort_pool=(), comfort_bonus=0.03,
                  enemy_weight=1.0, top_k=None, banned=()):
        """Ranks every eligible champion for `user_role`.

        allies / enemies: {role: champion or None}. `allies` must not include
        user_role. Works with any number of picks, including none.
        If enemies has no pick in user_role, candidates are scored as blind
        picks (vs. the likely field, penalizing the worst likely matchup);
        otherwise as counter picks vs. that lane opponent.
        comfort_bonus: win-probability points added to comfort picks for ranking.
        enemy_weight: scales every enemy interaction term (0 = ignore enemies).
        banned: champions that can't be picked this game.
        """
        allies, enemies = self._clean(allies), self._clean(enemies)
        allies.pop(user_role, None)
        taken = set(allies.values()) | set(enemies.values()) | set(banned or ())
        comfort = set(comfort_pool or [])

        ally_tokens = {c: self._token(c, r) for r, c in allies.items()}
        enemy_tokens = {c: (r, self._token(c, r)) for r, c in enemies.items()}
        lane_opp = enemies.get(user_role)
        blind = lane_opp is None

        # Blind pick: the most-played champions in the role that are still available
        field_pool = []
        if blind:
            available = sorted((c for c in self.eligible(user_role) if c not in taken),
                               key=lambda c: self.role_games.get((c, user_role), 0), reverse=True)
            field_pool = [(c, max(self.role_games.get((c, user_role), 0), 1)) for c in available[:BLIND_POOL]]

        recs, field_logits = [], []
        for champ in self.eligible(user_role):
            if champ in taken:
                continue
            t = self._token(champ, user_role)
            team = {**allies, user_role: champ}
            syn_terms = {a: t @ self.S @ ta for a, ta in ally_tokens.items()}
            ctr_terms = {}
            for e, (r, te) in enemy_tokens.items():
                term = t @ self.K @ te
                if r == user_role:
                    term += t @ self.L @ te + self._lane_effect(champ, e, r)
                ctr_terms[e] = enemy_weight * term

            if blind and field_pool:
                # logits vs. every likely opponent (NaN where it's the same champion)
                vs_field = np.array([np.nan if o == champ else
                                     self._draft_logit(team, {**enemies, user_role: o}, enemy_weight)
                                     for o, _ in field_pool])
                ok = ~np.isnan(vs_field)
                weights = np.array([w for _, w in field_pool], dtype=float)[ok]
                p = float(_sigmoid(vs_field[ok]) @ weights / weights.sum())
                field_logits.append(vs_field)
            else:
                p = float(_sigmoid(self._draft_logit(team, enemies, enemy_weight)))

            comp, comp_feature = self._comp_explain(allies, enemies, user_role, champ, enemy_weight)
            is_comfort = champ in comfort
            recs.append(Recommendation(
                champion=champ,
                win_prob=p,
                score=p + (comfort_bonus if is_comfort else 0.0),
                is_comfort=is_comfort,
                synergy=syn_terms,
                counters=ctr_terms,
                lane_record=self.lane_record(champ, user_role, lane_opp) if lane_opp else None,
                meta=self._power(champ, user_role, t),
                role_games=int(self.role_games.get((champ, user_role), 0)),
                comp=comp,
                comp_reason=comp_feature,
                blind=blind,
            ))

        # Blind-pick counter risk: compare each candidate's result vs. each
        # likely opponent with how the average candidate does vs. that same
        # opponent. The most negative gap is the pick-specific counter.
        if field_logits:
            specific = np.array(field_logits) - np.nanmean(field_logits, axis=0)
            for r, row in zip(recs, specific):
                if np.all(np.isnan(row)):
                    continue
                worst = int(np.nanargmin(row))
                r.counter_risk = float(min(row[worst], 0.0) * r.win_prob * (1 - r.win_prob))
                r.counter_risk_by = field_pool[worst][0]
                r.score += BLIND_RISK_WEIGHT * r.counter_risk

        # Express each effect RELATIVE to the average candidate. A strong ally
        # (e.g. a meta tank) raises every candidate's win rate; that says
        # nothing about which pick to make, and used to put the same
        # "Synergy with X" on every card. Only the difference between options
        # is decision-relevant. Converted to win-prob points at each p.
        def relative(attr):
            keys = list(getattr(recs[0], attr)) if recs else []
            means = {k: np.mean([getattr(r, attr)[k] for r in recs]) for k in keys}
            for r in recs:
                slope = r.win_prob * (1 - r.win_prob)
                rel = ((k, float((v - means[k]) * slope)) for k, v in getattr(r, attr).items())
                setattr(r, attr, sorted(rel, key=lambda x: -x[1]))
        relative('synergy')
        relative('counters')
        for attr in ('meta', 'comp'):
            mean = np.mean([getattr(r, attr) for r in recs]) if recs else 0.0
            for r in recs:
                setattr(r, attr, float((getattr(r, attr) - mean) * r.win_prob * (1 - r.win_prob)))

        recs.sort(key=lambda r: r.score, reverse=True)
        return recs[:top_k] if top_k else recs

    @staticmethod
    def reasons(rec, display=lambda n: n):
        """Short human-readable reasons for a recommendation."""
        out = []
        if abs(rec.meta) >= REASON_THRESHOLD and rec.role_games:
            kind, label = ("meta", "Strong") if rec.meta > 0 else ("weak", "Below-average")
            out.append((kind, f"{label} in this role ({rec.meta * 100:+.1f}, {rec.role_games} games)"))
        if rec.comp >= REASON_THRESHOLD and rec.comp_reason:
            label = COMP_REASONS.get(rec.comp_reason, 'Fits team composition')
            out.append(("comp", f"{label} ({rec.comp * 100:+.1f})"))
        good_syn = [(a, v) for a, v in rec.synergy if v >= REASON_THRESHOLD]
        if good_syn:
            out.append(("synergy", "Synergy with " + ", ".join(f"{display(a)} ({v * 100:+.1f})" for a, v in good_syn[:2])))
        strong = [(e, v) for e, v in rec.counters if v >= REASON_THRESHOLD]
        if strong:
            out.append(("strong", "Strong vs " + ", ".join(f"{display(e)} ({v * 100:+.1f})" for e, v in strong[:2])))
        weak = [(e, v) for e, v in reversed(rec.counters) if v <= -REASON_THRESHOLD]
        if weak:
            out.append(("weak", "Weak vs " + ", ".join(f"{display(e)} ({v * 100:+.1f})" for e, v in weak[:2])))
        if rec.lane_record:
            opp, wr, games = rec.lane_record
            out.append(("lane", f"{wr:.0%} win rate vs {display(opp)} in lane ({games} games)"))
        if rec.blind:
            if rec.counter_risk <= -BLIND_SAFE_THRESHOLD:
                out.append(("risk", f"Countered by {display(rec.counter_risk_by)} ({rec.counter_risk * 100:+.1f})"))
            else:
                out.append(("safe", "Blind-safe: no strong counter among likely picks"))
        return out
