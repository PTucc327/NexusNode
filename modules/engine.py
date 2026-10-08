import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'SUPPORT']

# Below this many head-to-head games, an observed matchup win rate is too
# noisy to quote as evidence (a 1-0 "matchup" is not a counter relationship).
MATCHUP_CONFIDENCE_GAMES = 8
# Contributions smaller than this (in win-probability points) aren't worth
# surfacing as reasons.
REASON_THRESHOLD = 0.002


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Recommendation:
    champion: str
    win_prob: float              # model win probability with this pick in the draft
    score: float                 # ranking score (win_prob + comfort bonus)
    is_comfort: bool
    # [(ally|enemy, Δwin-prob vs. the average candidate)], best first (+ = good for us)
    synergy: list = field(default_factory=list)
    counters: list = field(default_factory=list)
    lane_record: tuple = None                     # (opponent, win_rate, games) observed
    meta: float = 0.0                             # role strength, Δwin-prob vs average candidate
    role_games: int = 0                           # games on this champion in this role


class DraftingEngine:
    """Scores picks with the NexusNode draft model (see train_gnn.py).

    For a draft A (your team) vs B (enemies), the model's logit is additive:
    power(A) - power(B) + synergy(A) - synergy(B) + Σ cross-team counter terms
    + Σ lane terms. A candidate's effect is therefore its own power, plus its
    synergy with each ally, plus its counter interaction with EVERY enemy
    (not just the lane opponent). That decomposition drives both the ranking
    and the explanations.
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
    def _draft_logit(self, allies, enemies, enemy_weight=1.0):
        """Logit that `allies` beat `enemies` (both {role: champ}, may be partial)."""
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

    def win_probability(self, allies, enemies, enemy_weight=1.0):
        allies, enemies = self._clean(allies), self._clean(enemies)
        if not allies and not enemies:
            return None
        return float(_sigmoid(self._draft_logit(allies, enemies, enemy_weight)))

    def recommend(self, user_role, allies, enemies, comfort_pool=(), comfort_bonus=0.03,
                  enemy_weight=1.0, top_k=None, banned=()):
        """Ranks every eligible champion for `user_role`.

        allies / enemies: {role: champion or None}. `allies` must not include
        user_role. Works with any number of picks, including none (blind pick).
        comfort_bonus: win-probability points added to comfort picks for ranking.
        enemy_weight: scales every enemy interaction term (0 = ignore enemies).
        banned: champions that can't be picked this game.
        """
        allies, enemies = self._clean(allies), self._clean(enemies)
        allies.pop(user_role, None)
        taken = set(allies.values()) | set(enemies.values()) | set(banned or ())
        comfort = set(comfort_pool or [])

        base_logit = self._draft_logit(allies, enemies, enemy_weight)
        ally_tokens = {c: self._token(c, r) for r, c in allies.items()}
        enemy_tokens = {c: (r, self._token(c, r)) for r, c in enemies.items()}
        lane_opp = enemies.get(user_role)

        recs = []
        for champ in self.eligible(user_role):
            if champ in taken:
                continue
            t = self._token(champ, user_role)
            syn_terms = {a: t @ self.S @ ta for a, ta in ally_tokens.items()}
            ctr_terms = {}
            for e, (r, te) in enemy_tokens.items():
                term = t @ self.K @ te
                if r == user_role:
                    term += t @ self.L @ te + self._lane_effect(champ, e, r)
                ctr_terms[e] = enemy_weight * term
            delta = self._power(champ, user_role, t) + sum(syn_terms.values()) + sum(ctr_terms.values())
            p = float(_sigmoid(base_logit + delta))
            meta = self._power(champ, user_role, t)

            is_comfort = champ in comfort
            recs.append(Recommendation(
                champion=champ,
                win_prob=p,
                score=p + (comfort_bonus if is_comfort else 0.0),
                is_comfort=is_comfort,
                synergy=syn_terms,
                counters=ctr_terms,
                lane_record=self.lane_record(champ, user_role, lane_opp) if lane_opp else None,
                meta=meta,
                role_games=int(self.role_games.get((champ, user_role), 0)),
            ))

        # Express each ally/enemy effect RELATIVE to the average candidate.
        # A strong ally (e.g. a meta tank) raises every candidate's win rate;
        # that says nothing about which pick to make, and used to put the same
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
        mean_meta = np.mean([r.meta for r in recs]) if recs else 0.0
        for r in recs:
            r.meta = float((r.meta - mean_meta) * r.win_prob * (1 - r.win_prob))

        recs.sort(key=lambda r: r.score, reverse=True)
        return recs[:top_k] if top_k else recs

    @staticmethod
    def reasons(rec, display=lambda n: n):
        """Short human-readable reasons for a recommendation."""
        out = []
        if abs(rec.meta) >= REASON_THRESHOLD and rec.role_games:
            kind, label = ("meta", "Strong") if rec.meta > 0 else ("weak", "Below-average")
            out.append((kind, f"{label} in this role ({rec.meta * 100:+.1f}, {rec.role_games} games)"))
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
        return out
