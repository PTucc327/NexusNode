import numpy
import numpy as np
import torch
import json
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

torch.serialization.add_safe_globals([
    numpy._core.multiarray._reconstruct, 
    numpy.ndarray,
    numpy.dtype,
    numpy.dtypes.Float32DType
])

# Below this many head-to-head games, a matchup's win rate is treated as
# largely unreliable and gets scaled down rather than trusted outright
# (a 1-0 "matchup" is not a real counter relationship).
MATCHUP_CONFIDENCE_GAMES = 8

class DraftingEngine:
    def __init__(self, embeddings_path='./data/processed/champion_embeddings.pt',
                 roles_path='./data/processed/champion_roles.json',
                 matchups_path='./data/processed/champion_matchups.csv'):
        self.embeddings = self._load_pt(embeddings_path)
        self.roles_map = self._load_json(roles_path)
        self.matchups = self._load_matchups(matchups_path)

    def _load_pt(self, path):
        if os.path.exists(path):
            return torch.load(path, weights_only=True)
        return {}

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def _load_matchups(self, path):
        """Loads real lane matchup win rates into a dict keyed by
        (champion_name, opponent_name, role) -> (win_rate, games)."""
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        return {
            (row.champion_name, row.opponent_name, row.role): (row.win_rate, row.games)
            for row in df.itertuples()
        }

    def get_reasoning(self, target_champ, allies, role=None, lane_opponent=None):
        """XAI: Explains why a champion was recommended based on vector proximity
        and, if a same-role enemy is known, real lane matchup data."""
        counter_reason = None
        if role and lane_opponent and lane_opponent not in ("None", None):
            wr, games = self.matchups.get((target_champ, lane_opponent, role), (None, 0))
            if wr is not None and games >= MATCHUP_CONFIDENCE_GAMES:
                if wr >= 0.53:
                    counter_reason = f"Favored vs {lane_opponent} ({wr:.0%} in {games} games)"
                elif wr <= 0.47:
                    counter_reason = f"Risky vs {lane_opponent} ({wr:.0%} in {games} games)"

        if not allies or target_champ not in self.embeddings:
            return counter_reason or "Strong Meta Pick"
        
        target_vec = self.embeddings[target_champ].reshape(1, -1)
        best_partner, max_sim = None, -1
        
        for ally in allies:
            if ally in self.embeddings and ally != "None":
                ally_vec = self.embeddings[ally].reshape(1, -1)
                sim = cosine_similarity(target_vec, ally_vec)[0][0]
                if sim > max_sim:
                    max_sim, best_partner = sim, ally

        synergy_reason = f"Synergy with {best_partner}" if best_partner else "Balanced Fit"
        if counter_reason:
            return f"{synergy_reason} · {counter_reason}"
        return synergy_reason

    def get_counter_score(self, champ, role, lane_opponent):
        """Confidence-weighted lane matchup edge for `champ` against the
        enemy in the SAME role, e.g. your BOTTOM pick vs their BOTTOM pick.
        Returns a value in roughly [-0.5, 0.5]: positive means historically
        favored, negative means historically losing that matchup. Matchups
        with few recorded games are scaled toward 0 (no strong opinion)
        rather than trusted at face value.
        """
        if not lane_opponent or lane_opponent == "None":
            return 0.0
        wr, games = self.matchups.get((champ, lane_opponent, role), (None, 0))
        if wr is None:
            return 0.0
        confidence = min(games / MATCHUP_CONFIDENCE_GAMES, 1.0)
        return (wr - 0.5) * confidence

    def run_synthesis(self, user_role, allies, enemies, comfort_pool, loyalty_boost=1.2, enemy_weight=1.0):
        """
        The core DS logic: 
        1. Filters by role
        2. Calculates team centroid (ally synergy)
        3. Applies comfort multipliers
        4. Applies a real lane-matchup counter score against the enemy
           laner in the same role

        `enemies` can be either:
          - a dict {role: champion_or_None} for all 5 roles (preferred --
            enables real same-role matchup scoring), or
          - a flat list of champion names (legacy behavior -- still used
            for exclusion, but no matchup scoring is possible without
            knowing which role each enemy occupies)
        """
        eligible_champs = self.roles_map.get(user_role, [])
        active_allies = [a for a in allies if a != "None" and a in self.embeddings]
        
        if not active_allies:
            return []

        if isinstance(enemies, dict):
            enemy_by_role = enemies
            all_enemies = list(enemies.values())
        else:
            enemy_by_role = {}
            all_enemies = list(enemies)
        lane_opponent = enemy_by_role.get(user_role)

        # Create Team Centroid (Mean of teammate vectors)
        ally_vectors = [self.embeddings[a] for a in active_allies]
        team_centroid = np.mean(ally_vectors, axis=0).reshape(1, -1)
        
        scores = []
        for champ in eligible_champs:
            if champ in self.embeddings and champ not in allies and champ not in all_enemies:
                champ_vec = self.embeddings[champ].reshape(1, -1)
                base_sim = cosine_similarity(team_centroid, champ_vec)[0][0]
                
                # Apply Loyalty Bonus
                # NOTE: cosine similarity can be negative here, so a naive
                # `base_sim * loyalty_boost` would make comfort picks with a
                # negative fit score WORSE instead of better. Boost additively
                # instead, scaled by the magnitude of the base score, so the
                # bonus always pushes the score in the favorable direction.
                if champ in comfort_pool:
                    ally_score = base_sim + (loyalty_boost - 1) * abs(base_sim)
                else:
                    ally_score = base_sim

                # Apply enemy lane-matchup counter score
                counter_score = self.get_counter_score(champ, user_role, lane_opponent)
                final_score = ally_score + enemy_weight * counter_score

                scores.append((champ, final_score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:5]