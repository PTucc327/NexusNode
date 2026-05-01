import numpy as np
import torch
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

class DraftingEngine:
    def __init__(self, embeddings_path='./champion_embeddings.pt', roles_path='./data/processed/champion_roles.json'):
        self.embeddings = self._load_pt(embeddings_path)
        self.roles_map = self._load_json(roles_path)

    def _load_pt(self, path):
        return torch.load(path) if os.path.exists(path) else {}

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def get_reasoning(self, target_champ, allies):
        """XAI: Explains why a champion was recommended based on vector proximity."""
        if not allies or target_champ not in self.embeddings:
            return "Strong Meta Pick"
        
        target_vec = self.embeddings[target_champ].reshape(1, -1)
        best_partner, max_sim = None, -1
        
        for ally in allies:
            if ally in self.embeddings and ally != "None":
                ally_vec = self.embeddings[ally].reshape(1, -1)
                sim = cosine_similarity(target_vec, ally_vec)[0][0]
                if sim > max_sim:
                    max_sim, best_partner = sim, ally
                    
        return f"Synergy with {best_partner}" if best_partner else "Balanced Fit"

    def run_synthesis(self, user_role, allies, enemies, comfort_pool, loyalty_boost=1.2):
        """
        The core DS logic: 
        1. Filters by role
        2. Calculates team centroid
        3. Applies comfort multipliers
        """
        eligible_champs = self.roles_map.get(user_role, [])
        active_allies = [a for a in allies if a != "None" and a in self.embeddings]
        
        if not active_allies:
            return []

        # Create Team Centroid (Mean of teammate vectors)
        ally_vectors = [self.embeddings[a] for a in active_allies]
        team_centroid = np.mean(ally_vectors, axis=0).reshape(1, -1)
        
        scores = []
        for champ in eligible_champs:
            if champ in self.embeddings and champ not in allies and champ not in enemies:
                champ_vec = self.embeddings[champ].reshape(1, -1)
                base_sim = cosine_similarity(team_centroid, champ_vec)[0][0]
                
                # Apply Loyalty Bonus
                final_score = base_sim * loyalty_boost if champ in comfort_pool else base_sim
                scores.append((champ, final_score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:5]