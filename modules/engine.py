import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_detailed_reasoning(target_champ, allies, embeddings):
    """
    Identifies which ally has the strongest mathematical bond 
    with the recommended champion.
    """
    if not allies or target_champ not in embeddings:
        return "General Meta Synergy"
    
    # Find which specific ally has the highest cosine similarity to the pick
    target_vec = embeddings[target_champ].reshape(1, -1)
    best_partner = None
    max_sim = -1
    
    for ally in allies:
        if ally in embeddings and ally != "None":
            ally_vec = embeddings[ally].reshape(1, -1)
            sim = cosine_similarity(target_vec, ally_vec)[0][0]
            if sim > max_sim:
                max_sim = sim
                best_partner = ally
                
    if best_partner:
        return f"High synergy with {best_partner}"
    return "Balanced Team Fit"