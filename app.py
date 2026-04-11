import streamlit as st
import torch
import numpy as np
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="NexusNode | AI Draft Tool", page_icon="🎮", layout="wide")

# --- LOAD DATA ---
@st.cache_resource
def load_model_data():
    # Load the TRAINED embeddings
    embeddings = torch.load('champion_embeddings.pt', weights_only=False)
    champ_list = sorted(list(embeddings.keys()))
    
    # Load the filtered roles
    with open('champion_roles.json', 'r') as f:
        role_mapping = json.load(f)
        
    return embeddings, champ_list, role_mapping

embeddings, champ_list, role_mapping = load_model_data()

# --- HEADER ---
st.title("🔗 NexusNode")
st.caption("GNN-Powered Win-Condition Synthesis")
st.divider()

# --- SIDEBAR ---
st.sidebar.header("Draft Configuration")
user_role = st.sidebar.selectbox("Your Role", ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"])

st.sidebar.subheader("Your Team")
t1 = st.sidebar.selectbox("Ally 1", champ_list, index=0)
t2 = st.sidebar.selectbox("Ally 2", champ_list, index=1)
t3 = st.sidebar.selectbox("Ally 3", champ_list, index=2)
t4 = st.sidebar.selectbox("Ally 4", champ_list, index=3)

run_btn = st.sidebar.button("Synthesize Recommendation", type="primary", use_container_width=True)

# --- MAIN DISPLAY ---
if run_btn:
    # 1. Prepare input
    team = [t1, t2, t3, t4]
    team_vectors = [embeddings[name] for name in team]
    team_centroid = np.mean(team_vectors, axis=0).reshape(1, -1)
    
    # 2. Filter by Role
    valid_role_champs = role_mapping.get(user_role, [])
    
    # 3. Calculate Scores
    scores = []
    for champ in valid_role_champs:
        if champ in embeddings and champ not in team:
            # We reshape to 2D for cosine_similarity
            champ_vector = embeddings[champ].reshape(1, -1)
            sim = cosine_similarity(team_centroid, champ_vector)[0][0]
            scores.append((champ, sim))
    
    # 4. Sort and Display
    results = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    
    if results:
        st.write(f"### Optimal {user_role} Picks for Current Comp")
        cols = st.columns(5)
        
        for i, (champ, score) in enumerate(results):
            # Clean score for progress bar (0.0 to 1.0)
            clean_score = max(0.0, min(1.0, float(score)))
            
            with cols[i]:
                st.metric(label=f"Rank {i+1}", value=champ, delta=f"{clean_score:.2%}")
                st.progress(clean_score)
                
        st.success("Synergy Analysis Complete.")
    else:
        st.warning(f"No meta-data found for {user_role}. Please update your dataset.")

else:
    st.info("Configure the draft in the sidebar to begin GNN synthesis.")

# --- FOOTER ---
with st.expander("Model Stats"):
    st.write(f"Current Graph Size: {len(champ_list)} Champions")
    st.write(f"Embedding Dimensions: 64 (Latent Space)")