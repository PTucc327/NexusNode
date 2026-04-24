import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="NexusNode | AI Draft Tool", page_icon="🎮", layout="wide")

# --- LOAD DATA ---
@st.cache_resource
def load_model_data():
    # Load the TRAINED embeddings
    embeddings = torch.load('champion_embeddings.pt', weights_only=False)
    # Convert all tensors to numpy arrays immediately for faster math in Streamlit
    embeddings_np = {k: v.detach().numpy() if torch.is_tensor(v) else v for k, v in embeddings.items()}
    
    champ_list = sorted(list(embeddings_np.keys()))
    
    # Load the filtered roles
    with open('champion_roles.json', 'r') as f:
        role_mapping = json.load(f)
        
    return embeddings_np, champ_list, role_mapping

embeddings, champ_list, role_mapping = load_model_data()

# --- HEADER ---
st.title("🔗 NexusNode")
st.caption("GNN-Powered Win-Condition Synthesis")
st.divider()

# --- SIDEBAR ---
st.sidebar.header("Draft Configuration")
user_role = st.sidebar.selectbox("Your Role", ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"], index=3)

st.sidebar.subheader("Visible Allies")
st.sidebar.info("Select only the teammates you can currently see.")

# Adding a 'None' option to handle early-pick scenarios
options = ["None"] + champ_list
t1 = st.sidebar.selectbox("Ally 1", options, index=0)
t2 = st.sidebar.selectbox("Ally 2", options, index=0)
t3 = st.sidebar.selectbox("Ally 3", options, index=0)
t4 = st.sidebar.selectbox("Ally 4", options, index=0)

run_btn = st.sidebar.button("Synthesize Recommendation", type="primary", use_container_width=True)

# --- MAIN DISPLAY ---
if run_btn:
    # 1. Filter out 'None' selections
    team = [name for name in [t1, t2, t3, t4] if name != "None"]
    
    if not team:
        st.warning("Please select at least one teammate to calculate synergy.")
    else:
        # 2. Prepare Team Centroid
        team_vectors = [embeddings[name] for name in team]
        team_centroid = np.mean(team_vectors, axis=0).reshape(1, -1)
        
        # 3. Filter by Role
        valid_role_champs = role_mapping.get(user_role, [])
        
        # 4. Calculate Scores
        scores = []
        for champ in valid_role_champs:
            # Don't recommend someone already on the team
            if champ in embeddings and champ not in team:
                champ_vector = embeddings[champ].reshape(1, -1)
                # Calculate similarity
                sim = cosine_similarity(team_centroid, champ_vector)[0][0]
                scores.append((champ, sim))
        
        # 5. Sort and Display
        results = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
        
        if results:
            st.write(f"### Optimal {user_role} Picks for Current Comp")
            cols = st.columns(5)
            
            for i, (champ, score) in enumerate(results):
                # Normalize score for display (assuming cosine similarity ranges mostly 0-1 here)
                display_score = max(0.0, float(score))
                
                with cols[i]:
                    st.metric(label=f"Rank {i+1}", value=champ, delta=f"{display_score:.2%}")
                    st.progress(display_score)
            
            st.success("Synergy Analysis Complete.")
        else:
            st.warning(f"No meta-data found for {user_role} in this patch.")

else:
    st.info("Configure the visible teammates in the sidebar to begin GNN synthesis.")

# --- FOOTER ---
with st.expander("Model Context"):
    st.write(f"**Current Patch Data:** Training localized to current meta.")
    st.write(f"**Dimensionality:** 2D PCA Input -> 64D GNN Latent Space")
    st.write(f"**Strategy:** Targeted at early-pick optimization.")