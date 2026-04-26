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
    try:
        # Load the TRAINED embeddings
        embeddings = torch.load('champion_embeddings.pt', weights_only=False)
        # Convert all tensors to numpy arrays immediately for stability
        embeddings_np = {k: v.detach().numpy() if torch.is_tensor(v) else v for k, v in embeddings.items()}
        
        champ_list = sorted(list(embeddings_np.keys()))
        
        # Load the filtered roles
        with open('champion_roles.json', 'r') as f:
            role_mapping = json.load(f)
            
        return embeddings_np, champ_list, role_mapping
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}, [], {}

embeddings, champ_list, role_mapping = load_model_data()

# --- HEADER ---
st.title("🔗 NexusNode")
st.subheader("GNN-Powered Win-Condition Synthesis")
st.caption("Blending global challenger meta-patterns with personal playstyle.")
st.divider()

# --- SIDEBAR ---
st.sidebar.header("Draft Configuration")

# The UI uses user-friendly names
user_role = st.sidebar.selectbox(
    "Your Role", 
    ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"], 
    index=3
)

# --- PERSONALIZATION SECTION ---
st.sidebar.divider()
st.sidebar.subheader("Personalization")
use_personal = st.sidebar.checkbox("Apply Comfort Influence", value=False)

# Defensive logic to ensure default values exist in the current champ_list
preferred_defaults = ["Jinx", "Ezreal", "Kaisa", "Kai'Sa", "Vayne"]
available_defaults = [c for c in preferred_defaults if c in champ_list]

my_comfort_picks = st.sidebar.multiselect(
    "Your Comfort Pool", 
    options=champ_list,
    default=available_defaults
)

comfort_weight = st.sidebar.slider("Comfort Weight", 0.0, 1.0, 0.3) if use_personal else 0.0

st.sidebar.divider()
st.sidebar.subheader("Visible Allies")
st.sidebar.info("Select the teammates you currently see in the draft.")

# Handle early-pick scenarios with "None"
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
        # --- ROLE TRANSLATION LAYER ---
        # Maps UI selection to JSON keys
        role_translator = {
            "TOP": "TOP",
            "JUNGLE": "JUNGLE",
            "MIDDLE": "MIDDLE", 
            "BOTTOM": "BOTTOM",
            "SUPPORT": "SUPPORT"
        }
        
        target_key = role_translator.get(user_role, user_role)
        valid_role_champs = role_mapping.get(target_key, [])

        # 2. Prepare Team Centroid (Global Meta Needs)
        team_vectors = [embeddings[name] for name in team if name in embeddings]
        
        if not team_vectors:
            st.error("Selected allies not found in embedding database.")
        else:
            team_centroid = np.mean(team_vectors, axis=0).reshape(1, -1)
            
            # 3. Prepare User Anchor (Personal Playstyle)
            user_anchor = None
            if use_personal and my_comfort_picks:
                personal_vectors = [embeddings[c] for c in my_comfort_picks if c in embeddings]
                if personal_vectors:
                    user_anchor = np.mean(personal_vectors, axis=0).reshape(1, -1)

            # 4. Calculate Blended Scores
            scores = []
            for champ in valid_role_champs:
                if champ in embeddings and champ not in team:
                    champ_vector = embeddings[champ].reshape(1, -1)
                    
                    # Global Similarity (Team Synergy)
                    global_sim = cosine_similarity(team_centroid, champ_vector)[0][0]
                    
                    # Personalized Similarity (Comfort)
                    if user_anchor is not None:
                        personal_sim = cosine_similarity(user_anchor, champ_vector)[0][0]
                        # Blend the two scores based on user weight
                        final_sim = ((1 - comfort_weight) * global_sim) + (comfort_weight * personal_sim)
                    else:
                        final_sim = global_sim
                        
                    scores.append((champ, final_sim))
            
            # 5. Sort and Display Results
            results = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
            
            if results:
                st.write(f"### Optimal {user_role} Picks for Current Comp")
                
                # Dynamic Scaling for visual impact
                raw_scores = [r[1] for r in results]
                min_s, max_s = min(raw_scores), max(raw_scores)
                
                cols = st.columns(5)
                for i, (champ, score) in enumerate(results):
                    # Relative progress bar logic
                    if max_s != min_s:
                        norm_score = (score - min_s) / (max_s - min_s)
                        display_progress = float(0.4 + (norm_score * 0.6))
                    else:
                        display_progress = 1.0

                    with cols[i]:
                        # Metric shows raw blended similarity
                        st.metric(label=f"Rank {i+1}", value=champ, delta=f"{score:.4f} Sim")
                        # Explicit float cast to prevent StreamlitAPIException
                        st.progress(float(display_progress))
                
                status_text = f"Synergy Analysis Complete for {len(team)} teammates."
                if use_personal:
                    status_text += " (Personalization Influence Applied)"
                st.success(status_text)
            else:
                st.warning(f"No meta-data found for {user_role} in the current dataset.")

else:
    st.info("Configure the visible teammates in the sidebar to begin GNN synthesis.")

# --- FOOTER ---
with st.expander("Model Context & Methodology"):
    st.write("**Architecture:** Graph Neural Network (NexusGNN) with Weighted Link Prediction.")
    st.write("**Feature Engineering:** PCA (87.33% Variance) + 64D Latent Embeddings.")
    st.write("**Personalization:** Blended centroid logic (Global Synergy + Player Anchor).")
    st.write("**Scaling:** Progress bars indicate relative synergy strength among top 5 candidates.")