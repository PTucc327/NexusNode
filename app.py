import streamlit as st
import torch
import numpy as np
import json
import os
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from NexusNode.modules.engine import get_detailed_reasoning
from NexusNode.modules.riot_api import RiotInterface, get_champ_name_map

# --- 1. SECURITY & ENVIRONMENT ---
load_dotenv()
RIOT_KEY = os.getenv("RIOT_KEY")


@st.cache_resource
def load_model_data():
    """Loads GNN embeddings and role mapping metadata."""
    try:
        # weights_only=False used for compatibility with your saved tensors
        embeddings = torch.load('champion_embeddings.pt', weights_only=False)
        embeddings_np = {k: v.detach().numpy() if torch.is_tensor(v) else v for k, v in embeddings.items()}
        champ_list = sorted(list(embeddings_np.keys()))
        with open('champion_roles.json', 'r') as f:
            role_mapping = json.load(f)
        return embeddings_np, champ_list, role_mapping
    except Exception as e:
        st.error(f"Critical Data Load Error: {e}")
        return {}, [], {}

# Initialize Data
embeddings, champ_list, role_mapping = load_model_data()

# --- 3. THE SYNTHESIS ENGINE (The "Brain") ---
def run_synthesis(user_role, allies, enemies, comfort_pool, loyalty_bonus):
    """
    Core GNN logic isolated for speed and live drafting reactivity.
    """
    active_allies = [p for p in allies if p != "None"]
    active_enemies = [p for p in enemies if p != "None"]
    
    if not active_allies:
        return []

    # Calculate centroids in the GNN vector space
    team_centroid = np.mean([embeddings[n] for n in active_allies if n in embeddings], axis=0).reshape(1, -1)
    enemy_centroid = np.mean([embeddings[n] for n in active_enemies if n in embeddings], axis=0).reshape(1, -1) if active_enemies else None

    scores = []
    # Only iterate through champions relevant to the user's selected role
    for champ in role_mapping.get(user_role, []):
        if champ in embeddings and champ not in active_allies and champ not in active_enemies:
            c_vec = embeddings[champ].reshape(1, -1)
            
            # Base GNN Synergy (Cosine Similarity to team center)
            score = cosine_similarity(team_centroid, c_vec)[0][0]
            
            # Counter-Pick Penalty (Distance from enemy center)
            if enemy_centroid is not None:
                score -= (0.15 * cosine_similarity(enemy_centroid, c_vec)[0][0])
            
            # THE "TUCCINATOR" FACTOR: Apply the Personal Loyalty Bonus
            if champ in comfort_pool:
                score += loyalty_bonus
            
            scores.append((champ, score))
            
    return sorted(scores, key=lambda x: x[1], reverse=True)[:5]

# --- 4. PAGE CONFIG & UI ---
st.set_page_config(page_title="NexusNode | Tactical Draft", layout="wide", page_icon="🎮")

with st.sidebar:
    st.title("⚙️ Global Settings")
    user_role = st.selectbox("Your Role", ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"], index=3)
    
    st.divider()
    st.subheader("👤 NexusID Profile")
    riot_id = st.text_input("Riot ID", placeholder="Name#Tag")
    
    if st.button("🔄 Sync via NexusID"):
        if RIOT_KEY and "#" in riot_id:
            try:
                name, tag = riot_id.split("#")
                ri = RiotInterface(RIOT_KEY)
                puuid = ri.get_puuid(name, tag)
                if puuid:
                    masteries = ri.get_top_masteries(puuid)
                    name_map = get_champ_name_map()
                    st.session_state['comfort_picks'] = [name_map[m['championId']] for m in masteries if m['championId'] in name_map]
                    st.success("Mastery Profile Injected.")
            except Exception as e:
                st.error(f"Sync failed: {e}")

    st.divider()
    st.subheader("🎯 Personalization")
    comfort_boost = st.slider("Loyalty Bonus", 0.0, 0.5, 0.10, help="Weight given to your comfort pool.")
    
    # Safe defaults to prevent StreamlitAPIException
    raw_picks = st.session_state.get('comfort_picks', ["Jinx", "Kai'Sa", "Kaisa"])
    safe_defaults = [c for c in raw_picks if c in champ_list]
    my_comfort = st.multiselect("Active Comfort Pool", options=champ_list, default=safe_defaults)

# --- 5. THE VERSUS BOARD ---
st.title("🎮 NexusNode Tactical Engine")
st.divider()

col_a, col_v, col_e = st.columns([4, 1, 4])
options = ["None"] + champ_list

with col_a:
    st.subheader("💙 Allies")
    a1 = st.selectbox("Ally 1", options, key="a1")
    a2 = st.selectbox("Ally 2", options, key="a2")
    a3 = st.selectbox("Ally 3", options, key="a3")
    a4 = st.selectbox("Ally 4", options, key="a4")

with col_v:
    st.markdown("<h1 style='text-align: center; color: gray; margin-top: 100px;'>VS</h1>", unsafe_allow_html=True)

with col_e:
    st.subheader("❤️ Enemies")
    e1 = st.selectbox("Enemy 1", options, key="e1")
    e2 = st.selectbox("Enemy 2", options, key="e2")
    e3 = st.selectbox("Enemy 3", options, key="e3")
    e4 = st.selectbox("Enemy 4", options, key="e4")
    e5 = st.selectbox("Enemy 5", options, key="e5")

st.divider()

# --- 6. EXECUTION ---
if st.button("🚀 EXECUTE TACTICAL SYNTHESIS", type="primary", use_container_width=True):
    # Call the isolated synthesis function
    results = run_synthesis(
        user_role, 
        [a1, a2, a3, a4], 
        [e1, e2, e3, e4, e5], 
        my_comfort, 
        comfort_boost
    )
    
    if not results:
        st.warning("Please select at least one ally to generate synergy recommendations.")
    else:
        st.write(f"### Predicted Optimal {user_role} Picks")
        res_cols = st.columns(5)

        for i, (name, final_val) in enumerate(results):
            with res_cols[i]:
                # Get the reason from our new module
                reason =  get_detailed_reasoning(name, [a1, a2, a3, a4], embeddings)
                
                is_comfort = "⭐" if name in my_comfort else ""
                st.metric(label=f"Rank {i+1} {is_comfort}", value=f"{final_val:.3f}", delta=name)
                
                # Display the explanation in a subtle way
                st.caption(f"💡 {reason}")
                st.progress(max(0.0, min(1.0, float(final_val))))