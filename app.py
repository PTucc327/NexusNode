import streamlit as st
import torch
import numpy as np
import json
import os
from dotenv import load_dotenv

# --- 1. MODULAR IMPORTS ---
# Ensure your PYTHONPATH includes the project root
from modules.engine import DraftingEngine
from modules.riot_api import RiotInterface

# --- 2. SECURITY & ENVIRONMENT ---
load_dotenv()
RIOT_KEY = os.getenv("RIOT_KEY")

# --- 3. RESOURCE CACHING ---
@st.cache_resource
def init_engine():
    """Initializes the Brain once and caches it for the session."""
    try:
        # Note: adjust these paths if your file structure differs on your server
        engine = DraftingEngine(
            embeddings_path='champion_embeddings.pt', 
            roles_path='data/processed/champion_roles.json'
        )
        # Extract champion list for the dropdowns
        champ_list = sorted(list(engine.embeddings.keys()))
        return engine, champ_list
    except Exception as e:
        st.error(f"Critical Data Load Error: {e}")
        return None, []

# Initialize the Engine
engine, champ_list = init_engine()

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
                    # Uses our new helper method in RiotInterface
                    comfort_list = ri.get_user_comfort_pool(puuid)
                    st.session_state['comfort_picks'] = comfort_list
                    st.success(f"Mastery Profile Injected: {len(comfort_list)} champs.")
                else:
                    st.error("Could not find PUUID for that ID.")
            except Exception as e:
                st.error(f"Sync failed: {e}")

    st.divider()
    st.subheader("🎯 Personalization")
    comfort_boost = st.slider("Loyalty Bonus", 1.0, 1.5, 1.10, step=0.05, help="Multiplier for comfort pool (e.g. 1.1 = 10% boost).")
    
    # Safe defaults
    raw_picks = st.session_state.get('comfort_picks', ["Jinx", "Kai'Sa", "Vayne"])
    safe_defaults = [c for c in raw_picks if c in champ_list]
    my_comfort = st.multiselect("Active Comfort Pool", options=champ_list, default=safe_defaults)

# --- 5. THE VERSUS BOARD ---
st.title("🎮 NexusNode Tactical Engine")
st.caption("GNN-Powered Drafting Recommendations")
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
    if not engine:
        st.error("Engine not initialized. Check your model files.")
    else:
        # Use our engine's new synthesis method
        results = engine.run_synthesis(
            user_role=user_role, 
            allies=[a1, a2, a3, a4], 
            enemies=[e1, e2, e3, e4, e5], 
            comfort_pool=my_comfort, 
            loyalty_boost=comfort_boost
        )
        
        if not results:
            st.warning("Please select teammates to generate synergy recommendations.")
        else:
            st.write(f"### Predicted Optimal {user_role} Picks")
            res_cols = st.columns(5)

            for i, (name, final_val) in enumerate(results):
                with res_cols[i]:
                    # Using the reasoning logic from the engine
                    reason = engine.get_reasoning(name, [a1, a2, a3, a4])
                    
                    is_comfort = "⭐" if name in my_comfort else ""
                    st.metric(label=f"Rank {i+1} {is_comfort}", value=f"{final_val:.3f}", delta=name)
                    
                    st.caption(f"💡 {reason}")
                    # Progress normalized assuming similarity stays between 0 and 1
                    st.progress(max(0.0, min(1.0, float(final_val))))