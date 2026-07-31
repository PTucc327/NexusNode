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
            embeddings_path='data/processed/champion_embeddings.pt', 
            roles_path='data/processed/champion_roles.json',
            matchups_path='data/processed/champion_matchups.csv'
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
    enemy_weight = st.slider("Counter-Pick Weight", 0.0, 2.0, 1.0, step=0.1, help="How much real lane matchup history (your role vs the enemy in that same role) should influence the ranking. 0 = ignore matchups entirely.")
    
    # Safe defaults
    # NOTE: champion names throughout the dataset/embeddings/champ_list use
    # Riot's internal API naming convention (e.g. "Kaisa", not "Kai'Sa"),
    # matching Data Dragon's `key` field and the match API's `championName`
    # field. A punctuated "Kai'Sa" here will never match champ_list, so it
    # silently fails to appear as a pre-selected default every session.
    raw_picks = st.session_state.get('comfort_picks', ["Jinx", "Kaisa", "Vayne"])
    safe_defaults = [c for c in raw_picks if c in champ_list]
    my_comfort = st.multiselect("Active Comfort Pool", options=champ_list, default=safe_defaults)

# --- 5. THE VERSUS BOARD ---
st.title("🎮 NexusNode Tactical Engine")
st.caption("GNN-Powered Drafting Recommendations")
st.divider()

col_a, col_v, col_e = st.columns([4, 1, 4])
options = ["None"] + champ_list
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]
ally_roles = [r for r in ROLES if r != user_role]  # your 4 teammates' roles

with col_a:
    st.subheader("💙 Allies")
    ally_by_role = {}
    for r in ally_roles:
        ally_by_role[r] = st.selectbox(f"Ally ({r.title()})", options, key=f"ally_{r}")

with col_v:
    st.markdown("<h1 style='text-align: center; color: gray; margin-top: 100px;'>VS</h1>", unsafe_allow_html=True)

with col_e:
    st.subheader("❤️ Enemies")
    st.caption("Tagged by role so we can pull real lane matchup data for your lane specifically.")
    enemy_by_role = {}
    for r in ROLES:
        label = f"Enemy ({r.title()})" + (" 🎯 your lane" if r == user_role else "")
        enemy_by_role[r] = st.selectbox(label, options, key=f"enemy_{r}")

st.divider()

# --- 6. EXECUTION ---
if st.button("🚀 EXECUTE TACTICAL SYNTHESIS", type="primary", use_container_width=True):
    if not engine:
        st.error("Engine not initialized. Check your model files.")
    else:
        allies_list = list(ally_by_role.values())
        # Use our engine's new synthesis method
        results = engine.run_synthesis(
            user_role=user_role, 
            allies=allies_list, 
            enemies=enemy_by_role, 
            comfort_pool=my_comfort, 
            loyalty_boost=comfort_boost,
            enemy_weight=enemy_weight
        )
        
        if not results:
            st.warning("Please select teammates to generate synergy recommendations.")
        else:
            lane_opponent = enemy_by_role.get(user_role)
            st.write(f"### Predicted Optimal {user_role} Picks")
            if lane_opponent and lane_opponent != "None":
                st.caption(f"Factoring in lane matchup history vs {lane_opponent}")
            res_cols = st.columns(5)

            for i, (name, final_val) in enumerate(results):
                with res_cols[i]:
                    # Using the reasoning logic from the engine
                    reason = engine.get_reasoning(name, allies_list, role=user_role, lane_opponent=lane_opponent)
                    
                    is_comfort = "⭐" if name in my_comfort else ""
                    st.metric(label=f"Rank {i+1} {is_comfort}", value=name, delta=f"{final_val:.3f}")
                    
                    st.caption(f"💡 {reason}")
                    # Progress normalized assuming similarity stays between 0 and 1
                    st.progress(max(0.0, min(1.0, float(final_val))))