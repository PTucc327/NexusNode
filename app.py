import streamlit as st
import torch
import numpy as np
import json
import os
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from riot_api import RiotInterface

# --- SECURITY & ENVIRONMENT ---
load_dotenv()
RIOT_KEY = os.getenv("RIOT_KEY")

# --- UTILS ---
def get_champ_name_map():
    """Fetches latest champion ID-to-Name mapping from Riot Data Dragon."""
    ddragon_url = "https://ddragon.leagueoflegends.com/cdn/14.8.1/data/en_US/champion.json"
    try:
        data = requests.get(ddragon_url).json()
        return {int(v['key']): k for k, v in data['data'].items()}
    except:
        return {}

# --- DATA LOADING ---
@st.cache_resource
def load_model_data():
    try:
        embeddings = torch.load('champion_embeddings.pt', weights_only=False)
        embeddings_np = {k: v.detach().numpy() if torch.is_tensor(v) else v for k, v in embeddings.items()}
        champ_list = sorted(list(embeddings_np.keys()))
        with open('champion_roles.json', 'r') as f:
            role_mapping = json.load(f)
        return embeddings_np, champ_list, role_mapping
    except Exception as e:
        st.error(f"Critical Data Load Error: {e}")
        return {}, [], {}

embeddings, champ_list, role_mapping = load_model_data()

# --- TACTICAL CONSTANTS ---
ARCHETYPES = {
    "Dive/Engage": ["Malphite", "JarvanIV", "Vi", "Nocturne", "Diana", "Yasuo", "Yone", "Leona", "Amumu", "Rengar", "LeeSin"],
    "Poke/Siege": ["Xerath", "Ziggs", "Jayce", "Varus", "Ezreal", "Lux", "VelKoz", "Nidalee", "Hwei", "Caitlyn"],
    "Frontline/Tank": ["Ornn", "Maokai", "Sion", "KSante", "Braum", "Alistar", "TahmKench", "Poppy", "Sejuani"],
    "Enchanter/Utility": ["Lulu", "Janna", "Soraka", "Sona", "Milio", "Nami", "Karma", "Ivern", "Renata"],
    "Scale/Carry": ["Vayne", "Kayle", "Kassadin", "Jinx", "Aphelios", "Smolder", "Vladimir", "Ryze", "KogMaw"]
}

AP_CHAMPS = ["Ahri", "Anivia", "Annie", "Azir", "Brand", "Cassiopeia", "Diana", "Ekko", "Evelynn", "Fizz", "Hwei", "Karthus", "Kassadin", "Katarina", "LeBlanc", "Lux", "Mordekaiser", "Ryze", "Sylas", "Syndra", "Taliyah", "Viktor", "Vladimir", "Xerath", "Ziggs", "Zoe"]
AD_CHAMPS = ["Aatrox", "Aphelios", "Ashe", "Caitlyn", "Draven", "Ezreal", "Graves", "Irelia", "JarvanIV", "Jinx", "KaiSa", "Kai'Sa", "KhaZix", "LeeSin", "Lucian", "Pantheon", "Rengar", "Samira", "Talon", "Tristana", "Varus", "Vayne", "Yasuo", "Yone", "Zed"]

# --- SIDEBAR: SETTINGS & NEXUSID ---
with st.sidebar:
    st.title("⚙️ Global Settings")
    user_role = st.selectbox("Your Role", ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"], index=3)
    
    st.divider()
    st.subheader("👤 NexusID Profile")
    riot_id = st.text_input("Riot ID", placeholder="Name#Tag", help="Used to sync your champion masteries.")
    
    if st.button("🔄 Sync via NexusID"):
        if not RIOT_KEY:
            st.error("Riot API Key missing in .env file.")
        elif "#" not in riot_id:
            st.error("Please enter ID in 'Name#Tag' format.")
        else:
            try:
                name, tag = riot_id.split("#")
                ri = RiotInterface(RIOT_KEY)
                puuid = ri.get_puuid(name, tag)
                if puuid:
                    masteries = ri.get_top_masteries(puuid)
                    name_map = get_champ_name_map()
                    st.session_state['comfort_picks'] = [name_map[m['championId']] for m in masteries if m['championId'] in name_map]
                    st.success(f"Successfully injected {len(st.session_state['comfort_picks'])} champions into engine.")
                else:
                    st.error("Riot ID not found.")
            except Exception as e:
                st.error(f"Sync failed: {e}")

    st.divider()
    st.subheader("🎯 Personalization")
    # LOYALTY BONUS: This makes your profile matter
    comfort_boost = st.slider("Loyalty Bonus", 0.0, 0.5, 0.2, help="How much to prioritize champions in your comfort pool.")
    
    raw_picks = st.session_state.get('comfort_picks', ["Jinx", "Kai'Sa", "Kaisa"])
    safe_defaults = [c for c in raw_picks if c in champ_list]
    my_comfort = st.multiselect("Active Comfort Pool", options=champ_list, default=safe_defaults)

# --- MAIN UI: THE VERSUS BOARD ---
st.title("🎮 NexusNode")
st.caption("GNN-powered draft synthesis with live profile integration and archetype detection.")
st.divider()

col_allies, col_vs, col_enemies = st.columns([4, 1, 4])
options = ["None"] + champ_list

with col_allies:
    st.subheader("💙 Allies")
    a1 = st.selectbox("Ally 1", options, key="a1")
    a2 = st.selectbox("Ally 2", options, key="a2")
    a3 = st.selectbox("Ally 3", options, key="a3")
    a4 = st.selectbox("Ally 4", options, key="a4")

with col_vs:
    st.markdown("<h1 style='text-align: center; color: #555; padding-top: 100px;'>VS</h1>", unsafe_allow_html=True)

with col_enemies:
    st.subheader("❤️ Enemies")
    e1 = st.selectbox("Enemy 1", options, key="e1")
    e2 = st.selectbox("Enemy 2", options, key="e2")
    e3 = st.selectbox("Enemy 3", options, key="e3")
    e4 = st.selectbox("Enemy 4", options, key="e4")
    e5 = st.selectbox("Enemy 5", options, key="e5")

st.divider()

# --- SYNTHESIS ENGINE ---
if st.button("🚀 EXECUTE TACTICAL SYNTHESIS", type="primary", use_container_width=True):
    active_allies = [p for p in [a1, a2, a3, a4] if p != "None"]
    active_enemies = [p for p in [e1, e2, e3, e4, e5] if p != "None"]
    
    if not active_allies:
        st.warning("Please select at least one teammate to calculate team-based synergy.")
    else:
        # 1. Archetype Detection
        comp_counts = {arch: sum(1 for a in active_allies if a in members) for arch, members in ARCHETYPES.items()}
        primary_arch = max(comp_counts, key=comp_counts.get) if any(comp_counts.values()) else None

        # 2. Centroid Math
        team_centroid = np.mean([embeddings[n] for n in active_allies if n in embeddings], axis=0).reshape(1, -1)
        enemy_centroid = np.mean([embeddings[n] for n in active_enemies if n in embeddings], axis=0).reshape(1, -1) if active_enemies else None

        # 3. Scoring Loop
        scores = []
        for champ in role_mapping.get(user_role, []):
            if champ in embeddings and champ not in active_allies and champ not in active_enemies:
                c_vec = embeddings[champ].reshape(1, -1)
                
                # Base GNN Synergy
                val = cosine_similarity(team_centroid, c_vec)[0][0]
                
                # Counter Penalty (Influence from GNN space)
                if enemy_centroid is not None:
                    val -= (0.15 * cosine_similarity(enemy_centroid, c_vec)[0][0])
                
                # Archetype Alignment Bonus
                if primary_arch and champ in ARCHETYPES[primary_arch]:
                    val += 0.05
                
                # Damage Balance Logic
                ap_n = sum(1 for a in active_allies if a in AP_CHAMPS)
                ad_n = sum(1 for a in active_allies if a in AD_CHAMPS)
                if ap_n >= 2 and champ in AP_CHAMPS: val -= 0.05 * (ap_n - 1)
                if ad_n >= 2 and champ in AD_CHAMPS: val -= 0.05 * (ad_n - 1)

                # LOYALTY BONUS (Directly favors your comfort pool)
                if champ in my_comfort:
                    val += comfort_boost
                
                scores.append((champ, val))

        # 4. Display Results
        top_picks = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
        
        st.write(f"### Predicted Optimal {user_role} Picks")
        if primary_arch: 
            st.info(f"Team Synergy Identified: **{primary_arch}** composition.")
        
        res_cols = st.columns(5)
        for i, (name, final_val) in enumerate(top_picks):
            with res_cols[i]:
                is_comfort = "⭐" if name in my_comfort else ""
                
                # RESTORED: Numeric value is now the main 'value' 
                # and the 'delta' shows the name for high visibility
                st.metric(
                    label=f"Rank {i+1} {is_comfort}", 
                    value=name, 
                    delta=f"{final_val:.3f}",
                    delta_color="off" # Keeps the name color neutral
                )
                
                # Normalize progress bar for visualization (clamped 0-1)
                norm_val = max(0.0, min(1.0, float(final_val)))
                st.progress(norm_val)