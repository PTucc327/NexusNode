import html
import os

import streamlit as st
from dotenv import load_dotenv

from modules.champions import ChampionCatalog, fetch_ddragon
from modules.engine import DraftingEngine, ROLES
from modules.riot_api import RiotInterface, RiotAPIError, PLATFORM_ROUTING

# --- 1. PAGE CONFIG (must be the first Streamlit call) ---
st.set_page_config(page_title="NexusNode | Draft Assistant", layout="wide", page_icon="🎮")

# --- 2. SECURITY & ENVIRONMENT ---
load_dotenv()


def get_riot_key():
    key = os.getenv("RIOT_KEY")
    if key:
        return key
    try:
        return st.secrets.get("RIOT_KEY")
    except Exception:
        return None


RIOT_KEY = get_riot_key()
ROLE_LABELS = {"TOP": "Top", "JUNGLE": "Jungle", "MIDDLE": "Mid", "BOTTOM": "Bot", "SUPPORT": "Support"}
ROLE_ICONS = {"TOP": "🛡️", "JUNGLE": "🌲", "MIDDLE": "✨", "BOTTOM": "🏹", "SUPPORT": "💠"}


# --- 3. RESOURCE CACHING ---
@st.cache_resource
def init_engine():
    """Loads the draft model once per server process."""
    return DraftingEngine()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_ddragon():
    return fetch_ddragon()


try:
    engine = init_engine()
except FileNotFoundError:
    st.error("Model files not found. Run the pipeline first: `python modules/eda.py`, "
             "`python modules/preprocess.py`, `python modules/train_gnn.py`.")
    st.stop()

catalog = ChampionCatalog(load_ddragon())
champ_list = sorted(engine.champions, key=catalog.display_name)


def name(champ):
    return catalog.display_name(champ)


def portrait(champ, size=44, team="neutral", placeholder="?"):
    """HTML for a champion portrait, or an empty slot placeholder."""
    url = catalog.icon_url(champ) if champ else None
    if url:
        return (f'<img class="portrait {team}" src="{url}" width="{size}" height="{size}" '
                f'alt="{html.escape(name(champ))}" title="{html.escape(name(champ))}">')
    label = html.escape(placeholder if not champ else name(champ)[:2])
    return f'<div class="portrait empty {team}" style="width:{size}px;height:{size}px">{label}</div>'


# --- 4. STYLES ---
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.portrait { border-radius: 10px; border: 2px solid rgba(128,128,128,.35); object-fit: cover; display: block; }
.portrait.blue { border-color: #3b82f6; }
.portrait.red { border-color: #ef4444; }
.portrait.empty { display:flex; align-items:center; justify-content:center; font-size:.75rem;
  font-weight:600; opacity:.6; background: rgba(128,128,128,.12); border-style: dashed; }
.team-title { font-weight: 700; font-size: 1.1rem; margin-bottom: .25rem; }
.team-title.blue { color: #3b82f6; } .team-title.red { color: #ef4444; }
.wp-wrap { margin: .5rem 0 1rem 0; }
.wp-labels { display:flex; justify-content:space-between; font-weight:600; font-size:.9rem; margin-bottom:.25rem; }
.wp-bar { display:flex; height: 14px; border-radius: 999px; overflow:hidden; background: rgba(128,128,128,.2); }
.wp-blue { background: linear-gradient(90deg,#2563eb,#3b82f6); }
.wp-red { background: linear-gradient(90deg,#f87171,#dc2626); }
.wp-note { font-size: .78rem; opacity: .7; margin-top: .3rem; }
.rec-card { border: 1px solid rgba(128,128,128,.3); border-radius: 14px; padding: .9rem .8rem;
  text-align:center; height: 100%; background: rgba(128,128,128,.05); }
.rec-card.top { border-color: #f59e0b; box-shadow: 0 0 0 1px #f59e0b inset; }
.rec-card .portrait { margin: 0 auto .5rem auto; }
.rec-rank { font-size:.75rem; font-weight:700; opacity:.65; letter-spacing:.05em; }
.rec-name { font-weight: 700; font-size: 1.05rem; }
.rec-wp { font-size: 1.6rem; font-weight: 800; line-height:1.2; }
.rec-delta { font-size: .8rem; font-weight:600; }
.pos { color: #16a34a; } .neg { color: #dc2626; }
.reasons { list-style:none; padding:0; margin:.6rem 0 0 0; text-align:left; font-size:.8rem; }
.reasons li { padding: .2rem .4rem; border-radius: 6px; margin-bottom: .25rem; background: rgba(128,128,128,.08); }
.reasons li.synergy { border-left: 3px solid #3b82f6; }
.reasons li.strong { border-left: 3px solid #16a34a; }
.reasons li.weak { border-left: 3px solid #dc2626; }
.reasons li.lane { border-left: 3px solid #a855f7; }
.pool-row { display:flex; flex-wrap:wrap; gap:4px; margin:.25rem 0 .5rem 0; }
</style>
""", unsafe_allow_html=True)


# --- 5. SESSION STATE / CALLBACKS ---
def reset_draft():
    for r in ROLES:
        st.session_state[f"blue_{r}"] = None
        st.session_state[f"red_{r}"] = None


def lock_in(role, champ):
    st.session_state[f"blue_{role}"] = champ


def sync_profile():
    # Read from session state: callback args are bound at render time and
    # would miss an ID typed just before clicking.
    riot_id = st.session_state.get("riot_id", "")
    region = st.session_state.get("riot_region", "na1")
    game_name, _, tag = riot_id.strip().rpartition("#")
    if not game_name or not tag:
        st.session_state["sync_msg"] = ("error", "Enter your Riot ID as Name#Tag.")
        return
    try:
        ri = RiotInterface(RIOT_KEY, region=region, catalog=catalog)
        puuid = ri.get_puuid(game_name, tag)
        if not puuid:
            st.session_state["sync_msg"] = ("error", f"No account found for {riot_id}.")
            return
        picks = [catalog.resolve(c, engine.champions) for c in ri.get_user_comfort_pool(puuid)]
        picks = [p for p in picks if p]
        st.session_state["comfort_pool"] = picks
        st.session_state["sync_msg"] = ("success", f"Loaded {len(picks)} champions from your mastery.")
    except RiotAPIError as e:
        st.session_state["sync_msg"] = ("error", str(e))
    except Exception as e:
        st.session_state["sync_msg"] = ("error", f"Sync failed: {e}")


# --- 6. SIDEBAR ---
with st.sidebar:
    st.header("🎯 Your role")
    user_role = st.segmented_control(
        "Your role", ROLES, default="BOTTOM", key="user_role", label_visibility="collapsed",
        format_func=lambda r: f"{ROLE_ICONS[r]} {ROLE_LABELS[r]}",
    ) or "BOTTOM"

    st.divider()
    st.header("⭐ Champion pool")
    st.caption("Champions you're comfortable on get a ranking bonus.")
    comfort = st.multiselect("Comfort picks", champ_list, key="comfort_pool", format_func=name,
                             placeholder="Add champions…", label_visibility="collapsed")
    if comfort:
        st.markdown('<div class="pool-row">' + "".join(portrait(c, 32) for c in comfort) + "</div>",
                    unsafe_allow_html=True)

    with st.expander("🔄 Import from Riot account"):
        if not RIOT_KEY:
            st.info("Add `RIOT_KEY=...` to a `.env` file to enable account sync.")
        else:
            st.text_input("Riot ID", placeholder="Name#Tag", key="riot_id")
            st.selectbox("Server", list(PLATFORM_ROUTING), index=0, format_func=str.upper, key="riot_region")
            st.button("Import top mastery champions", on_click=sync_profile,
                      width="stretch")
        msg = st.session_state.pop("sync_msg", None)
        if msg:
            getattr(st, msg[0])(msg[1])

    st.divider()
    with st.expander("⚙️ Tuning"):
        enemy_weight = st.slider(
            "Enemy awareness", 0.0, 2.0, 1.0, step=0.1,
            help="Scales how much the enemy team's champions affect the ranking. "
                 "1.0 = as learned from match data, 0 = ignore enemies entirely.")
        comfort_bonus = st.slider(
            "Comfort bonus (win-prob points)", 0.0, 10.0, 3.0, step=0.5,
            help="Ranking boost for champions in your pool. Displayed win probability is unaffected.") / 100

    st.button("🧹 Reset draft", on_click=reset_draft, width="stretch")


# --- 7. DRAFT BOARD ---
st.title("🎮 NexusNode Draft Assistant")
st.caption("Fill in the picks you know. Recommendations for your role update instantly, "
           "factoring in your allies' synergy **and** every enemy champion.")


def slot_options(role):
    """Role-eligible champions first, then everyone else (all searchable)."""
    eligible = set(engine.eligible(role))
    return sorted(champ_list, key=lambda c: (c not in eligible, name(c)))


def draft_slot(team, role):
    key = f"{team}_{role}"
    is_you = team == "blue" and role == user_role
    current = st.session_state.get(key)
    c_icon, c_select = st.columns([1, 6], vertical_alignment="center")
    with c_icon:
        st.markdown(portrait(current, 44, team, ROLE_LABELS[role][:3]), unsafe_allow_html=True)
    with c_select:
        label = f"{ROLE_ICONS[role]} {ROLE_LABELS[role]}" + ("  ·  🎯 You" if is_you else "")
        placeholder = "Your pick (see suggestions below)" if is_you else "Not picked yet"
        return st.selectbox(label, slot_options(role), index=None, key=key, format_func=name,
                            placeholder=placeholder)


col_blue, col_red = st.columns(2, gap="large")
with col_blue:
    with st.container(border=True):
        st.markdown('<div class="team-title blue">💙 Your team</div>', unsafe_allow_html=True)
        blue = {r: draft_slot("blue", r) for r in ROLES}
with col_red:
    with st.container(border=True):
        st.markdown('<div class="team-title red">❤️ Enemy team</div>', unsafe_allow_html=True)
        red = {r: draft_slot("red", r) for r in ROLES}

# Duplicate guard: a champion can only be picked once per game
picked = [c for c in list(blue.values()) + list(red.values()) if c]
dupes = sorted({c for c in picked if picked.count(c) > 1})
if dupes:
    st.warning("Each champion can only be picked once per game: "
               + ", ".join(name(c) for c in dupes) + " is selected more than once.")

# --- 8. LIVE WIN PROBABILITY ---
p_now = engine.win_probability(blue, red, enemy_weight)
n_picks = len(picked)
if p_now is not None:
    st.markdown(f"""
<div class="wp-wrap">
  <div class="wp-labels"><span style="color:#3b82f6">Your team {p_now:.1%}</span>
  <span style="color:#ef4444">{1 - p_now:.1%} Enemy team</span></div>
  <div class="wp-bar"><div class="wp-blue" style="width:{p_now * 100:.1f}%"></div>
  <div class="wp-red" style="width:{(1 - p_now) * 100:.1f}%"></div></div>
  <div class="wp-note">Model-estimated win probability from {n_picks} of 10 picks. Drafts typically swing
  outcomes by a few percent; execution decides the rest.</div>
</div>""", unsafe_allow_html=True)

# --- 9. RECOMMENDATIONS ---
allies = {r: c for r, c in blue.items() if r != user_role}
recs = engine.recommend(user_role, allies, red, comfort_pool=comfort,
                        comfort_bonus=comfort_bonus, enemy_weight=enemy_weight)

st.subheader(f"{ROLE_ICONS[user_role]} Best {ROLE_LABELS[user_role]} picks for this draft")
if not recs:
    st.info(f"No eligible {ROLE_LABELS[user_role]} champions found in the model data.")
else:
    avg_p = sum(r.win_prob for r in recs) / len(recs)
    context = []
    if any(allies.values()):
        context.append(f"{sum(1 for c in allies.values() if c)} allies")
    if any(red.values()):
        context.append(f"{sum(1 for c in red.values() if c)} enemies")
    st.caption("Considering " + (" and ".join(context) if context else "no picks yet (blind pick: overall strength)")
               + f". Δ is compared with the average {ROLE_LABELS[user_role]} option; "
               "numbers in reasons are each champion's effect in win-probability points.")

    cols = st.columns(5)
    for i, (rec, col) in enumerate(zip(recs[:5], cols)):
        delta = rec.win_prob - avg_p
        reasons = DraftingEngine.reasons(rec, display=name)
        reason_html = "".join(f'<li class="{k}">{html.escape(t)}</li>' for k, t in reasons) \
            or '<li>Solid overall pick</li>'
        star = " ⭐" if rec.is_comfort else ""
        with col:
            st.markdown(f"""
<div class="rec-card {'top' if i == 0 else ''}">
  <div class="rec-rank">#{i + 1}</div>
  {portrait(rec.champion, 72, 'blue')}
  <div class="rec-name">{html.escape(name(rec.champion))}{star}</div>
  <div class="rec-wp">{rec.win_prob:.1%}</div>
  <div class="rec-delta {'pos' if delta >= 0 else 'neg'}">{delta * 100:+.1f} pts Δ</div>
  <ul class="reasons">{reason_html}</ul>
</div>""", unsafe_allow_html=True)
            locked = blue.get(user_role) == rec.champion
            st.button("✅ Locked in" if locked else "Lock in", key=f"lock_{rec.champion}",
                      on_click=lock_in, args=(user_role, rec.champion), disabled=locked,
                      width="stretch")

    with st.expander(f"📋 Full {ROLE_LABELS[user_role]} ranking ({len(recs)} champions)"):
        def top_name(pairs, positive=True):
            pairs = [p for p in pairs if (p[1] > 0 if positive else p[1] < 0)]
            if not positive:
                pairs = list(reversed(pairs))
            return name(pairs[0][0]) if pairs else ""

        rows = [{
            "icon": catalog.icon_url(r.champion),
            "Champion": name(r.champion) + (" ⭐" if r.is_comfort else ""),
            "Win %": r.win_prob * 100,
            "Δ pts": (r.win_prob - avg_p) * 100,
            "Best synergy": top_name(r.synergy),
            "Strong vs": top_name(r.counters),
            "Weak vs": top_name(r.counters, positive=False),
        } for r in recs]
        st.dataframe(rows, hide_index=True, width="stretch", column_config={
            "icon": st.column_config.ImageColumn("", width="small"),
            "Win %": st.column_config.NumberColumn(format="%.1f%%"),
            "Δ pts": st.column_config.NumberColumn(format="%+.1f"),
        })

# --- 10. ABOUT THE MODEL ---
with st.expander("🧠 How NexusNode works"):
    st.markdown("""
A **relational graph neural network** learns a vector for every champion from two kinds of
relationships seen in high-Elo ranked games: champions played **together** (synergy) and champions
played **against each other** (counters). It's trained end-to-end to predict which team wins, scoring a
draft as:

- **Power**: each champion's individual strength in the current meta
- **Synergy**: how well each pair of teammates works together
- **Counters**: how every one of your champions fares against every enemy champion
- **Lane matchup**: an extra term for the direct opponent in the same role

Recommendations are the picks that maximize your team's predicted win probability.
""")
    m = engine.metrics
    if m:
        v = m.get("validation", {})
        full, ally_only, base = v.get("full_model", {}), v.get("allies_only_ablation", {}), v.get("baseline_side_only", {})
        st.caption(f"Trained {m.get('trained_at', '?')} on {m.get('matches', '?')} matches, "
                   f"{m.get('champions', '?')} champions. Held-out validation ({m.get('validation_matches', '?')} matches):")
        st.dataframe([
            {"Model": "Full model (allies + enemies)", "Accuracy": full.get("accuracy"), "AUC": full.get("auc"), "Log loss": full.get("log_loss")},
            {"Model": "Allies only (ablation)", "Accuracy": ally_only.get("accuracy"), "AUC": ally_only.get("auc"), "Log loss": ally_only.get("log_loss")},
            {"Model": "No draft info (side only)", "Accuracy": base.get("accuracy"), "AUC": base.get("auc"), "Log loss": base.get("log_loss")},
        ], hide_index=True, width="stretch", column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.3f"),
            "AUC": st.column_config.NumberColumn(format="%.3f"),
            "Log loss": st.column_config.NumberColumn(format="%.4f"),
        })

st.caption("NexusNode isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games "
           "or anyone officially involved in producing or managing League of Legends.")
