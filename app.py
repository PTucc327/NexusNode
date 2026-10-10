import html
import logging
import os
import re
import threading
import time
from collections import deque

import streamlit as st
from dotenv import load_dotenv

from modules.champions import ChampionCatalog, fetch_ddragon
from modules.engine import DraftingEngine, ROLES
from modules.riot_api import RiotInterface, RiotAPIError, PLATFORM_ROUTING
from modules.lcu import LCUClient, LCUError, parse_session, assign_enemy_roles

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
# Desktop companion mode: reads the LOCAL League client's champ select
# (read-only; see modules/lcu.py). Off unless explicitly enabled, and never
# meaningful on a hosted server. Riot must acknowledge League Client API use
# before this is released to players.
DESKTOP_MODE = os.getenv('NEXUSNODE_DESKTOP') == '1'
LEGAL_NOTICE = ("NexusNode isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot "
                "Games or anyone officially involved in producing or managing Riot Games properties. Riot "
                "Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.")
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
        return (f'<img class="portrait {team}" src="{html.escape(url, quote=True)}" width="{size}" height="{size}" '
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
.dmg { margin: .1rem 0 .6rem 0; font-size: .75rem; }
.dmg-bar { display:flex; height: 8px; border-radius: 999px; overflow:hidden; margin: .2rem 0; }
.dmg-ad { background: #f97316; } .dmg-ap { background: #8b5cf6; }
.dmg-labels { display:flex; justify-content:space-between; opacity:.8; }
.dmg-warn { color: #dc2626; font-weight: 600; }
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
.reasons li.meta { border-left: 3px solid #f59e0b; }
.reasons li.comp { border-left: 3px solid #0ea5e9; }
.reasons li.risk { border-left: 3px solid #dc2626; }
.reasons li.safe { border-left: 3px solid #16a34a; }
.pool-row { display:flex; flex-wrap:wrap; gap:4px; margin:.25rem 0 .5rem 0; }
</style>
""", unsafe_allow_html=True)


# --- 5. SESSION STATE / CALLBACKS ---
def reset_draft():
    for r in ROLES:
        st.session_state[f"blue_{r}"] = None
        st.session_state[f"red_{r}"] = None
    st.session_state["bans"] = []


def lock_in(role, champ):
    st.session_state[f"blue_{role}"] = champ


# Riot ID lookups spend the app's API key, which Riot holds us responsible
# for. Limit them per visitor and across the whole server, and cache results
# so repeat lookups don't hit Riot at all.
SYNC_COOLDOWN_S = 20          # per browser session
SYNC_GLOBAL_PER_MINUTE = 30   # across all visitors of this server
RIOT_NAME_RE = re.compile(r"^.{3,16}$")
RIOT_TAG_RE = re.compile(r"^[A-Za-z0-9]{3,5}$")


class _RateLimiter:
    """Sliding-window limiter shared by all sessions in this process."""
    def __init__(self, limit, window_s):
        self.limit, self.window_s = limit, window_s
        self.calls, self.lock = deque(), threading.Lock()

    def allow(self):
        now = time.monotonic()
        with self.lock:
            while self.calls and now - self.calls[0] > self.window_s:
                self.calls.popleft()
            if len(self.calls) >= self.limit:
                return False
            self.calls.append(now)
            return True


@st.cache_resource
def sync_limiter():
    return _RateLimiter(SYNC_GLOBAL_PER_MINUTE, 60)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=1000)
def fetch_comfort_pool(game_name, tag, region):
    """Top-mastery champions for a Riot ID (None if the account doesn't exist).
    Cached for an hour; mastery data is public and changes slowly."""
    ri = RiotInterface(RIOT_KEY, region=region, catalog=catalog)
    puuid = ri.get_puuid(game_name, tag)
    if not puuid:
        return None
    return ri.get_user_comfort_pool(puuid)


def sync_profile():
    # Read from session state: callback args are bound at render time and
    # would miss an ID typed just before clicking.
    riot_id = st.session_state.get("riot_id", "")
    region = st.session_state.get("riot_region", "na1")
    game_name, _, tag = riot_id.strip().rpartition("#")
    if not RIOT_NAME_RE.match(game_name) or not RIOT_TAG_RE.match(tag) or region not in PLATFORM_ROUTING:
        st.session_state["sync_msg"] = ("error", "Enter your Riot ID as Name#Tag (e.g. Faker#KR1).")
        return
    wait = SYNC_COOLDOWN_S - (time.monotonic() - st.session_state.get("last_sync", -1e9))
    if wait > 0:
        st.session_state["sync_msg"] = ("warning", f"Please wait {wait:.0f}s before importing again.")
        return
    if not sync_limiter().allow():
        st.session_state["sync_msg"] = ("warning", "Lots of imports right now; please try again in a minute.")
        return
    st.session_state["last_sync"] = time.monotonic()
    try:
        names = fetch_comfort_pool(game_name, tag, region)
        if names is None:
            st.session_state["sync_msg"] = ("error", "No account found for that Riot ID on this server.")
            return
        picks = [p for p in (catalog.resolve(c, engine.champions) for c in names) if p]
        st.session_state["comfort_pool"] = picks
        st.session_state["sync_msg"] = ("success", f"Loaded {len(picks)} champions from your mastery.")
    except RiotAPIError as e:
        st.session_state["sync_msg"] = ("error", str(e))  # our own user-safe messages
    except Exception:
        # Don't surface internal error details to visitors
        logging.exception("Riot sync failed")
        st.session_state["sync_msg"] = ("error", "Couldn't reach Riot's servers. Please try again later.")


# --- 5b. LEAGUE CLIENT SYNC (desktop mode) ---
@st.cache_resource
def champion_ids():
    """Numeric champion id (as the client reports it) -> this model's champion name."""
    ids = {}
    for cid, dd_name in catalog._by_numeric_id.items():
        name_ = catalog.resolve(dd_name, engine.champions)
        if name_:
            ids[cid] = name_
    return ids


def draft_from_client(state):
    """Widget values for a parsed champ-select state (full sync, so slots the
    client shows as empty are cleared too)."""
    updates = {}
    # Blind modes (Practice Tool, Blind Pick) don't assign roles: keep the role
    # the player chose in the app and infer allies' roles from play rates.
    my_role = state.my_role or st.session_state.get("user_role") or "BOTTOM"
    if state.my_role:
        updates["user_role"] = state.my_role
    allies = dict(state.allies)
    open_roles = [r for r in ROLES if r != my_role and r not in allies]
    allies.update(assign_enemy_roles(state.unassigned_allies, engine.role_games, open_roles))
    for r in ROLES:
        updates[f"blue_{r}"] = allies.get(r)
    updates[f"blue_{my_role}"] = state.my_pick
    enemy_roles = assign_enemy_roles(state.enemies, engine.role_games)
    for r in ROLES:
        updates[f"red_{r}"] = enemy_roles.get(r)
    updates["bans"] = [b for b in state.bans if b in engine.index]
    return updates


# Apply client updates BEFORE any widget is created this run (Streamlit only
# allows setting a widget's value before it's instantiated).
for _key, _value in st.session_state.pop("lcu_pending", {}).items():
    st.session_state[_key] = _value


@st.fragment(run_every=2)
def league_client_sync():
    if not st.session_state.get("lcu_on"):
        return
    client = LCUClient.from_lockfile()
    if client is None:
        st.caption("⚪ League client isn't running.")
        return
    try:
        session = client.champ_select_session()
    except (LCUError, OSError, ValueError) as e:
        logging.warning("League client read failed: %s", e)
        st.caption("🟠 Can't read the League client right now.")
        return
    if session is None:
        st.caption("⚪ Waiting for champion select…")
        return
    updates = draft_from_client(parse_session(session, champion_ids().get))
    changed = {k: v for k, v in updates.items() if st.session_state.get(k) != v}
    if changed:
        st.session_state["lcu_pending"] = changed
        st.rerun(scope="app")
    st.caption("🟢 Live: synced with champion select (read-only).")


# --- 6. SIDEBAR ---
with st.sidebar:
    if DESKTOP_MODE:
        st.header("📡 League client")
        st.toggle("Sync with champion select", key="lcu_on",
                  help="Reads your own champ select (picks, bans, your role) from the League client on "
                       "this computer. Read-only: it never picks, bans or clicks anything for you.")
        lcu_status = st.container()  # filled by league_client_sync() at the end of the page
        st.divider()
    st.header("🎯 Your role")
    # The starting role lives in session state (the League client sync can
    # also set it); the widget's own parameters stay constant so Streamlit
    # keeps treating it as the same widget across reruns.
    st.session_state.setdefault("user_role", "BOTTOM")
    user_role = st.segmented_control(
        "Your role", ROLES, key="user_role", label_visibility="collapsed",
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
st.caption("Fill in bans and the picks you know. Recommendations for your role update instantly, "
           "factoring in role strength, your lane matchup, "
           + ("team composition, " if engine.profiles is not None else "") + "and the rest of both teams.")
_m = engine.metrics
if _m.get("patches"):
    _hl = _m.get("patch_half_life")
    st.caption(f"📅 Based on {_m.get('matches', '?'):,} high-Elo Ranked Solo games from patches "
               f"{_m['patches'][0]}–{_m['patches'][1]} (model trained {_m.get('trained_at', '?')})"
               + (f"; recent patches count more (half-life {_hl} patches)." if _hl else "."))


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


bans = st.multiselect("🚫 Bans", champ_list, key="bans", format_func=name, max_selections=10,
                      placeholder="Add banned champions (they won't be recommended)…")

def damage_meter(picks):
    """AD/AP split of a team's picks, from the damage each champion deals in games."""
    dmg = engine.team_damage(picks)
    if dmg is None:
        return '<div class="dmg"><div class="dmg-labels"><span>Damage: no picks yet</span></div></div>'
    ad, n = dmg
    warn = ""
    if ad >= 0.75:
        warn = '<span class="dmg-warn">Very AD-heavy: enemies can stack armor</span>'
    elif ad <= 0.25:
        warn = '<span class="dmg-warn">Very AP-heavy: enemies can stack magic resist</span>'
    return (f'<div class="dmg"><div class="dmg-labels"><span>Physical {ad:.0%}</span>'
            f'<span>{n} pick{"s" if n != 1 else ""}</span><span>Magic {1 - ad:.0%}</span></div>'
            f'<div class="dmg-bar"><div class="dmg-ad" style="width:{ad * 100:.0f}%"></div>'
            f'<div class="dmg-ap" style="width:{(1 - ad) * 100:.0f}%"></div></div>{warn}</div>')


col_blue, col_red = st.columns(2, gap="large")
with col_blue:
    with st.container(border=True):
        st.markdown('<div class="team-title blue">💙 Your team</div>', unsafe_allow_html=True)
        blue_meter = st.empty()
        blue = {r: draft_slot("blue", r) for r in ROLES}
        blue_meter.markdown(damage_meter(blue), unsafe_allow_html=True)
with col_red:
    with st.container(border=True):
        st.markdown('<div class="team-title red">❤️ Enemy team</div>', unsafe_allow_html=True)
        red_meter = st.empty()
        red = {r: draft_slot("red", r) for r in ROLES}
        red_meter.markdown(damage_meter(red), unsafe_allow_html=True)

# Duplicate guard: a champion can only be picked once per game
picked = [c for c in list(blue.values()) + list(red.values()) if c]
dupes = sorted({c for c in picked if picked.count(c) > 1})
banned_picks = sorted(set(picked) & set(bans))
if banned_picks:
    st.warning(", ".join(name(c) for c in banned_picks) + " is banned but also picked.")
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
recs = engine.recommend(user_role, allies, red, comfort_pool=comfort, banned=bans,
                        comfort_bonus=comfort_bonus, enemy_weight=enemy_weight)

lane_opp = red.get(user_role)
st.subheader(f"{ROLE_ICONS[user_role]} Best {ROLE_LABELS[user_role]} picks for this draft")
if lane_opp:
    st.info(f"🎯 **Counter pick**: your lane opponent is **{name(lane_opp)}**. Picks are scored directly "
            "against them and the rest of the enemy team.")
else:
    st.info(f"🛡️ **Blind pick**: the enemy {ROLE_LABELS[user_role]} hasn't picked yet. Win % is the average "
            f"against the most-played {ROLE_LABELS[user_role]} champions still available, and picks that "
            "get hard-countered by one of them are ranked lower.")
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
            **({"Countered by": name(r.counter_risk_by) if r.counter_risk <= -0.005 else "",
                "Counter risk": r.counter_risk * 100} if r.blind else {}),
        } for r in recs]
        st.dataframe(rows, hide_index=True, width="stretch", column_config={
            "icon": st.column_config.ImageColumn("", width="small"),
            "Win %": st.column_config.NumberColumn(format="%.1f%%"),
            "Δ pts": st.column_config.NumberColumn(format="%+.1f"),
            "Counter risk": st.column_config.NumberColumn(format="%+.1f"),
        })

# --- 10. ABOUT THE MODEL ---
with st.expander("🧠 How NexusNode works"):
    comp_line = ("- **Team composition**: class mix, AD/AP balance and frontline for both teams (e.g. a full-AD "
                 "team or one with no frontline), learned from match outcomes.\n") if engine.profiles is not None else ""
    st.markdown(f"""
NexusNode predicts which team wins from the draft, trained on high-Elo Ranked Solo games, and recommends
the pick that maximizes your team's predicted win probability. A draft is scored from:

- **Role strength**: how each champion performs *in that specific role*, shrunk toward average when
  there are few games (so a 12-game off-role pick can't top the list). Recent patches count more.
- **Lane matchup**: your champion vs. the enemy in the same role, from observed head-to-head games,
  again weighted by how many games back it up.
{comp_line}- **Synergy & cross-lane counters**: a relational graph neural network learns these from champions played
  together and against each other. With the current amount of data these effects are still small; they
  grow automatically as the weekly pipeline collects more matches.

**Blind vs. counter pick.** If your lane opponent is already locked in, picks are scored directly against
them. If not, each pick is scored against the most-played champions still available for that role, and
picks that one of them specifically counters are ranked lower.

Each component was kept only if it improved predictions on held-out matches.
""")
    m = engine.metrics
    if m:
        v = m.get("validation", {})
        rows = [
            ("Full model + lane matchup evidence (used)", v.get("full_model_plus_lane_evidence")),
            ("Full model, no lane evidence", v.get("full_model")),
            ("Without enemy terms (ablation)", v.get("allies_only_ablation")),
            ("No draft information (side only)", v.get("baseline_side_only")),
        ]
        st.caption(f"Trained {m.get('trained_at', '?')} on {m.get('matches', '?')} matches, "
                   f"{m.get('champions', '?')} champions. Held-out validation on the "
                   f"{'newest ' if m.get('validation_split') == 'newest' else ''}{m.get('validation_matches', '?')} matches. "
                   "Lower log loss and higher AUC are better; drafts alone only shift outcomes by a few percent.")
        st.dataframe([
            {"Model": label, "Accuracy": r.get("accuracy"), "AUC": r.get("auc"), "Log loss": r.get("log_loss")}
            for label, r in rows if r
        ], hide_index=True, width="stretch", column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.3f"),
            "AUC": st.column_config.NumberColumn(format="%.3f"),
            "Log loss": st.column_config.NumberColumn(format="%.4f"),
        })

# Exact legal boilerplate required by Riot's developer policies
st.caption(LEGAL_NOTICE)

# Run the League client sync LAST: it may trigger a rerun, and widgets that
# haven't rendered yet in the interrupted run would lose their values
# (your role, comfort pool). Its status still shows in the sidebar slot.
if DESKTOP_MODE:
    with lcu_status:
        league_client_sync()
