import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="NBA Points Over/Under", page_icon="🏀", layout="wide")

# anime.js (v3) is loaded from a CDN inside the animated components below.
ANIME_CDN = "https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"

# --------------------------------------------------------------------------- #
# Theme: reds, maroon, blush and white
# --------------------------------------------------------------------------- #
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&display=swap');

:root {
  --maroon: #5B0F1A;
  --maroon-deep: #3E0A12;
  --crimson: #C8102E;
  --red: #E03A4B;
  --rose: #F6B8BD;
  --blush: #FDECEC;
  --cream: #FFF8F7;
  --ink: #2B0A10;
}

html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: linear-gradient(180deg, var(--cream) 0%, var(--blush) 100%); color: var(--ink); }
.block-container { padding-top: 1.5rem; max-width: 1200px; }

/* hide Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

/* headings */
h1, h2, h3, h4 { font-family: 'Bebas Neue', 'Inter', sans-serif !important; letter-spacing: 0.04em;
                 color: var(--maroon) !important; font-weight: 400 !important; }
h2 { font-size: 2rem !important; }
h3 { font-size: 1.5rem !important; }

/* sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, var(--maroon) 0%, var(--maroon-deep) 100%); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] small { color: #fff !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #F9D5D8 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"] { background: #fff !important; border-radius: 10px; }
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] input { color: var(--ink) !important; }
[data-testid="stSidebar"] button { background: var(--blush) !important; color: var(--maroon) !important; }

/* cards: every column becomes a white card */
[data-testid="column"], [data-testid="stColumn"] {
  background: #fff; border: 1px solid var(--rose); border-radius: 18px;
  padding: 1rem 1.3rem 1.2rem; box-shadow: 0 6px 20px rgba(91, 15, 26, 0.08);
  transition: transform .25s ease, box-shadow .25s ease;
}
[data-testid="column"]:hover, [data-testid="stColumn"]:hover {
  transform: translateY(-3px); box-shadow: 0 12px 28px rgba(200, 16, 46, 0.18);
}

/* labels and text inside the main area */
.stApp label, .stApp p, .stApp li { color: var(--ink); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #fff !important; }

/* predict button */
.stButton > button {
  background: linear-gradient(135deg, var(--crimson) 0%, var(--maroon) 100%);
  color: #fff; border: none; border-radius: 14px; padding: 0.8rem 1rem;
  font-family: 'Bebas Neue', sans-serif; font-size: 1.5rem; letter-spacing: 0.12em;
  box-shadow: 0 8px 20px rgba(200, 16, 46, 0.35);
  transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
}
.stButton > button:hover { transform: translateY(-2px) scale(1.01); filter: brightness(1.08);
  box-shadow: 0 12px 26px rgba(200, 16, 46, 0.45); color: #fff; border: none; }
.stButton > button:active { transform: translateY(0) scale(0.99); }
.stButton > button p, .stButton > button span, .stButton > button div { color: #fff !important; }
.stButton > button:focus:not(:active) { color: #fff; border: none; }

/* alerts */
[data-testid="stAlert"] { border-radius: 14px; border-left: 5px solid var(--crimson); background: #fff !important;
  border-top: 1px solid var(--rose); border-right: 1px solid var(--rose); border-bottom: 1px solid var(--rose); }
[data-testid="stAlert"], [data-testid="stAlert"] > div, [data-testid="stAlertContainer"] { background: #fff !important; }
[data-testid="stAlert"] * { color: var(--maroon) !important; }

/* expander */
[data-testid="stExpander"] { border: 1px solid var(--rose); border-radius: 14px; background: #fff; }
[data-testid="stExpander"] summary { color: var(--maroon); font-weight: 600; }

hr { border-color: var(--rose) !important; }

/* gentle entrance animation (disabled for reduced-motion users) */
@keyframes riseIn { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: no-preference) {
  [data-testid="column"], [data-testid="stColumn"] { animation: riseIn .6s ease both; }
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Animated hero banner (anime.js): court lines draw themselves, title staggers in,
# and a ball bounces.
# --------------------------------------------------------------------------- #
HERO_HTML = """
<style>
  html, body { margin: 0; background: transparent; }
  .hero { position: relative; overflow: hidden; border-radius: 22px; height: 190px;
          background: linear-gradient(120deg, #3E0A12 0%, #5B0F1A 45%, #C8102E 120%);
          font-family: 'Bebas Neue', 'Arial Narrow', Impact, sans-serif; color: #fff; }
  svg.court { position: absolute; right: -10px; top: -6px; height: 210px; opacity: .55; }
  svg.court * { fill: none; stroke: #F6B8BD; stroke-width: 2.5; stroke-linecap: round; }
  .text { position: absolute; left: 34px; top: 34px; }
  .title { font-size: min(54px, 6.4vw); letter-spacing: 3px; line-height: 1; margin: 0; white-space: nowrap; }
  .title span { display: inline-block; opacity: 0; }
  .sub { font-family: Inter, Arial, sans-serif; font-size: 14px; color: #F6B8BD; margin-top: 12px;
         opacity: 0; max-width: 520px; line-height: 1.5; }
  .tag { display: inline-block; margin-top: 14px; padding: 5px 14px; border-radius: 999px; opacity: 0;
         background: rgba(255,255,255,.14); border: 1px solid rgba(246,184,189,.6);
         font-family: Inter, Arial, sans-serif; font-size: 12px; letter-spacing: .06em; }
  .ball { position: absolute; right: 70px; bottom: 18px; width: 46px; height: 46px; opacity: 0; }
  .ball circle.b { fill: #E03A4B; stroke: #fff; stroke-width: 2; }
  .ball path { fill: none; stroke: #fff; stroke-width: 2; }
  .shadow { position: absolute; right: 66px; bottom: 10px; width: 54px; height: 8px; border-radius: 50%;
            background: rgba(0,0,0,.35); opacity: 0; }
</style>
<div class="hero">
  <svg class="court" viewBox="0 0 300 210">
    <rect class="l" x="10" y="10" width="280" height="190" rx="4"/>
    <rect class="l" x="95" y="10" width="110" height="105"/>
    <circle class="l" cx="150" cy="115" r="38"/>
    <path class="l" d="M 40 10 L 40 62 A 125 125 0 0 0 260 62 L 260 10"/>
    <circle class="l" cx="150" cy="30" r="9"/>
    <path class="l" d="M 128 10 L 172 10"/>
  </svg>
  <div class="text">
    <h1 class="title" id="title">NBA POINTS OVER/UNDER</h1>
    <div class="sub">XGBoost classifier trained on 2023-24 game logs and tested on 2024-25
      for 5 players. A learning project, not betting advice.</div>
    <div class="tag">CHRONOLOGICAL HOLDOUT &middot; PRIOR GAMES ONLY</div>
  </div>
  <div class="shadow" id="shadow"></div>
  <svg class="ball" id="ball" viewBox="0 0 46 46">
    <circle class="b" cx="23" cy="23" r="20"/>
    <path d="M3 23 H43 M23 3 V43 M8 8 Q23 23 8 38 M38 8 Q23 23 38 38"/>
  </svg>
</div>
<script src="__ANIME__"></script>
<script>
  (function () {
    var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var t = document.getElementById('title');
    t.innerHTML = t.textContent.split('').map(function (c) {
      return '<span>' + (c === ' ' ? '&nbsp;' : c) + '</span>';
    }).join('');
    if (typeof anime === 'undefined' || reduce) {
      document.querySelectorAll('.title span,.sub,.tag,.ball,.shadow').forEach(function (e) { e.style.opacity = 1; });
      return;
    }
    var tl = anime.timeline({ easing: 'easeOutExpo' });
    tl.add({ targets: '.court .l', strokeDashoffset: [anime.setDashoffset, 0],
             duration: 1800, delay: anime.stagger(140) }, 0)
      .add({ targets: '.title span', opacity: [0, 1], translateY: [34, 0], rotate: [8, 0],
             duration: 900, delay: anime.stagger(32) }, 200)
      .add({ targets: '.sub', opacity: [0, 1], translateY: [14, 0], duration: 800 }, 900)
      .add({ targets: '.tag', opacity: [0, 1], scale: [0.8, 1], duration: 700, easing: 'easeOutBack' }, 1100)
      .add({ targets: '.ball', opacity: [0, 1], duration: 400 }, 1000)
      .add({ targets: '.shadow', opacity: [0, 1], duration: 400 }, 1000);
    anime({ targets: '#ball', translateY: [0, -70], duration: 620, direction: 'alternate',
            loop: true, easing: 'easeOutQuad', delay: 1400 });
    anime({ targets: '#ball', rotate: 360, duration: 1240, loop: true, easing: 'linear', delay: 1400 });
    anime({ targets: '#shadow', scaleX: [1, 0.55], duration: 620, direction: 'alternate',
            loop: true, easing: 'easeOutQuad', delay: 1400 });
  })();
</script>
"""

# --------------------------------------------------------------------------- #
# Animated result card: verdict pops in, probability counts up, bars fill.
# --------------------------------------------------------------------------- #
RESULT_HTML = """
<style>
  html, body { margin: 0; background: transparent; }
  .card { border-radius: 20px; padding: 22px 26px; background: #fff; border: 1px solid #F6B8BD;
          box-shadow: 0 10px 28px rgba(91,15,26,.12); font-family: Inter, Arial, sans-serif; color: #2B0A10; }
  .top { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
  .who { font-family: 'Bebas Neue', Impact, sans-serif; font-size: 26px; letter-spacing: 1.5px; color: #5B0F1A; }
  .chip { font-family: 'Bebas Neue', Impact, sans-serif; font-size: 24px; letter-spacing: 2px; color: #fff;
          padding: 6px 20px; border-radius: 999px; opacity: 0; background: __CHIP__; }
  .big { font-family: 'Bebas Neue', Impact, sans-serif; font-size: 72px; line-height: 1; margin: 6px 0 2px;
         color: __BIG__; }
  .big small { font-size: 28px; color: #5B0F1A; margin-left: 6px; }
  .bars { margin-top: 12px; }
  .row { display: flex; align-items: center; gap: 12px; margin: 10px 0; }
  .lab { width: 64px; font-size: 13px; font-weight: 600; letter-spacing: .06em; color: #5B0F1A; }
  .track { position: relative; flex: 1; height: 18px; background: #FDECEC; border-radius: 999px; overflow: hidden; }
  .fill { height: 100%; width: 0; border-radius: 999px; }
  .over  { background: linear-gradient(90deg, #E03A4B, #C8102E); }
  .under { background: linear-gradient(90deg, #8A1C2B, #5B0F1A); }
  .mid { position: absolute; left: 50%; top: -2px; bottom: -2px; width: 2px; background: rgba(43,10,16,.35); }
  .pct { width: 58px; text-align: right; font-weight: 600; font-size: 14px; }
</style>
<div class="card">
  <div class="top">
    <div class="who">__WHO__ &middot; LINE __LINE__</div>
    <div class="chip" id="chip">LEANS __SIDE__</div>
  </div>
  <div class="big"><span id="num">50.0</span><small>% __SIDE__</small></div>
  <div class="bars">
    <div class="row"><div class="lab">OVER</div>
      <div class="track"><div class="fill over" id="fo"></div><div class="mid"></div></div>
      <div class="pct" id="po">0.0%</div></div>
    <div class="row"><div class="lab">UNDER</div>
      <div class="track"><div class="fill under" id="fu"></div><div class="mid"></div></div>
      <div class="pct" id="pu">0.0%</div></div>
  </div>
</div>
<script src="__ANIME__"></script>
<script>
  (function () {
    var over = __OVER__, under = 100 - over, lead = Math.max(over, under);
    var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    function setFinal() {
      document.getElementById('chip').style.opacity = 1;
      document.getElementById('num').textContent = lead.toFixed(1);
      document.getElementById('fo').style.width = over + '%';
      document.getElementById('fu').style.width = under + '%';
      document.getElementById('po').textContent = over.toFixed(1) + '%';
      document.getElementById('pu').textContent = under.toFixed(1) + '%';
    }
    if (typeof anime === 'undefined' || reduce) { setFinal(); return; }
    var counter = { n: 50, o: 0, u: 0 };
    anime({ targets: '#chip', opacity: [0, 1], scale: [0.4, 1], duration: 900, easing: 'easeOutElastic(1, .6)' });
    anime({ targets: counter, n: lead, o: over, u: under, duration: 1400, easing: 'easeOutExpo',
            update: function () {
              document.getElementById('num').textContent = counter.n.toFixed(1);
              document.getElementById('po').textContent = counter.o.toFixed(1) + '%';
              document.getElementById('pu').textContent = counter.u.toFixed(1) + '%';
            } });
    anime({ targets: '#fo', width: over + '%', duration: 1400, easing: 'easeOutElastic(1, .8)' });
    anime({ targets: '#fu', width: under + '%', duration: 1400, easing: 'easeOutElastic(1, .8)', delay: 120 });
  })();
</script>
"""


def render_hero() -> None:
    components.html(HERO_HTML.replace("__ANIME__", ANIME_CDN), height=200)


def render_result(player: str, line: float, p_over: float) -> None:
    side = "OVER" if p_over >= 0.5 else "UNDER"
    chip = "linear-gradient(135deg,#E03A4B,#C8102E)" if side == "OVER" else "linear-gradient(135deg,#8A1C2B,#5B0F1A)"
    big = "#C8102E" if side == "OVER" else "#5B0F1A"
    html = (
        RESULT_HTML.replace("__ANIME__", ANIME_CDN)
        .replace("__WHO__", json.dumps(player)[1:-1])
        .replace("__LINE__", f"{line:g}")
        .replace("__SIDE__", side)
        .replace("__CHIP__", chip)
        .replace("__BIG__", big)
        .replace("__OVER__", f"{p_over * 100:.4f}")
    )
    components.html(html, height=290)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model():
    p = Path("data/models/multi_player_model.pkl")
    return joblib.load(p) if p.exists() else None


@st.cache_data
def load_feature_columns():
    p = Path("data/models/feature_columns.txt")
    return p.read_text().strip().split("\n") if p.exists() else None


model = load_model()
FEAT_COLS = load_feature_columns()

render_hero()

if model is None or FEAT_COLS is None:
    st.error("Model not found. Run the pipeline first:")
    st.code(
        "python fetch_multi_player_data.py\n"
        "python feature_engineering_multi_v2.py\n"
        "python model_training_multi_v2.py"
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Config (encodings must match feature_engineering_multi_v2.py)
# --------------------------------------------------------------------------- #
PLAYER_CONFIG = {
    "LeBron James":    {"encoded": 0, "avg_pts": 24.4},
    "Anthony Edwards": {"encoded": 1, "avg_pts": 27.6},
    "Stephen Curry":   {"encoded": 2, "avg_pts": 24.5},
    "Kevin Durant":    {"encoded": 3, "avg_pts": 26.6},
    "Luka Doncic":     {"encoded": 4, "avg_pts": 28.2},
}

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Player and line")
    player = st.selectbox("Player", list(PLAYER_CONFIG.keys()))
    cfg = PLAYER_CONFIG[player]

    default_line = float(int(cfg["avg_pts"])) + 0.5
    line = st.number_input("Over/Under line", 10.0, 55.0, default_line, 0.5)
    st.caption(
        "The model was trained on lines built from each player's running "
        "average (x.5). Lines far from recent form are outside what it saw."
    )

st.header("Game scenario")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Recent form")
    pts5 = st.slider("Last 5-game avg pts", 10.0, 45.0, float(cfg["avg_pts"]), 0.5)
    pts10 = st.slider("Last 10-game avg pts", 10.0, 45.0, float(cfg["avg_pts"]), 0.5)
    min5 = st.slider("Last 5-game avg minutes", 20.0, 42.0, 34.0, 0.5)
    fga5 = st.slider("Last 5-game avg FGA", 8.0, 28.0, 18.0, 0.5)

with c2:
    st.subheader("Game context")
    is_home = 1 if st.radio("Location", ["Home", "Away"]) == "Home" else 0
    rest = st.slider("Days rest", 0, 4, 1)

pts_trend = pts5 - pts10
st.info(f"Form trend (5-game minus 10-game): {pts_trend:+.1f} pts")
st.markdown("---")

# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
if st.button("PREDICT", type="primary", use_container_width=True):
    values = {
        "PLAYER_ENCODED":  cfg["encoded"],
        "PTS_ROLLING_5":   pts5,
        "PTS_ROLLING_10":  pts10,
        "PTS_TREND":       pts_trend,
        "MIN_ROLLING_5":   min5,
        "FGA_ROLLING_5":   fga5,
        "DAYS_REST":       rest,
        "IS_HOME":         is_home,
        "PLAYER_LINE":     line,
        "LINE_MINUS_FORM": line - pts5,
    }
    X = pd.DataFrame([[values[c] for c in FEAT_COLS]], columns=FEAT_COLS)
    proba = model.predict_proba(X)[0]
    p_under, p_over = float(proba[0]), float(proba[1])

    render_result(player, line, p_over)

    st.caption(
        "These probabilities are not calibrated. On held-out 2024-25 games the model's "
        "accuracy was about the same as always guessing the majority class, so treat "
        "high percentages as the model extrapolating, not as real confidence."
    )

    if abs(p_over - 0.5) < 0.08:
        st.warning("Close to a coin flip. The model has little edge here.")
    if abs(line - pts5) > 5:
        st.warning("This line is far from recent form, outside the range the model trained on.")

    with st.expander("Model inputs"):
        st.dataframe(X.T.rename(columns={0: "value"}))

st.markdown("---")
st.caption(
    "Limitations: five players, two seasons, points only. No injury, lineup or opponent data. "
    "See README for holdout results."
)