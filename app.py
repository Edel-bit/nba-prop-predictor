import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="NBA Points Over/Under", page_icon="🏀", layout="wide")

st.title("🏀 NBA Points Over/Under Predictor")
st.caption(
    "XGBoost classifier trained on 2023-24 game logs and tested on 2024-25 "
    "for 5 players. A learning project, not betting advice."
)


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
if st.button("Predict", type="primary", use_container_width=True):
    values = {
        "PLAYER_ENCODED":     cfg["encoded"],
        "PTS_ROLLING_5":      pts5,
        "PTS_ROLLING_10":     pts10,
        "PTS_TREND":          pts_trend,
        "MIN_ROLLING_5":      min5,
        "FGA_ROLLING_5":      fga5,
        "DAYS_REST":          rest,
        "IS_HOME":            is_home,
        "PLAYER_LINE":        line,
        "LINE_MINUS_FORM":    line - pts5,
    }
    X = pd.DataFrame([[values[c] for c in FEAT_COLS]], columns=FEAT_COLS)
    proba = model.predict_proba(X)[0]
    p_under, p_over = float(proba[0]), float(proba[1])

    st.subheader(f"{player}  ·  Line {line}")
    if p_over >= 0.5:
        st.success(f"Leans OVER {line}  ({p_over*100:.1f}%)")
    else:
        st.error(f"Leans UNDER {line}  ({p_under*100:.1f}%)")

    m1, m2 = st.columns(2)
    m1.metric("OVER probability", f"{p_over*100:.1f}%")
    m2.metric("UNDER probability", f"{p_under*100:.1f}%")
    
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