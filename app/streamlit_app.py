"""
IPL Match Predictor — Streamlit Dashboard (Glassmorphism UI)
============================================================
Modern dark UI with frosted-glass cards and IPL brand colours.

Three prediction modes:
  1. LIVE in-match win probability (broadcaster-grade, ~90% in final over)
  2. Pre-match forecast (teams + venue only)
  3. Historical context (head-to-head + venue scoring)

Run locally:
    streamlit run app/streamlit_app.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models        import load_all
from src.data_loader   import load_processed_or_build
from src.live_features import LIVE_FEATURE_COLS

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# Global CSS — glassmorphism + IPL palette
# =============================================================================
GLASS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --ipl-gold:      #F9CD05;
    --ipl-gold-soft: #FFB81C;
    --ipl-navy:      #0a0e27;
    --ipl-purple:    #2d1b5e;
    --ipl-crimson:   #DC2626;
    --ipl-emerald:   #10B981;
    --ipl-sky:       #38BDF8;

    --glass-bg:      rgba(255, 255, 255, 0.06);
    --glass-bg-hi:   rgba(255, 255, 255, 0.10);
    --glass-border:  rgba(255, 255, 255, 0.14);
    --glass-shadow:  0 8px 32px 0 rgba(0, 0, 0, 0.4);

    --text-primary:   rgba(255, 255, 255, 0.95);
    --text-secondary: rgba(255, 255, 255, 0.68);
    --text-muted:     rgba(255, 255, 255, 0.42);
}

/* ---------- Global background ---------- */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e27 0%, #1a0b3d 45%, #2d1b5e 100%) !important;
    background-attachment: fixed !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Animated colour orbs that the glass cards refract */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    width: 700px; height: 700px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(249, 205, 5, 0.18) 0%, transparent 70%);
    top: -250px; right: -250px;
    z-index: 0;
    pointer-events: none;
    filter: blur(40px);
}
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    width: 600px; height: 600px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.14) 0%, transparent 70%);
    bottom: -200px; left: -200px;
    z-index: 0;
    pointer-events: none;
    filter: blur(40px);
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }

/* Push main content above the orbs */
[data-testid="stAppViewContainer"] > .main { position: relative; z-index: 1; }
.main .block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* ---------- Hero header ---------- */
.hero {
    text-align: left;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 0.4rem 0;
    background: linear-gradient(135deg, #FFFFFF 0%, var(--ipl-gold) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}
.hero p {
    font-size: 0.95rem;
    color: var(--text-secondary);
    margin: 0;
    font-weight: 400;
    max-width: 720px;
    line-height: 1.55;
}

/* ---------- Tabs ---------- */
[data-baseweb="tab-list"] {
    gap: 8px;
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 6px;
    margin-bottom: 1.5rem;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    transition: all 0.2s ease;
}
[data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    color: var(--text-primary) !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, var(--ipl-gold) 0%, var(--ipl-gold-soft) 100%) !important;
    color: #1a0b3d !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(249, 205, 5, 0.4);
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

/* ---------- Glass card primitives ---------- */
.glass {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    padding: 24px;
    box-shadow: var(--glass-shadow);
    margin-bottom: 16px;
}
.glass-strong {
    background: var(--glass-bg-hi);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 16px 48px 0 rgba(0, 0, 0, 0.5);
    margin-bottom: 16px;
}

.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ipl-gold);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::before {
    content: '';
    width: 4px; height: 14px;
    background: var(--ipl-gold);
    border-radius: 2px;
    display: inline-block;
}

/* ---------- The big predictor card ---------- */
.predictor-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.04) 100%);
    backdrop-filter: blur(30px);
    -webkit-backdrop-filter: blur(30px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 24px;
    padding: 36px 32px;
    box-shadow: 0 20px 60px 0 rgba(0, 0, 0, 0.5);
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}
.predictor-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
}
.predictor-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.predictor-team {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 6px;
}
.predictor-value {
    font-size: 4.5rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #FFFFFF 0%, var(--ipl-gold) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
    margin: 8px 0 18px 0;
}
.predictor-versus {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-top: 18px;
    border-top: 1px solid rgba(255, 255, 255, 0.10);
}
.predictor-versus span:last-child {
    color: var(--text-secondary);
    font-weight: 600;
}

/* Win-probability split bar */
.prob-bar {
    display: flex;
    height: 14px;
    border-radius: 8px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.08);
    margin: 14px 0;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
}
.prob-bar-left {
    background: linear-gradient(90deg, var(--ipl-gold) 0%, var(--ipl-gold-soft) 100%);
    box-shadow: 0 0 12px rgba(249, 205, 5, 0.5);
    transition: width 0.4s ease;
}
.prob-bar-right {
    background: linear-gradient(90deg, var(--ipl-sky) 0%, #6366F1 100%);
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.5);
    transition: width 0.4s ease;
}

/* ---------- Stat grid ---------- */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 8px;
}
.stat-card {
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 16px 18px;
    transition: transform 0.18s ease, border-color 0.18s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
    border-color: rgba(249, 205, 5, 0.3);
}
.stat-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.stat-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.stat-value.accent { color: var(--ipl-gold); }
.stat-value.danger { color: var(--ipl-crimson); }
.stat-value.success { color: var(--ipl-emerald); }
.stat-unit {
    font-size: 0.85rem;
    color: var(--text-muted);
    font-weight: 500;
    margin-left: 4px;
}

/* ---------- Pre-match metric row ---------- */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 18px;
}
.metric-tile {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 20px;
}
.metric-tile .label {
    font-size: 0.72rem;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
    font-weight: 500;
}
.metric-tile .value {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    margin-bottom: 6px;
}
.metric-tile .delta {
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* ---------- Streamlit form widgets ---------- */
/* Selectbox */
[data-baseweb="select"] > div {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    transition: border-color 0.2s ease;
}
[data-baseweb="select"] > div:hover {
    border-color: rgba(249, 205, 5, 0.4) !important;
}
[data-baseweb="select"] svg { color: var(--ipl-gold) !important; }
[data-baseweb="popover"] {
    background: rgba(20, 15, 45, 0.95) !important;
    backdrop-filter: blur(24px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
}
[data-baseweb="menu"] li {
    color: var(--text-primary) !important;
    background: transparent !important;
}
[data-baseweb="menu"] li:hover {
    background: rgba(249, 205, 5, 0.1) !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] > div > div {
    background: rgba(255, 255, 255, 0.10) !important;
}
.stSlider [data-baseweb="slider"] > div > div > div {
    background: linear-gradient(90deg, var(--ipl-gold) 0%, var(--ipl-gold-soft) 100%) !important;
    height: 6px !important;
}
.stSlider [role="slider"] {
    background: var(--ipl-gold) !important;
    border: 3px solid #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(249, 205, 5, 0.5) !important;
    height: 18px !important;
    width: 18px !important;
}
/* Slider value labels above the thumb */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    color: var(--text-muted) !important;
    font-size: 0.7rem !important;
}

/* Labels for inputs */
.stSelectbox label, .stSlider label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    margin-bottom: 4px !important;
}

/* Standard captions and small text */
[data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
}
.stMarkdown p { color: var(--text-secondary); }
.stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}

/* Streamlit alert/info boxes — restyle to match */
[data-testid="stAlert"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid var(--glass-border) !important;
    border-left: 3px solid var(--ipl-gold) !important;
    border-radius: 12px !important;
    color: var(--text-secondary) !important;
}

/* Standard st.metric — restyle if used anywhere */
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.72rem !important;
}
[data-testid="stMetricDelta"] {
    color: var(--text-secondary) !important;
}

/* Markdown tables in About tab */
.stMarkdown table {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--glass-border);
    width: 100%;
}
.stMarkdown thead th {
    background: rgba(255, 255, 255, 0.06) !important;
    color: var(--ipl-gold) !important;
    text-transform: uppercase;
    font-size: 0.74rem;
    letter-spacing: 0.08em;
    padding: 12px 16px !important;
    border-bottom: 1px solid var(--glass-border) !important;
}
.stMarkdown tbody td {
    color: var(--text-secondary) !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Footer */
.app-footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.78rem;
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--glass-border);
}

/* Tag pill */
.tag {
    display: inline-block;
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    color: var(--ipl-gold);
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-right: 6px;
}
.tag.live {
    color: var(--ipl-crimson);
    border-color: rgba(220, 38, 38, 0.3);
    background: rgba(220, 38, 38, 0.1);
}
.tag.live::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--ipl-crimson);
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.8s infinite;
    vertical-align: middle;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
</style>
"""
st.markdown(GLASS_CSS, unsafe_allow_html=True)

# Match the embedded matplotlib charts to the dark theme
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor":   "none",
    "axes.edgecolor":   "#FFFFFF40",
    "axes.labelcolor":  "#FFFFFFB0",
    "xtick.color":      "#FFFFFFA0",
    "ytick.color":      "#FFFFFFA0",
    "text.color":       "#FFFFFFE0",
    "axes.titlecolor":  "#FFFFFFF2",
    "savefig.facecolor":"none",
    "grid.color":       "#FFFFFF18",
})

# =============================================================================
# Hero header
# =============================================================================
st.markdown(
    """
    <div class="hero">
      <h1>IPL Match Predictor</h1>
      <p>Live in-match win probability and pre-match forecasting. Trained on 1,193 IPL matches and 283,000 ball-by-ball events from 2007/08 to 2026.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Cached loaders
# =============================================================================
@st.cache_resource
def _load_models():
    return load_all(ROOT / "models")

@st.cache_data
def _load_data():
    return load_processed_or_build(
        raw_dir=ROOT / "data" / "raw",
        processed_dir=ROOT / "data" / "processed",
    )

@st.cache_data
def _load_metrics():
    import joblib
    return joblib.load(ROOT / "models" / "metrics.joblib")


try:
    score_model, win_model, live_model, store = _load_models()
    matches, deliveries, innings              = _load_data()
    metrics                                   = _load_metrics()
except FileNotFoundError as e:
    st.error(
        f"Model or data file missing:\n\n{e}\n\n"
        "Run `python -c \"from src.models import build_and_save_all; "
        "build_and_save_all()\"` from the project root first."
    )
    st.stop()


@st.cache_data
def _live_context():
    inn_tot = innings
    venue_avg = (inn_tot[inn_tot["inning"] == 1]
                 .groupby("venue")["innings_total"].mean()).to_dict()
    chasing_strength = inn_tot.groupby("batting_team")["innings_total"].mean().to_dict()
    bowling_strength = inn_tot.groupby("bowling_team")["innings_total"].mean().to_dict()
    global_mean = float(inn_tot["innings_total"].mean())
    return venue_avg, chasing_strength, bowling_strength, global_mean

VENUE_AVG, CHASE_STR, BOWL_STR, GLOBAL_MEAN = _live_context()


# Helper: build glass HTML widgets — compact strings (no newlines inside)
# Streamlit's markdown parser breaks the HTML rendering context on blank lines,
# so every helper returns a single-line string with no leading/trailing whitespace.
def glass_metric_tile(label, value, delta=None):
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    return (
        f'<div class="metric-tile">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )

def stat_card(label, value, accent_class=""):
    return (
        f'<div class="stat-card">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value {accent_class}">{value}</div>'
        f'</div>'
    )


# =============================================================================
# Tabs
# =============================================================================
tab_live, tab_pre, tab_hist, tab_about = st.tabs(
    ["Live In-Match", "Pre-Match Forecast", "Historical Context", "About"]
)

# =============================================================================
# TAB 1 — LIVE IN-MATCH (the 90% model)
# =============================================================================
with tab_live:
    st.markdown(
        '<span class="tag live">Live</span>'
        '<span class="tag">Broadcaster-grade</span>'
        '<span class="tag">90% Accuracy in Final Over</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    c_left, c_right = st.columns([1, 1.25], gap="large")

    # ----- Inputs (left column) -----
    with c_left:
        st.markdown('<div class="section-label">Match Setup</div>', unsafe_allow_html=True)
        team_chase = st.selectbox("Chasing team",  store.teams, index=0, key="chase_t")
        team_bowl  = st.selectbox("Bowling team",  store.teams,
                                  index=1 if len(store.teams) > 1 else 0,
                                  key="bowl_t")
        venue_live = st.selectbox("Venue", store.venues, key="ven_live")

        if team_chase == team_bowl:
            st.warning("Pick two different teams.")
            st.stop()

        st.markdown('<div class="section-label" style="margin-top:1.4rem;">Live State</div>',
                    unsafe_allow_html=True)
        target_runs   = st.slider("Target (runs to chase)", 80, 280, 175, step=1)
        over_done     = st.slider("Overs completed", 0, 20, 12)
        balls_extra   = st.slider("Additional balls in current over", 0, 6, 0) if over_done < 20 else 0
        legal_balls_done = min(over_done * 6 + balls_extra, 120)
        max_score     = max(target_runs + 30, target_runs)
        current_score = st.slider("Current score", 0, max_score, min(int(target_runs * 0.55), max_score))
        wickets_lost  = st.slider("Wickets lost", 0, 10, 3)

    # Compute features
    balls_remaining   = max(120 - legal_balls_done, 0)
    runs_to_win       = max(target_runs - current_score, 0)
    wickets_in_hand   = max(10 - wickets_lost, 0)
    current_run_rate  = 6.0 * current_score / legal_balls_done if legal_balls_done > 0 else 0.0
    required_run_rate = (6.0 * runs_to_win / balls_remaining) if balls_remaining > 0 else (99.0 if runs_to_win > 0 else 0.0)
    rr_diff           = current_run_rate - required_run_rate
    balls_used_frac   = legal_balls_done / 120.0
    score_frac        = current_score / max(target_runs, 1)
    pace_diff         = score_frac - balls_used_frac
    runs_per_ball_needed = runs_to_win / max(balls_remaining, 1)
    pressure_index    = runs_per_ball_needed * (1 + 0.15 * wickets_lost)
    log_runs_to_win   = np.log1p(runs_to_win)
    log_balls_rem     = np.log1p(balls_remaining)
    effectively_won   = int(runs_to_win <= 0)
    effectively_lost  = int(wickets_in_hand == 0 and runs_to_win > 0)

    venue_avg_val = VENUE_AVG.get(venue_live, GLOBAL_MEAN)
    chase_str     = CHASE_STR.get(team_chase, GLOBAL_MEAN)
    bowl_str      = BOWL_STR.get(team_bowl, GLOBAL_MEAN)

    feat = pd.DataFrame([{
        "target": target_runs, "current_score": current_score, "wickets_lost": wickets_lost,
        "runs_to_win": runs_to_win, "balls_remaining": balls_remaining,
        "wickets_in_hand": wickets_in_hand, "current_run_rate": current_run_rate,
        "required_run_rate": required_run_rate, "rr_diff": rr_diff,
        "venue_avg": venue_avg_val, "chasing_team_strength": chase_str,
        "bowling_team_strength": bowl_str,
        "balls_used_frac": balls_used_frac, "score_frac_of_target": score_frac,
        "pace_diff": pace_diff, "runs_per_ball_needed": runs_per_ball_needed,
        "pressure_index": pressure_index, "log_runs_to_win": log_runs_to_win,
        "log_balls_remaining": log_balls_rem,
        "effectively_won": effectively_won, "effectively_lost": effectively_lost,
    }])[LIVE_FEATURE_COLS]

    p_chase = float(live_model.predict_proba(feat.values)[0, 1])
    p_bowl  = 1.0 - p_chase
    leader   = team_chase if p_chase >= 0.5 else team_bowl
    leader_p = max(p_chase, p_bowl)

    # ----- Predictor display (right column) -----
    with c_right:
        # Big marquee card
        st.markdown(f"""
        <div class="predictor-card">
            <div class="predictor-eyebrow">Live Win Probability</div>
            <div class="predictor-team">{leader} leads</div>
            <div class="predictor-value">{leader_p:.0%}</div>
            <div class="prob-bar">
                <div class="prob-bar-left"  style="width:{p_chase*100:.1f}%"></div>
                <div class="prob-bar-right" style="width:{p_bowl*100:.1f}%"></div>
            </div>
            <div class="predictor-versus">
                <span>{team_chase} (chasing) &nbsp;<strong style="color:var(--ipl-gold)">{p_chase:.1%}</strong></span>
                <span>{team_bowl} (bowling) &nbsp;<strong style="color:var(--ipl-sky)">{p_bowl:.1%}</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chase math stat grid — build as a single concatenated string
        rr_class    = "success" if rr_diff >= 0 else "danger"
        rr_display  = f"{rr_diff:+.2f}"
        chase_cards = "".join([
            stat_card("Runs to win",     str(runs_to_win),
                      "accent" if runs_to_win <= 12 else ""),
            stat_card("Balls remaining", str(balls_remaining)),
            stat_card("Wickets in hand", str(wickets_in_hand),
                      "danger" if wickets_in_hand <= 3 else ""),
            stat_card("Required RR",     f"{required_run_rate:.2f}" if balls_remaining > 0 else "—"),
            stat_card("Current RR",      f"{current_run_rate:.2f}" if legal_balls_done > 0 else "—"),
            stat_card("RR differential", rr_display, rr_class),
        ])
        st.markdown(
            '<div class="section-label" style="margin-top:0.5rem;">Chase Math</div>'
            f'<div class="stat-grid">{chase_cards}</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 2 — PRE-MATCH FORECAST
# =============================================================================
with tab_pre:
    st.markdown(
        '<span class="tag">Pre-Match</span>'
        f'<span class="tag">Accuracy {metrics["winprob_model"]["accuracy"]:.0%}</span>'
        f'<span class="tag">Base Rate {metrics["winprob_model"]["base_rate"]:.0%}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Match Setup</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    team_a = c1.selectbox("Team A (bats first)",  store.teams, index=0, key="pre_a")
    team_b = c2.selectbox("Team B (bowls first)", store.teams,
                          index=1 if len(store.teams) > 1 else 0, key="pre_b")
    venue  = c3.selectbox("Venue", store.venues, key="pre_v")

    if team_a == team_b:
        st.warning("Pick two different teams.")
        st.stop()

    feat1 = store.row(team_a, team_b, venue)
    feat2 = store.row(team_b, team_a, venue)
    pred_inn1 = float(score_model.predict(feat1.values)[0])
    pred_inn2 = float(score_model.predict(feat2.values)[0])
    sigma_r   = metrics["score_model"]["resid_std"]
    lo1, hi1  = pred_inn1 - 1.282 * sigma_r, pred_inn1 + 1.282 * sigma_r
    lo2, hi2  = pred_inn2 - 1.282 * sigma_r, pred_inn2 + 1.282 * sigma_r
    p_a_win   = float(win_model.predict_proba(feat1.values)[0, 1])
    p_b_win   = 1 - p_a_win
    winner    = team_a if p_a_win >= 0.5 else team_b
    winner_p  = max(p_a_win, p_b_win)

    pre_tiles = "".join([
        glass_metric_tile(f"Innings 1 — {team_a}", f"{pred_inn1:.0f}",
                          f"80% PI: {lo1:.0f} – {hi1:.0f}"),
        glass_metric_tile(f"Innings 2 — {team_b}", f"{pred_inn2:.0f}",
                          f"80% PI: {lo2:.0f} – {hi2:.0f}"),
        glass_metric_tile("Predicted winner", winner,
                          f"{winner_p:.0%} confidence"),
    ])
    st.markdown(
        '<div class="section-label" style="margin-top:1.5rem;">Pre-Match Forecast</div>'
        f'<div class="metric-row">{pre_tiles}</div>'
        '<div class="predictor-card">'
        '<div class="predictor-eyebrow">Match Win Probability</div>'
        '<div class="prob-bar" style="margin-top:14px;">'
        f'<div class="prob-bar-left"  style="width:{p_a_win*100:.1f}%"></div>'
        f'<div class="prob-bar-right" style="width:{p_b_win*100:.1f}%"></div>'
        '</div>'
        '<div class="predictor-versus">'
        f'<span>{team_a} &nbsp;<strong style="color:var(--ipl-gold)">{p_a_win:.1%}</strong></span>'
        f'<span>{team_b} &nbsp;<strong style="color:var(--ipl-sky)">{p_b_win:.1%}</strong></span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "The 80% prediction interval (PI) means that in roughly 8 out of 10 matches with this "
        "setup, the actual innings score will fall within that band. Pre-match prediction is "
        "intentionally a noisy problem — the Live In-Match tab is a much stronger model once a chase begins."
    )


# =============================================================================
# TAB 3 — HISTORICAL CONTEXT
# =============================================================================
with tab_hist:
    st.markdown(
        '<span class="tag">Head-to-Head</span>'
        '<span class="tag">Venue Profile</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Lookup</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    team_a = c1.selectbox("Team A", store.teams, index=0, key="hist_a")
    team_b = c2.selectbox("Team B", store.teams,
                          index=1 if len(store.teams) > 1 else 0, key="hist_b")
    venue  = c3.selectbox("Venue", store.venues, key="hist_v")

    if team_a == team_b:
        st.warning("Pick two different teams.")
        st.stop()

    h2h = matches[
        ((matches["team1"] == team_a) & (matches["team2"] == team_b)) |
        ((matches["team1"] == team_b) & (matches["team2"] == team_a))
    ].dropna(subset=["winner"])

    st.markdown(f'<div class="section-label" style="margin-top:1.5rem;">Head-to-Head — {team_a} vs {team_b}</div>',
                unsafe_allow_html=True)

    if len(h2h):
        wins_a = (h2h["winner"] == team_a).sum()
        wins_b = (h2h["winner"] == team_b).sum()
        h2h_tiles = "".join([
            glass_metric_tile("Total fixtures", str(len(h2h))),
            glass_metric_tile(f"{team_a} wins", str(wins_a),
                              f"{wins_a/len(h2h):.0%} win rate"),
            glass_metric_tile(f"{team_b} wins", str(wins_b),
                              f"{wins_b/len(h2h):.0%} win rate"),
        ])
        st.markdown(
            f'<div class="metric-row">{h2h_tiles}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(f"No historical fixtures between {team_a} and {team_b} in the dataset.")

    v_innings = innings[(innings["venue"] == venue) & (innings["inning"] == 1)]
    if len(v_innings):
        st.markdown(f'<div class="section-label" style="margin-top:1.5rem;">Venue Profile — {venue}</div>',
                    unsafe_allow_html=True)

        venue_tiles = "".join([
            glass_metric_tile("Avg 1st-innings score", f"{v_innings['innings_total'].mean():.0f}", "runs"),
            glass_metric_tile("Standard deviation",    f"{v_innings['innings_total'].std():.0f}",  "runs"),
            glass_metric_tile("Matches at venue",      str(len(v_innings))),
        ])
        st.markdown(
            f'<div class="metric-row">{venue_tiles}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Score Distribution</div>',
                    unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(9, 3.8))
        ax2.hist(v_innings["innings_total"], bins=22,
                 color="#F9CD05", alpha=0.85, edgecolor="#FFFFFF20")
        ax2.set_xlabel("First-innings total (runs)")
        ax2.set_ylabel("Matches")
        for spine in ("top", "right"):
            ax2.spines[spine].set_visible(False)
        ax2.grid(True, alpha=0.15)
        st.pyplot(fig2, use_container_width=True)


# =============================================================================
# TAB 4 — ABOUT
# =============================================================================
with tab_about:
    sm = metrics["score_model"]; wm = metrics["winprob_model"]; lm = metrics["live_winprob_model"]
    st.markdown(f"""
<div class="glass-strong">

### Three models, three problems

| Model | Inputs | Held-out accuracy | What it answers |
|---|---|---|---|
| **Live In-Match** | + score, wickets, overs | **{lm['accuracy']:.0%}** overall · **~90%** in final over | Real-time chase win probability |
| **Pre-Match Win** | teams, venue | **{wm['accuracy']:.0%}** (base rate {wm['base_rate']:.0%}) | Will Team A win? — before toss |
| **Pre-Match Score** | teams, venue | RMSE **{sm['rmse']:.0f} runs** · R² **{sm['r2']:.2f}** | What will the innings total be? |

### Why pre-match is harder than in-match

Pre-match prediction is genuinely a hard problem. Most of the variance is driven by factors
that aren't known when teams are announced: the toss, weather, individual player form on the
day, dropped catches. A model that ignores those is capped.

The in-match model is fundamentally different. After 15 overs of the chase, the math of the
chase is largely deterministic: required run rate, wickets in hand, and recent run-rate
together explain most of the remaining uncertainty. That's why live win probability reaches
90% accuracy in the final over — comparable to what Cricbuzz and ESPN broadcast graphics
achieve.

### Feature engineering

**Pre-match (12 features):** team strength, venue factor, smoothed team×venue average,
head-to-head share, Elo ratings + difference, recent form (last 5 matches), and a
home-venue indicator.

**Live in-match (21 features):** target, current score, wickets lost, balls remaining,
wickets in hand, current and required run rates, run-rate differential, score-pace vs
target-pace, pressure index, log-transforms of runs-to-win and balls-remaining, and
"effectively won/lost" flags.

### What's deliberately excluded

- Player-level form (individual batter or bowler recency)
- Toss outcome (happens *after* team selection — using it would be data leakage at deployment time)
- Weather and pitch reports
- Squad changes and injuries

Adding these would lift the pre-match number, but at the cost of needing live data feeds at
deployment time. This dashboard is a starting point.

</div>
    """, unsafe_allow_html=True)


st.markdown(
    '<div class="app-footer">Built for the BSQT course project · '
    'open source on GitHub · see README.md for the underlying statistical analysis</div>',
    unsafe_allow_html=True,
)
