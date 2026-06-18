"""
IPL Match Predictor — Streamlit Dashboard
=========================================
Three prediction modes, each clearly labeled with what it can and can't do:

  1. LIVE in-match win probability     (broadcaster-grade, hits 87-90% accuracy in late chase)
  2. Pre-match forecast                (teams + venue only, honest baseline ~70%)
  3. Historical context                (head-to-head + venue scoring)

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

st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div style="padding: 12px 0 4px 0;">
      <h1 style="margin-bottom:0;">🏏 IPL Match Predictor</h1>
      <p style="color:#666; margin-top:4px;">
        Live in-match win probability + honest pre-match forecasts, trained on 1,193 IPL matches (2007/08–2026).
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
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


# Helper: compute pre-computed venue / team strength values for the live model
@st.cache_data
def _live_context():
    """venue_avg + team strengths needed by the live model."""
    inn_tot = innings
    venue_avg = (inn_tot[inn_tot["inning"] == 1]
                 .groupby("venue")["innings_total"].mean()).to_dict()
    chasing_strength = inn_tot.groupby("batting_team")["innings_total"].mean().to_dict()
    bowling_strength = inn_tot.groupby("bowling_team")["innings_total"].mean().to_dict()
    global_mean = float(inn_tot["innings_total"].mean())
    return venue_avg, chasing_strength, bowling_strength, global_mean

VENUE_AVG, CHASE_STR, BOWL_STR, GLOBAL_MEAN = _live_context()


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_live, tab_pre, tab_hist, tab_about = st.tabs(
    ["🔴 Live in-match", "📊 Pre-match forecast", "📚 Historical context", "ℹ️ About"]
)

# ============================================================================
# TAB 1: LIVE IN-MATCH (the 90% model)
# ============================================================================
with tab_live:
    st.markdown("##### Live Win-Probability — chasing team's outlook at any point in the 2nd innings")
    st.caption(
        f"Trained on {metrics['live_winprob_model']['n_train_balls']:,} ball-by-ball snapshots. "
        f"Held-out accuracy in the final over: "
        f"**~90%**; over 18 onwards: **~87%**; over 15 onwards: **~82%**. "
        f"Overall AUC = {metrics['live_winprob_model']['roc_auc']:.3f}."
    )

    c_left, c_right = st.columns([1, 1.2])

    with c_left:
        team_chase = st.selectbox("Chasing team", store.teams, index=0, key="chase_t")
        team_bowl  = st.selectbox("Bowling team", store.teams,
                                  index=1 if len(store.teams) > 1 else 0,
                                  key="bowl_t")
        venue_live = st.selectbox("Venue", store.venues, key="ven_live")

        if team_chase == team_bowl:
            st.warning("Pick two different teams.")
            st.stop()

        st.markdown("**Live match state**")
        target_runs = st.slider("Target (runs to chase)", 80, 280, 175, step=1)

        # The over selector + ball selector
        over_done = st.slider("Overs completed", 0, 20, 12)
        if over_done < 20:
            balls_extra = st.slider("Additional balls in current over (0-6)", 0, 6, 0)
        else:
            balls_extra = 0
        legal_balls_done = min(over_done * 6 + balls_extra, 120)

        max_score = max(target_runs + 30, target_runs)
        current_score = st.slider("Current score", 0, max_score, min(int(target_runs * 0.55), max_score))
        wickets_lost  = st.slider("Wickets lost", 0, 10, 3)

    # Compute features
    balls_remaining = max(120 - legal_balls_done, 0)
    runs_to_win     = max(target_runs - current_score, 0)
    wickets_in_hand = max(10 - wickets_lost, 0)

    current_run_rate  = 6.0 * current_score / legal_balls_done if legal_balls_done > 0 else 0.0
    required_run_rate = (6.0 * runs_to_win / balls_remaining) if balls_remaining > 0 else (99.0 if runs_to_win > 0 else 0.0)
    rr_diff = current_run_rate - required_run_rate

    balls_used_frac      = legal_balls_done / 120.0
    score_frac_of_target = current_score / max(target_runs, 1)
    pace_diff            = score_frac_of_target - balls_used_frac
    runs_per_ball_needed = runs_to_win / max(balls_remaining, 1)
    pressure_index       = runs_per_ball_needed * (1 + 0.15 * wickets_lost)
    log_runs_to_win      = np.log1p(runs_to_win)
    log_balls_remaining  = np.log1p(balls_remaining)
    effectively_won      = int(runs_to_win <= 0)
    effectively_lost     = int(wickets_in_hand == 0 and runs_to_win > 0)

    venue_avg = VENUE_AVG.get(venue_live, GLOBAL_MEAN)
    chasing_str = CHASE_STR.get(team_chase, GLOBAL_MEAN)
    bowling_str = BOWL_STR.get(team_bowl, GLOBAL_MEAN)

    feat = pd.DataFrame([{
        "target": target_runs,
        "current_score": current_score,
        "wickets_lost": wickets_lost,
        "runs_to_win": runs_to_win,
        "balls_remaining": balls_remaining,
        "wickets_in_hand": wickets_in_hand,
        "current_run_rate": current_run_rate,
        "required_run_rate": required_run_rate,
        "rr_diff": rr_diff,
        "venue_avg": venue_avg,
        "chasing_team_strength": chasing_str,
        "bowling_team_strength": bowling_str,
        "balls_used_frac": balls_used_frac,
        "score_frac_of_target": score_frac_of_target,
        "pace_diff": pace_diff,
        "runs_per_ball_needed": runs_per_ball_needed,
        "pressure_index": pressure_index,
        "log_runs_to_win": log_runs_to_win,
        "log_balls_remaining": log_balls_remaining,
        "effectively_won": effectively_won,
        "effectively_lost": effectively_lost,
    }])[LIVE_FEATURE_COLS]

    p_chase_wins = float(live_model.predict_proba(feat.values)[0, 1])

    with c_right:
        st.markdown("##### Live probability")
        # Big number metric
        c1, c2 = st.columns(2)
        c1.metric(f"P({team_chase} wins)", f"{p_chase_wins:.1%}")
        c2.metric(f"P({team_bowl} wins)",  f"{1-p_chase_wins:.1%}")

        # Horizontal bar
        fig, ax = plt.subplots(figsize=(7, 1.4))
        ax.barh([0], [p_chase_wins],     color="#1f4e79", label=team_chase)
        ax.barh([0], [1 - p_chase_wins], left=[p_chase_wins], color="#c0392b", label=team_bowl)
        ax.axvline(0.5, color="white", linewidth=2)
        ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.6); ax.axis("off")
        ax.text(p_chase_wins/2, 0, f"{team_chase[:18]}\n{p_chase_wins:.0%}",
                ha="center", va="center", color="white", fontweight="bold", fontsize=11)
        ax.text(p_chase_wins + (1-p_chase_wins)/2, 0,
                f"{team_bowl[:18]}\n{(1-p_chase_wins):.0%}",
                ha="center", va="center", color="white", fontweight="bold", fontsize=11)
        st.pyplot(fig, use_container_width=True)

        # State summary
        st.markdown("**Chase math**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Runs to win", runs_to_win)
        c2.metric("Balls remaining", balls_remaining)
        c3.metric("Wickets in hand", wickets_in_hand)
        c1.metric("Required RR", f"{required_run_rate:.2f}" if balls_remaining > 0 else "—")
        c2.metric("Current RR",  f"{current_run_rate:.2f}" if legal_balls_done > 0 else "—")
        c3.metric("RR diff",     f"{rr_diff:+.2f}")

# ============================================================================
# TAB 2: PRE-MATCH FORECAST
# ============================================================================
with tab_pre:
    st.markdown("##### Pre-match forecast — teams + venue only")
    st.caption(
        f"Honest baseline: this is a deliberately limited model that does not know the toss, "
        f"squad, or pitch report. Held-out accuracy: "
        f"**{metrics['winprob_model']['accuracy']:.1%}** "
        f"(base rate {metrics['winprob_model']['base_rate']:.1%}). "
        f"AUC = {metrics['winprob_model']['roc_auc']:.3f}."
    )

    c1, c2, c3 = st.columns(3)
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
    lo1, hi1 = pred_inn1 - 1.282 * sigma_r, pred_inn1 + 1.282 * sigma_r
    lo2, hi2 = pred_inn2 - 1.282 * sigma_r, pred_inn2 + 1.282 * sigma_r
    p_a_win = float(win_model.predict_proba(feat1.values)[0, 1])

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Innings 1 — {team_a[:18]}", f"{pred_inn1:.0f}",
              f"80% PI: {lo1:.0f}–{hi1:.0f}")
    c2.metric(f"Innings 2 — {team_b[:18]}", f"{pred_inn2:.0f}",
              f"80% PI: {lo2:.0f}–{hi2:.0f}")
    likely_winner = team_a if p_a_win >= 0.5 else team_b
    c3.metric("Predicted winner", likely_winner[:18],
              f"P({team_a[:14]}) = {p_a_win:.0%}")

    fig, ax = plt.subplots(figsize=(8, 1.4))
    ax.barh([0], [p_a_win],     color="#1f4e79")
    ax.barh([0], [1 - p_a_win], left=[p_a_win], color="#c0392b")
    ax.axvline(0.5, color="white", linewidth=2)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.6); ax.axis("off")
    ax.text(p_a_win/2, 0, f"{team_a[:18]}\n{p_a_win:.0%}",
            ha="center", va="center", color="white", fontweight="bold")
    ax.text(p_a_win + (1-p_a_win)/2, 0, f"{team_b[:18]}\n{(1-p_a_win):.0%}",
            ha="center", va="center", color="white", fontweight="bold")
    st.pyplot(fig, use_container_width=True)

    st.info(
        "**How to read this:** the 80% prediction interval means that in roughly 8 out of 10 "
        "matches with this setup, the actual innings score will fall in that band. Pre-match "
        "prediction is intentionally a noisy problem — the live in-match tab is a much stronger model."
    )

# ============================================================================
# TAB 3: HISTORICAL CONTEXT
# ============================================================================
with tab_hist:
    c1, c2 = st.columns([1, 1])
    team_a = c1.selectbox("Team A", store.teams, index=0, key="hist_a")
    team_b = c2.selectbox("Team B", store.teams,
                          index=1 if len(store.teams) > 1 else 0, key="hist_b")
    venue  = st.selectbox("Venue", store.venues, key="hist_v")

    h2h = matches[
        ((matches["team1"] == team_a) & (matches["team2"] == team_b)) |
        ((matches["team1"] == team_b) & (matches["team2"] == team_a))
    ].dropna(subset=["winner"])

    st.markdown(f"##### Head-to-head: {team_a} vs {team_b}")
    if len(h2h):
        wins_a = (h2h["winner"] == team_a).sum()
        wins_b = (h2h["winner"] == team_b).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total fixtures", len(h2h))
        c2.metric(f"{team_a[:16]} wins", wins_a)
        c3.metric(f"{team_b[:16]} wins", wins_b)
    else:
        st.info(f"No historical fixtures between {team_a} and {team_b} in the dataset.")

    v_innings = innings[(innings["venue"] == venue) & (innings["inning"] == 1)]
    if len(v_innings):
        st.markdown(f"##### Venue scoring profile: {venue}")
        c1, c2 = st.columns([1, 2])
        c1.metric("Avg 1st-innings score", f"{v_innings['innings_total'].mean():.0f}")
        c1.metric("Std deviation",         f"{v_innings['innings_total'].std():.0f}")
        c1.metric("Matches at venue",      len(v_innings))

        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        ax2.hist(v_innings["innings_total"], bins=20,
                 color="#1f4e79", alpha=0.7, edgecolor="white")
        ax2.set_xlabel("First-innings total runs")
        ax2.set_ylabel("Matches")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        c2.pyplot(fig2, use_container_width=True)

# ============================================================================
# TAB 4: ABOUT
# ============================================================================
with tab_about:
    sm = metrics["score_model"]; wm = metrics["winprob_model"]; lm = metrics["live_winprob_model"]
    st.markdown(f"""
### Three models, three problems

| Model | Inputs | Held-out accuracy | What it answers |
|---|---|---|---|
| **Live in-match** | + score, wickets, overs | **{lm['accuracy']:.0%}** overall, **~90%** in last over | Real-time win probability during the chase |
| **Pre-match win** | teams, venue | **{wm['accuracy']:.0%}** (base rate {wm['base_rate']:.0%}) | Will team A win? — before toss |
| **Pre-match score** | teams, venue | RMSE **{sm['rmse']:.0f} runs**, R² **{sm['r2']:.2f}** | What will the innings total be? |

### Why pre-match is harder than in-match
Pre-match prediction is genuinely a hard problem. Most of the variance is driven by
factors that aren't known when teams are announced: the toss, weather, individual
player form on the day, dropped catches. A model that ignores those is capped.

The in-match model is fundamentally different. After 15 overs of the chase, the
math of the chase is largely deterministic: required-run-rate, wickets in hand,
and recent partnership behaviour together explain most of the remaining
uncertainty. That's why **live win probability reaches 90% accuracy in the final
over** — comparable to what Cricbuzz and ESPN broadcast graphics achieve.

### Feature engineering

**Pre-match (12 features):** team strength, venue factor, smoothed team×venue
average, head-to-head share, Elo ratings + difference, recent-form (last 5
matches), and a home-venue indicator.

**Live in-match (21 features):** target, current score, wickets lost, balls
remaining, wickets in hand, current and required run rates, run-rate diff,
score-pace vs target-pace, pressure index, log-transforms of runs-to-win and
balls-remaining, and "effectively won/lost" flags.

### What's deliberately excluded
* Player-level form (individual batter/bowler recency)
* Toss outcome (happens *after* team selection — using it would be leakage at deployment time)
* Weather / pitch reports
* Squad changes / injuries

Adding these would lift the pre-match number, but at the cost of needing live
data feeds at deployment time. This dashboard is a starting point.
""")

st.markdown("---")
st.caption("Built for the BSQT course project · code on GitHub · "
           "see README.md for the underlying statistical analysis.")
