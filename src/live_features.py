"""
src.live_features
=================
Feature engineering for SECOND-INNINGS live win-probability prediction.

This is a fundamentally different problem from pre-match forecasting:
during a chase, most of the match has already happened. The features
below capture the live state of the chase, and a model trained on them
typically reaches 85-90% accuracy because the math of the chase is
largely deterministic in the back half of the innings.

This is the same formulation broadcast win-probability graphics use.

Public API
----------
build_live_dataset(matches, deliveries)
    Returns one row per (match_id, ball_no_innings2) with:
        target               - innings-1 total + 1
        current_score        - chasing team's runs after this ball
        wickets_lost         - chasing team's wickets after this ball
        balls_remaining      - balls left in the 20 overs
        runs_to_win          - target - current_score
        required_run_rate    - 6 * runs_to_win / balls_remaining
        current_run_rate     - 6 * current_score / balls_bowled_so_far
        rr_diff              - current_rr - required_rr
        wickets_in_hand      - 10 - wickets_lost
        venue_avg            - average first-innings score at venue
        chasing_team_strength
        bowling_team_strength
        chasing_won          - the LABEL (1 if chaser won, 0 otherwise)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import build_innings_totals


# ---------------------------------------------------------------------------
def build_live_dataset(matches:    pd.DataFrame,
                       deliveries: pd.DataFrame) -> pd.DataFrame:
    """One snapshot per ball of the 2nd innings, with the chase outcome."""
    # Innings 1 totals (the chase target)
    inn1 = (deliveries[deliveries["inning"] == 1]
            .groupby("match_id")["total_runs"].sum()
            .rename("inn1_total").reset_index())

    # Innings 2 ball-by-ball, sorted
    inn2 = deliveries[deliveries["inning"] == 2].copy()
    inn2 = inn2.sort_values(["match_id", "over", "ball"]).reset_index(drop=True)
    inn2["delivery_no"] = inn2.groupby("match_id").cumcount() + 1

    # Use VALID balls (excludes wides / no-balls) for chase math.
    # Without this, the raw delivery count exceeds 120 on extras-heavy
    # innings, breaking balls_remaining and required_run_rate.
    if "valid_ball" in inn2.columns:
        inn2["ball_no"] = inn2.groupby("match_id")["valid_ball"].cumsum()
    else:
        inn2["ball_no"] = inn2["delivery_no"]
    inn2["ball_no"] = inn2["ball_no"].astype(int)

    inn2["cum_runs"]   = inn2.groupby("match_id")["total_runs"].cumsum()
    inn2["is_wicket"]  = inn2["player_dismissed"].notna().astype(int)
    inn2["cum_wkts"]   = inn2.groupby("match_id")["is_wicket"].cumsum()

    # Merge target
    inn2 = inn2.merge(inn1, on="match_id", how="left")
    inn2["target"]          = inn2["inn1_total"] + 1
    inn2["runs_to_win"]     = (inn2["target"] - inn2["cum_runs"]).clip(lower=0)
    inn2["balls_remaining"] = (120 - inn2["ball_no"]).clip(lower=0)
    inn2["wickets_in_hand"] = (10 - inn2["cum_wkts"]).clip(lower=0)

    # Run-rate features (guard against division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        inn2["current_run_rate"]  = np.where(
            inn2["ball_no"] > 0,
            6.0 * inn2["cum_runs"] / inn2["ball_no"],
            0.0,
        )
        inn2["required_run_rate"] = np.where(
            inn2["balls_remaining"] > 0,
            6.0 * inn2["runs_to_win"] / inn2["balls_remaining"],
            np.where(inn2["runs_to_win"] > 0, 99.0, 0.0),
        )
    inn2["rr_diff"] = inn2["current_run_rate"] - inn2["required_run_rate"]

    # --- Additional chase-math features that strongly predict outcome ---
    # Score "pace" features: where the chase is vs. a linear par-line
    inn2["balls_used_frac"]      = inn2["ball_no"] / 120.0
    inn2["score_frac_of_target"] = inn2["cum_runs"] / inn2["target"].clip(lower=1)
    # Distance from par-line (positive = ahead of pace)
    inn2["pace_diff"]            = inn2["score_frac_of_target"] - inn2["balls_used_frac"]
    # Pressure index: runs needed per ball, scaled by wickets lost
    inn2["runs_per_ball_needed"] = inn2["runs_to_win"] / inn2["balls_remaining"].clip(lower=1)
    inn2["pressure_index"]       = (
        inn2["runs_per_ball_needed"] *
        (1 + 0.15 * inn2["cum_wkts"])
    )
    # Log-transforms compress the long tail of hopeless chases
    inn2["log_runs_to_win"]      = np.log1p(inn2["runs_to_win"])
    inn2["log_balls_remaining"]  = np.log1p(inn2["balls_remaining"])
    # Already-won / already-lost indicators (these become very strong late)
    inn2["effectively_won"]      = (inn2["runs_to_win"] <= 0).astype(int)
    inn2["effectively_lost"]     = (
        (inn2["wickets_in_hand"] == 0) & (inn2["runs_to_win"] > 0)
    ).astype(int)

    # Venue average (1st-innings) — context feature
    inn_tot = build_innings_totals(deliveries, matches)
    venue_avg = (inn_tot[inn_tot["inning"] == 1]
                 .groupby("venue")["innings_total"].mean()
                 .rename("venue_avg").reset_index())

    # Team strengths
    chasing_strength = (
        inn_tot.groupby("batting_team")["innings_total"].mean()
        .rename("chasing_team_strength")
    )
    bowling_strength = (
        inn_tot.groupby("bowling_team")["innings_total"].mean()
        .rename("bowling_team_strength")
    )

    # Attach venue + winner from matches
    m_lite = matches.rename(columns={"id": "match_id"})[
        ["match_id", "venue", "winner"]
    ]
    inn2 = inn2.merge(m_lite, on="match_id", how="left")
    inn2 = inn2.merge(venue_avg, on="venue", how="left")
    inn2 = inn2.merge(chasing_strength, left_on="batting_team", right_index=True, how="left")
    inn2 = inn2.merge(bowling_strength, left_on="bowling_team", right_index=True, how="left")

    # Label
    inn2["chasing_won"] = (inn2["batting_team"] == inn2["winner"]).astype(int)
    # Drop rows with no winner (rain/abandoned)
    inn2 = inn2.dropna(subset=["winner"])

    # Final tidy column set
    feature_cols = [
        "target",
        "cum_runs", "cum_wkts", "ball_no",
        "runs_to_win", "balls_remaining", "wickets_in_hand",
        "current_run_rate", "required_run_rate", "rr_diff",
        "venue_avg", "chasing_team_strength", "bowling_team_strength",
        # Chase-math additions
        "balls_used_frac", "score_frac_of_target", "pace_diff",
        "runs_per_ball_needed", "pressure_index",
        "log_runs_to_win", "log_balls_remaining",
        "effectively_won", "effectively_lost",
    ]
    out = inn2[["match_id", "batting_team", "bowling_team", "venue"]
               + feature_cols + ["chasing_won"]].copy()
    out = out.rename(columns={
        "cum_runs": "current_score",
        "cum_wkts": "wickets_lost",
    })
    return out


# Columns in the order the model expects
LIVE_FEATURE_COLS = [
    "target",
    "current_score", "wickets_lost",
    "runs_to_win", "balls_remaining", "wickets_in_hand",
    "current_run_rate", "required_run_rate", "rr_diff",
    "venue_avg", "chasing_team_strength", "bowling_team_strength",
    "balls_used_frac", "score_frac_of_target", "pace_diff",
    "runs_per_ball_needed", "pressure_index",
    "log_runs_to_win", "log_balls_remaining",
    "effectively_won", "effectively_lost",
]
