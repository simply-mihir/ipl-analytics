"""
src.data_loader
===============
Single source of truth for loading and cleaning IPL data.

Supports TWO input formats automatically:

  (A) NEW combined format  : data/raw/IPL.csv
      one row per ball, with match-level metadata denormalised.
      ~1,193 matches, ~283k ball-by-ball rows.

  (B) LEGACY split format  : data/raw/matches.csv + data/raw/deliveries.csv
      the classic Kaggle "IPL Complete Dataset" structure.

In both cases the public API returns three pandas DataFrames with the
same canonical schema, so all downstream notebooks / features / models
work unchanged:

    matches     : one row per match     (id, season, team1, team2,
                                         toss_winner, toss_decision,
                                         winner, venue, city, ...)

    deliveries  : one row per ball      (match_id, inning, batting_team,
                                         bowling_team, over, ball,
                                         batsman, bowler, batsman_runs,
                                         extra_runs, total_runs,
                                         player_dismissed, dismissal_kind)

    innings     : one row per (match, inning) with innings_total + wickets
"""
from __future__ import annotations

from pathlib import Path
import warnings
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical team-name map (only TRUE re-brands; defunct franchises kept
# separate from their successors).
# ---------------------------------------------------------------------------
TEAM_NAME_MAP = {
    # Delhi rebrand 2018
    "Delhi Daredevils":            "Delhi Capitals",
    # Punjab rebrand 2021
    "Kings XI Punjab":             "Punjab Kings",
    # Bangalore -> Bengaluru rebrand 2024
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    # Typo fix
    "Rising Pune Supergiants":     "Rising Pune Supergiant",
}

# ---------------------------------------------------------------------------
# Venue canonicalisation. Many stadiums appear in the data with different
# suffixes (", Mumbai", ", Chennai", "Uppal", etc.) -- collapse to one
# canonical name per stadium so analyses don't double-count.
# ---------------------------------------------------------------------------
VENUE_NAME_MAP = {
    # M Chinnaswamy Stadium, Bengaluru
    "M.Chinnaswamy Stadium":                         "M Chinnaswamy Stadium",
    "M Chinnaswamy Stadium, Bengaluru":              "M Chinnaswamy Stadium",
    # Wankhede Stadium, Mumbai
    "Wankhede Stadium, Mumbai":                      "Wankhede Stadium",
    # Eden Gardens, Kolkata
    "Eden Gardens, Kolkata":                         "Eden Gardens",
    # MA Chidambaram Stadium, Chennai
    "MA Chidambaram Stadium, Chepauk":               "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai":      "MA Chidambaram Stadium",
    # Rajiv Gandhi International Stadium, Hyderabad
    "Rajiv Gandhi International Stadium, Uppal":     "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium",
    # Punjab Cricket Association Stadium, Mohali (re-named IS Bindra)
    "Punjab Cricket Association Stadium, Mohali":               "Punjab Cricket Association Stadium",
    "Punjab Cricket Association IS Bindra Stadium":             "Punjab Cricket Association Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali":     "Punjab Cricket Association Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association Stadium",
    # Sawai Mansingh Stadium, Jaipur
    "Sawai Mansingh Stadium, Jaipur":                "Sawai Mansingh Stadium",
    # Arun Jaitley / Feroz Shah Kotla -- same ground, re-named 2019
    "Arun Jaitley Stadium, Delhi":                   "Arun Jaitley Stadium",
    "Feroz Shah Kotla":                              "Arun Jaitley Stadium",
    # Narendra Modi Stadium -- formerly Sardar Patel
    "Sardar Patel Stadium, Motera":                  "Narendra Modi Stadium",
    "Narendra Modi Stadium, Ahmedabad":              "Narendra Modi Stadium",
    # Maharashtra Cricket Association Stadium, Pune (formerly Subrata Roy Sahara)
    "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium",
    "Subrata Roy Sahara Stadium":                    "Maharashtra Cricket Association Stadium",
    # Dr DY Patil, Mumbai
    "Dr DY Patil Sports Academy, Mumbai":            "Dr DY Patil Sports Academy",
    # Brabourne, Mumbai
    "Brabourne Stadium, Mumbai":                     "Brabourne Stadium",
    # Visakhapatnam
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam":
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    # Dharamsala
    "Himachal Pradesh Cricket Association Stadium, Dharamsala":
        "Himachal Pradesh Cricket Association Stadium",
    # Mullanpur / New Chandigarh (Maharaja Yadavindra Singh)
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur":
        "Maharaja Yadavindra Singh International Cricket Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh":
        "Maharaja Yadavindra Singh International Cricket Stadium",
    # Lucknow (Ekana)
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow":
        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    # Nagpur
    "Vidarbha Cricket Association Stadium, Jamtha":
        "Vidarbha Cricket Association Stadium",
    # Guwahati
    "Barsapara Cricket Stadium, Guwahati":           "Barsapara Cricket Stadium",
    # Sheikh Zayed (Abu Dhabi)
    "Zayed Cricket Stadium, Abu Dhabi":              "Sheikh Zayed Stadium",
}

# Default paths
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise_season(s):
    """'2007/08' -> 2007 ; '2009' -> 2009 ; NaN -> NaN."""
    if pd.isna(s):
        return s
    s = str(s).strip()
    if "/" in s:
        return int(s.split("/")[0])
    try:
        return int(s)
    except ValueError:
        return s


def _apply_team_map(series):
    return series.replace(TEAM_NAME_MAP)


def _apply_venue_map(series):
    return series.replace(VENUE_NAME_MAP)


# ---------------------------------------------------------------------------
# (A) NEW combined format loader
# ---------------------------------------------------------------------------
def _load_combined_format(raw_dir: Path):
    """Read data/raw/IPL.csv (single combined file)."""
    path = Path(raw_dir) / "IPL.csv"
    df = pd.read_csv(path, low_memory=False)

    # Drop the Kaggle index column if it sneaks in
    df = df.drop(columns=[c for c in ("Unnamed: 0",) if c in df.columns])

    # Restrict to actual IPL men's matches (the file sometimes contains
    # off-season trial matches with different team_type).
    if "event_name" in df.columns:
        df = df[df["event_name"].fillna("").str.contains("Indian Premier League", case=False)]
    if "match_type" in df.columns:
        df = df[df["match_type"].fillna("") == "T20"]

    # Apply team & venue maps
    for col in ("batting_team", "bowling_team", "toss_winner", "match_won_by"):
        if col in df.columns:
            df[col] = _apply_team_map(df[col])
    if "venue" in df.columns:
        df["venue"] = _apply_venue_map(df["venue"])
    if "season" in df.columns:
        df["season"] = df["season"].map(_normalise_season).astype("Int64")

    # ----- Build canonical `matches` DataFrame -----
    # For team1/team2 we use the first innings batting & bowling teams of
    # each match (deterministic given the data).
    inn1 = (df[df["innings"] == 1]
            .groupby("match_id")[["batting_team", "bowling_team"]]
            .first()
            .reset_index()
            .rename(columns={"batting_team": "team1", "bowling_team": "team2"}))

    match_meta = (df.groupby("match_id").first()[[
        "date", "season", "venue", "city",
        "toss_winner", "toss_decision",
        "match_won_by", "win_outcome",
        "player_of_match", "result_type",
    ]].reset_index()
        .rename(columns={"match_id": "id", "match_won_by": "winner"}))

    matches = match_meta.merge(inn1, left_on="id", right_on="match_id", how="left") \
                        .drop(columns=["match_id"])

    # Tag no-result matches so downstream win-prob analyses can skip them
    if "result_type" in matches.columns:
        no_result = matches["result_type"].fillna("").eq("no result")
        matches.loc[no_result, "winner"] = pd.NA

    # ----- Build canonical `deliveries` DataFrame -----
    deliveries = df.rename(columns={
        "innings":      "inning",
        "batter":       "batsman",
        "wicket_kind":  "dismissal_kind",
        "player_out":   "player_dismissed",
        "runs_batter":  "batsman_runs",
        "runs_extras":  "extra_runs",
        "runs_total":   "total_runs",
    })

    # Keep only the canonical columns + a few useful extras
    keep = [
        "match_id", "inning", "batting_team", "bowling_team",
        "over", "ball", "batsman", "non_striker", "bowler",
        "batsman_runs", "extra_runs", "total_runs",
        "player_dismissed", "dismissal_kind",
        # extras useful for richer analysis
        "team_runs", "team_wicket", "team_balls",
        "valid_ball", "extra_type",
    ]
    keep = [c for c in keep if c in deliveries.columns]
    deliveries = deliveries[keep].reset_index(drop=True)

    return matches, deliveries


# ---------------------------------------------------------------------------
# (B) LEGACY split-format loader
# ---------------------------------------------------------------------------
def _load_legacy_format(raw_dir: Path):
    """Read data/raw/matches.csv + data/raw/deliveries.csv (older Kaggle layout)."""
    raw_dir = Path(raw_dir)
    matches    = pd.read_csv(raw_dir / "matches.csv")
    deliveries = pd.read_csv(raw_dir / "deliveries.csv")

    matches.columns    = [c.lower() for c in matches.columns]
    deliveries.columns = [c.lower() for c in deliveries.columns]

    if "season" in matches.columns:
        matches["season"] = matches["season"].map(_normalise_season).astype("Int64")

    # column rename in deliveries
    if "batter" in deliveries.columns and "batsman" not in deliveries.columns:
        deliveries = deliveries.rename(columns={"batter": "batsman"})

    for col in ("team1", "team2", "toss_winner", "winner"):
        if col in matches.columns:
            matches[col] = _apply_team_map(matches[col])
    for col in ("batting_team", "bowling_team"):
        if col in deliveries.columns:
            deliveries[col] = _apply_team_map(deliveries[col])
    if "venue" in matches.columns:
        matches["venue"] = _apply_venue_map(matches["venue"])

    return matches, deliveries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_matches_and_deliveries(raw_dir: Path = RAW_DIR):
    """
    Auto-detect the input format and return (matches, deliveries) DataFrames
    in canonical schema.
    """
    raw_dir = Path(raw_dir)

    combined  = raw_dir / "IPL.csv"
    legacy_m  = raw_dir / "matches.csv"
    legacy_d  = raw_dir / "deliveries.csv"

    if combined.exists():
        return _load_combined_format(raw_dir)
    if legacy_m.exists() and legacy_d.exists():
        return _load_legacy_format(raw_dir)

    raise FileNotFoundError(
        "Could not find raw IPL data in {0!s}. Expected ONE of:\n"
        "  - {0!s}/IPL.csv                (new combined format)\n"
        "  - {0!s}/matches.csv + {0!s}/deliveries.csv  (legacy Kaggle format)"
        .format(raw_dir)
    )


def build_innings_totals(deliveries: pd.DataFrame,
                         matches:    pd.DataFrame) -> pd.DataFrame:
    """One row per (match, innings) with total runs and wickets."""
    agg = (
        deliveries.groupby(["match_id", "inning", "batting_team", "bowling_team"],
                           as_index=False)
        .agg(innings_total=("total_runs", "sum"),
             wickets=("player_dismissed", "count"))
    )
    # Attach season + venue
    agg = agg.merge(
        matches[["id", "season", "venue"]],
        left_on="match_id", right_on="id", how="left",
    ).drop(columns=["id"])
    return agg


def load_processed_or_build(raw_dir: Path = RAW_DIR,
                            processed_dir: Path = PROCESSED_DIR,
                            force_rebuild: bool = False):
    """Return cleaned DataFrames; cache them to data/processed/ as parquet."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    m_path  = processed_dir / "matches_clean.parquet"
    d_path  = processed_dir / "deliveries_clean.parquet"
    in_path = processed_dir / "innings_totals.parquet"

    if (not force_rebuild
            and m_path.exists() and d_path.exists() and in_path.exists()):
        return (pd.read_parquet(m_path),
                pd.read_parquet(d_path),
                pd.read_parquet(in_path))

    matches, deliveries = load_matches_and_deliveries(raw_dir)
    innings             = build_innings_totals(deliveries, matches)

    matches.to_parquet(m_path,   index=False)
    deliveries.to_parquet(d_path, index=False)
    innings.to_parquet(in_path,  index=False)
    return matches, deliveries, innings


# Backwards-compatible aliases so existing notebooks keep working
def load_matches(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return only the matches DataFrame."""
    m, _ = load_matches_and_deliveries(raw_dir)
    return m


def load_deliveries(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return only the deliveries DataFrame."""
    _, d = load_matches_and_deliveries(raw_dir)
    return d
