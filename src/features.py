"""
src.features
============
Feature engineering for the PRE-MATCH prediction problem.

The user picks: (batting_team, bowling_team, venue) ONLY.
We engineer numeric features from historical data:

    --- baseline (v1) ---
    batting_team_strength : avg runs scored by batting_team
    bowling_team_strength : avg runs conceded by bowling_team
    venue_factor          : avg first-innings score at venue
    bat_at_venue          : batting_team avg at venue (Bayesian-smoothed)
    bowl_at_venue         : bowling_team avg conceded at venue
    h2h_bat_share         : head-to-head win share, smoothed toward 0.5

    --- upgrade (v2) ---
    batting_elo           : Elo rating (sequential, decay 0.997 per match)
    bowling_elo           : Elo rating of opponent
    elo_diff              : batting_elo - bowling_elo
    batting_recent_form   : avg runs in last 5 matches (smoothed)
    bowling_recent_form   : avg conceded in last 5 matches (smoothed)
    bat_at_home_venue     : 1 if venue is batting team's home, 0 else

All features are computed at training time and re-used at inference time
via the TeamVenueFeatureStore class below.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from collections import defaultdict, deque


def _smoothed_mean(group_sum, group_n, global_mean, k=5.0):
    """Bayesian shrinkage."""
    return (group_sum + k * global_mean) / (group_n + k)


# ---------------------------------------------------------------------------
# Elo ratings — sequential through history. We use a simple Elo update with
# a K-factor of 24 and a logistic expected-score model. The Elo isn't
# competitive with chess-strength estimates; it just captures cross-team
# strength in an easy-to-feature numeric form.
# ---------------------------------------------------------------------------
def _compute_elo_history(matches: pd.DataFrame, k_factor: float = 24.0,
                         init_rating: float = 1500.0) -> pd.DataFrame:
    """Return matches with team1_elo_pre / team2_elo_pre columns."""
    m = matches.dropna(subset=["winner"]).copy()
    # Use 'date' if present, else fall back to id order
    if "date" in m.columns:
        m["date"] = pd.to_datetime(m["date"], errors="coerce")
        m = m.sort_values(["date", "id"]).reset_index(drop=True)
    else:
        m = m.sort_values("id").reset_index(drop=True)

    rating = defaultdict(lambda: init_rating)
    t1_pre, t2_pre = [], []
    for _, r in m.iterrows():
        t1, t2 = r["team1"], r["team2"]
        r1, r2 = rating[t1], rating[t2]
        t1_pre.append(r1)
        t2_pre.append(r2)
        # expected scores
        e1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400))
        e2 = 1.0 - e1
        s1 = 1.0 if r["winner"] == t1 else 0.0
        s2 = 1.0 - s1
        rating[t1] = r1 + k_factor * (s1 - e1)
        rating[t2] = r2 + k_factor * (s2 - e2)
    m["team1_elo_pre"] = t1_pre
    m["team2_elo_pre"] = t2_pre
    m["_final_rating_t1"] = m["team1"].map(dict(rating))
    m["_final_rating_t2"] = m["team2"].map(dict(rating))
    return m, dict(rating)


# ---------------------------------------------------------------------------
# Home-venue inference — pick each team's most-frequent venue as home
# (this matches the IPL franchise structure surprisingly well; e.g.,
# Mumbai's most-frequent venue is Wankhede, CSK's is Chepauk, etc.)
# ---------------------------------------------------------------------------
def _infer_home_venues(matches: pd.DataFrame) -> dict:
    home = {}
    for team in set(matches["team1"]).union(set(matches["team2"])):
        sub_t1 = matches.loc[matches["team1"] == team, "venue"]
        sub_t2 = matches.loc[matches["team2"] == team, "venue"]
        all_v  = pd.concat([sub_t1, sub_t2]).dropna()
        if len(all_v):
            home[team] = all_v.value_counts().idxmax()
    return home


# ---------------------------------------------------------------------------
class TeamVenueFeatureStore:
    """Pre-computes pre-match features for prediction."""

    FEATURES = [
        # baseline
        "batting_team_strength", "bowling_team_strength",
        "venue_factor", "bat_at_venue", "bowl_at_venue",
        "h2h_bat_share",
        # upgrade
        "batting_elo", "bowling_elo", "elo_diff",
        "batting_recent_form", "bowling_recent_form",
        "bat_at_home_venue",
    ]

    def fit(self, matches: pd.DataFrame, innings: pd.DataFrame):
        self._global_mean = float(innings["innings_total"].mean())

        # --- v1 features ---
        self._bat_strength  = innings.groupby("batting_team")["innings_total"].mean()
        self._bowl_strength = innings.groupby("bowling_team")["innings_total"].mean()

        first_inn = innings[innings["inning"] == 1]
        self._venue_factor = first_inn.groupby("venue")["innings_total"].mean()

        agg = innings.groupby(["batting_team", "venue"])["innings_total"]
        bat_v = pd.DataFrame({"n": agg.count(), "sum": agg.sum()})
        bat_v["mean_smoothed"] = _smoothed_mean(
            bat_v["sum"], bat_v["n"], self._global_mean, k=5
        )
        self._bat_at_venue = bat_v["mean_smoothed"]

        agg = innings.groupby(["bowling_team", "venue"])["innings_total"]
        bowl_v = pd.DataFrame({"n": agg.count(), "sum": agg.sum()})
        bowl_v["mean_smoothed"] = _smoothed_mean(
            bowl_v["sum"], bowl_v["n"], self._global_mean, k=5
        )
        self._bowl_at_venue = bowl_v["mean_smoothed"]

        # Head-to-head
        m = matches.dropna(subset=["winner"]).copy()
        pair = m[["team1", "team2", "winner"]].copy()
        pair["team_a"] = pair[["team1", "team2"]].min(axis=1)
        pair["team_b"] = pair[["team1", "team2"]].max(axis=1)
        pair["a_won"] = (pair["winner"] == pair["team_a"]).astype(int)
        h2h = (pair.groupby(["team_a", "team_b"])["a_won"]
                   .agg(["sum", "count"]).reset_index())
        h2h["share_a"] = (h2h["sum"] + 0.5 * 5) / (h2h["count"] + 5)
        self._h2h = h2h.set_index(["team_a", "team_b"])["share_a"]

        # --- v2 features ---
        # Elo
        _, final_elo = _compute_elo_history(matches)
        self._elo = final_elo

        # Recent form (last 5 matches) -- for each team, the mean of their
        # last 5 innings totals (batting) and last 5 conceded (bowling).
        bat_recent  = {}
        bowl_recent = {}
        for team in set(innings["batting_team"]).union(set(innings["bowling_team"])):
            bat_runs = innings.loc[
                innings["batting_team"] == team, "innings_total"
            ].tail(5).mean()
            bowl_runs = innings.loc[
                innings["bowling_team"] == team, "innings_total"
            ].tail(5).mean()
            bat_recent[team]  = bat_runs  if not pd.isna(bat_runs)  else self._global_mean
            bowl_recent[team] = bowl_runs if not pd.isna(bowl_runs) else self._global_mean
        self._bat_recent  = bat_recent
        self._bowl_recent = bowl_recent

        # Home venues
        self._home_venue = _infer_home_venues(matches)

        # Public catalogues
        self.teams  = sorted(
            set(innings["batting_team"].unique())
            | set(innings["bowling_team"].unique())
        )
        self.venues = sorted(self._venue_factor.index)
        return self

    # -----------------------------------------------------------------
    def row(self, batting_team: str, bowling_team: str, venue: str) -> pd.DataFrame:
        bat_strength  = self._bat_strength.get(batting_team,  self._global_mean)
        bowl_strength = self._bowl_strength.get(bowling_team, self._global_mean)
        venue_factor  = self._venue_factor.get(venue,         self._global_mean)
        bat_at_venue  = self._bat_at_venue.get((batting_team,  venue), bat_strength)
        bowl_at_venue = self._bowl_at_venue.get((bowling_team, venue), bowl_strength)

        a, b = sorted([batting_team, bowling_team])
        if batting_team == a:
            h2h_share = self._h2h.get((a, b), 0.5)
        else:
            h2h_share = 1 - self._h2h.get((a, b), 0.5)

        bat_elo  = self._elo.get(batting_team, 1500.0)
        bowl_elo = self._elo.get(bowling_team, 1500.0)

        bat_form  = self._bat_recent.get(batting_team,  self._global_mean)
        bowl_form = self._bowl_recent.get(bowling_team, self._global_mean)

        bat_home  = int(self._home_venue.get(batting_team)  == venue)

        return pd.DataFrame([{
            "batting_team_strength": bat_strength,
            "bowling_team_strength": bowl_strength,
            "venue_factor":          venue_factor,
            "bat_at_venue":          bat_at_venue,
            "bowl_at_venue":         bowl_at_venue,
            "h2h_bat_share":         h2h_share,
            "batting_elo":           bat_elo,
            "bowling_elo":           bowl_elo,
            "elo_diff":              bat_elo - bowl_elo,
            "batting_recent_form":   bat_form,
            "bowling_recent_form":   bowl_form,
            "bat_at_home_venue":     bat_home,
        }])

    # -----------------------------------------------------------------
    def transform(self, innings: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in innings.iterrows():
            rows.append(self.row(r["batting_team"], r["bowling_team"], r["venue"]).iloc[0])
        feat = pd.DataFrame(rows).reset_index(drop=True)
        feat["match_id"]     = innings["match_id"].values
        feat["inning"]       = innings["inning"].values
        feat["innings_total"]= innings["innings_total"].values
        win = matches.set_index("id")["winner"]
        feat["winner"]       = feat["match_id"].map(win)
        feat["batting_team"] = innings["batting_team"].values
        feat["bowling_team"] = innings["bowling_team"].values
        feat["venue"]        = innings["venue"].values
        return feat
