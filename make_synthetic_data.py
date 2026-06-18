"""
Generate synthetic IPL-shaped data for offline testing.

Drops matches.csv and deliveries.csv into data/raw/ — same format as Kaggle.
Delete those files (or just don't run this script) when you have the real data.
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings",
    "Rajasthan Royals", "Sunrisers Hyderabad", "Gujarat Titans",
    "Lucknow Super Giants",
]
# Each team has a "true" batting strength multiplier (1.0 = league avg)
TEAM_BAT_STRENGTH = {t: rng.normal(1.0, 0.07) for t in TEAMS}
TEAM_BOWL_STRENGTH = {t: rng.normal(1.0, 0.06) for t in TEAMS}

VENUES = [
    "M Chinnaswamy Stadium", "Wankhede Stadium", "Eden Gardens",
    "MA Chidambaram Stadium", "Arun Jaitley Stadium",
    "Punjab Cricket Association Stadium",
    "Sawai Mansingh Stadium", "Rajiv Gandhi International Stadium",
]
VENUE_FACTOR = {v: rng.normal(1.0, 0.06) for v in VENUES}

SEASONS  = list(range(2015, 2025))
N_MATCH  = 750

# ---------- matches.csv ----------
rows = []
for mid in range(1, N_MATCH + 1):
    t1, t2  = rng.choice(TEAMS, size=2, replace=False)
    season  = int(rng.choice(SEASONS))
    venue   = str(rng.choice(VENUES))
    toss_w  = str(rng.choice([t1, t2]))
    toss_d  = str(rng.choice(["bat", "field"], p=[0.32, 0.68]))

    # batting first team
    if toss_d == "bat": bat_first = toss_w
    else:               bat_first = t1 if toss_w == t2 else t2

    bowl_first = t1 if bat_first == t2 else t2

    # innings 1 score driven by team batting * team bowling * venue
    base = 165
    mu1 = base * TEAM_BAT_STRENGTH[bat_first] / TEAM_BOWL_STRENGTH[bowl_first] * VENUE_FACTOR[venue]
    score1 = max(80, int(rng.normal(mu1, 22)))

    # innings 2: chasing factor + their strength
    mu2 = base * TEAM_BAT_STRENGTH[bowl_first] / TEAM_BOWL_STRENGTH[bat_first] * VENUE_FACTOR[venue]
    score2 = max(60, int(rng.normal(mu2, 22)))
    # cap chase at score1+1 if win, else lower
    if score2 >= score1: score2 = score1 + rng.integers(1, 6)
    else:               score2 = min(score2, score1 - 1)

    winner = bat_first if score1 > score2 else bowl_first

    rows.append({
        "id": mid, "season": season, "city": "City",
        "date": f"{season}-04-01",
        "team1": t1, "team2": t2,
        "toss_winner": toss_w, "toss_decision": toss_d,
        "result": "runs" if winner == bat_first else "wickets",
        "winner": winner, "venue": venue,
        "_bat_first": bat_first, "_score1": score1, "_score2": score2,
    })
matches = pd.DataFrame(rows)
matches_out = matches.drop(columns=["_bat_first", "_score1", "_score2"])
matches_out.to_csv(OUT / "matches.csv", index=False)
print(f"matches.csv: {matches_out.shape}")

# ---------- deliveries.csv ----------
del_rows = []
for _, m in matches.iterrows():
    for inning, (bat_team, bowl_team, target_total) in enumerate(
        [(m["_bat_first"], (m["team2"] if m["team1"] == m["_bat_first"] else m["team1"]), m["_score1"]),
         ((m["team2"] if m["team1"] == m["_bat_first"] else m["team1"]), m["_bat_first"], m["_score2"])],
        start=1,
    ):
        runs_so_far, wkts, ball_no = 0, 0, 0
        while ball_no < 120 and wkts < 10:
            ball_no += 1
            over = (ball_no - 1) // 6
            ball = ((ball_no - 1) % 6) + 1
            r = rng.choice([0, 1, 2, 3, 4, 6], p=[0.42, 0.30, 0.08, 0.01, 0.13, 0.06])
            dismissed = "BatsmanX" if rng.random() < (0.04 + 0.0006 * over) else None
            if dismissed: wkts += 1
            runs_so_far += r
            del_rows.append({
                "match_id": m["id"], "inning": inning,
                "batting_team": bat_team, "bowling_team": bowl_team,
                "over": over, "ball": ball,
                "batsman": "BatsmanA", "bowler": "BowlerA",
                "batsman_runs": r, "extra_runs": 0, "total_runs": r,
                "player_dismissed": dismissed,
                "dismissal_kind": "caught" if dismissed else None,
            })

deliveries = pd.DataFrame(del_rows)
deliveries.to_csv(OUT / "deliveries.csv", index=False)
print(f"deliveries.csv: {deliveries.shape}")
