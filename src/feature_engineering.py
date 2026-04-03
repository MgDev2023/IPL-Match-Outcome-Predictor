"""
Feature engineering for IPL match outcome prediction.
Expects: data/raw/matches.csv from Kaggle "IPL Complete Dataset 2008-2024"
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/matches.csv")
OUT_PATH = Path("data/processed/features.csv")

from predictor import HOME_VENUES


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Normalise column names across different dataset versions
    rename = {
        "dl_applied": "method",
        "city": "city",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop no-result / tie / superover-decided matches for cleaner labels
    df = df[df["result"].isin(["normal", "runs", "wickets", "tie"]) | df["result"].isna()]
    df = df[df["winner"].notna()].copy()

    return df


def is_home(team: str, venue: str) -> int:
    for home_venue in HOME_VENUES.get(team, []):
        if home_venue.lower() in str(venue).lower() or str(venue).lower() in home_venue.lower():
            return 1
    return 0


def rolling_win_pct(df: pd.DataFrame, team_col: str, n: int = 5) -> pd.Series:
    """Win % for each team over their last n matches (computed before current match)."""
    records = []
    team_history: dict[str, list[int]] = {}

    for _, row in df.iterrows():
        team = row[team_col]
        hist = team_history.get(team, [])
        if len(hist) >= n:
            pct = sum(hist[-n:]) / n
        elif len(hist) > 0:
            pct = sum(hist) / len(hist)
        else:
            pct = 0.5  # prior

        records.append(pct)

        # Update history: 1 if this team won
        won = 1 if row["winner"] == team else 0
        team_history.setdefault(team, []).append(won)

    return pd.Series(records, index=df.index)


def head_to_head(df: pd.DataFrame) -> pd.Series:
    """team1 win % in all prior head-to-head meetings vs team2."""
    records = []
    h2h: dict[tuple, list[int]] = {}

    for _, row in df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        key = tuple(sorted([t1, t2]))
        hist = h2h.get(key, [])  # 1 = team1-of-key wins, 0 = team2-of-key wins

        if len(hist) == 0:
            pct = 0.5
        else:
            # figure out which side team1 (current row) is
            canonical_t1 = key[0]
            if t1 == canonical_t1:
                pct = sum(hist) / len(hist)
            else:
                pct = 1 - sum(hist) / len(hist)

        records.append(pct)

        # Update
        canonical_t1 = key[0]
        won = 1 if row["winner"] == canonical_t1 else 0
        h2h.setdefault(key, []).append(won)

    return pd.Series(records, index=df.index)


def toss_venue_advantage(df: pd.DataFrame) -> pd.Series:
    """Historical rate at which toss winner wins at this venue (computed before current match)."""
    records = []
    venue_history: dict[str, list[int]] = {}

    for _, row in df.iterrows():
        venue = row["venue"]
        hist = venue_history.get(venue, [])
        pct = sum(hist) / len(hist) if hist else 0.5
        records.append(pct)

        won = 1 if row["toss_winner"] == row["winner"] else 0
        venue_history.setdefault(venue, []).append(won)

    return pd.Series(records, index=df.index)


def team_venue_win_rate(df: pd.DataFrame, team_col: str) -> pd.Series:
    """Historical win rate for each team at each venue (computed before current match)."""
    records = []
    venue_team_history: dict[tuple, list[int]] = {}

    for _, row in df.iterrows():
        team = row[team_col]
        venue = row["venue"]
        key = (team, venue)
        hist = venue_team_history.get(key, [])
        pct = sum(hist) / len(hist) if hist else 0.5
        records.append(pct)

        won = 1 if row["winner"] == team else 0
        venue_team_history.setdefault(key, []).append(won)

    return pd.Series(records, index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    # Target: 1 if team1 wins
    feat["target"] = (df["winner"] == df["team1"]).astype(int)

    # Toss feature: did team1 win the toss?
    feat["team1_won_toss"] = (df["toss_winner"] == df["team1"]).astype(int)

    # Toss decision: bat=1, field=0
    feat["toss_bat"] = (df["toss_decision"].str.lower() == "bat").astype(int)

    # Interaction: did TEAM1 specifically bat first / field first?
    # Much stronger signal than raw toss_bat alone
    feat["team1_bats_first"]   = ((df["toss_winner"] == df["team1"]) & (df["toss_decision"].str.lower() == "bat")).astype(int)
    feat["team1_fields_first"] = ((df["toss_winner"] == df["team1"]) & (df["toss_decision"].str.lower() == "field")).astype(int)
    feat["team2_bats_first"]   = ((df["toss_winner"] == df["team2"]) & (df["toss_decision"].str.lower() == "bat")).astype(int)
    feat["team2_fields_first"] = ((df["toss_winner"] == df["team2"]) & (df["toss_decision"].str.lower() == "field")).astype(int)

    # Home advantage
    feat["team1_home"] = df.apply(lambda r: is_home(r["team1"], r["venue"]), axis=1)
    feat["team2_home"] = df.apply(lambda r: is_home(r["team2"], r["venue"]), axis=1)

    # Rolling form (last 5 and last 10)
    feat["team1_form5"] = rolling_win_pct(df, "team1", n=5)
    feat["team2_form5"] = rolling_win_pct(df, "team2", n=5)
    feat["team1_form10"] = rolling_win_pct(df, "team1", n=10)
    feat["team2_form10"] = rolling_win_pct(df, "team2", n=10)
    feat["form_diff"] = feat["team1_form5"] - feat["team2_form5"]

    # Head-to-head
    feat["h2h_team1_win_pct"] = head_to_head(df)

    # Toss × venue historical advantage
    feat["toss_venue_adv"] = toss_venue_advantage(df)

    # Team-specific venue win rate (strongest new feature)
    feat["team1_venue_win_rate"] = team_venue_win_rate(df, "team1")
    feat["team2_venue_win_rate"] = team_venue_win_rate(df, "team2")
    feat["venue_win_diff"] = feat["team1_venue_win_rate"] - feat["team2_venue_win_rate"]

    # Season — extract starting year correctly (e.g. "2009/10" → 2009, "2024" → 2024)
    feat["season"] = df["season"].astype(str).str.split("/").str[0].astype(int)

    # Keep identifiers for display (not used in model)
    feat["team1"] = df["team1"].values
    feat["team2"] = df["team2"].values
    feat["venue"] = df["venue"].values
    feat["date"] = df["date"].values
    feat["winner"] = df["winner"].values

    return feat


def run():
    print("Loading matches...")
    df = load_matches()
    print(f"  {len(df)} valid matches loaded")

    print("Engineering features...")
    features = build_features(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_PATH, index=False)
    print(f"  Saved to {OUT_PATH}")
    return features


if __name__ == "__main__":
    run()
