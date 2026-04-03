"""
Prediction helper used by both the Streamlit app and CLI.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODELS_DIR = Path("models")
FEATURE_COLS = [
    "team1_won_toss",
    "toss_bat",
    "team1_home",
    "team2_home",
    "team1_form5",
    "team2_form5",
    "form_diff",
    "h2h_team1_win_pct",
    "toss_venue_adv",
    "season",
]

HOME_VENUES = {
    "Mumbai Indians": ["Wankhede Stadium", "Brabourne Stadium"],
    "Chennai Super Kings": ["MA Chidambaram Stadium, Chepauk", "MA Chidambaram Stadium"],
    "Royal Challengers Bangalore": ["M Chinnaswamy Stadium"],
    "Royal Challengers Bengaluru": ["M Chinnaswamy Stadium"],
    "Kolkata Knight Riders": ["Eden Gardens"],
    "Delhi Capitals": ["Arun Jaitley Stadium", "Feroz Shah Kotla"],
    "Delhi Daredevils": ["Arun Jaitley Stadium", "Feroz Shah Kotla"],
    "Punjab Kings": ["Punjab Cricket Association IS Bindra Stadium, Mohali",
                     "Punjab Cricket Association Stadium, Mohali"],
    "Kings XI Punjab": ["Punjab Cricket Association IS Bindra Stadium, Mohali",
                        "Punjab Cricket Association Stadium, Mohali"],
    "Rajasthan Royals": ["Sawai Mansingh Stadium"],
    "Sunrisers Hyderabad": ["Rajiv Gandhi International Stadium, Uppal",
                            "Rajiv Gandhi International Stadium"],
    "Deccan Chargers": ["Rajiv Gandhi International Stadium, Uppal"],
    "Gujarat Titans": ["Narendra Modi Stadium, Ahmedabad", "Narendra Modi Stadium"],
    "Lucknow Super Giants": ["BRSABV Ekana Cricket Stadium",
                             "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium"],
}


def _is_home(team: str, venue: str) -> int:
    for hv in HOME_VENUES.get(team, []):
        if hv.lower() in str(venue).lower() or str(venue).lower() in hv.lower():
            return 1
    return 0


def load_models():
    lr = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")
    return lr, rf


def load_historical_stats():
    """Pre-compute per-team and head-to-head stats from the processed features file."""
    feat_path = Path("data/processed/features.csv")
    if not feat_path.exists():
        return {}, {}, {}

    df = pd.read_csv(feat_path, parse_dates=["date"])
    df = df.sort_values("date")

    # Latest form5 per team (use last row each team appears)
    form_t1 = df.groupby("team1")["team1_form5"].last().to_dict()
    form_t2 = df.groupby("team2")["team2_form5"].last().to_dict()
    team_form = {}
    for k, v in form_t1.items():
        team_form[k] = v
    for k, v in form_t2.items():
        if k not in team_form:
            team_form[k] = v
        else:
            team_form[k] = (team_form[k] + v) / 2

    # Head-to-head: most recent value
    h2h = {}
    for _, row in df.iterrows():
        key = (row["team1"], row["team2"])
        h2h[key] = row["h2h_team1_win_pct"]

    # Toss-venue advantage: most recent per venue
    toss_venue = df.groupby("venue")["toss_venue_adv"].last().to_dict()

    return team_form, h2h, toss_venue


def predict_match(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    season: int,
    lr_model,
    rf_model,
    team_form: dict,
    h2h: dict,
    toss_venue: dict,
) -> dict:
    t1_form = team_form.get(team1, 0.5)
    t2_form = team_form.get(team2, 0.5)

    h2h_pct = h2h.get((team1, team2), None)
    if h2h_pct is None:
        h2h_pct = 1 - h2h.get((team2, team1), 0.5)

    tva = toss_venue.get(venue, 0.5)

    row = {
        "team1_won_toss": int(toss_winner == team1),
        "toss_bat": int(toss_decision.lower() == "bat"),
        "team1_home": _is_home(team1, venue),
        "team2_home": _is_home(team2, venue),
        "team1_form5": t1_form,
        "team2_form5": t2_form,
        "form_diff": t1_form - t2_form,
        "h2h_team1_win_pct": h2h_pct,
        "toss_venue_adv": tva,
        "season": season,
    }

    X = pd.DataFrame([row])[FEATURE_COLS]

    lr_prob = lr_model.predict_proba(X)[0][1]
    rf_prob = rf_model.predict_proba(X)[0][1]
    ensemble_prob = (lr_prob + rf_prob) / 2

    return {
        "team1": team1,
        "team2": team2,
        "lr_prob_team1": lr_prob,
        "rf_prob_team1": rf_prob,
        "ensemble_prob_team1": ensemble_prob,
        "predicted_winner": team1 if ensemble_prob >= 0.5 else team2,
        "confidence": max(ensemble_prob, 1 - ensemble_prob),
    }
