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
    "team1_bats_first",
    "team1_fields_first",
    "team2_bats_first",
    "team2_fields_first",
    "team1_home",
    "team2_home",
    "team1_form5",
    "team2_form5",
    "team1_form10",
    "team2_form10",
    "form_diff",
    "h2h_team1_win_pct",
    "toss_venue_adv",
    "team1_venue_win_rate",
    "team2_venue_win_rate",
    "venue_win_diff",
    "team1_elo",
    "team2_elo",
    "elo_diff",
    "team1_season_form",
    "team2_season_form",
    "team1_streak",
    "team2_streak",
    "is_playoff",
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


def _get_toss_venue_adv(venue: str, toss_venue: dict) -> float:
    """Partial-match lookup so 'Eden Gardens' matches 'Eden Gardens, Kolkata'."""
    v = venue.lower()
    # exact match first
    if venue in toss_venue:
        return toss_venue[venue]
    # partial match
    for key, val in toss_venue.items():
        if v in key.lower() or key.lower() in v:
            return val
    return 0.5


def load_models():
    lr = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")
    return lr, rf


def _fuzzy_get(lookup: dict, key: str, default=0.5):
    """Partial string match for venue-keyed dicts."""
    if key in lookup:
        return lookup[key]
    kl = key.lower()
    for k, v in lookup.items():
        if kl in k.lower() or k.lower() in kl:
            return v
    return default


def load_historical_stats():
    """Pre-compute per-team and head-to-head stats from the processed features file."""
    feat_path = Path("data/processed/features.csv")
    if not feat_path.exists():
        return {}, {}, {}, {}, {}

    df = pd.read_csv(feat_path, parse_dates=["date"])
    df = df.sort_values("date")

    # Latest form5 and form10 per team
    def _merge_team_stat(col_t1, col_t2):
        d1 = df.groupby("team1")[col_t1].last().to_dict()
        d2 = df.groupby("team2")[col_t2].last().to_dict()
        merged = dict(d1)
        for k, v in d2.items():
            merged[k] = (merged[k] + v) / 2 if k in merged else v
        return merged

    team_form5  = _merge_team_stat("team1_form5",  "team2_form5")
    team_form10 = _merge_team_stat("team1_form10", "team2_form10")

    # Head-to-head: most recent value
    h2h = {}
    for _, row in df.iterrows():
        h2h[(row["team1"], row["team2"])] = row["h2h_team1_win_pct"]

    # Toss-venue advantage per venue
    toss_venue = df.groupby("venue")["toss_venue_adv"].last().to_dict()

    # Team venue win rate: (team, venue) → win rate
    team_venue = {}
    for col, rate_col in [("team1", "team1_venue_win_rate"), ("team2", "team2_venue_win_rate")]:
        for _, row in df.iterrows():
            team_venue[(row[col], row["venue"])] = row[rate_col]

    # Latest ELO per team
    elo = {}
    for col, elo_col in [("team1", "team1_elo"), ("team2", "team2_elo")]:
        elo.update(df.groupby(col)[elo_col].last().to_dict())

    # Latest season form per team
    season_form = {}
    for col, sf_col in [("team1", "team1_season_form"), ("team2", "team2_season_form")]:
        d = df.groupby(col)[sf_col].last().to_dict()
        for k, v in d.items():
            season_form[k] = (season_form[k] + v) / 2 if k in season_form else v

    # Latest win streak per team
    streak = {}
    for col, st_col in [("team1", "team1_streak"), ("team2", "team2_streak")]:
        d = df.groupby(col)[st_col].last().to_dict()
        for k, v in d.items():
            streak[k] = max(streak.get(k, 0), v)

    return team_form5, team_form10, h2h, toss_venue, team_venue, elo, season_form, streak


def _get_venue_win_rate(team: str, venue: str, team_venue: dict) -> float:
    """Fuzzy venue match for team-venue win rate."""
    if (team, venue) in team_venue:
        return team_venue[(team, venue)]
    vl = venue.lower()
    for (t, v), val in team_venue.items():
        if t == team and (vl in v.lower() or v.lower() in vl):
            return val
    return 0.5


def predict_match(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    season: int,
    is_playoff: int,
    lr_model,
    rf_model,
    team_form: dict,
    team_form10: dict,
    h2h: dict,
    toss_venue: dict,
    team_venue: dict,
    elo: dict,
    season_form: dict,
    streak: dict,
) -> dict:
    t1_form   = team_form.get(team1, 0.5)
    t2_form   = team_form.get(team2, 0.5)
    t1_form10 = team_form10.get(team1, 0.5)
    t2_form10 = team_form10.get(team2, 0.5)

    h2h_pct = h2h.get((team1, team2), None)
    if h2h_pct is None:
        h2h_pct = 1 - h2h.get((team2, team1), 0.5)

    tva  = _get_toss_venue_adv(venue, toss_venue)
    t1vr = _get_venue_win_rate(team1, venue, team_venue)
    t2vr = _get_venue_win_rate(team2, venue, team_venue)

    t1_elo = elo.get(team1, 1500.0)
    t2_elo = elo.get(team2, 1500.0)
    t1_sf  = season_form.get(team1, 0.5)
    t2_sf  = season_form.get(team2, 0.5)
    t1_str = streak.get(team1, 0)
    t2_str = streak.get(team2, 0)

    t1_toss = int(toss_winner == team1)
    t2_toss = int(toss_winner == team2)
    is_bat  = int(toss_decision.lower() == "bat")
    is_fld  = 1 - is_bat

    row = {
        "team1_won_toss":       t1_toss,
        "toss_bat":             is_bat,
        "team1_bats_first":     t1_toss * is_bat,
        "team1_fields_first":   t1_toss * is_fld,
        "team2_bats_first":     t2_toss * is_bat,
        "team2_fields_first":   t2_toss * is_fld,
        "team1_home":           _is_home(team1, venue),
        "team2_home":           _is_home(team2, venue),
        "team1_form5":          t1_form,
        "team2_form5":          t2_form,
        "team1_form10":         t1_form10,
        "team2_form10":         t2_form10,
        "form_diff":            t1_form - t2_form,
        "h2h_team1_win_pct":    h2h_pct,
        "toss_venue_adv":       tva,
        "team1_venue_win_rate": t1vr,
        "team2_venue_win_rate": t2vr,
        "venue_win_diff":       t1vr - t2vr,
        "team1_elo":            t1_elo,
        "team2_elo":            t2_elo,
        "elo_diff":             t1_elo - t2_elo,
        "team1_season_form":    t1_sf,
        "team2_season_form":    t2_sf,
        "team1_streak":         t1_str,
        "team2_streak":         t2_str,
        "is_playoff":           is_playoff,
        "season":               season,
    }

    X = pd.DataFrame([row])[FEATURE_COLS]

    lr_prob       = lr_model.predict_proba(X)[0][1]
    rf_prob       = rf_model.predict_proba(X)[0][1]
    ensemble_prob = (lr_prob + rf_prob) / 2

    return {
        "team1":               team1,
        "team2":               team2,
        "lr_prob_team1":       lr_prob,
        "rf_prob_team1":       rf_prob,
        "ensemble_prob_team1": ensemble_prob,
        "predicted_winner":    team1 if ensemble_prob >= 0.5 else team2,
        "confidence":          max(ensemble_prob, 1 - ensemble_prob),
    }
