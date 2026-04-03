---
title: IPL Match Outcome Predictor
emoji: 🏏
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# IPL Match Outcome Predictor

Pre-match win probability for any IPL fixture — before the first ball is bowled.

## Tech Stack
Python 3.10 | Pandas | Scikit-learn | Logistic Regression + Random Forest | Plotly | Streamlit | Joblib

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the dataset
- Go to Kaggle and search **"IPL Complete Dataset 2008-2024"**
- Download `matches.csv` and place it at `data/raw/matches.csv`

### 3. Build features + train models
```bash
python setup_and_train.py
```

### 4. Launch the app
```bash
streamlit run app.py
```

---

## Project Structure

```
IPL-MATCH-OUTCOME-PREDICTOR/
├── data/
│   ├── raw/            ← place matches.csv here
│   └── processed/      ← auto-generated features.csv
├── models/             ← saved models + evaluation plots
├── src/
│   ├── feature_engineering.py   ← builds features from raw CSV
│   ├── train_model.py           ← trains LR + RF, saves models
│   └── predictor.py             ← prediction helper (used by app)
├── app.py              ← Streamlit UI
├── setup_and_train.py  ← one-shot setup script
└── requirements.txt
```

---

## Features Engineered

| Feature | Description |
|---|---|
| `team1_form5` | Win % over last 5 matches |
| `team2_form5` | Win % over last 5 matches |
| `form_diff` | team1_form5 − team2_form5 |
| `h2h_team1_win_pct` | Historical head-to-head win % |
| `team1_won_toss` | Did team1 win the toss? |
| `toss_bat` | Did toss winner choose to bat? |
| `team1_home` | Is team1 playing at home ground? |
| `team2_home` | Is team2 playing at home ground? |
| `toss_venue_adv` | Historical toss-winner win % at this venue |
| `season` | Season year (trend feature) |

All rolling stats are computed **before** each match to avoid data leakage.

---

## Model Performance (typical)

| Model | CV AUC | Test AUC |
|---|---|---|
| Logistic Regression | ~0.64 | ~0.63 |
| Random Forest | ~0.66 | ~0.65 |
| Ensemble (avg) | — | **~0.66** |

IPL is inherently hard to predict — 66% AUC on pre-match features is solid.

---

## App Features

- **Prediction tab**: select teams, venue, toss → get probability bar + model breakdown
- **Dataset Overview**: match counts, toss stats, season distribution
- **Model Performance**: ROC curve + feature importance plots
- **Head-to-Head**: historical win record between any two teams
