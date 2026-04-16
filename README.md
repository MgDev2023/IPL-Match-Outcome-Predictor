# IPL Match Outcome Predictor

A web app that predicts which team is more likely to win an IPL match — before the match even starts.

---

## What does it do?

You pick two teams, a venue, and the toss result. The app shows you the win probability for each team based on past IPL data.

---

## How it works

I used historical IPL match data (2008–2024) to build a machine learning model. The model looks at:
- How each team has been performing recently (last 5 matches)
- Head-to-head record between the two teams
- Whether the toss winner has an advantage at that venue
- Home ground advantage

Two models are trained — Logistic Regression and Random Forest — and their predictions are averaged.

---

## Tech used

- Python
- Scikit-learn (machine learning)
- Pandas (data processing)
- Streamlit (web app)
- Plotly (charts)

---

## How to run it locally

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Get the dataset**
- Download `matches.csv` from Kaggle (search "IPL Complete Dataset 2008-2024")
- Place it at `data/raw/matches.csv`

**Step 3 — Build features and train**
```bash
python setup_and_train.py
```

**Step 4 — Run the app**
```bash
streamlit run app.py
```

---

## Model accuracy

~66% AUC. IPL is unpredictable by nature, so this is actually decent for pre-match prediction.

---

## Project structure

```
IPL-Match-Outcome-Predictor/
├── data/
│   ├── raw/            ← put matches.csv here
│   └── processed/      ← auto-generated
├── models/             ← saved models
├── src/
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predictor.py
├── app.py
├── setup_and_train.py
└── requirements.txt
```

---

## Made by

Megan — fresher portfolio project to practice machine learning and sports data analysis.
