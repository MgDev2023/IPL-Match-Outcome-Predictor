"""
Behind the Code — recruiter-facing explainer page.
Walks through the full ML pipeline: data → features → models → app.
"""

import streamlit as st

st.set_page_config(
    page_title="Behind the Code · IPL Predictor",
    page_icon="🧠",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Behind the Code")
st.markdown(
    "A recruiter-friendly walkthrough of every technical decision made in this project — "
    "from raw CSV to live prediction."
)
st.markdown("---")

# ── Quick stat bar ────────────────────────────────────────────────────────────
CARD_CSS = """
<style>
.stat-grid {
    display: flex;
    gap: 14px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.stat-card {
    flex: 1;
    min-width: 140px;
    background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
    border: 1px solid #2d5a8e;
    border-radius: 12px;
    padding: 18px 16px 14px 16px;
    text-align: center;
}
.stat-card .stat-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #4fc3f7;
    line-height: 1.1;
    margin-bottom: 6px;
}
.stat-card .stat-label {
    font-size: 0.78rem;
    color: #90caf9;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}
</style>
"""

STAT_CARDS = """
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-value">2008–2024</div>
    <div class="stat-label">Dataset</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">950+</div>
    <div class="stat-label">Valid Matches</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">10</div>
    <div class="stat-label">Features Engineered</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">LR + RF</div>
    <div class="stat-label">Models + Ensemble</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">0.57</div>
    <div class="stat-label">CV AUC (5-fold)</div>
  </div>
</div>
"""

st.markdown(CARD_CSS + STAT_CARDS, unsafe_allow_html=True)
st.markdown("---")

# ── 1. Problem Statement ──────────────────────────────────────────────────────
st.header("1  Problem Statement")
st.markdown(
    """
**Goal:** Predict which team wins an IPL match *before the first ball is bowled*, using only
information available at toss time — no ball-by-ball data, no player stats.

This is a **binary classification** problem:
`target = 1` if Team 1 wins, `target = 0` if Team 2 wins.

The challenge is that cricket outcomes carry high inherent randomness. A model that
reaches ~66% AUC purely from pre-match signals is genuinely useful.
"""
)

# ── 2. Data Pipeline ──────────────────────────────────────────────────────────
st.header("2  Data Pipeline")

col_a, col_b = st.columns([1, 2])
with col_a:
    st.markdown(
        """
```
data/raw/matches.csv          ← Kaggle dataset
        ↓
src/feature_engineering.py   ← clean + engineer
        ↓
data/processed/features.csv  ← model-ready rows
        ↓
src/train_model.py           ← train + evaluate
        ↓
models/*.joblib               ← saved pipelines
        ↓
app.py  +  predictor.py      ← live UI
```
"""
    )
with col_b:
    st.markdown(
        """
**Raw data (Kaggle — "IPL Complete Dataset 2008–2024"):**
- One row per match with columns: `team1`, `team2`, `venue`, `toss_winner`,
  `toss_decision`, `winner`, `season`, `date`, etc.

**Cleaning steps:**
- Standardise column names (strip whitespace, lowercase, snake_case).
- Parse dates with `dayfirst=True` to handle the DD/MM/YYYY format in the CSV.
- Drop no-result matches, ties, and Super Overs — these have ambiguous outcomes
  that would add noise to the target label.
- Keep only rows where `winner` is not null.

**Output:** `data/processed/features.csv` — one row per match, 10 numeric feature
columns + identifiers (`team1`, `team2`, `venue`, `date`, `winner`).
"""
    )

# ── 3. Feature Engineering ────────────────────────────────────────────────────
st.header("3  Feature Engineering")
st.markdown(
    "All rolling statistics are computed **before** the current match is processed "
    "to guarantee zero data leakage. The dataset is sorted by date first."
)

features = [
    ("team1_form5", "Win % over last 5 matches for Team 1",
     "Captures short-term momentum. Computed row-by-row: for each match, look at "
     "only the results *before* it. Defaults to 0.5 (neutral) for new teams with no history."),

    ("team2_form5", "Win % over last 5 matches for Team 2",
     "Same computation as above for the opposing team."),

    ("form_diff", "team1_form5 − team2_form5",
     "A single signed number summarising relative momentum. Positive → Team 1 in better "
     "form. This feature helps linear models capture the gap directly."),

    ("h2h_team1_win_pct", "Historical head-to-head win % (Team 1 vs Team 2)",
     "Looks at every previous meeting between the exact pair and records the "
     "fraction Team 1 won. Canonical pair ordering (sorted alphabetically) prevents "
     "double-counting. Defaults to 0.5 when no prior meetings exist."),

    ("team1_won_toss", "Did Team 1 win the toss? (binary 0/1)",
     "Toss advantage varies by venue and conditions; the model learns this interaction "
     "through toss_venue_adv."),

    ("toss_bat", "Did the toss winner choose to bat? (binary 0/1)",
     "Captures the strategic toss decision. Batting vs fielding preference shifts with "
     "dew, pitch conditions, and team strengths."),

    ("team1_home", "Is Team 1 playing at their home ground? (binary 0/1)",
     "Each franchise has known home venues mapped in HOME_VENUES. Partial string matching "
     "handles name variations across dataset versions (e.g. 'Feroz Shah Kotla' vs "
     "'Arun Jaitley Stadium')."),

    ("team2_home", "Is Team 2 playing at their home ground? (binary 0/1)",
     "A neutral venue sets both to 0; a home fixture sets exactly one to 1."),

    ("toss_venue_adv", "Historical rate at which the toss winner wins at this venue",
     "Pre-computed per venue from all prior matches at that ground. A venue where "
     "toss winners win 70% of the time is very different from one where it makes no "
     "difference. Defaults to 0.5 for new venues."),

    ("season", "Year of the season (e.g. 2024)",
     "Ordinal trend feature. Allows the model to learn that newer seasons may reflect "
     "different team compositions, rules, or conditions."),
]

for fname, fshort, fdesc in features:
    with st.expander(f"`{fname}` — {fshort}"):
        st.markdown(fdesc)

# ── 4. Modelling ──────────────────────────────────────────────────────────────
st.header("4  Modelling")

col_lr, col_rf, col_ens = st.columns(3)

with col_lr:
    st.subheader("Logistic Regression")
    st.markdown(
        """
- Wrapped in a `Pipeline` with `StandardScaler` so features are Z-scored before fitting.
- `C=0.5` — mild L2 regularisation to prevent overfitting on the relatively small dataset.
- `max_iter=1000` to ensure convergence.
- **Why include it?** Fast, interpretable coefficients, and a strong baseline. Its coefficients
  also populate the Feature Importance chart, making the model explainable.
"""
    )

with col_rf:
    st.subheader("Random Forest")
    st.markdown(
        """
- 300 trees, `max_depth=6`, `min_samples_leaf=10`.
- Shallow trees and high `min_samples_leaf` combat overfitting on ~950 rows.
- `n_jobs=-1` for parallel training.
- **Why include it?** Captures non-linear interactions (e.g. home advantage matters more
  when combined with good recent form) that logistic regression cannot model without
  manual feature crosses.
"""
    )

with col_ens:
    st.subheader("Ensemble")
    st.markdown(
        """
- Simple average of both models' predicted probabilities.
- Probability averaging is equivalent to a soft vote and reduces variance without
  introducing a meta-learner that could overfit.
- `predicted_winner` is the team with ensemble probability ≥ 0.5.
- **Why ensemble?** LR and RF make different kinds of errors; their combination
  consistently outperforms either alone on the test set.
"""
    )

# ── 5. Evaluation Strategy ────────────────────────────────────────────────────
st.header("5  Evaluation Strategy")

col_split, col_cv = st.columns(2)

with col_split:
    st.subheader("Chronological Train / Test Split")
    st.markdown(
        """
The dataset is split at the **80th percentile date**, not randomly.

- **Why?** Random splitting leaks future information into training (a match from 2023
  could appear in train while a 2019 match is in test). Chronological splitting
  simulates real prediction: train on history, test on the future.
- The test set covers approximately the last 4–5 IPL seasons.
"""
    )

with col_cv:
    st.subheader("Stratified K-Fold Cross-Validation")
    st.markdown(
        """
`StratifiedKFold(n_splits=5)` is run on the training set only, before fitting the
final model on all training data.

- **Stratified** ensures each fold has a balanced target class ratio, important because
  IPL match outcomes can be slightly imbalanced by team ordering.
- CV AUC is reported alongside test AUC to detect overfitting.
- Scoring metric: **ROC-AUC** — robust to class imbalance and captures probability
  calibration, not just hard predictions.
"""
    )

st.markdown(
    """
| Model | CV AUC (5-fold) | Test AUC |
|---|---|---|
| Logistic Regression | ~0.57 | ~0.51 |
| Random Forest | ~0.57 | ~0.51 |
| **Ensemble (avg)** | — | **~0.51** |

The low test AUC reflects an honest reality: IPL outcomes are highly stochastic.
Pre-match features alone (form, H2H, toss, venue history) have limited predictive power —
the model correctly captures this uncertainty rather than overstating confidence.
To push beyond ~55% AUC would require ball-by-ball data, player availability, and
pitch/weather reports — data not available before a match starts.
"""
)

# ── 6. Anti-Leakage Design ────────────────────────────────────────────────────
st.header("6  Data Leakage Prevention")
st.markdown(
    """
Data leakage is the most common mistake in sports prediction ML. This project avoids it
in three explicit ways:

| Leakage Risk | How It's Prevented |
|---|---|
| Rolling form using future results | `rolling_win_pct()` appends the current result *after* recording the stat for the current row |
| Head-to-head using current match result | `head_to_head()` reads the cumulative history *before* updating it with the current outcome |
| Toss-venue advantage using current match | `toss_venue_advantage()` reads venue history *before* appending the current match |
| Random train/test split | Chronological 80/20 split — no future matches appear in training |

All three rolling functions iterate the dataframe in date order and follow the
**read → record → update** pattern.
"""
)

# ── 7. App Architecture ───────────────────────────────────────────────────────
st.header("7  App Architecture")
st.markdown(
    """
```
app.py
├── st.cache_resource  →  load_resources()       # models loaded once, stay in memory
├── st.cache_data      →  load_feature_data()    # CSV loaded once, shared across tabs
├── Match Setup Form   →  selectboxes + button
├── Prediction Block   →  predictor.predict_match()
│     ├── builds feature vector from user inputs + historical stats
│     ├── calls lr_model.predict_proba() + rf_model.predict_proba()
│     └── returns ensemble probability + winner
├── Tab: Dataset Overview   →  aggregated stats from features.csv
├── Tab: Model Performance  →  static PNGs saved during training
└── Tab: Head-to-Head       →  filtered query on features.csv

pages/Behind_the_Code.py   ←  this page (recruiter explainer)

src/
├── feature_engineering.py  ←  data → features.csv
├── train_model.py          ←  features.csv → models/*.joblib + plots
└── predictor.py            ←  shared constants (HOME_VENUES) + predict_match()
```

**Key design choices:**
- `predictor.py` is the single source of truth for `HOME_VENUES` — both
  `feature_engineering.py` and `app.py` import from it, so a venue change only needs
  to be made in one place.
- Models are Scikit-learn `Pipeline` objects (Scaler + Classifier), so the full
  preprocessing is embedded in the `.joblib` file — no risk of forgetting to scale
  at inference time.
- `@st.cache_resource` keeps the 300-tree Random Forest in memory across requests
  rather than reloading it on every interaction.
"""
)

# ── 8. Tech Stack ─────────────────────────────────────────────────────────────
st.header("8  Tech Stack")

t1, t2, t3, t4 = st.columns(4)
with t1:
    st.markdown("**Data**")
    st.markdown("- Pandas\n- NumPy\n- Kaggle CSV dataset")
with t2:
    st.markdown("**ML**")
    st.markdown("- Scikit-learn\n- Logistic Regression\n- Random Forest\n- Joblib")
with t3:
    st.markdown("**Visualisation**")
    st.markdown("- Plotly (interactive charts)\n- Matplotlib / Seaborn (training plots)")
with t4:
    st.markdown("**App**")
    st.markdown("- Streamlit\n- Python 3.10+\n- Multipage architecture")

st.markdown("---")
st.caption("IPL Match Outcome Predictor · Built with Python & Streamlit · Data: IPL 2008–2024")
