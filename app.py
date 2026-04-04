"""
Streamlit app — IPL Match Outcome Predictor
Run: streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from predictor import HOME_VENUES, predict_match

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Match Outcome Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
TEAMS = sorted([
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Gujarat Titans",
    "Lucknow Super Giants",
])

VENUES = sorted([
    # Current / regular IPL venues
    "Wankhede Stadium",
    "MA Chidambaram Stadium, Chepauk",
    "M Chinnaswamy Stadium",
    "Eden Gardens",
    "Arun Jaitley Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Sawai Mansingh Stadium",
    "Rajiv Gandhi International Stadium, Uppal",
    "Narendra Modi Stadium, Ahmedabad",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "Brabourne Stadium",
    "Dr DY Patil Sports Academy, Mumbai",
    "Maharashtra Cricket Association Stadium, Pune",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
    "Holkar Cricket Stadium",
    "Saurashtra Cricket Association Stadium",
    "Barsa Para Cricket Stadium, Guwahati",
    # Historic / alternate IPL venues
    "Feroz Shah Kotla",
    "Subrata Roy Sahara Stadium",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
    "Sardar Patel Stadium, Motera",
    "JSCA International Stadium Complex",
    "Barabati Stadium",
    "Green Park",
    "Vidarbha Cricket Association Stadium, Jamtha",
    "Nehru Stadium",
    "Shaheed Veer Narayan Singh International Stadium",
    # UAE / overseas venues
    "Dubai International Cricket Stadium",
    "Sheikh Zayed Stadium",
    "Sharjah Cricket Stadium",
    "Zayed Cricket Stadium, Abu Dhabi",
    # South Africa venues (2009)
    "Newlands",
    "Kingsmead",
    "New Wanderers Stadium",
    "SuperSport Park",
    "OUTsurance Oval",
    "De Beers Diamond Oval",
    "Buffalo Park",
    "St George's Park",
])

MODELS_READY = (
    Path("models/logistic_regression.joblib").exists()
    and Path("models/random_forest.joblib").exists()
)


# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_resources():
    from predictor import load_models, load_historical_stats
    lr, rf = load_models()
    team_form5, team_form10, h2h, toss_venue, team_venue, elo, season_form, streak = load_historical_stats()
    return lr, rf, team_form5, team_form10, h2h, toss_venue, team_venue, elo, season_form, streak


@st.cache_data(show_spinner=False)
def load_feature_data():
    path = Path("data/processed/features.csv")
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding: 8px 0 4px 0;'>"
        "<span style='font-size:3rem;'>🏏</span><br>"
        "<span style='font-size:1.3rem; font-weight:700; letter-spacing:0.5px;'>IPL Predictor</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🏏 IPL Match Outcome Predictor")
st.markdown("Predict win probability **before the first ball is bowled**.")

if not MODELS_READY:
    st.error(
        "**Models not found.** Please run the setup steps first:\n\n"
        "```bash\n"
        "# 1. Place matches.csv in data/raw/\n"
        "# 2. python src/feature_engineering.py\n"
        "# 3. python src/train_model.py\n"
        "# 4. streamlit run app.py\n"
        "```\n\n"
        "See README.md for the Kaggle dataset link."
    )
    st.stop()

lr_model, rf_model, team_form, team_form10, h2h, toss_venue, team_venue, elo, season_form, streak = load_resources()

# ── Match setup form ──────────────────────────────────────────────────────────
st.markdown("### Match Setup")
col1, col2, col3 = st.columns(3)

with col1:
    team1 = st.selectbox("Team 1", TEAMS, index=0)

with col2:
    team2_options = [t for t in TEAMS if t != team1]
    team2 = st.selectbox("Team 2", team2_options, index=0)

with col3:
    venue = st.selectbox("Venue", VENUES, index=0)

col4, col5, col6 = st.columns(3)

with col4:
    toss_winner = st.selectbox("Toss Winner", [team1, team2])

with col5:
    toss_decision_label = st.selectbox("Toss Decision", ["Bat First", "Field First"])
    toss_decision = "bat" if toss_decision_label == "Bat First" else "field"

with col6:
    season = st.selectbox("Season", list(range(2026, 2007, -1)), index=0)

col7, _ = st.columns([1, 2])
with col7:
    match_type_label = st.selectbox("Match Type", ["League Match", "Playoff / Knockout"])
    is_playoff = 1 if match_type_label == "Playoff / Knockout" else 0

predict_btn = st.button("🔮 Predict Outcome", type="primary", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    result = predict_match(
        team1=team1,
        team2=team2,
        venue=venue,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
        season=season,
        is_playoff=is_playoff,
        lr_model=lr_model,
        rf_model=rf_model,
        team_form=team_form,
        team_form10=team_form10,
        h2h=h2h,
        toss_venue=toss_venue,
        team_venue=team_venue,
        elo=elo,
        season_form=season_form,
        streak=streak,
    )

    st.markdown("---")
    st.markdown("### Prediction Results")

    # Winner banner
    winner = result["predicted_winner"]
    conf = result["confidence"]
    st.success(f"**Predicted Winner: {winner}** — Confidence: {conf:.1%}")

    # Probability gauge chart
    p1 = result["ensemble_prob_team1"]
    p2 = 1 - p1

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=["Win Probability"],
        x=[p1 * 100],
        name=team1,
        orientation="h",
        marker_color="#1f77b4",
        text=f"{p1:.1%}",
        textposition="inside",
        insidetextanchor="middle",
    ))
    fig.add_trace(go.Bar(
        y=["Win Probability"],
        x=[p2 * 100],
        name=team2,
        orientation="h",
        marker_color="#ff7f0e",
        text=f"{p2:.1%}",
        textposition="inside",
        insidetextanchor="middle",
    ))

    fig.update_layout(
        barmode="stack",
        height=120,
        margin=dict(l=0, r=0, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model breakdown
    col_lr, col_rf, col_ens = st.columns(3)
    with col_lr:
        st.metric("Logistic Regression", f"{result['lr_prob_team1']:.1%}",
                  help=f"{team1} win probability from LR")
    with col_rf:
        st.metric("Random Forest", f"{result['rf_prob_team1']:.1%}",
                  help=f"{team1} win probability from RF")
    with col_ens:
        st.metric("Ensemble (avg)", f"{result['ensemble_prob_team1']:.1%}",
                  help="Average of both models")

    # Feature context
    with st.expander("Feature values used for this prediction"):
        t1_form = team_form.get(team1, 0.5)
        t2_form = team_form.get(team2, 0.5)
        _h2h_direct = h2h.get((team1, team2), None)
        _h2h_reverse = h2h.get((team2, team1), None)
        if _h2h_direct is not None:
            h2h_pct = _h2h_direct
        elif _h2h_reverse is not None:
            h2h_pct = 1 - _h2h_reverse
        else:
            h2h_pct = None
        from predictor import _get_toss_venue_adv
        tva = _get_toss_venue_adv(venue, toss_venue)

        feat_df = pd.DataFrame({
            "Feature": [
                f"{team1} form (last 5)",
                f"{team2} form (last 5)",
                "Head-to-head (team1 win%)",
                "Toss winner = team1",
                "Toss decision = bat",
                "Team1 home ground",
                "Team2 home ground",
                "Toss×venue advantage",
            ],
            "Value": [
                f"{t1_form:.1%}",
                f"{t2_form:.1%}",
                f"{h2h_pct:.1%}" if h2h_pct is not None else "N/A",
                "Yes" if toss_winner == team1 else "No",
                "Yes" if toss_decision == "bat" else "No",
                "Yes" if any(v.lower() in venue.lower() for v in HOME_VENUES.get(team1, [])) else "No",
                "Yes" if any(v.lower() in venue.lower() for v in HOME_VENUES.get(team2, [])) else "No",
                f"{tva:.1%}",
            ],
        })
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

# ── EDA / Model tabs ──────────────────────────────────────────────────────────
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "📈 Model Performance", "🔍 Head-to-Head"])

with tab1:
    df_feat = load_feature_data()
    if df_feat is not None:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Matches", len(df_feat))
        col_b.metric("Seasons", df_feat["season"].nunique() if "season" in df_feat.columns else "—")
        col_c.metric("Venues", df_feat["venue"].nunique())

        df_feat["toss_winner_won"] = (df_feat["team1_won_toss"] == df_feat["target"]).astype(int)
        toss_agg = df_feat.groupby("toss_bat")["toss_winner_won"].agg(["mean", "count"]).reset_index()
        toss_agg["toss_bat"] = toss_agg["toss_bat"].map({1: "Chose to Bat", 0: "Chose to Field"})

        field_win  = toss_agg.loc[toss_agg["toss_bat"] == "Chose to Field",  "mean"].values[0]
        bat_win    = toss_agg.loc[toss_agg["toss_bat"] == "Chose to Bat",    "mean"].values[0]
        field_cnt  = int(toss_agg.loc[toss_agg["toss_bat"] == "Chose to Field", "count"].values[0])
        bat_cnt    = int(toss_agg.loc[toss_agg["toss_bat"] == "Chose to Bat",   "count"].values[0])
        overall    = df_feat["toss_winner_won"].mean()

        st.markdown("**Toss Impact Analysis**")
        ta, tb, tc = st.columns(3)
        ta.metric("Overall: toss winner wins", f"{overall:.1%}", help="Across all IPL matches")
        tb.metric("Win rate — chose to Field", f"{field_win:.1%}", f"{field_cnt} matches",
                  help="Toss winner elected to field first")
        tc.metric("Win rate — chose to Bat",   f"{bat_win:.1%}",  f"{bat_cnt} matches",
                  help="Toss winner elected to bat first")

        st.caption(
            f"**Key insight:** Captains who win the toss and **choose to field** win "
            f"**{field_win:.1%}** of the time vs only **{bat_win:.1%}** when batting first — "
            f"chasing is clearly the preferred IPL strategy."
        )

        fig_toss = go.Figure(go.Bar(
            x=["Chose to Field", "Chose to Bat"],
            y=[field_win * 100, bat_win * 100],
            text=[f"{field_win:.1%}", f"{bat_win:.1%}"],
            textposition="outside",
            marker_color=["#2ecc71", "#e74c3c"],
            width=0.4,
        ))
        fig_toss.add_hline(y=50, line_dash="dash", line_color="gray",
                           annotation_text="50% baseline", annotation_position="right")
        fig_toss.update_layout(
            height=320,
            yaxis=dict(range=[0, 70], ticksuffix="%", title="Toss Winner Win Rate"),
            xaxis_title="Toss Decision",
            margin=dict(l=0, r=0, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_toss, use_container_width=True)

        st.markdown("**Matches per Season**")
        season_counts = df_feat.groupby("season").size().reset_index(name="Matches")
        st.bar_chart(season_counts.set_index("season"))
    else:
        st.info("Run feature engineering first to see dataset stats.")

with tab2:
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
    import numpy as np

    df_perf = load_feature_data()
    if df_perf is not None:
        FEATURE_COLS = [
            "team1_won_toss","toss_bat",
            "team1_bats_first","team1_fields_first",
            "team2_bats_first","team2_fields_first",
            "team1_home","team2_home",
            "team1_form5","team2_form5",
            "team1_form10","team2_form10",
            "form_diff","h2h_team1_win_pct",
            "toss_venue_adv",
            "team1_venue_win_rate","team2_venue_win_rate","venue_win_diff",
            "team1_elo","team2_elo","elo_diff",
            "team1_season_form","team2_season_form",
            "team1_streak","team2_streak",
            "is_playoff","season",
        ]
        df_perf["date"] = pd.to_datetime(df_perf["date"])
        split_date = df_perf["date"].quantile(0.8)
        test_df = df_perf[df_perf["date"] >= split_date].copy()
        X_te = test_df[FEATURE_COLS].fillna(0.5)
        y_te = test_df["target"]

        lr_prob  = lr_model.predict_proba(X_te)[:, 1]
        rf_prob  = rf_model.predict_proba(X_te)[:, 1]
        ens_prob = (lr_prob + rf_prob) / 2

        lr_auc  = roc_auc_score(y_te, lr_prob)
        rf_auc  = roc_auc_score(y_te, rf_prob)
        ens_auc = roc_auc_score(y_te, ens_prob)

        ma, mb, mc = st.columns(3)
        ma.metric("LR Test AUC",       f"{lr_auc:.3f}")
        mb.metric("RF Test AUC",       f"{rf_auc:.3f}")
        mc.metric("Ensemble Test AUC", f"{ens_auc:.3f}")

        # ROC Curve
        st.markdown("**ROC Curve — Test Set**")
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="gray"))
        for name, prob, auc in [
            ("Logistic Regression", lr_prob,  lr_auc),
            ("Random Forest",       rf_prob,  rf_auc),
            ("Ensemble",            ens_prob, ens_auc),
        ]:
            fpr, tpr, _ = roc_curve(y_te, prob)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"{name} (AUC={auc:.3f})"))
        fig_roc.update_layout(
            height=380,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # Feature Importance
        st.markdown("**Feature Importance**")
        fi_col1, fi_col2 = st.columns(2)

        with fi_col1:
            st.caption("Logistic Regression Coefficients")
            lr_clf = lr_model.named_steps["clf"]
            coef_df = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Coefficient": lr_clf.coef_[0],
            }).sort_values("Coefficient")
            fig_lr = go.Figure(go.Bar(
                x=coef_df["Coefficient"], y=coef_df["Feature"],
                orientation="h",
                marker_color=["#d73027" if c < 0 else "#4575b4"
                              for c in coef_df["Coefficient"]],
            ))
            fig_lr.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_lr, use_container_width=True)

        with fi_col2:
            st.caption("Random Forest Importance")
            rf_clf = rf_model.named_steps["clf"]
            imp_df = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Importance": rf_clf.feature_importances_,
            }).sort_values("Importance")
            fig_rf = go.Figure(go.Bar(
                x=imp_df["Importance"], y=imp_df["Feature"],
                orientation="h", marker_color="#4575b4",
            ))
            fig_rf.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rf, use_container_width=True)

        # Confusion Matrices
        st.markdown("**Confusion Matrices**")
        cm_col1, cm_col2 = st.columns(2)
        for col, name, prob in [
            (cm_col1, "Logistic Regression", lr_prob),
            (cm_col2, "Random Forest",       rf_prob),
        ]:
            cm = confusion_matrix(y_te, (prob >= 0.5).astype(int))
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=["team2 wins", "team1 wins"],
                y=["team2 wins", "team1 wins"],
                colorscale="Blues",
                text=cm, texttemplate="%{text}",
                showscale=False,
            ))
            fig_cm.update_layout(
                title=name, height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            col.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Run feature engineering first to see model performance.")

with tab3:
    df_feat = load_feature_data()
    if df_feat is not None:
        h2h_t1 = st.selectbox("Team A", TEAMS, key="h2h_t1")
        h2h_t2 = st.selectbox("Team B", [t for t in TEAMS if t != h2h_t1], key="h2h_t2")

        matches = df_feat[
            ((df_feat["team1"] == h2h_t1) & (df_feat["team2"] == h2h_t2)) |
            ((df_feat["team1"] == h2h_t2) & (df_feat["team2"] == h2h_t1))
        ].copy()

        if len(matches) == 0:
            st.info("No historical head-to-head data found for this pair.")
        else:
            t1_wins = (
                ((matches["team1"] == h2h_t1) & (matches["target"] == 1)) |
                ((matches["team2"] == h2h_t1) & (matches["target"] == 0))
            ).sum()
            t2_wins = len(matches) - t1_wins

            c1, c2, c3 = st.columns(3)
            c1.metric(f"{h2h_t1} wins", t1_wins)
            c2.metric("Total matches", len(matches))
            c3.metric(f"{h2h_t2} wins", t2_wins)

            fig_h2h = go.Figure(go.Pie(
                labels=[h2h_t1, h2h_t2],
                values=[t1_wins, t2_wins],
                hole=0.4,
                marker_colors=["#1f77b4", "#ff7f0e"],
            ))
            fig_h2h.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig_h2h, use_container_width=True)
    else:
        st.info("Run feature engineering first.")
