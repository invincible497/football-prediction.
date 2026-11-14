"""
Football Match Outcome Prediction System - Streamlit Web Application.
A professional, modern dashboard for football match prediction, team analysis,
machine learning model evaluation, and prediction tracking.
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config FIRST before any other streamlit calls
st.set_page_config(
    page_title="PredicKick | Football Outcome AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

from database import (
    init_db, get_dashboard_stats, get_leagues_list,
    get_teams_by_league, get_all_matches, get_predictions_history,
    update_prediction_actual, DB_FILE
)
from data_loader import load_and_sync_all_data, LEAGUES
from model_trainer import (
    train_and_evaluate_all_models, get_stored_metrics,
    MODELS_DIR, BEST_MODEL_FILE
)
from predictor import predict_match


# Custom CSS styling
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        color: #f8fafc;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-val {
        font-size: 2.1rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Prediction Banner */
    .prediction-banner-h {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        color: #ecfdf5;
        margin-bottom: 20px;
    }
    .prediction-banner-d {
        background: linear-gradient(135deg, #78350f 0%, #451a03 100%);
        border: 2px solid #f59e0b;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        color: #fffbeb;
        margin-bottom: 20px;
    }
    .prediction-banner-a {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        border: 2px solid #3b82f6;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        color: #eff6ff;
        margin-bottom: 20px;
    }

    /* Outcome Badges */
    .badge-win {
        background-color: #10b981;
        color: #022c22;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-draw {
        background-color: #f59e0b;
        color: #451a03;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-loss {
        background-color: #ef4444;
        color: #450a0a;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Prob Cards */
    .prob-box {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #334155;
    }
    .prob-pct {
        font-size: 1.8rem;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def setup_initial_system():
    """Initializes SQLite database, syncs data, and trains models if needed."""
    init_db()
    load_and_sync_all_data(force_refresh=False)
    metrics = get_stored_metrics()
    if not metrics or not os.path.exists(BEST_MODEL_FILE):
        try:
            train_and_evaluate_all_models()
        except Exception as e:
            st.warning(f"Initial model training notice: {e}")
    return True


# Sidebar Navigation
setup_initial_system()

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/football2.png", width=64)
    st.title("PredicKick AI")
    st.caption("Intelligent Football Match Prediction System")
    st.markdown("---")

    menu_selection = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "⚽ Match Predictor",
            "📈 Team Analysis",
            "🤖 Model Performance",
            "📜 Prediction History",
            "⚙️ Data & Model Settings"
        ],
        index=0
    )

    st.markdown("---")
    st.info(
        "⚠️ **Disclaimer**\n\n"
        "Predictions are probabilistic statistical estimates based on historical machine learning models. "
        "They do not guarantee match results."
    )


# ---------------------------------------------------------------------------
# PAGE 1: DASHBOARD
# ---------------------------------------------------------------------------
if menu_selection == "📊 Dashboard":
    st.title("📊 System Dashboard & Overview")
    st.markdown("Real-time monitoring of historical datasets, model readiness, and predictive performance.")

    stats = get_dashboard_stats()
    metrics = get_stored_metrics()

    best_acc = 0.0
    best_model_name = "N/A"
    if metrics and "models" in metrics:
        best_model_name = metrics.get("best_model", "N/A")
        if best_model_name in metrics["models"]:
            best_acc = metrics["models"][best_model_name]["accuracy"] * 100

    # Top KPI Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Matches</div>
            <div class="metric-val">{stats['total_matches']:,}</div>
            <div class="metric-sub">5 Top Leagues</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Teams Tracked</div>
            <div class="metric-val">{stats['total_teams']}</div>
            <div class="metric-sub">Across Europe</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Model Accuracy</div>
            <div class="metric-val">{best_acc:.1f}%</div>
            <div class="metric-sub">{best_model_name}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predictions Made</div>
            <div class="metric-val">{stats['total_predictions']}</div>
            <div class="metric-sub">Saved in SQLite</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        acc_text = f"{stats['prediction_accuracy']:.1f}%" if stats['prediction_accuracy'] is not None else "Pending"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Live Win Rate</div>
            <div class="metric-val">{acc_text}</div>
            <div class="metric-sub">{stats['verified_predictions']} Verified</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🏆 Historical Match Distribution & League Insights")
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.subheader("Outcome Frequency (Home Advantage)")
        outcomes = stats.get("match_outcomes", {})
        if outcomes:
            labels = ["Home Win (1)", "Draw (X)", "Away Win (2)"]
            counts = [outcomes.get("H", 0), outcomes.get("D", 0), outcomes.get("A", 0)]
            colors = ["#10b981", "#f59e0b", "#3b82f6"]

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            wedges, texts, autotexts = ax.pie(
                counts,
                labels=labels,
                autopct='%1.1f%%',
                colors=colors,
                startangle=140,
                textprops=dict(color="#f8fafc")
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
            ax.axis('equal')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No match distribution data available yet.")

    with c_right:
        st.subheader("Active Leagues Tracked")
        leagues = get_leagues_list()
        league_df = pd.DataFrame(leagues)
        if not league_df.empty:
            league_df.columns = ["Code", "League Name", "Country"]
            st.dataframe(league_df, use_container_width=True, hide_index=True)
        st.markdown("""
        **Data Highlights:**
        - High-density coverage of **English Premier League**, **La Liga**, **Bundesliga**, **Serie A**, and **Ligue 1**.
        - Strict temporal ordering guarantees **zero future match data leakage**.
        - Full rolling features: last 5-match points, venue win rates, head-to-head records, and rest day fatigue.
        """)

    st.markdown("### ⏱️ Most Recent Predictions")
    preds_df = get_predictions_history()
    if not preds_df.empty:
        display_df = preds_df[[
            'created_at', 'match_date', 'league_name', 'home_team', 'away_team',
            'predicted_outcome', 'prob_home', 'prob_draw', 'prob_away', 'confidence', 'actual_outcome', 'is_correct'
        ]].head(5).copy()
        display_df.columns = [
            'Generated At', 'Fixture Date', 'League', 'Home Team', 'Away Team',
            'Pick', 'Home %', 'Draw %', 'Away %', 'Confidence %', 'Actual', 'Result'
        ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions recorded yet. Jump to the **Match Predictor** to generate your first match outcome prediction!")


# ---------------------------------------------------------------------------
# PAGE 2: MATCH PREDICTOR
# ---------------------------------------------------------------------------
elif menu_selection == "⚽ Match Predictor":
    st.title("⚽ Match Outcome Predictor")
    st.markdown("Select two teams to run probabilistic machine learning inference and examine key influencing factors.")

    leagues = get_leagues_list()
    league_options = {f"{l['name']} ({l['country']})": l['code'] for l in leagues}

    col_league, col_date = st.columns([2, 1])
    with col_league:
        selected_league_name = st.selectbox("1. Select Football League", list(league_options.keys()))
        selected_league_code = league_options[selected_league_name]

    with col_date:
        match_date = st.date_input("2. Match Date", value=datetime.now())

    teams = get_teams_by_league(selected_league_code)
    if not teams:
        teams = get_teams_by_league("ALL")

    col_home, col_away = st.columns(2)
    with col_home:
        home_team = st.selectbox("🏠 Home Team", teams, index=0 if len(teams) > 0 else None)
    with col_away:
        away_default_idx = 1 if len(teams) > 1 else 0
        away_team = st.selectbox("✈️ Away Team", teams, index=away_default_idx)

    # Optional model selection
    metrics = get_stored_metrics()
    model_choices = ["Best Model (Auto-selected)", "Logistic Regression", "Random Forest", "Gradient Boosting"]
    selected_model_choice = st.selectbox("Machine Learning Algorithm", model_choices, index=0)
    actual_model_choice = None if "Auto-selected" in selected_model_choice else selected_model_choice

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Generate Match Prediction", type="primary", use_container_width=True)

    if predict_btn:
        if home_team == away_team:
            st.error("Error: Home Team and Away Team cannot be the same club. Please select two distinct teams.")
        else:
            with st.spinner(f"Analysing rolling form, venue dynamics, and H2H records for {home_team} vs {away_team}..."):
                try:
                    res = predict_match(
                        home_team=home_team,
                        away_team=away_team,
                        league_code=selected_league_code,
                        match_date=str(match_date),
                        model_name=actual_model_choice,
                        save_to_history=True
                    )

                    pred_code = res['predicted_outcome']
                    p_home = res['prob_home']
                    p_draw = res['prob_draw']
                    p_away = res['prob_away']
                    conf = res['confidence']

                    # Outcome Banner
                    banner_class = "prediction-banner-h" if pred_code == '1' else ("prediction-banner-a" if pred_code == '2' else "prediction-banner-d")
                    st.markdown(f"""
                    <div class="{banner_class}">
                        <h4 style="margin:0; text-transform:uppercase; letter-spacing:1px; opacity:0.85;">Model Prediction Outcome</h4>
                        <h1 style="margin: 8px 0; font-size: 2.8rem;">{res['predicted_label']}</h1>
                        <p style="font-size: 1.15rem; margin: 0;">
                            Confidence Level: <strong>{conf}% ({res['confidence_tier']})</strong> | Model: <strong>{res['model_name']}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 3 Outcome Probability Cards
                    p_c1, p_c2, p_c3 = st.columns(3)
                    with p_c1:
                        st.markdown(f"""
                        <div class="prob-box" style="border-top: 4px solid #10b981;">
                            <div style="color: #94a3b8; font-weight:600;">HOME WIN (1)</div>
                            <div class="prob-pct" style="color: #10b981;">{p_home}%</div>
                            <div style="font-size: 0.85rem; color: #cbd5e1;">{home_team}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with p_c2:
                        st.markdown(f"""
                        <div class="prob-box" style="border-top: 4px solid #f59e0b;">
                            <div style="color: #94a3b8; font-weight:600;">DRAW (X)</div>
                            <div class="prob-pct" style="color: #f59e0b;">{p_draw}%</div>
                            <div style="font-size: 0.85rem; color: #cbd5e1;">Tie Outcome</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with p_c3:
                        st.markdown(f"""
                        <div class="prob-box" style="border-top: 4px solid #3b82f6;">
                            <div style="color: #94a3b8; font-weight:600;">AWAY WIN (2)</div>
                            <div class="prob-pct" style="color: #3b82f6;">{p_away}%</div>
                            <div style="font-size: 0.85rem; color: #cbd5e1;">{away_team}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Visual horizontal stacked probability bar
                    st.markdown(f"""
                    <div style="width: 100%; background: #334155; border-radius: 8px; overflow: hidden; height: 26px; display: flex; font-size: 0.8rem; font-weight: bold; line-height: 26px; text-align: center; color: white;">
                        <div style="width: {p_home}%; background: #10b981;">{p_home}%</div>
                        <div style="width: {p_draw}%; background: #f59e0b;">{p_draw}%</div>
                        <div style="width: {p_away}%; background: #3b82f6;">{p_away}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Influencing Factors Breakdown
                    st.markdown("### 🔍 Key Prediction Influencing Factors")
                    exp = res['explanation']

                    f_col1, f_col2 = st.columns([1, 1])
                    with f_col1:
                        st.markdown("#### Factor Breakdown")
                        for f in exp['influencing_factors']:
                            st.markdown(f"""
                            - **{f['factor']}** (Advantage: *{f['advantage']}* | Impact: `{f['weight']}`):  
                              {f['detail']}
                            """)

                    with f_col2:
                        st.markdown("#### Analytical Match Summary")
                        for bullet in exp['bullet_points']:
                            st.markdown(f"- {bullet}")

                    st.success("✅ This prediction has been logged to your persistent **Prediction History**.")

                except Exception as ex:
                    st.error(f"Error executing prediction: {ex}")


# ---------------------------------------------------------------------------
# PAGE 3: TEAM ANALYSIS
# ---------------------------------------------------------------------------
elif menu_selection == "📈 Team Analysis":
    st.title("📈 In-Depth Team Analytics")
    st.markdown("Examine historical performance, home vs away split, goal efficiency, and head-to-head dynamics.")

    leagues = get_leagues_list()
    league_options = {l['name']: l['code'] for l in leagues}
    sel_league_name = st.selectbox("Filter League", list(league_options.keys()))
    sel_league_code = league_options[sel_league_name]

    teams = get_teams_by_league(sel_league_code)
    if not teams:
        teams = get_teams_by_league("ALL")

    sel_team = st.selectbox("Select Team", teams)

    all_matches = get_all_matches()
    if not all_matches.empty and sel_team:
        # Filter matches involving selected team
        team_matches = all_matches[
            (all_matches['home_team'] == sel_team) | (all_matches['away_team'] == sel_team)
        ].copy()

        if not team_matches.empty:
            # Calculate team records
            total_games = len(team_matches)
            wins = 0
            draws = 0
            losses = 0
            gf = 0
            ga = 0

            home_games = 0
            home_wins = 0
            away_games = 0
            away_wins = 0

            recent_results = []

            for _, row in team_matches.iterrows():
                is_home = (row['home_team'] == sel_team)
                ftr = row['ftr']
                if is_home:
                    home_games += 1
                    gf += row['fthg']
                    ga += row['ftag']
                    if ftr == 'H':
                        wins += 1
                        home_wins += 1
                        recent_results.append(('W', row['match_date'], f"vs {row['away_team']} (H)", f"{row['fthg']}-{row['ftag']}"))
                    elif ftr == 'D':
                        draws += 1
                        recent_results.append(('D', row['match_date'], f"vs {row['away_team']} (H)", f"{row['fthg']}-{row['ftag']}"))
                    else:
                        losses += 1
                        recent_results.append(('L', row['match_date'], f"vs {row['away_team']} (H)", f"{row['fthg']}-{row['ftag']}"))
                else:
                    away_games += 1
                    gf += row['ftag']
                    ga += row['fthg']
                    if ftr == 'A':
                        wins += 1
                        away_wins += 1
                        recent_results.append(('W', row['match_date'], f"@ {row['home_team']} (A)", f"{row['ftag']}-{row['fthg']}"))
                    elif ftr == 'D':
                        draws += 1
                        recent_results.append(('D', row['match_date'], f"@ {row['home_team']} (A)", f"{row['ftag']}-{row['fthg']}"))
                    else:
                        losses += 1
                        recent_results.append(('L', row['match_date'], f"@ {row['home_team']} (A)", f"{row['ftag']}-{row['fthg']}"))

            win_pct = (wins / total_games * 100) if total_games > 0 else 0
            home_win_pct = (home_wins / home_games * 100) if home_games > 0 else 0
            away_win_pct = (away_wins / away_games * 100) if away_games > 0 else 0

            # Overview Cards
            tc1, tc2, tc3, tc4, tc5 = st.columns(5)
            with tc1:
                st.metric("Total Matches", total_games)
            with tc2:
                st.metric("Win Rate", f"{win_pct:.1f}%", f"{wins}W {draws}D {losses}L")
            with tc3:
                st.metric("Home Win Rate", f"{home_win_pct:.1f}%", f"{home_wins}/{home_games}")
            with tc4:
                st.metric("Away Win Rate", f"{away_win_pct:.1f}%", f"{away_wins}/{away_games}")
            with tc5:
                st.metric("Goal Difference", f"{gf - ga:+d}", f"{gf} GF, {ga} GA")

            st.markdown("### 📅 Recent Match Form (Last 5 Fixtures)")
            last5 = recent_results[-5:]
            badge_cols = st.columns(5)
            for idx, res in enumerate(reversed(last5)):
                badge_type = "badge-win" if res[0] == 'W' else ("badge-draw" if res[0] == 'D' else "badge-loss")
                with badge_cols[idx]:
                    st.markdown(f"""
                    <div style="background:#1e293b; padding:12px; border-radius:8px; border:1px solid #334155; text-align:center;">
                        <span class="{badge_type}">{res[0]}</span>
                        <div style="font-weight:600; margin-top:8px; font-size:0.9rem;">{res[2]}</div>
                        <div style="font-size:0.8rem; color:#94a3b8;">Score: {res[3]}</div>
                        <div style="font-size:0.75rem; color:#64748b;">{res[1]}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Performance Charts
            ch1, ch2 = st.columns(2)
            with ch1:
                st.subheader("Outcome Breakdown")
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('#0f172a')
                ax.set_facecolor('#0f172a')
                ax.pie(
                    [wins, draws, losses],
                    labels=["Wins", "Draws", "Losses"],
                    autopct="%1.1f%%",
                    colors=["#10b981", "#f59e0b", "#ef4444"],
                    textprops=dict(color="#f8fafc")
                )
                st.pyplot(fig)
                plt.close(fig)

            with ch2:
                st.subheader("Home vs Away Win Percentage")
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('#0f172a')
                ax.set_facecolor('#0f172a')
                bars = ax.bar(["Home Venue", "Away Venue"], [home_win_pct, away_win_pct], color=["#10b981", "#3b82f6"])
                ax.set_ylabel("Win Rate (%)", color="#f8fafc")
                ax.tick_params(colors="#f8fafc")
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', color="white", fontweight='bold')
                ax.set_ylim(0, 100)
                st.pyplot(fig)
                plt.close(fig)

            # Head to head explorer
            st.markdown("### ⚔️ Head-to-Head Comparison Tool")
            opponents = [t for t in teams if t != sel_team]
            sel_opp = st.selectbox("Select Opponent to Compare", opponents)
            if sel_opp:
                h2h_matches = team_matches[
                    (team_matches['home_team'] == sel_opp) | (team_matches['away_team'] == sel_opp)
                ].sort_values(by='match_date', ascending=False)

                if not h2h_matches.empty:
                    st.write(f"Found **{len(h2h_matches)}** previous encounters:")
                    display_h2h = h2h_matches[['match_date', 'home_team', 'away_team', 'fthg', 'ftag', 'ftr']].copy()
                    display_h2h.columns = ['Date', 'Home', 'Away', 'HG', 'AG', 'Result']
                    st.dataframe(display_h2h, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No direct matches recorded between {sel_team} and {sel_opp} in current database.")


# ---------------------------------------------------------------------------
# PAGE 4: MODEL PERFORMANCE
# ---------------------------------------------------------------------------
elif menu_selection == "🤖 Model Performance":
    st.title("🤖 Machine Learning Model Performance & Benchmark")
    st.markdown("Rigorous chronological evaluation of **Logistic Regression**, **Random Forest**, and **Gradient Boosting**.")

    metrics = get_stored_metrics()
    if not metrics or "models" not in metrics:
        st.warning("Model metrics not found. Train the models below to view comprehensive benchmarks.")
        if st.button("🚀 Train & Benchmark All Models Now"):
            with st.spinner("Engineering chronological features and training models..."):
                metrics = train_and_evaluate_all_models()
                st.success("Training and benchmark completed!")
                st.rerun()
    else:
        best_model = metrics.get("best_model", "N/A")
        st.success(f"🏆 **Champion Model:** **{best_model}** (Selected automatically based on highest weighted F1 score and calibrated log loss).")

        # Comparison Table
        st.subheader("Model Comparison Matrix")
        model_rows = []
        for name, m in metrics["models"].items():
            model_rows.append({
                "Algorithm": name,
                "Accuracy (%)": round(m["accuracy"] * 100, 2),
                "F1 Score (Weighted)": round(m["f1_weighted"], 4),
                "F1 Score (Macro)": round(m["f1_macro"], 4),
                "Precision (Weighted)": round(m["precision_weighted"], 4),
                "Recall (Weighted)": round(m["recall_weighted"], 4),
                "Log Loss": round(m["log_loss"], 4),
                "Brier Score": round(m["brier_score"], 4)
            })

        df_comparison = pd.DataFrame(model_rows)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

        # Comparative Bar Charts
        st.markdown("### 📊 Metrics Comparison Visualizer")
        mc1, mc2 = st.columns(2)

        with mc1:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            sns.barplot(data=df_comparison, x="Algorithm", y="Accuracy (%)", palette=["#3b82f6", "#10b981", "#8b5cf6"], ax=ax)
            ax.set_ylabel("Accuracy (%)", color="white")
            ax.set_xlabel("", color="white")
            ax.tick_params(colors="white")
            ax.set_ylim(0, 100)
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() - 12),
                            ha='center', va='center', color='white', fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

        with mc2:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            sns.barplot(data=df_comparison, x="Algorithm", y="Log Loss", palette=["#f59e0b", "#ef4444", "#ec4899"], ax=ax)
            ax.set_ylabel("Log Loss (Lower is Better)", color="white")
            ax.set_xlabel("", color="white")
            ax.tick_params(colors="white")
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                            ha='center', va='center', color='white', fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

        # Confusion Matrix Heatmap
        st.markdown("### 🎯 Confusion Matrix")
        selected_model_cm = st.selectbox("Select Model to Inspect Confusion Matrix", list(metrics["models"].keys()))
        cm_data = np.array(metrics["models"][selected_model_cm]["confusion_matrix"])

        labels = ["Home Win (1)", "Draw (X)", "Away Win (2)"]
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        sns.heatmap(
            cm_data,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=False
        )
        ax.set_xlabel("Predicted Outcome", color="white", fontweight='bold')
        ax.set_ylabel("Actual Outcome", color="white", fontweight='bold')
        ax.tick_params(colors="white")
        st.pyplot(fig)
        plt.close(fig)

        # Top Feature Importances
        st.markdown("### 🌟 Feature Importance Ranking")
        feat_imp = metrics["models"][selected_model_cm].get("feature_importances", {})
        if feat_imp:
            top_features = list(feat_imp.items())[:10]
            f_names = [f[0].replace("_", " ").title() for f in top_features]
            f_vals = [f[1] for f in top_features]

            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            y_pos = np.arange(len(f_names))
            ax.barh(y_pos, f_vals, color="#38bdf8", align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(f_names, color="white")
            ax.invert_yaxis()
            ax.set_xlabel("Relative Importance Weight", color="white")
            ax.tick_params(colors="white")
            st.pyplot(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# PAGE 5: PREDICTION HISTORY
# ---------------------------------------------------------------------------
elif menu_selection == "📜 Prediction History":
    st.title("📜 Prediction History & Result Verification")
    st.markdown("Audit previously generated predictions stored in SQLite and compare predicted outcomes against actual match results.")

    preds_df = get_predictions_history()

    if preds_df.empty:
        st.info("No predictions recorded yet. Use the **Match Predictor** to generate predictions.")
    else:
        # Verification Summary
        verified_count = preds_df['is_correct'].notnull().sum()
        correct_count = (preds_df['is_correct'] == 1).sum()
        accuracy_rate = (correct_count / verified_count * 100) if verified_count > 0 else 0.0

        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            st.metric("Total Saved Predictions", len(preds_df))
        with hc2:
            st.metric("Verified Predictions", verified_count)
        with hc3:
            st.metric("Historical Accuracy", f"{accuracy_rate:.1f}%", f"{correct_count} Correct")

        st.markdown("---")

        # Result verification tool
        st.subheader("Verify / Update Actual Match Result")
        col_sel, col_act, col_save = st.columns([3, 2, 2])

        with col_sel:
            pred_options = {
                f"ID {row['id']}: {row['home_team']} vs {row['away_team']} ({row['match_date']}) - Pick: {row['predicted_outcome']}": row['id']
                for _, row in preds_df.iterrows()
            }
            selected_pred_label = st.selectbox("Select Prediction to Verify", list(pred_options.keys()))
            selected_pred_id = pred_options[selected_pred_label]

        with col_act:
            actual_pick = st.selectbox("Actual Match Result", ["1 (Home Win)", "X (Draw)", "2 (Away Win)"])
            actual_code = actual_pick.split()[0]

        with col_save:
            st.write("&nbsp;")
            if st.button("Submit Result Verification", type="primary", use_container_width=True):
                success = update_prediction_actual(selected_pred_id, actual_code)
                if success:
                    st.success("Prediction verified and updated in database!")
                    st.rerun()

        st.markdown("### All Stored Predictions")
        # Filter controls
        f_league, f_outcome = st.columns(2)
        with f_league:
            all_leagues = ["ALL"] + list(preds_df['league_name'].dropna().unique())
            sel_l_filter = st.selectbox("Filter by League", all_leagues)
        with f_outcome:
            sel_o_filter = st.selectbox("Filter by Pick", ["ALL", "1", "X", "2"])

        filtered_df = preds_df.copy()
        if sel_l_filter != "ALL":
            filtered_df = filtered_df[filtered_df['league_name'] == sel_l_filter]
        if sel_o_filter != "ALL":
            filtered_df = filtered_df[filtered_df['predicted_outcome'] == sel_o_filter]

        display_cols = [
            'id', 'created_at', 'match_date', 'league_name', 'home_team', 'away_team',
            'predicted_outcome', 'prob_home', 'prob_draw', 'prob_away', 'confidence',
            'model_name', 'actual_outcome', 'is_correct'
        ]
        sub_df = filtered_df[display_cols].copy()
        sub_df.columns = [
            'ID', 'Created', 'Match Date', 'League', 'Home Team', 'Away Team',
            'Pick', 'Home %', 'Draw %', 'Away %', 'Conf %', 'Model', 'Actual', 'Correct'
        ]
        st.dataframe(sub_df, use_container_width=True, hide_index=True)

        # CSV Download Button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Prediction History (CSV)",
            data=csv_data,
            file_name=f"football_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


# ---------------------------------------------------------------------------
# PAGE 6: SETTINGS & MODEL RETRAINING
# ---------------------------------------------------------------------------
elif menu_selection == "⚙️ Data & Model Settings":
    st.title("⚙️ Data Sync & Model Retraining")
    st.markdown("Manage historical football datasets, resynchronize from football-data.co.uk, or retrain classifiers on demand.")

    stats = get_dashboard_stats()
    st.info(f"Database location: `{DB_FILE}` | Current Match Count: **{stats['total_matches']:,}** matches.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Synchronization")
        st.write("Download latest historical match records from football-data.co.uk across all 5 major European leagues.")
        if st.button("🔄 Resync Historical Data"):
            with st.spinner("Downloading and processing league datasets..."):
                count = load_and_sync_all_data(force_refresh=True)
                st.success(f"Synchronization complete! {count:,} matches stored in SQLite database.")
                st.rerun()

    with col2:
        st.subheader("Model Retraining")
        st.write("Re-engineer chronological features and retrain Logistic Regression, Random Forest, and Gradient Boosting.")
        if st.button("🚀 Retrain All Models"):
            with st.spinner("Training models with chronological train/test split..."):
                summary = train_and_evaluate_all_models()
                st.success(f"Models successfully trained! Champion model: {summary['best_model']}")
                st.rerun()
