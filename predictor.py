"""
Prediction Engine Module.
Generates match outcome predictions (1, X, 2), calibrated probabilities summing to 100%,
confidence scoring, and human-readable analytical explanations.
"""

from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from database import get_all_matches, insert_prediction, DB_FILE
from feature_engineering import (
    extract_features_for_new_match,
    FEATURE_COLUMNS,
    TARGET_INV_MAP,
    TARGET_LABEL_MAP
)
from model_trainer import load_model_and_scaler, get_stored_metrics


def generate_factor_explanations(
    home_team: str,
    away_team: str,
    features: Dict[str, float],
    context: Dict[str, Any],
    pred_outcome: str
) -> Dict[str, Any]:
    """
    Creates structured and human-readable insights explaining why
    the model favored the predicted outcome.
    """
    h_stats = context['h_stats']
    a_stats = context['a_stats']
    h2h = context['h2h_stats']
    h_venue_pct = context['home_venue_win_pct'] * 100
    a_venue_pct = context['away_venue_win_pct'] * 100

    insights = []
    influencing_factors = []

    # 1. Recent Form Analysis
    h_pts = round(features['home_last5_pts'], 1)
    a_pts = round(features['away_last5_pts'], 1)
    pts_diff = round(features['pts_diff_last5'], 1)
    h_streak = "".join(context.get('home_recent_streak', []))
    a_streak = "".join(context.get('away_recent_streak', []))

    if pts_diff >= 3:
        influencing_factors.append({
            "factor": "Recent Form Momentum",
            "advantage": home_team,
            "weight": "High",
            "detail": f"{home_team} took {h_pts}/15 points in recent matches vs {a_pts}/15 for {away_team}."
        })
        insights.append(
            f"**Form Momentum:** {home_team} enters this fixture in superior form ({h_pts} points in their last 5, streak: {h_streak or 'N/A'}), "
            f"compared to {away_team}'s {a_pts} points (streak: {a_streak or 'N/A'})."
        )
    elif pts_diff <= -3:
        influencing_factors.append({
            "factor": "Recent Form Momentum",
            "advantage": away_team,
            "weight": "High",
            "detail": f"{away_team} gathered {a_pts}/15 points recently vs {h_pts}/15 for {home_team}."
        })
        insights.append(
            f"**Form Momentum:** {away_team} has been sharper recently, collecting {a_pts} points from their last 5 matches "
            f"compared to {home_team}'s {h_pts} points."
        )
    else:
        influencing_factors.append({
            "factor": "Recent Form Momentum",
            "advantage": "Balanced",
            "weight": "Moderate",
            "detail": f"Both teams show comparable recent form ({h_pts} vs {a_pts} points)."
        })
        insights.append(
            f"**Form Parity:** Both clubs show balanced recent form ({home_team}: {h_pts} pts, {away_team}: {a_pts} pts)."
        )

    # 2. Attacking & Defensive Firepower
    h_gf = round(features['home_last5_gf'], 2)
    h_ga = round(features['home_last5_ga'], 2)
    a_gf = round(features['away_last5_gf'], 2)
    a_ga = round(features['away_last5_ga'], 2)

    insights.append(
        f"**Goal Statistics:** {home_team} averages {h_gf} goals scored and {h_ga} conceded per game recently. "
        f"{away_team} averages {a_gf} goals scored and {a_ga} conceded."
    )

    if h_gf - h_ga > a_gf - a_ga + 0.5:
        influencing_factors.append({
            "factor": "Net Goal Efficiency",
            "advantage": home_team,
            "weight": "Moderate",
            "detail": f"{home_team} has a positive goal difference margin (+{round(h_gf-h_ga, 2)}) vs {away_team} ({round(a_gf-a_ga, 2)})."
        })
    elif a_gf - a_ga > h_gf - h_ga + 0.5:
        influencing_factors.append({
            "factor": "Net Goal Efficiency",
            "advantage": away_team,
            "weight": "Moderate",
            "detail": f"{away_team} holds superior goal differential margin (+{round(a_gf-a_ga, 2)}) vs {home_team} ({round(h_gf-h_ga, 2)})."
        })

    # 3. Venue & Home Advantage
    influencing_factors.append({
        "factor": "Home Ground Advantage",
        "advantage": home_team,
        "weight": "High" if h_venue_pct > 55 else "Moderate",
        "detail": f"{home_team} has a {round(h_venue_pct, 1)}% win rate at home. {away_team} wins {round(a_venue_pct, 1)}% of away fixtures."
    })
    insights.append(
        f"**Venue Dynamics:** Playing at home provides a historical boost. {home_team} records a {round(h_venue_pct, 1)}% "
        f"win rate on home turf, while {away_team} has won {round(a_venue_pct, 1)}% of away matches."
    )

    # 4. Head-to-Head History
    if h2h['count'] > 0:
        h_h2h_win = round(h2h['home_win_pct'] * 100, 1)
        a_h2h_win = round(h2h['away_win_pct'] * 100, 1)
        draw_h2h = round(h2h['draw_pct'] * 100, 1)
        influencing_factors.append({
            "factor": "Head-to-Head Record",
            "advantage": home_team if h_h2h_win > a_h2h_win else (away_team if a_h2h_win > h_h2h_win else "Even"),
            "weight": "Moderate",
            "detail": f"Over {int(h2h['count'])} past meetings: {home_team} {h_h2h_win}%, Draws {draw_h2h}%, {away_team} {a_h2h_win}%."
        })
        insights.append(
            f"**Head-to-Head:** Across {int(h2h['count'])} previous encounters, {home_team} has won {h_h2h_win}%, "
            f"{draw_h2h}% ended in draws, and {away_team} has claimed {a_h2h_win}%."
        )
    else:
        insights.append(
            "**Head-to-Head:** No direct previous meetings recorded in recent database records; prediction relies primarily on recent league form and venue dynamics."
        )

    # 5. Rest / Schedule Fatigue
    h_rest = int(features['home_rest_days'])
    a_rest = int(features['away_rest_days'])
    if abs(h_rest - a_rest) >= 4:
        fresher = home_team if h_rest > a_rest else away_team
        influencing_factors.append({
            "factor": "Rest & Schedule",
            "advantage": fresher,
            "weight": "Low",
            "detail": f"{fresher} had {max(h_rest, a_rest)} rest days compared to {min(h_rest, a_rest)} for the opponent."
        })
        insights.append(f"**Physical Rest:** {fresher} had {max(h_rest, a_rest)} days of recovery compared to {min(h_rest, a_rest)} days.")

    return {
        "summary": " ".join(insights[:3]),
        "bullet_points": insights,
        "influencing_factors": influencing_factors
    }


def predict_match(
    home_team: str,
    away_team: str,
    league_code: str,
    match_date: Optional[str] = None,
    model_name: Optional[str] = None,
    save_to_history: bool = True,
    db_path: str = DB_FILE
) -> Dict[str, Any]:
    """
    Main prediction pipeline:
    Extracts features chronologically, feeds into trained classifier,
    calibrates probabilities to exact 100%, calculates confidence,
    generates contextual sports analytics explanations, and saves to SQLite.
    """
    if home_team == away_team:
        raise ValueError("Home team and Away team cannot be identical.")

    if not match_date:
        match_date = datetime.now().strftime("%Y-%m-%d")

    # Load historical matches
    matches_df = get_all_matches(db_path)
    if matches_df.empty:
        raise ValueError("No historical match data found in database. Please initialize dataset first.")

    # Extract leak-free features strictly before match_date
    features_dict, context = extract_features_for_new_match(
        home_team=home_team,
        away_team=away_team,
        match_date_str=match_date,
        historical_matches_df=matches_df
    )

    # Load model and scaler
    model, scaler = load_model_and_scaler(model_name)
    if model is None or scaler is None:
        raise RuntimeError("No trained model available. Please train models first.")

    # Model name detection
    actual_model_name = getattr(model, '__class__', type(model)).__name__
    if hasattr(model, 'classes_'):
        # Map class names nicely
        if 'Logistic' in actual_model_name:
            friendly_name = "Logistic Regression"
        elif 'Forest' in actual_model_name:
            friendly_name = "Random Forest"
        elif 'Gradient' in actual_model_name:
            friendly_name = "Gradient Boosting"
        else:
            friendly_name = actual_model_name
    else:
        friendly_name = "Ensemble Predictor"

    # Format vector into dataframe matching training feature order
    feature_row = [features_dict[col] for col in FEATURE_COLUMNS]
    X_input = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)

    # Scale
    X_scaled = scaler.transform(X_input)

    # Model inference
    raw_probs = model.predict_proba(X_scaled)[0]

    # Strictly calibrate / normalize probabilities to sum to exactly 100%
    total_prob = float(np.sum(raw_probs))
    if total_prob > 0:
        norm_probs = raw_probs / total_prob
    else:
        norm_probs = np.array([0.3333, 0.3333, 0.3334])

    p_home = float(norm_probs[0])
    p_draw = float(norm_probs[1])
    p_away = float(norm_probs[2])

    # Round nicely while guaranteeing exact 100.0% sum
    p_home_pct = round(p_home * 100.0, 1)
    p_draw_pct = round(p_draw * 100.0, 1)
    p_away_pct = round(100.0 - (p_home_pct + p_draw_pct), 1)

    # Outcome determination
    if p_home >= p_draw and p_home >= p_away:
        predicted_code = "1"
        predicted_label = "Home Win (1)"
        max_p = p_home
    elif p_away >= p_home and p_away >= p_draw:
        predicted_code = "2"
        predicted_label = "Away Win (2)"
        max_p = p_away
    else:
        predicted_code = "X"
        predicted_label = "Draw (X)"
        max_p = p_draw

    # Confidence calculation: how significantly top probability exceeds uniform 33.3%
    # Scaled to realistic 40% - 90% range for sports betting / analytics
    excess = max(0.0, max_p - 0.3333)
    confidence = min(92.0, max(42.0, (excess / 0.6667) * 55.0 + 40.0 + (max_p * 15.0)))
    confidence = round(confidence, 1)

    # Confidence qualitative tier
    if confidence >= 70.0:
        confidence_tier = "High"
    elif confidence >= 55.0:
        confidence_tier = "Moderate"
    else:
        confidence_tier = "Competitive / Balanced"

    # Explanations
    explanation = generate_factor_explanations(
        home_team=home_team,
        away_team=away_team,
        features=features_dict,
        context=context,
        pred_outcome=predicted_code
    )

    pred_result = {
        "home_team": home_team,
        "away_team": away_team,
        "league_code": league_code,
        "match_date": match_date,
        "predicted_outcome": predicted_code,
        "predicted_label": predicted_label,
        "prob_home": p_home_pct,
        "prob_draw": p_draw_pct,
        "prob_away": p_away_pct,
        "confidence": confidence,
        "confidence_tier": confidence_tier,
        "model_name": friendly_name,
        "explanation": explanation,
        "features": features_dict,
        "disclaimer": "Predictions are probabilistic estimates generated via machine learning and do not guarantee actual sporting outcomes."
    }

    # Save to SQLite database
    if save_to_history:
        record_id = insert_prediction(
            league_code=league_code,
            home_team=home_team,
            away_team=away_team,
            predicted_outcome=predicted_code,
            prob_home=p_home_pct,
            prob_draw=p_draw_pct,
            prob_away=p_away_pct,
            confidence=confidence,
            model_name=friendly_name,
            factors=explanation,
            match_date=match_date,
            db_path=db_path
        )
        pred_result["prediction_id"] = record_id

    return pred_result
