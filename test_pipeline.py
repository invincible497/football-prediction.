"""
Comprehensive pipeline integration and verification test for Football Match Prediction System.
"""

import sys
import os
import json

from database import (
    init_db, get_dashboard_stats, get_all_matches,
    get_predictions_history, update_prediction_actual, DB_FILE
)
from data_loader import load_and_sync_all_data
from model_trainer import (
    train_and_evaluate_all_models, get_stored_metrics,
    BEST_MODEL_FILE, SCALER_FILE
)
from predictor import predict_match


def run_full_pipeline_test():
    print("==================================================")
    print("STEP 1: Initializing Database & Loading Matches")
    print("==================================================")
    init_db()
    count = load_and_sync_all_data()
    print(f"Total historical matches synchronized: {count}")
    assert count > 500, f"Expected >500 matches, got {count}"

    stats = get_dashboard_stats()
    print(f"Stats: Total Matches={stats['total_matches']}, Teams={stats['total_teams']}, Leagues={stats['total_leagues']}")
    assert stats['total_matches'] > 0
    assert stats['total_teams'] > 0

    print("\n==================================================")
    print("STEP 2: Chronological Feature Engineering & Training")
    print("==================================================")
    metrics_summary = train_and_evaluate_all_models()
    print(f"Champion Model selected: {metrics_summary['best_model']}")
    print(f"Training samples: {metrics_summary['train_samples']}, Test samples: {metrics_summary['test_samples']}")

    for name, m in metrics_summary['models'].items():
        print(f"\n--- {name} ---")
        print(f"  Accuracy:            {m['accuracy'] * 100:.2f}%")
        print(f"  Weighted F1:         {m['f1_weighted']:.4f}")
        print(f"  Macro F1:            {m['f1_macro']:.4f}")
        print(f"  Weighted Precision:  {m['precision_weighted']:.4f}")
        print(f"  Weighted Recall:     {m['recall_weighted']:.4f}")
        print(f"  Log Loss:            {m['log_loss']:.4f}")
        print(f"  Brier Score:         {m['brier_score']:.4f}")
        print(f"  Confusion Matrix:    {m['confusion_matrix']}")
        top_3 = list(m['feature_importances'].items())[:3]
        print(f"  Top 3 Features:      {top_3}")

    assert os.path.exists(BEST_MODEL_FILE), "Best model file was not saved!"
    assert os.path.exists(SCALER_FILE), "Scaler file was not saved!"

    print("\n==================================================")
    print("STEP 3: End-to-End Prediction Test")
    print("==================================================")
    # Test fixture 1: Arsenal vs Chelsea (Premier League)
    pred_res = predict_match(
        home_team="Arsenal",
        away_team="Chelsea",
        league_code="E0",
        match_date="2024-05-01",
        save_to_history=True
    )
    print("Match Prediction Test (Arsenal vs Chelsea):")
    print(f"  Predicted Outcome: {pred_res['predicted_label']} (Code: {pred_res['predicted_outcome']})")
    print(f"  Probabilities: Home={pred_res['prob_home']}%, Draw={pred_res['prob_draw']}%, Away={pred_res['prob_away']}%")
    prob_sum = round(pred_res['prob_home'] + pred_res['prob_draw'] + pred_res['prob_away'], 1)
    print(f"  Sum of Probabilities: {prob_sum}%")
    assert abs(prob_sum - 100.0) < 0.1, f"Probabilities do not sum to 100%! Got {prob_sum}"
    print(f"  Confidence: {pred_res['confidence']}% ({pred_res['confidence_tier']})")
    print(f"  Model Used: {pred_res['model_name']}")
    print(f"  Explanation Factor Count: {len(pred_res['explanation']['influencing_factors'])}")
    print(f"  Explanation Summary: {pred_res['explanation']['summary']}")

    # Test fixture 2: Real Madrid vs Barcelona (La Liga)
    pred_res2 = predict_match(
        home_team="Real Madrid",
        away_team="Barcelona",
        league_code="SP1",
        match_date="2024-04-21",
        save_to_history=True
    )
    print("\nMatch Prediction Test (Real Madrid vs Barcelona):")
    print(f"  Predicted Outcome: {pred_res2['predicted_label']} (Code: {pred_res2['predicted_outcome']})")
    print(f"  Probabilities: Home={pred_res2['prob_home']}%, Draw={pred_res2['prob_draw']}%, Away={pred_res2['prob_away']}%")

    print("\n==================================================")
    print("STEP 4: Database History & Outcome Verification")
    print("==================================================")
    history_df = get_predictions_history()
    print(f"Stored predictions in history: {len(history_df)}")
    assert len(history_df) >= 2, "Predictions were not saved to history!"

    latest_id = int(history_df.iloc[0]['id'])
    print(f"Verifying latest prediction ID {latest_id} with actual outcome '1'...")
    update_prediction_actual(latest_id, "1")

    updated_history = get_predictions_history()
    verified_row = updated_history[updated_history['id'] == latest_id].iloc[0]
    print(f"Verified row actual outcome: {verified_row['actual_outcome']}, is_correct: {verified_row['is_correct']}")
    assert verified_row['actual_outcome'] == '1'
    assert verified_row['is_correct'] in (0, 1)

    print("\n==================================================")
    print("ALL TESTS PASSED WITH ZERO ERRORS!")
    print("==================================================")


if __name__ == "__main__":
    run_full_pipeline_test()
