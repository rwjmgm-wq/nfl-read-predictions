"""
Deep dive into WHY the model predicts what it does for showcase plays.
Uses SHAP values and raw features to explain the probability trajectories.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import shap
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
ORGANIZED_DIR = BASE_DIR / "organized_plays"


def load_model_and_explainer():
    """Load model and create SHAP explainer."""
    model = lgb.Booster(model_file=str(MODEL_DIR / "completion_model.lgb"))
    with open(MODEL_DIR / "feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)

    explainer = shap.TreeExplainer(model)
    return model, feature_names, explainer


def get_play_features_and_probs(play_path, model, feature_names):
    """Get frame-by-frame features and probabilities."""
    from frame_features import calculate_frame_features

    features = calculate_frame_features(play_path)
    if features is None or len(features) == 0:
        return None, None

    # Keep raw features for analysis
    X = features[[col for col in feature_names if col in features.columns]].copy()
    X = X.fillna(X.median())

    # Get probabilities
    probs = model.predict(X)

    probs_df = pd.DataFrame({
        'frame': range(len(probs)),
        'p_incomplete': probs[:, 0],
        'p_complete': probs[:, 1],
        'p_interception': probs[:, 2]
    })

    return X, probs_df


def explain_play(play_path, play_name, model, feature_names, explainer):
    """Generate detailed explanation for a play's probability trajectory."""

    print(f"\n{'='*80}")
    print(f"EXPLAINING: {play_name}")
    print(f"Path: {play_path}")
    print(f"{'='*80}")

    X, probs_df = get_play_features_and_probs(play_path, model, feature_names)

    if X is None:
        print("ERROR: Could not load play features")
        return

    n_frames = len(X)
    print(f"\nTotal frames: {n_frames}")

    # Get SHAP values for all frames
    shap_values = explainer.shap_values(X)
    # shap_values is a list of 3 arrays (one per class)
    # Each array is (n_frames, n_features)

    # Class indices: 0=incomplete, 1=complete, 2=interception
    class_names = ['Incomplete', 'Complete', 'Interception']

    print("\n" + "-"*60)
    print("FRAME-BY-FRAME PROBABILITIES")
    print("-"*60)
    print(probs_df.round(3).to_string(index=False))

    # Analyze key frames
    key_frames = [0, n_frames//2, n_frames-1]  # Start, middle, end
    if n_frames > 2:
        key_frames = [0, n_frames//2, n_frames-1]
    else:
        key_frames = list(range(n_frames))

    for frame_idx in key_frames:
        print(f"\n{'='*60}")
        print(f"FRAME {frame_idx} ANALYSIS")
        print(f"{'='*60}")

        probs = probs_df.iloc[frame_idx]
        print(f"\nProbabilities at frame {frame_idx}:")
        print(f"  P(Incomplete):   {probs['p_incomplete']:.1%}")
        print(f"  P(Complete):     {probs['p_complete']:.1%}")
        print(f"  P(Interception): {probs['p_interception']:.1%}")

        # Get feature values at this frame
        features_at_frame = X.iloc[frame_idx]

        # Key features to highlight
        key_features = [
            'receiver_to_catch_point',
            'defender_to_catch_point',
            'separation',
            'second_defender_dist',
            'receiver_speed',
            'defender_speed',
            'ball_to_catch_point',
            'receiver_defender_speed_diff',
            'closing_speed',
            'time_to_catch',
            'receiver_will_arrive_first',
            'defender_can_arrive'
        ]

        print(f"\nKey features at frame {frame_idx}:")
        for feat in key_features:
            if feat in features_at_frame.index:
                val = features_at_frame[feat]
                print(f"  {feat}: {val:.3f}" if isinstance(val, float) else f"  {feat}: {val}")

        # Top SHAP contributors for each class
        print(f"\nTop SHAP drivers at frame {frame_idx}:")

        for class_idx, class_name in enumerate(class_names):
            shap_frame = shap_values[class_idx][frame_idx]
            feature_importance = list(zip(X.columns, shap_frame))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            print(f"\n  {class_name}:")
            for feat, shap_val in feature_importance[:5]:
                direction = "+" if shap_val > 0 else "-"
                feat_val = features_at_frame[feat] if feat in features_at_frame.index else "N/A"
                print(f"    {direction} {feat}: SHAP={shap_val:+.3f} (value={feat_val:.3f})" if isinstance(feat_val, float) else f"    {direction} {feat}: SHAP={shap_val:+.3f} (value={feat_val})")

    # Explain probability changes
    if n_frames > 1:
        print(f"\n{'='*60}")
        print("PROBABILITY CHANGE ANALYSIS")
        print(f"{'='*60}")

        for class_idx, class_name in enumerate(class_names):
            col = f'p_{class_name.lower()}'
            start_prob = probs_df[col].iloc[0]
            end_prob = probs_df[col].iloc[-1]
            change = end_prob - start_prob

            if abs(change) > 0.05:  # Significant change
                print(f"\n{class_name}: {start_prob:.1%} -> {end_prob:.1%} ({change:+.1%})")

                # What features changed most?
                shap_start = shap_values[class_idx][0]
                shap_end = shap_values[class_idx][-1]
                shap_change = shap_end - shap_start

                feature_changes = list(zip(X.columns, shap_change))
                feature_changes.sort(key=lambda x: abs(x[1]), reverse=True)

                print(f"  Top drivers of this change:")
                for feat, shap_delta in feature_changes[:5]:
                    if abs(shap_delta) > 0.01:
                        feat_start = X[feat].iloc[0]
                        feat_end = X[feat].iloc[-1]
                        direction = "+" if shap_delta > 0 else "-"
                        print(f"    {direction} {feat}: SHAP change={shap_delta:+.3f}")
                        print(f"       Feature: {feat_start:.2f} -> {feat_end:.2f}")


def main():
    """Analyze the specific showcase plays."""

    model, feature_names, explainer = load_model_and_explainer()

    # The plays to analyze
    plays_to_explain = [
        {
            'name': '1. High Drama Opener - INTERCEPTION (GB vs CHI, Q4 12:53)',
            'path': ORGANIZED_DIR / 'Week_01' / 'GB_vs_CHI' / '038_Q4_1253_CHI'
        },
        {
            'name': '2. Dramatic Completion - COMPLETE (GB vs CHI, Q4 4:57)',
            'path': ORGANIZED_DIR / 'Week_01' / 'GB_vs_CHI' / '043_Q4_457_CHI'
        },
        {
            'name': '4. Late Suspense - COMPLETE (TB vs CAR, Q2 3:18)',
            'path': ORGANIZED_DIR / 'Week_18' / 'TB_vs_CAR' / '021_Q2_318_CAR'
        }
    ]

    for play_info in plays_to_explain:
        explain_play(play_info['path'], play_info['name'], model, feature_names, explainer)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
