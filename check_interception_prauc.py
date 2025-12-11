"""
Check PR-AUC for interception predictions to verify model isn't just guessing.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"


def check_interception_prauc(y_test, y_probs, interception_label_index=2):
    """
    y_test: Actual labels (0=Incomplete, 1=Complete, 2=Interception)
    y_probs: Model probability output matrix (n_samples, 3)
    interception_label_index: Which column in y_probs corresponds to INTs? (Usually 2)
    """

    # 1. Create a binary target just for Interceptions
    y_test_binary = (y_test == interception_label_index).astype(int)

    # 2. Extract the probability specifically for Interceptions
    y_probs_int = y_probs[:, interception_label_index]

    # 3. Calculate PR-AUC (Average Precision)
    pr_auc = average_precision_score(y_test_binary, y_probs_int)

    print(f"Interception PR-AUC: {pr_auc:.4f}")

    # 4. Calculate Baseline (No-Skill)
    baseline = y_test_binary.mean()
    print(f"Baseline (Random Guessing): {baseline:.4f}")
    print(f"Model Lift: {pr_auc / baseline:.1f}x better than random")

    # 5. Plot the Curve
    precision, recall, thresholds = precision_recall_curve(y_test_binary, y_probs_int)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.', markersize=2, label=f'Model (PR-AUC={pr_auc:.3f})')
    plt.axhline(y=baseline, linestyle='--', color='red', label=f'Baseline (No Skill) = {baseline:.3f}')
    plt.xlabel('Recall (How many INTs did we catch?)')
    plt.ylabel('Precision (When we predict INT, are we right?)')
    plt.title('Interception Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Save the plot
    plt.savefig(BASE_DIR / "interception_pr_curve.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {BASE_DIR / 'interception_pr_curve.png'}")
    plt.show()

    # Additional analysis: what are the model's highest INT probability predictions?
    print("\n" + "="*60)
    print("HIGHEST INTERCEPTION PROBABILITY PREDICTIONS")
    print("="*60)

    # Get top 20 predictions by INT probability
    top_indices = np.argsort(y_probs_int)[-20:][::-1]

    print(f"\nTop 20 frames by P(Interception):")
    print(f"{'P(INT)':>8} | {'Actual':>12} | {'P(Comp)':>8} | {'P(Inc)':>8}")
    print("-" * 45)

    label_names = {0: 'Incomplete', 1: 'Complete', 2: 'INTERCEPTION'}
    correct_count = 0

    for idx in top_indices:
        actual = int(y_test.iloc[idx]) if hasattr(y_test, 'iloc') else int(y_test[idx])
        is_correct = actual == 2
        if is_correct:
            correct_count += 1
        marker = " <-- CORRECT" if is_correct else ""
        print(f"{y_probs_int[idx]:>8.3f} | {label_names[actual]:>12} | {y_probs[idx, 1]:>8.3f} | {y_probs[idx, 0]:>8.3f}{marker}")

    print(f"\nPrecision @ top 20: {correct_count}/20 = {correct_count/20:.1%}")

    return pr_auc, baseline


def main():
    # Load the features
    print("Loading features...")
    df = pd.read_csv(BASE_DIR / "all_frame_features.csv")

    # Load the model
    print("Loading model...")
    model = lgb.Booster(model_file=str(MODEL_DIR / "completion_model.lgb"))

    with open(MODEL_DIR / "feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)

    # Prepare data (same as training)
    exclude_cols = ['frame_id', 'week', 'game', 'play_folder', 'play_path',
                    'outcome', 'game_id', 'ball_catchable', 'contested']

    X = df[[col for col in df.columns if col not in exclude_cols]].copy()
    X = X.fillna(X.median())

    # Ensure correct column order
    X = X[feature_names]

    outcome_map = {'I': 0, 'C': 1, 'IN': 2}
    y = df['outcome'].map(outcome_map)

    # Use game-based split (same as training)
    from sklearn.model_selection import GroupKFold
    groups = df['game_id']
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups))
    train_idx, test_idx = splits[0]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print(f"Test set size: {len(X_test)} frames")
    print(f"Interceptions in test set: {(y_test == 2).sum()}")

    # Get predictions
    y_probs = model.predict(X_test)

    print("\n" + "="*60)
    print("INTERCEPTION PR-AUC ANALYSIS")
    print("="*60 + "\n")

    pr_auc, baseline = check_interception_prauc(y_test, y_probs)

    # Also check Complete and Incomplete for comparison
    print("\n" + "="*60)
    print("COMPARISON: ALL OUTCOME PR-AUCs")
    print("="*60)

    for label_idx, label_name in [(0, 'Incomplete'), (1, 'Complete'), (2, 'Interception')]:
        y_binary = (y_test == label_idx).astype(int)
        pr_auc = average_precision_score(y_binary, y_probs[:, label_idx])
        baseline = y_binary.mean()
        lift = pr_auc / baseline
        print(f"{label_name:>12}: PR-AUC = {pr_auc:.4f}, Baseline = {baseline:.4f}, Lift = {lift:.1f}x")


if __name__ == "__main__":
    main()
