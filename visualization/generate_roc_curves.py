"""
Generate ROC (Receiver Operating Characteristic) curves for all three outcomes.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

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

    # Get predictions
    print("Generating predictions...")
    y_probs = model.predict(X_test)

    # Colors for each outcome
    colors = {
        'Complete': '#00CED1',      # Cyan
        'Incomplete': '#FF4444',    # Red
        'Interception': '#FF00FF'   # Magenta
    }

    outcome_names = ['Incomplete', 'Complete', 'Interception']

    # =============================================================================
    # INDIVIDUAL ROC CURVES (3 subplots)
    # =============================================================================

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (label_idx, label_name) in enumerate([(0, 'Incomplete'), (1, 'Complete'), (2, 'Interception')]):
        ax = axes[idx]

        # Create binary labels
        y_binary = (y_test == label_idx).astype(int)
        y_pred_probs = y_probs[:, label_idx]

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_binary, y_pred_probs)
        roc_auc = roc_auc_score(y_binary, y_pred_probs)

        # Plot ROC curve
        ax.plot(fpr, tpr, color=colors[label_name], linewidth=3,
                label=f'{label_name} (AUC = {roc_auc:.4f})')

        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2,
                label='Random Classifier (AUC = 0.50)')

        # Fill area under curve
        ax.fill_between(fpr, tpr, alpha=0.2, color=colors[label_name])

        # Formatting
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'{label_name}\nROC-AUC = {roc_auc:.4f}',
                     fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.suptitle('ROC Curves by Outcome (One-vs-Rest)', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "roc_curves_individual.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {BASE_DIR / 'roc_curves_individual.png'}")
    plt.close()

    # =============================================================================
    # COMBINED ROC CURVES (all on one plot)
    # =============================================================================

    fig, ax = plt.subplots(figsize=(10, 8))

    for label_idx, label_name in [(0, 'Incomplete'), (1, 'Complete'), (2, 'Interception')]:
        # Create binary labels
        y_binary = (y_test == label_idx).astype(int)
        y_pred_probs = y_probs[:, label_idx]

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_binary, y_pred_probs)
        roc_auc = roc_auc_score(y_binary, y_pred_probs)

        # Plot ROC curve
        ax.plot(fpr, tpr, color=colors[label_name], linewidth=3,
                label=f'{label_name} (AUC = {roc_auc:.4f})')

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2,
            label='Random Classifier (AUC = 0.50)')

    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves: All Outcomes (One-vs-Rest)\nDashed line shows random classifier baseline',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(BASE_DIR / "roc_curves_combined.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {BASE_DIR / 'roc_curves_combined.png'}")
    plt.close()

    print("\nDone! Generated ROC curves.")

if __name__ == "__main__":
    main()
