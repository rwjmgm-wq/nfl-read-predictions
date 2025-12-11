"""
Train completion probability model using frame-by-frame features.

Predicts P(Completion), P(Incompletion), P(Interception) at each frame during ball flight.
Uses gradient boosting (LightGBM) with game-based train/test split to avoid leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

from frame_features import calculate_frame_features

# Paths
BASE_DIR = Path(__file__).parent
ORGANIZED_DIR = BASE_DIR / "organized_plays"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def get_play_outcome(play_folder):
    """Get pass result for a play."""
    supp_file = play_folder / "supplementary.csv"
    if not supp_file.exists():
        return None
    supp = pd.read_csv(supp_file)
    if len(supp) == 0:
        return None
    return supp['pass_result'].iloc[0]


def extract_all_features(max_plays=None, save_intermediate=True):
    """
    Extract frame-by-frame features for all plays.

    Returns:
        DataFrame with all frame features plus play metadata
    """
    all_features = []
    play_count = 0
    failed_count = 0

    # Get all play folders
    play_folders = []
    for week_folder in sorted(ORGANIZED_DIR.iterdir()):
        if not week_folder.is_dir() or not week_folder.name.startswith("Week"):
            continue
        for game_folder in sorted(week_folder.iterdir()):
            if not game_folder.is_dir():
                continue
            for play_folder in sorted(game_folder.iterdir()):
                if not play_folder.is_dir():
                    continue
                play_folders.append((week_folder.name, game_folder.name, play_folder))

    print(f"Found {len(play_folders)} plays to process")

    if max_plays:
        play_folders = play_folders[:max_plays]

    for week_name, game_name, play_folder in tqdm(play_folders, desc="Extracting features"):
        # Get outcome
        outcome = get_play_outcome(play_folder)
        if outcome not in ['C', 'I', 'IN']:
            failed_count += 1
            continue

        # Calculate features
        features = calculate_frame_features(play_folder)
        if features is None or len(features) == 0:
            failed_count += 1
            continue

        # Add metadata
        features['week'] = week_name
        features['game'] = game_name
        features['play_folder'] = play_folder.name
        features['play_path'] = str(play_folder)
        features['outcome'] = outcome

        # Create game_id for grouping (used for train/test split)
        features['game_id'] = f"{week_name}_{game_name}"

        all_features.append(features)
        play_count += 1

    print(f"\nProcessed {play_count} plays successfully, {failed_count} failed")

    # Combine all
    df = pd.concat(all_features, ignore_index=True)

    if save_intermediate:
        df.to_csv(BASE_DIR / "all_frame_features.csv", index=False)
        print(f"Saved features to all_frame_features.csv ({len(df)} rows)")

    return df


def prepare_data(df):
    """
    Prepare data for training.

    Returns:
        X: feature matrix
        y: target labels (0=Incomplete, 1=Complete, 2=Interception)
        groups: game_id for group-based splitting
        feature_names: list of feature names
    """
    # Feature columns (exclude metadata and target)
    exclude_cols = ['frame_id', 'week', 'game', 'play_folder', 'play_path',
                    'outcome', 'game_id', 'ball_catchable', 'contested']

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Handle any remaining NaN values
    X = df[feature_cols].copy()
    X = X.fillna(X.median())

    # Create target: 0=Incomplete, 1=Complete, 2=Interception
    outcome_map = {'I': 0, 'C': 1, 'IN': 2}
    y = df['outcome'].map(outcome_map)

    groups = df['game_id']

    return X, y, groups, feature_cols


def train_model(X, y, groups, feature_names):
    """
    Train LightGBM model with game-based cross-validation.

    Returns:
        Trained model, test indices, predictions
    """
    # Use GroupKFold to split by game
    gkf = GroupKFold(n_splits=5)

    # Get one split for final train/test
    splits = list(gkf.split(X, y, groups))
    train_idx, test_idx = splits[0]  # Use first fold as test

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train size: {len(X_train)} frames from {groups.iloc[train_idx].nunique()} games")
    print(f"Test size: {len(X_test)} frames from {groups.iloc[test_idx].nunique()} games")

    # Check class distribution
    print(f"\nTrain class distribution:")
    print(y_train.value_counts().sort_index())
    print(f"\nTest class distribution:")
    print(y_test.value_counts().sort_index())

    # LightGBM parameters
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    valid_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_names, reference=train_data)

    # Train
    print("\nTraining model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    # Predictions (probabilities for each class)
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    return model, test_idx, y_pred_proba, y_test, X_test


def evaluate_model(model, y_test, y_pred_proba, feature_names):
    """Evaluate model performance."""
    y_pred = np.argmax(y_pred_proba, axis=1)

    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    # Classification report
    target_names = ['Incomplete', 'Complete', 'Interception']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=target_names, columns=target_names))

    # AUC for each class (one-vs-rest)
    print("\nAUC Scores (One-vs-Rest):")
    for i, name in enumerate(target_names):
        y_binary = (y_test == i).astype(int)
        auc = roc_auc_score(y_binary, y_pred_proba[:, i])
        print(f"  {name}: {auc:.4f}")

    # Feature importance
    print("\nTop 15 Feature Importance:")
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    for i, row in importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.1f}")

    return importance


def get_example_plays(df, model, X, feature_names, n_examples=1):
    """
    Get example plays for each outcome type showing probability evolution.

    Returns dict with example plays and their frame-by-frame probabilities.
    """
    examples = {}

    for outcome, label in [('C', 'Complete'), ('I', 'Incomplete'), ('IN', 'Interception')]:
        # Find plays with this outcome
        outcome_plays = df[df['outcome'] == outcome]['play_path'].unique()

        if len(outcome_plays) == 0:
            continue

        # Pick a representative play (one with reasonable number of frames)
        for play_path in outcome_plays[:20]:  # Check first 20
            play_frames = df[df['play_path'] == play_path]
            if 10 <= len(play_frames) <= 30:  # Good length for visualization
                break

        play_frames = df[df['play_path'] == play_path].copy()
        play_X = X.loc[play_frames.index]

        # Get probabilities for each frame
        probs = model.predict(play_X[feature_names])

        play_frames['P_incomplete'] = probs[:, 0]
        play_frames['P_complete'] = probs[:, 1]
        play_frames['P_interception'] = probs[:, 2]

        # Load supplementary for context
        supp = pd.read_csv(Path(play_path) / "supplementary.csv")

        examples[outcome] = {
            'label': label,
            'play_path': play_path,
            'play_description': supp['play_description'].iloc[0] if len(supp) > 0 else "N/A",
            'frames': play_frames[['frame_id', 'time_remaining', 'separation',
                                   'ball_to_receiver', 'P_incomplete', 'P_complete',
                                   'P_interception']].copy()
        }

    return examples


def print_example_plays(examples):
    """Print example plays with probability evolution."""
    print("\n" + "="*80)
    print("EXAMPLE PLAYS - PROBABILITY EVOLUTION")
    print("="*80)

    for outcome, data in examples.items():
        print(f"\n{'='*80}")
        print(f"OUTCOME: {data['label']} ({outcome})")
        print(f"{'='*80}")
        print(f"Play: {data['play_description'][:100]}...")
        print(f"Path: {data['play_path']}")
        print(f"\nFrame-by-frame probabilities:")

        frames = data['frames']
        pd.set_option('display.float_format', '{:.3f}'.format)
        print(frames.to_string(index=False))

        # Summary
        print(f"\nProbability evolution:")
        print(f"  P(Complete):     {frames['P_complete'].iloc[0]:.3f} -> {frames['P_complete'].iloc[-1]:.3f}")
        print(f"  P(Incomplete):   {frames['P_incomplete'].iloc[0]:.3f} -> {frames['P_incomplete'].iloc[-1]:.3f}")
        print(f"  P(Interception): {frames['P_interception'].iloc[0]:.3f} -> {frames['P_interception'].iloc[-1]:.3f}")


def main(extract_features=True, max_plays=None):
    """Main training pipeline."""

    # Step 1: Extract features (or load if already done)
    features_file = BASE_DIR / "all_frame_features.csv"

    if extract_features or not features_file.exists():
        print("="*60)
        print("STEP 1: EXTRACTING FEATURES")
        print("="*60)
        df = extract_all_features(max_plays=max_plays)
    else:
        print("Loading pre-extracted features...")
        df = pd.read_csv(features_file)
        print(f"Loaded {len(df)} rows")

    # Step 2: Prepare data
    print("\n" + "="*60)
    print("STEP 2: PREPARING DATA")
    print("="*60)
    X, y, groups, feature_names = prepare_data(df)
    print(f"Features: {len(feature_names)}")
    print(f"Samples: {len(X)}")
    print(f"Games: {groups.nunique()}")

    # Outcome distribution
    print(f"\nOutcome distribution:")
    print(df['outcome'].value_counts())

    # Step 3: Train model
    print("\n" + "="*60)
    print("STEP 3: TRAINING MODEL")
    print("="*60)
    model, test_idx, y_pred_proba, y_test, X_test = train_model(X, y, groups, feature_names)

    # Step 4: Evaluate
    importance = evaluate_model(model, y_test, y_pred_proba, feature_names)

    # Step 5: Example plays
    print("\n" + "="*60)
    print("STEP 5: EXAMPLE PLAYS")
    print("="*60)

    # Get test set plays only
    test_df = df.iloc[test_idx].copy()
    test_X = X.iloc[test_idx].copy()

    examples = get_example_plays(test_df, model, test_X, feature_names)
    print_example_plays(examples)

    # Save model and metadata
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)

    model.save_model(str(MODEL_DIR / "completion_model.lgb"))

    with open(MODEL_DIR / "feature_names.pkl", 'wb') as f:
        pickle.dump(feature_names, f)

    importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    print(f"Model saved to {MODEL_DIR / 'completion_model.lgb'}")
    print(f"Feature names saved to {MODEL_DIR / 'feature_names.pkl'}")
    print(f"Feature importance saved to {MODEL_DIR / 'feature_importance.csv'}")

    return model, df, examples


if __name__ == "__main__":
    import sys

    # Parse arguments
    extract = True
    max_plays = None

    if "--load" in sys.argv:
        extract = False

    if "--test" in sys.argv:
        max_plays = 500  # Quick test with 500 plays

    model, df, examples = main(extract_features=extract, max_plays=max_plays)
