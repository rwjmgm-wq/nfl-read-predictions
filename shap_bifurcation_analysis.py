"""
SHAP Analysis for M3 vs M6 Bifurcation Detection.

Explains WHY each method triggered at different frames:
- M3 (Confidence Threshold): What features drove early confidence?
- M6 (Z-Score Breakout): What features caused the late anomaly/shift?
- Delta Analysis: What changed between the two trigger points?
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import shap
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
ORGANIZED_DIR = BASE_DIR / "organized_plays"


def load_model_and_features():
    """Load the trained LightGBM model and feature names."""
    model = lgb.Booster(model_file=str(MODEL_DIR / "completion_model.lgb"))

    with open(MODEL_DIR / "feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)

    return model, feature_names


def get_play_features(play_path, feature_names):
    """Get frame-by-frame features for a play."""
    from frame_features import calculate_frame_features

    features = calculate_frame_features(play_path)
    if features is None or len(features) == 0:
        return None

    # Prepare features for model (same as in bifurcation_detection.py)
    X = features[[col for col in feature_names if col in features.columns]].copy()
    X = X.fillna(X.median())

    return X, features


def analyze_bifurcation_why(model, play_features, trigger_frame_m3, trigger_frame_m6,
                            outcome_class=1, save_path=None):
    """
    Compare SHAP explanations at M3 vs M6 trigger frames.

    Args:
        model: Trained LightGBM model
        play_features: DataFrame of features for all frames of a play
        trigger_frame_m3: Frame index where M3 triggered
        trigger_frame_m6: Frame index where M6 triggered
        outcome_class: 0=Incomplete, 1=Complete, 2=Interception
        save_path: Optional path to save plots
    """
    # Create SHAP explainer for tree model
    explainer = shap.TreeExplainer(model)

    # Get features for the two critical frames
    # Ensure we're within bounds
    n_frames = len(play_features)
    frame_m3 = min(trigger_frame_m3, n_frames - 1)
    frame_m6 = min(trigger_frame_m6, n_frames - 1)

    features_m3 = play_features.iloc[[frame_m3]]
    features_m6 = play_features.iloc[[frame_m6]]

    # Calculate SHAP values
    # For multiclass, shap_values is a list of arrays (one per class)
    shap_values_m3 = explainer.shap_values(features_m3)
    shap_values_m6 = explainer.shap_values(features_m6)

    # Handle different SHAP output formats
    # For LightGBM multiclass, shap_values can be:
    # - List of 3 arrays (one per class), each shape (n_samples, n_features)
    # - Or single array of shape (n_samples, n_features, n_classes)
    if isinstance(shap_values_m3, list):
        # List format: [class0_array, class1_array, class2_array]
        shap_m3 = shap_values_m3[outcome_class][0]  # First row of class array
        shap_m6 = shap_values_m6[outcome_class][0]
    elif len(shap_values_m3.shape) == 3:
        # 3D array format: (n_samples, n_features, n_classes)
        shap_m3 = shap_values_m3[0, :, outcome_class]
        shap_m6 = shap_values_m6[0, :, outcome_class]
    else:
        # 2D array (single class or binary)
        shap_m3 = shap_values_m3[0]
        shap_m6 = shap_values_m6[0]

    feature_names = play_features.columns.tolist()

    # Create results DataFrame
    results = pd.DataFrame({
        'feature': feature_names,
        'value_at_m3': features_m3.iloc[0].values,
        'value_at_m6': features_m6.iloc[0].values,
        'shap_m3': shap_m3,
        'shap_m6': shap_m6,
        'shap_change': shap_m6 - shap_m3,
        'value_change': features_m6.iloc[0].values - features_m3.iloc[0].values
    })

    return results, explainer, shap_values_m3, shap_values_m6


def print_shap_analysis(results, frame_m3, frame_m6, outcome):
    """Print formatted SHAP analysis."""
    outcome_names = {0: 'Incomplete', 1: 'Complete', 2: 'Interception'}

    print("\n" + "="*80)
    print(f"SHAP ANALYSIS: M3 (Frame {frame_m3}) vs M6 (Frame {frame_m6})")
    print(f"Outcome: {outcome_names.get(outcome, outcome)}")
    print("="*80)

    # Top drivers at M3 (sort by absolute value)
    print(f"\n--- M3 Trigger (Frame {frame_m3}) - Top Drivers ---")
    results['abs_shap_m3'] = results['shap_m3'].abs()
    top_m3 = results.nlargest(5, 'abs_shap_m3')[['feature', 'value_at_m3', 'shap_m3']]
    for _, row in top_m3.iterrows():
        direction = "+" if row['shap_m3'] > 0 else ""
        print(f"  {row['feature']:30} = {row['value_at_m3']:8.3f} -> SHAP: {direction}{row['shap_m3']:.4f}")

    # Top drivers at M6
    print(f"\n--- M6 Trigger (Frame {frame_m6}) - Top Drivers ---")
    results['abs_shap_m6'] = results['shap_m6'].abs()
    top_m6 = results.nlargest(5, 'abs_shap_m6')[['feature', 'value_at_m6', 'shap_m6']]
    for _, row in top_m6.iterrows():
        direction = "+" if row['shap_m6'] > 0 else ""
        print(f"  {row['feature']:30} = {row['value_at_m6']:8.3f} -> SHAP: {direction}{row['shap_m6']:.4f}")

    # What changed the most (narrative shift)
    print(f"\n--- Narrative Shift (What Changed Most Between M3 and M6) ---")
    results['abs_shap_change'] = results['shap_change'].abs()
    top_change = results.nlargest(5, 'abs_shap_change')[['feature', 'value_change', 'shap_change']]
    for _, row in top_change.iterrows():
        direction = "+" if row['shap_change'] > 0 else ""
        val_dir = "+" if row['value_change'] > 0 else ""
        print(f"  {row['feature']:30} changed by {val_dir}{row['value_change']:8.3f} -> SHAP delta: {direction}{row['shap_change']:.4f}")


def create_shap_visualizations(results, explainer, shap_m3_raw, shap_m6_raw, features_m3, features_m6,
                                frame_m3, frame_m6, outcome_class, play_name, save_dir=None):
    """Create SHAP visualization plots."""
    outcome_names = {0: 'Incomplete', 1: 'Complete', 2: 'Interception'}
    outcome_name = outcome_names.get(outcome_class, str(outcome_class))

    # Extract SHAP values for the specific class (handle different formats)
    if isinstance(shap_m3_raw, list):
        shap_vals_m3 = shap_m3_raw[outcome_class][0]
        shap_vals_m6 = shap_m6_raw[outcome_class][0]
        base_value = explainer.expected_value[outcome_class]
    elif len(shap_m3_raw.shape) == 3:
        shap_vals_m3 = shap_m3_raw[0, :, outcome_class]
        shap_vals_m6 = shap_m6_raw[0, :, outcome_class]
        base_value = explainer.expected_value[outcome_class]
    else:
        shap_vals_m3 = shap_m3_raw[0]
        shap_vals_m6 = shap_m6_raw[0]
        base_value = explainer.expected_value

    # Figure 1: Side-by-side waterfall plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # M3 waterfall
    plt.sca(axes[0])
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_vals_m3,
            base_values=base_value,
            data=features_m3.iloc[0].values,
            feature_names=features_m3.columns.tolist()
        ),
        max_display=10,
        show=False
    )
    axes[0].set_title(f'M3 Trigger (Frame {frame_m3})\nP({outcome_name})')

    # M6 waterfall
    plt.sca(axes[1])
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_vals_m6,
            base_values=base_value,
            data=features_m6.iloc[0].values,
            feature_names=features_m6.columns.tolist()
        ),
        max_display=10,
        show=False
    )
    axes[1].set_title(f'M6 Trigger (Frame {frame_m6})\nP({outcome_name})')

    plt.suptitle(f'SHAP Analysis: {play_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / f"shap_waterfall_{play_name}.png", dpi=150, bbox_inches='tight')
        print(f"Saved: shap_waterfall_{play_name}.png")
    plt.close()

    # Figure 2: Delta bar chart (what changed)
    fig, ax = plt.subplots(figsize=(12, 8))

    results['abs_shap_change'] = results['shap_change'].abs()
    top_changes = results.nlargest(10, 'abs_shap_change').sort_values('shap_change')
    colors = ['green' if x > 0 else 'red' for x in top_changes['shap_change']]

    ax.barh(top_changes['feature'], top_changes['shap_change'], color=colors, alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('SHAP Change (M6 - M3)')
    ax.set_title(f'Narrative Shift: What Changed Between M3 and M6\n{play_name}')

    # Add value annotations
    for i, (idx, row) in enumerate(top_changes.iterrows()):
        val_change = row['value_change']
        ax.annotate(f'Δ={val_change:+.2f}',
                   xy=(row['shap_change'], i),
                   xytext=(5 if row['shap_change'] > 0 else -5, 0),
                   textcoords='offset points',
                   ha='left' if row['shap_change'] > 0 else 'right',
                   va='center', fontsize=8)

    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / f"shap_delta_{play_name}.png", dpi=150, bbox_inches='tight')
        print(f"Saved: shap_delta_{play_name}.png")
    plt.close()


def analyze_single_play(play_path, model, feature_names, save_plots=True):
    """Full SHAP analysis for a single play."""
    from bifurcation_detection import (
        get_play_probabilities,
        method_confidence_threshold,
        method_zscore_breakout
    )

    play_path = Path(play_path)

    # Load supplementary data for outcome
    supp = pd.read_csv(play_path / "supplementary.csv")
    outcome_code = supp['pass_result'].iloc[0]
    outcome_map = {'C': 'complete', 'I': 'incomplete', 'IN': 'interception'}
    outcome_class_map = {'C': 1, 'I': 0, 'IN': 2}
    outcome = outcome_map.get(outcome_code)
    outcome_class = outcome_class_map.get(outcome_code)

    # Get features
    play_features, raw_features = get_play_features(play_path, feature_names)
    if play_features is None:
        print(f"Failed to get features for {play_path}")
        return None

    # Get probabilities and bifurcation frames
    frame_probs = get_play_probabilities(play_path, model, feature_names)
    if frame_probs is None:
        print(f"Failed to get probabilities for {play_path}")
        return None

    frame_m3 = method_confidence_threshold(frame_probs, outcome)
    frame_m6 = method_zscore_breakout(frame_probs, outcome)

    play_name = play_path.name

    print(f"\n{'='*80}")
    print(f"PLAY: {play_name}")
    print(f"Outcome: {outcome.upper()}")
    print(f"Total Frames: {len(play_features)}")
    print(f"M3 Trigger Frame: {frame_m3}")
    print(f"M6 Trigger Frame: {frame_m6}")
    print(f"Frame Difference: {abs(frame_m6 - frame_m3)}")
    print(f"{'='*80}")

    # Run SHAP analysis
    results, explainer, shap_m3, shap_m6 = analyze_bifurcation_why(
        model, play_features, frame_m3, frame_m6, outcome_class
    )

    # Print analysis
    print_shap_analysis(results, frame_m3, frame_m6, outcome_class)

    # Create visualizations
    if save_plots:
        features_m3 = play_features.iloc[[min(frame_m3, len(play_features)-1)]]
        features_m6 = play_features.iloc[[min(frame_m6, len(play_features)-1)]]

        create_shap_visualizations(
            results, explainer, shap_m3, shap_m6,
            features_m3, features_m6,
            frame_m3, frame_m6, outcome_class,
            play_name, save_dir=BASE_DIR
        )

    return results


def analyze_conflict_plays(n_plays=5, min_frame_diff=3):
    """
    Find and analyze plays where M3 and M6 disagree significantly.
    These are the most interesting for understanding the difference.
    """
    print("Loading model and bifurcation results...")
    model, feature_names = load_model_and_features()

    # Load bifurcation results
    bif_results = pd.read_csv(BASE_DIR / "bifurcation_results.csv")

    # Find plays with significant M3 vs M6 disagreement
    bif_results['frame_diff'] = abs(bif_results['bifurcation_frame_m3'] - bif_results['bifurcation_frame_m6'])
    conflict_plays = bif_results[bif_results['frame_diff'] >= min_frame_diff].sort_values('frame_diff', ascending=False)

    print(f"\nFound {len(conflict_plays)} plays with M3/M6 disagreement >= {min_frame_diff} frames")

    # Analyze top N conflict plays (one per outcome type)
    analyzed = {'complete': 0, 'incomplete': 0, 'interception': 0}

    for _, row in conflict_plays.iterrows():
        outcome = row['outcome']
        if analyzed[outcome] >= n_plays // 3 + 1:
            continue

        play_path = ORGANIZED_DIR / row['week'] / row['game'] / row['play_folder']

        if play_path.exists():
            try:
                analyze_single_play(play_path, model, feature_names)
                analyzed[outcome] += 1
            except Exception as e:
                print(f"Error analyzing {play_path}: {e}")
                continue

        if sum(analyzed.values()) >= n_plays:
            break

    print(f"\nAnalyzed {sum(analyzed.values())} conflict plays")


def analyze_all_plays_shap(save_every=500):
    """
    Run SHAP analysis on ALL plays and aggregate results.

    This extracts SHAP values at M3 and M6 trigger frames for every play,
    then aggregates to find overall feature importance patterns.

    Args:
        save_every: Save intermediate results every N plays (for crash recovery)

    Returns:
        Aggregated SHAP results DataFrame
    """
    from bifurcation_detection import (
        get_play_probabilities,
        method_confidence_threshold,
        method_zscore_breakout
    )
    import time

    print("="*80)
    print("SHAP ANALYSIS ON ALL PLAYS")
    print("="*80)

    # Load model and bifurcation results
    model, feature_names = load_model_and_features()
    bif_results = pd.read_csv(BASE_DIR / "bifurcation_results.csv")

    print(f"\nTotal plays to analyze: {len(bif_results)}")

    # Create SHAP explainer once (reuse for all plays)
    explainer = shap.TreeExplainer(model)

    # Storage for all SHAP values
    all_shap_m3 = []  # List of dicts: {feature: shap_value}
    all_shap_m6 = []
    all_shap_delta = []
    all_metadata = []  # outcome, play_id, etc.

    start_time = time.time()
    successful = 0
    failed = 0

    for idx, row in bif_results.iterrows():
        # Progress update
        if idx % 100 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(bif_results) - idx) / rate if rate > 0 else 0
            print(f"Processing play {idx+1}/{len(bif_results)} ({successful} successful, {failed} failed) "
                  f"[{elapsed/60:.1f}min elapsed, ~{remaining/60:.1f}min remaining]")

        try:
            play_path = ORGANIZED_DIR / row['week'] / row['game'] / row['play_folder']

            if not play_path.exists():
                failed += 1
                continue

            # Get play features
            play_features, _ = get_play_features(play_path, feature_names)
            if play_features is None or len(play_features) == 0:
                failed += 1
                continue

            # Get trigger frames
            outcome_code = row['outcome']
            outcome_class_map = {'complete': 1, 'incomplete': 0, 'interception': 2}
            outcome_class = outcome_class_map.get(outcome_code, 1)

            frame_m3 = int(row['bifurcation_frame_m3'])
            frame_m6 = int(row['bifurcation_frame_m6'])

            # Clamp to valid range
            n_frames = len(play_features)
            frame_m3 = min(max(0, frame_m3), n_frames - 1)
            frame_m6 = min(max(0, frame_m6), n_frames - 1)

            # Get features at trigger frames
            features_m3 = play_features.iloc[[frame_m3]]
            features_m6 = play_features.iloc[[frame_m6]]

            # Calculate SHAP values
            shap_values_m3 = explainer.shap_values(features_m3)
            shap_values_m6 = explainer.shap_values(features_m6)

            # Extract values for the relevant outcome class
            if isinstance(shap_values_m3, list):
                shap_m3 = shap_values_m3[outcome_class][0]
                shap_m6 = shap_values_m6[outcome_class][0]
            elif len(shap_values_m3.shape) == 3:
                shap_m3 = shap_values_m3[0, :, outcome_class]
                shap_m6 = shap_values_m6[0, :, outcome_class]
            else:
                shap_m3 = shap_values_m3[0]
                shap_m6 = shap_values_m6[0]

            # Store as dictionaries
            shap_dict_m3 = {feat: val for feat, val in zip(feature_names, shap_m3)}
            shap_dict_m6 = {feat: val for feat, val in zip(feature_names, shap_m6)}
            shap_dict_delta = {feat: shap_m6[i] - shap_m3[i] for i, feat in enumerate(feature_names)}

            all_shap_m3.append(shap_dict_m3)
            all_shap_m6.append(shap_dict_m6)
            all_shap_delta.append(shap_dict_delta)

            all_metadata.append({
                'play_id': f"{row['week']}_{row['game']}_{row['play_folder']}",
                'outcome': outcome_code,
                'frame_m3': frame_m3,
                'frame_m6': frame_m6,
                'frame_diff': abs(frame_m6 - frame_m3),
                'total_frames': n_frames
            })

            successful += 1

            # Intermediate save
            if successful % save_every == 0:
                print(f"\n  Saving intermediate results ({successful} plays)...")
                _save_intermediate_shap_results(
                    all_shap_m3, all_shap_m6, all_shap_delta, all_metadata, feature_names
                )

        except Exception as e:
            failed += 1
            if failed <= 5:  # Only print first 5 errors
                print(f"  Error on play {idx}: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"SHAP ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}/{len(bif_results)} plays")
    print(f"Failed: {failed}")
    print(f"Time: {elapsed/60:.1f} minutes ({elapsed/successful:.2f} sec/play)")

    # Create aggregated results
    print("\nAggregating SHAP values...")
    aggregated = aggregate_shap_results(
        all_shap_m3, all_shap_m6, all_shap_delta, all_metadata, feature_names
    )

    return aggregated


def _save_intermediate_shap_results(shap_m3_list, shap_m6_list, shap_delta_list, metadata_list, feature_names):
    """Save intermediate results for crash recovery."""
    # Convert to DataFrames
    df_m3 = pd.DataFrame(shap_m3_list)
    df_m6 = pd.DataFrame(shap_m6_list)
    df_delta = pd.DataFrame(shap_delta_list)
    df_meta = pd.DataFrame(metadata_list)

    # Save
    df_m3.to_csv(BASE_DIR / "shap_all_plays_m3_intermediate.csv", index=False)
    df_m6.to_csv(BASE_DIR / "shap_all_plays_m6_intermediate.csv", index=False)
    df_delta.to_csv(BASE_DIR / "shap_all_plays_delta_intermediate.csv", index=False)
    df_meta.to_csv(BASE_DIR / "shap_all_plays_metadata_intermediate.csv", index=False)


def aggregate_shap_results(shap_m3_list, shap_m6_list, shap_delta_list, metadata_list, feature_names):
    """
    Aggregate SHAP values across all plays to find overall patterns.
    """
    # Convert lists to DataFrames
    df_m3 = pd.DataFrame(shap_m3_list)
    df_m6 = pd.DataFrame(shap_m6_list)
    df_delta = pd.DataFrame(shap_delta_list)
    df_meta = pd.DataFrame(metadata_list)

    # Save raw results
    print("Saving raw SHAP results...")
    df_m3.to_csv(BASE_DIR / "shap_all_plays_m3.csv", index=False)
    df_m6.to_csv(BASE_DIR / "shap_all_plays_m6.csv", index=False)
    df_delta.to_csv(BASE_DIR / "shap_all_plays_delta.csv", index=False)
    df_meta.to_csv(BASE_DIR / "shap_all_plays_metadata.csv", index=False)

    # Aggregate by outcome type
    results = []

    for outcome in ['complete', 'incomplete', 'interception']:
        mask = df_meta['outcome'] == outcome
        n_plays = mask.sum()

        if n_plays == 0:
            continue

        # Mean absolute SHAP at M3
        mean_abs_m3 = df_m3[mask].abs().mean()

        # Mean absolute SHAP at M6
        mean_abs_m6 = df_m6[mask].abs().mean()

        # Mean absolute SHAP change (narrative shift)
        mean_abs_delta = df_delta[mask].abs().mean()

        # Mean signed SHAP (direction of influence)
        mean_signed_m3 = df_m3[mask].mean()
        mean_signed_m6 = df_m6[mask].mean()
        mean_signed_delta = df_delta[mask].mean()

        for feat in feature_names:
            results.append({
                'outcome': outcome,
                'feature': feat,
                'n_plays': n_plays,
                'mean_abs_shap_m3': mean_abs_m3[feat],
                'mean_abs_shap_m6': mean_abs_m6[feat],
                'mean_abs_shap_delta': mean_abs_delta[feat],
                'mean_signed_shap_m3': mean_signed_m3[feat],
                'mean_signed_shap_m6': mean_signed_m6[feat],
                'mean_signed_shap_delta': mean_signed_delta[feat]
            })

    # Also aggregate across ALL plays
    mean_abs_m3_all = df_m3.abs().mean()
    mean_abs_m6_all = df_m6.abs().mean()
    mean_abs_delta_all = df_delta.abs().mean()
    mean_signed_m3_all = df_m3.mean()
    mean_signed_m6_all = df_m6.mean()
    mean_signed_delta_all = df_delta.mean()

    for feat in feature_names:
        results.append({
            'outcome': 'ALL',
            'feature': feat,
            'n_plays': len(df_meta),
            'mean_abs_shap_m3': mean_abs_m3_all[feat],
            'mean_abs_shap_m6': mean_abs_m6_all[feat],
            'mean_abs_shap_delta': mean_abs_delta_all[feat],
            'mean_signed_shap_m3': mean_signed_m3_all[feat],
            'mean_signed_shap_m6': mean_signed_m6_all[feat],
            'mean_signed_shap_delta': mean_signed_delta_all[feat]
        })

    aggregated_df = pd.DataFrame(results)
    aggregated_df.to_csv(BASE_DIR / "shap_aggregated_by_outcome.csv", index=False)
    print(f"Saved: shap_aggregated_by_outcome.csv")

    # Print summary
    print_shap_summary(aggregated_df)

    # Create summary visualizations
    create_aggregated_shap_plots(aggregated_df, df_m3, df_m6, df_delta, df_meta)

    return aggregated_df


def print_shap_summary(aggregated_df):
    """Print formatted summary of aggregated SHAP results."""
    print("\n" + "="*80)
    print("AGGREGATED SHAP SUMMARY")
    print("="*80)

    for outcome in ['complete', 'incomplete', 'interception', 'ALL']:
        subset = aggregated_df[aggregated_df['outcome'] == outcome]
        if len(subset) == 0:
            continue

        n_plays = subset['n_plays'].iloc[0]

        print(f"\n{'='*60}")
        print(f"{outcome.upper()} PLAYS (n={n_plays})")
        print(f"{'='*60}")

        # Top 5 features at M3 (by mean abs SHAP)
        print(f"\nTop Features at M3 Trigger (Early Detection):")
        top_m3 = subset.nlargest(5, 'mean_abs_shap_m3')[['feature', 'mean_abs_shap_m3', 'mean_signed_shap_m3']]
        for _, row in top_m3.iterrows():
            direction = "+" if row['mean_signed_shap_m3'] > 0 else ""
            print(f"  {row['feature']:35} | Abs: {row['mean_abs_shap_m3']:.4f} | Avg: {direction}{row['mean_signed_shap_m3']:.4f}")

        # Top 5 features at M6 (by mean abs SHAP)
        print(f"\nTop Features at M6 Trigger (Late/Breakout Detection):")
        top_m6 = subset.nlargest(5, 'mean_abs_shap_m6')[['feature', 'mean_abs_shap_m6', 'mean_signed_shap_m6']]
        for _, row in top_m6.iterrows():
            direction = "+" if row['mean_signed_shap_m6'] > 0 else ""
            print(f"  {row['feature']:35} | Abs: {row['mean_abs_shap_m6']:.4f} | Avg: {direction}{row['mean_signed_shap_m6']:.4f}")

        # Top 5 features by narrative shift (delta)
        print(f"\nTop Features for Narrative Shift (M6 - M3):")
        top_delta = subset.nlargest(5, 'mean_abs_shap_delta')[['feature', 'mean_abs_shap_delta', 'mean_signed_shap_delta']]
        for _, row in top_delta.iterrows():
            direction = "+" if row['mean_signed_shap_delta'] > 0 else ""
            print(f"  {row['feature']:35} | Abs: {row['mean_abs_shap_delta']:.4f} | Avg: {direction}{row['mean_signed_shap_delta']:.4f}")


def create_aggregated_shap_plots(aggregated_df, df_m3, df_m6, df_delta, df_meta):
    """Create visualization plots for aggregated SHAP analysis."""
    import matplotlib.pyplot as plt

    # Create output directory
    shap_dir = BASE_DIR / "shap_plots"
    shap_dir.mkdir(exist_ok=True)

    # Plot 1: Feature Importance by Outcome at M3 vs M6
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for i, outcome in enumerate(['complete', 'incomplete', 'interception']):
        ax = axes[i]
        subset = aggregated_df[aggregated_df['outcome'] == outcome]

        # Get top 10 features by combined importance
        subset = subset.copy()
        subset['combined'] = subset['mean_abs_shap_m3'] + subset['mean_abs_shap_m6']
        top_features = subset.nlargest(10, 'combined')

        x = np.arange(len(top_features))
        width = 0.35

        bars1 = ax.barh(x - width/2, top_features['mean_abs_shap_m3'], width, label='M3', color='blue', alpha=0.7)
        bars2 = ax.barh(x + width/2, top_features['mean_abs_shap_m6'], width, label='M6', color='orange', alpha=0.7)

        ax.set_yticks(x)
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Mean Absolute SHAP Value')
        ax.set_title(f'{outcome.upper()}\n(n={top_features["n_plays"].iloc[0]})')
        ax.legend()
        ax.invert_yaxis()

    plt.suptitle('Feature Importance: M3 (Early) vs M6 (Late/Breakout)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_feature_importance_m3_vs_m6.png", dpi=150, bbox_inches='tight')
    print(f"Saved: shap_plots/shap_feature_importance_m3_vs_m6.png")
    plt.close()

    # Plot 2: Narrative Shift - What changes between M3 and M6
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for i, outcome in enumerate(['complete', 'incomplete', 'interception']):
        ax = axes[i]
        subset = aggregated_df[aggregated_df['outcome'] == outcome]

        # Top 10 by absolute delta
        top_delta = subset.nlargest(10, 'mean_abs_shap_delta')[['feature', 'mean_signed_shap_delta']]

        colors = ['green' if x > 0 else 'red' for x in top_delta['mean_signed_shap_delta']]
        ax.barh(top_delta['feature'], top_delta['mean_signed_shap_delta'], color=colors, alpha=0.7)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Mean SHAP Change (M6 - M3)')
        ax.set_title(f'{outcome.upper()} - Narrative Shift')
        ax.invert_yaxis()

    plt.suptitle('Narrative Shift: Feature Importance Change from M3 to M6', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_narrative_shift_by_outcome.png", dpi=150, bbox_inches='tight')
    print(f"Saved: shap_plots/shap_narrative_shift_by_outcome.png")
    plt.close()

    # Plot 3: Overall feature importance heatmap
    fig, ax = plt.subplots(figsize=(14, 10))

    # Pivot to create heatmap data
    heatmap_data = []
    for outcome in ['complete', 'incomplete', 'interception']:
        subset = aggregated_df[aggregated_df['outcome'] == outcome]
        top_10 = subset.nlargest(10, 'mean_abs_shap_m6')['feature'].tolist()

        for feat in top_10:
            row = subset[subset['feature'] == feat].iloc[0]
            heatmap_data.append({
                'feature': feat,
                'outcome': outcome,
                'm3': row['mean_abs_shap_m3'],
                'm6': row['mean_abs_shap_m6'],
                'delta': row['mean_abs_shap_delta']
            })

    heatmap_df = pd.DataFrame(heatmap_data)

    # Create pivot table for heatmap
    pivot = heatmap_df.pivot_table(index='feature', columns='outcome', values='m6', aggfunc='first')

    # Sort by max importance across outcomes
    pivot['max'] = pivot.max(axis=1)
    pivot = pivot.sort_values('max', ascending=True).drop('max', axis=1)

    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([c.upper() for c in pivot.columns])
    ax.set_yticklabels(pivot.index)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Absolute SHAP at M6')

    ax.set_title('Feature Importance at M6 Trigger by Outcome', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_heatmap_m6_by_outcome.png", dpi=150, bbox_inches='tight')
    print(f"Saved: shap_plots/shap_heatmap_m6_by_outcome.png")
    plt.close()

    # Plot 4: Distribution of frame differences (M6 - M3) by outcome
    fig, ax = plt.subplots(figsize=(10, 6))

    for outcome in ['complete', 'incomplete', 'interception']:
        mask = df_meta['outcome'] == outcome
        diffs = df_meta[mask]['frame_diff']
        ax.hist(diffs, bins=20, alpha=0.5, label=f'{outcome} (n={mask.sum()})')

    ax.set_xlabel('Frame Difference (|M6 - M3|)')
    ax.set_ylabel('Number of Plays')
    ax.set_title('Distribution of M3/M6 Trigger Frame Differences')
    ax.legend()
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_frame_diff_distribution.png", dpi=150, bbox_inches='tight')
    print(f"Saved: shap_plots/shap_frame_diff_distribution.png")
    plt.close()

    print(f"\nAll plots saved to: {shap_dir}")


def main():
    """Main function - analyze sample plays with SHAP."""
    print("="*80)
    print("SHAP BIFURCATION ANALYSIS")
    print("="*80)

    # Load model
    model, feature_names = load_model_and_features()

    # Option 1: Analyze specific high-drama plays
    # These are plays from the bifurcation analysis with interesting characteristics

    # Load bifurcation results to find interesting plays
    bif_results = pd.read_csv(BASE_DIR / "bifurcation_results.csv")

    # Find plays by category
    print("\n" + "-"*60)
    print("ANALYZING SAMPLE PLAYS BY OUTCOME TYPE")
    print("-"*60)

    for outcome in ['complete', 'incomplete', 'interception']:
        subset = bif_results[bif_results['outcome'] == outcome]

        # Find a play with significant M3/M6 difference
        subset['frame_diff'] = abs(subset['bifurcation_frame_m3'] - subset['bifurcation_frame_m6'])
        interesting = subset[subset['frame_diff'] >= 3].sort_values('drama_score', ascending=False)

        if len(interesting) == 0:
            interesting = subset.sort_values('drama_score', ascending=False)

        if len(interesting) > 0:
            row = interesting.iloc[0]
            play_path = ORGANIZED_DIR / row['week'] / row['game'] / row['play_folder']

            if play_path.exists():
                analyze_single_play(play_path, model, feature_names)


if __name__ == "__main__":
    import sys

    if "--all" in sys.argv:
        # Run SHAP analysis on ALL plays (computationally intensive)
        analyze_all_plays_shap(save_every=500)
    elif "--conflict" in sys.argv:
        # Analyze plays where M3 and M6 significantly disagree
        analyze_conflict_plays(n_plays=6, min_frame_diff=5)
    else:
        main()
