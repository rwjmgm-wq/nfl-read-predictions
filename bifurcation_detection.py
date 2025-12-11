"""
Bifurcation Point Detection for Pass Play Probability Trajectories.

Detects the frame where a play's outcome becomes "determined" using two proven methods:
- M3: Confidence Threshold - when probability crosses outcome-specific threshold (80.5% accuracy)
- M6: Z-Score Breakout - when probability deviates from rolling baseline (best for interceptions)

These two methods were selected based on prediction accuracy analysis:
- M3 had highest overall accuracy at predicting actual outcomes at bifurcation point
- M6 had best accuracy for rare interception events (42.9% vs ~30% for other methods)

The combo model uses both methods to provide robust bifurcation detection.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import entropy
from tqdm import tqdm
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
ORGANIZED_DIR = BASE_DIR / "organized_plays"


def smooth_probability_trajectory(probs, window_length=5, polyorder=2):
    """Smooth probability trajectory using Savitzky-Golay filter."""
    if len(probs) < window_length:
        window_length = len(probs) if len(probs) % 2 == 1 else len(probs) - 1
        window_length = max(3, window_length)
    polyorder = min(polyorder, window_length - 1)
    smoothed = savgol_filter(probs, window_length, polyorder)
    return np.clip(smoothed, 0, 1)


def get_play_probabilities(play_path, model, feature_names):
    """Get frame-by-frame probabilities for a single play."""
    from frame_features import calculate_frame_features

    features = calculate_frame_features(play_path)
    if features is None or len(features) == 0:
        return None

    # Prepare features for model
    X = features[[col for col in feature_names if col in features.columns]].copy()
    X = X.fillna(X.median())

    # Get probabilities
    probs = model.predict(X)

    # Smooth each probability trajectory
    p_incomplete = smooth_probability_trajectory(probs[:, 0])
    p_complete = smooth_probability_trajectory(probs[:, 1])
    p_interception = smooth_probability_trajectory(probs[:, 2])

    return pd.DataFrame({
        'frame': range(len(probs)),
        'p_incomplete': p_incomplete,
        'p_complete': p_complete,
        'p_interception': p_interception
    })


# =============================================================================
# BIFURCATION DETECTION METHODS
# =============================================================================

def method_confidence_threshold(frame_probs, outcome):
    """
    M3: Confidence Threshold (80.5% accuracy - best overall)

    First frame where winning outcome's probability crosses a confidence threshold.
    Uses outcome-specific thresholds tuned for each outcome type.
    """
    # Outcome-specific thresholds (tuned for accuracy)
    thresholds = {
        'complete': 0.6,
        'incomplete': 0.5,
        'interception': 0.3
    }

    outcome_col = f'p_{outcome}'
    threshold = thresholds[outcome]
    probs = frame_probs[outcome_col].values

    # Find first frame crossing threshold
    crossings = np.where(probs > threshold)[0]

    if len(crossings) > 0:
        return crossings[0]

    # If threshold never reached, return frame of maximum probability
    return np.argmax(probs)


def method_zscore_breakout(frame_probs, outcome, window=10, z_thresh=2.0):
    """
    M6: Z-Score Breakout (best for interceptions - 42.9% accuracy)

    Detects when probability deviates significantly from its rolling baseline.
    Great for detecting sudden "realization" moments (like a ball tip or defender break).
    """
    outcome_col = f'p_{outcome}'
    series = frame_probs[outcome_col]

    # Calculate rolling statistics
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std().clip(lower=0.01)

    z_scores = (series - rolling_mean) / rolling_std

    # Look for first time Z-score spikes above threshold
    breakouts = np.where(z_scores > z_thresh)[0]

    if len(breakouts) > 0:
        # Filter for breakouts where probability is > 0.5 (meaningful)
        valid_breakouts = [b for b in breakouts if series.iloc[b] > 0.5]
        if valid_breakouts:
            return valid_breakouts[0]

    # Fallback to max velocity
    if len(series) > 1:
        return np.argmax(np.diff(series)) + 1
    return 0


def combo_bifurcation(frame_probs, outcome):
    """
    Combo Model: Uses both M3 and M6 with intelligent selection.

    Strategy:
    - For interceptions: Prefer M6 (Z-Score) as it catches sudden shifts better
    - For completions/incompletions: Prefer M3 (Confidence) as it's more reliable
    - If methods agree (within 2 frames), use the earlier one
    - If methods disagree significantly, use outcome-appropriate method
    """
    m3_frame = method_confidence_threshold(frame_probs, outcome)
    m6_frame = method_zscore_breakout(frame_probs, outcome)

    n_frames = len(frame_probs)

    # If methods agree (within 2 frames), use the earlier detection
    if abs(m3_frame - m6_frame) <= 2:
        return min(m3_frame, m6_frame)

    # For interceptions, prefer M6 (better at catching sudden events)
    if outcome == 'interception':
        # But only if M6 detected something reasonable (not fallback)
        if m6_frame < n_frames - 1:
            return m6_frame
        return m3_frame

    # For completions/incompletions, prefer M3 (higher overall accuracy)
    return m3_frame


# =============================================================================
# BIFURCATION ENGINE CLASS (Simplified)
# =============================================================================

class BifurcationEngine:
    """
    Simplified bifurcation detection using M3 (Confidence) and M6 (Z-Score).

    Also calculates entropy-based drama metrics for broadcast applications.
    """

    def __init__(self, frame_probs, outcome, smoothing_window=5):
        """
        Args:
            frame_probs: DataFrame with ['p_complete', 'p_incomplete', 'p_interception']
            outcome: str ('complete', 'incomplete', 'interception')
            smoothing_window: int, window size for Savitzky-Golay smoothing
        """
        self.raw_df = frame_probs.copy()
        self.outcome = outcome
        self.outcome_col = f'p_{outcome}'
        self.cols = ['p_complete', 'p_incomplete', 'p_interception']

        # Pre-process: Smooth and Normalize
        self.df = self._preprocess_probabilities(self.raw_df, smoothing_window)

        # Calculate Shannon Entropy (for drama metrics)
        self.df['entropy'] = self.df[self.cols].apply(lambda x: entropy(x, base=2), axis=1)

    def _preprocess_probabilities(self, df, window):
        """Smooths trajectories and ensures they sum to 1.0."""
        df_smooth = df.copy()

        window = min(window, len(df))
        if window % 2 == 0:
            window -= 1
        if window < 3:
            window = 3

        for col in self.cols:
            if col in df_smooth.columns:
                df_smooth[col] = savgol_filter(df[col], window, 2)

        # Clip and Normalize
        df_smooth[self.cols] = df_smooth[self.cols].clip(1e-6, 1.0)
        df_smooth[self.cols] = df_smooth[self.cols].div(df_smooth[self.cols].sum(axis=1), axis=0)

        return df_smooth

    def calculate_drama_integral(self):
        """Mean entropy over time - higher = more dramatic/uncertain play."""
        return self.df['entropy'].mean()

    def run_all(self):
        """
        Run M3 and M6 bifurcation detection plus drama metrics.

        Returns:
            dict with bifurcation frames, timings, and drama metrics
        """
        n_frames = len(self.df)

        # Run the two core methods
        m3_frame = method_confidence_threshold(self.raw_df, self.outcome)
        m6_frame = method_zscore_breakout(self.raw_df, self.outcome)
        combo_frame = combo_bifurcation(self.raw_df, self.outcome)

        # Drama metrics
        drama_integral = self.calculate_drama_integral()

        return {
            # Method results
            'bifurcation_frame_m3': m3_frame,
            'bifurcation_frame_m6': m6_frame,
            'bifurcation_frame_combo': combo_frame,

            # Timing metrics (normalized 0-1)
            'bifurcation_timing_m3': m3_frame / max(1, n_frames - 1),
            'bifurcation_timing_m6': m6_frame / max(1, n_frames - 1),
            'bifurcation_timing_combo': combo_frame / max(1, n_frames - 1),

            # Drama and entropy metrics
            'drama_integral': drama_integral,
            'entropy_at_release': self.df['entropy'].iloc[0],
            'entropy_at_end': self.df['entropy'].iloc[-1],
            'entropy_drop': self.df['entropy'].iloc[0] - self.df['entropy'].iloc[-1],
        }


def detect_bifurcation(frame_probs, outcome):
    """
    Run M3+M6 bifurcation detection on a play.

    Args:
        frame_probs: DataFrame with columns [frame, p_complete, p_incomplete, p_interception]
        outcome: str, the actual outcome ('complete', 'incomplete', 'interception')

    Returns:
        dict with all bifurcation metrics
    """
    outcome_col = f'p_{outcome}'
    other_cols = [c for c in ['p_complete', 'p_incomplete', 'p_interception'] if c != outcome_col]

    n_frames = len(frame_probs)

    # Run M3 and M6 methods
    m3_frame = method_confidence_threshold(frame_probs, outcome)
    m6_frame = method_zscore_breakout(frame_probs, outcome)
    combo_frame = combo_bifurcation(frame_probs, outcome)

    # Run engine for drama metrics
    engine = BifurcationEngine(frame_probs, outcome)
    engine_results = engine.run_all()

    # Use combo as primary bifurcation frame
    bif_frame = combo_frame

    # Timing metrics
    bif_timing_m3 = m3_frame / max(n_frames - 1, 1)
    bif_timing_m6 = m6_frame / max(n_frames - 1, 1)
    bif_timing_combo = combo_frame / max(n_frames - 1, 1)

    # Probability at key frames
    prob_at_release = frame_probs[outcome_col].iloc[0]
    prob_at_bifurcation = frame_probs[outcome_col].iloc[bif_frame]
    prob_swing = prob_at_bifurcation - prob_at_release

    # Margin at bifurcation
    max_other_at_bif = frame_probs[other_cols].iloc[bif_frame].max()
    margin_at_bifurcation = prob_at_bifurcation - max_other_at_bif

    # Pre-bifurcation leader (at release)
    release_probs = {
        'complete': frame_probs['p_complete'].iloc[0],
        'incomplete': frame_probs['p_incomplete'].iloc[0],
        'interception': frame_probs['p_interception'].iloc[0]
    }
    pre_bif_leader = max(release_probs, key=release_probs.get)

    # Check for lead changes
    def get_leader_at_frame(frame_idx):
        probs = {
            'complete': frame_probs['p_complete'].iloc[frame_idx],
            'incomplete': frame_probs['p_incomplete'].iloc[frame_idx],
            'interception': frame_probs['p_interception'].iloc[frame_idx]
        }
        return max(probs, key=probs.get)

    leaders = [get_leader_at_frame(i) for i in range(n_frames)]
    had_lead_change = len(set(leaders)) > 1

    # Drama score (simplified)
    drama_score = (1 - bif_timing_combo) + (1 if had_lead_change else 0) + (1 - max(0, margin_at_bifurcation))

    return {
        'total_frames': n_frames,

        # M3 (Confidence Threshold) results
        'bifurcation_frame_m3': m3_frame,
        'bifurcation_timing_m3': bif_timing_m3,

        # M6 (Z-Score Breakout) results
        'bifurcation_frame_m6': m6_frame,
        'bifurcation_timing_m6': bif_timing_m6,

        # Combo model (primary)
        'bifurcation_frame_combo': combo_frame,
        'bifurcation_timing_combo': bif_timing_combo,

        # Primary method metrics (using combo)
        'bifurcation_frame': bif_frame,
        'bifurcation_timing': bif_timing_combo,

        # Probability metrics
        'probability_at_release': prob_at_release,
        'probability_at_bifurcation': prob_at_bifurcation,
        'probability_swing': prob_swing,
        'margin_at_bifurcation': margin_at_bifurcation,

        # Lead change analysis
        'pre_bifurcation_leader': pre_bif_leader,
        'had_lead_change': had_lead_change,

        # Drama scores
        'drama_score': drama_score,
        'drama_integral': engine_results['drama_integral'],

        # Entropy metrics
        'entropy_at_release': engine_results['entropy_at_release'],
        'entropy_at_end': engine_results['entropy_at_end'],
        'entropy_drop': engine_results['entropy_drop'],
    }


def analyze_all_plays(max_plays=None):
    """
    Run bifurcation analysis on all plays.

    Returns:
        DataFrame with one row per play containing all bifurcation metrics
    """
    print("Loading model...")
    model = lgb.Booster(model_file=str(MODEL_DIR / "completion_model.lgb"))

    with open(MODEL_DIR / "feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)

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

                # Get outcome
                supp_file = play_folder / "supplementary.csv"
                if supp_file.exists():
                    supp = pd.read_csv(supp_file)
                    if len(supp) > 0 and supp['pass_result'].iloc[0] in ['C', 'I', 'IN']:
                        play_folders.append({
                            'path': play_folder,
                            'week': week_folder.name,
                            'game': game_folder.name,
                            'play_id': play_folder.name,
                            'outcome_code': supp['pass_result'].iloc[0],
                            'game_id': supp['game_id'].iloc[0],
                            'play_id_num': supp['play_id'].iloc[0]
                        })

    print(f"Found {len(play_folders)} plays to analyze")

    if max_plays:
        play_folders = play_folders[:max_plays]

    # Analyze each play
    results = []
    failed = 0

    for play_info in tqdm(play_folders, desc="Analyzing bifurcation"):
        try:
            # Get probabilities
            frame_probs = get_play_probabilities(play_info['path'], model, feature_names)

            if frame_probs is None or len(frame_probs) < 2:
                failed += 1
                continue

            # Map outcome code to name
            outcome_map = {'C': 'complete', 'I': 'incomplete', 'IN': 'interception'}
            outcome = outcome_map[play_info['outcome_code']]

            # Detect bifurcation
            bif_metrics = detect_bifurcation(frame_probs, outcome)

            # Combine with play info
            result = {
                'play_folder': play_info['play_id'],
                'week': play_info['week'],
                'game': play_info['game'],
                'game_id': play_info['game_id'],
                'play_id': play_info['play_id_num'],
                'outcome': outcome,
                **bif_metrics
            }
            results.append(result)

        except Exception as e:
            failed += 1
            continue

    print(f"\nAnalyzed {len(results)} plays successfully, {failed} failed")

    df = pd.DataFrame(results)
    return df


def print_summary_statistics(df):
    """Print summary statistics for M3+M6 bifurcation analysis."""
    print("\n" + "="*80)
    print("BIFURCATION ANALYSIS SUMMARY (M3 + M6 COMBO MODEL)")
    print("="*80)

    print(f"\nTotal plays analyzed: {len(df)}")
    print(f"Outcome distribution:")
    print(df['outcome'].value_counts().to_string())

    # Distribution of bifurcation timing
    print("\n" + "-"*60)
    print("BIFURCATION TIMING DISTRIBUTION")
    print("-"*60)

    for method, name in [('m3', 'Confidence Threshold'), ('m6', 'Z-Score Breakout'), ('combo', 'COMBO MODEL')]:
        col = f'bifurcation_timing_{method}'
        early = (df[col] < 0.33).mean() * 100
        mid = ((df[col] >= 0.33) & (df[col] < 0.67)).mean() * 100
        late = (df[col] >= 0.67).mean() * 100
        print(f"\n{name}:")
        print(f"  Early (first 33%): {early:.1f}%")
        print(f"  Mid (33-67%): {mid:.1f}%")
        print(f"  Late (final 33%): {late:.1f}%")
        print(f"  Mean timing: {df[col].mean():.3f}")
        print(f"  Median timing: {df[col].median():.3f}")

    # Method agreement
    print("\n" + "-"*60)
    print("M3 vs M6 AGREEMENT")
    print("-"*60)

    frame_diff = np.abs(df['bifurcation_frame_m3'] - df['bifurcation_frame_m6'])
    print(f"\nFrame difference between M3 and M6:")
    print(f"  Mean: {frame_diff.mean():.1f} frames")
    print(f"  Median: {frame_diff.median():.1f} frames")
    print(f"  Agree within 2 frames: {(frame_diff <= 2).mean()*100:.1f}%")
    print(f"  Agree within 5 frames: {(frame_diff <= 5).mean()*100:.1f}%")

    # Timing by outcome
    print("\n" + "-"*60)
    print("TIMING BY OUTCOME (COMBO MODEL)")
    print("-"*60)

    for outcome in ['complete', 'incomplete', 'interception']:
        subset = df[df['outcome'] == outcome]
        print(f"\n{outcome.upper()} ({len(subset)} plays):")
        print(f"  Mean bifurcation timing: {subset['bifurcation_timing_combo'].mean():.3f}")
        print(f"  Mean probability at release: {subset['probability_at_release'].mean():.3f}")
        print(f"  Mean probability swing: {subset['probability_swing'].mean():.3f}")
        print(f"  Had lead change: {subset['had_lead_change'].mean()*100:.1f}%")

    # Lead changes
    print("\n" + "-"*60)
    print("LEAD CHANGE ANALYSIS")
    print("-"*60)

    lead_changes = df['had_lead_change'].mean() * 100
    print(f"\nPlays with lead change: {lead_changes:.1f}%")

    for outcome in ['complete', 'incomplete', 'interception']:
        subset = df[df['outcome'] == outcome]
        lc_pct = subset['had_lead_change'].mean() * 100
        print(f"  {outcome}: {lc_pct:.1f}%")

    # Drama score
    print("\n" + "-"*60)
    print("DRAMA SCORE DISTRIBUTION")
    print("-"*60)

    print(f"\nDrama score stats:")
    print(f"  Mean: {df['drama_score'].mean():.3f}")
    print(f"  Median: {df['drama_score'].median():.3f}")
    print(f"  Std: {df['drama_score'].std():.3f}")

    # Top 5 most dramatic plays
    print("\nTop 5 most dramatic plays:")
    top_drama = df.nlargest(5, 'drama_score')[['play_folder', 'outcome', 'drama_score',
                                                'bifurcation_timing_combo', 'had_lead_change',
                                                'probability_swing']]
    print(top_drama.to_string(index=False))


def create_visualizations(df):
    """Create visualizations for M3+M6 bifurcation analysis."""

    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Histogram - bifurcation timing for M3, M6, and Combo
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    methods = [('m3', 'M3: Confidence Threshold'),
               ('m6', 'M6: Z-Score Breakout'),
               ('combo', 'COMBO MODEL')]

    for ax, (method, title) in zip(axes, methods):
        col = f'bifurcation_timing_{method}'

        for outcome, color in [('complete', 'green'), ('incomplete', 'red'), ('interception', 'purple')]:
            subset = df[df['outcome'] == outcome][col]
            ax.hist(subset, bins=20, alpha=0.5, color=color, label=outcome.capitalize(), density=True)

        ax.axvline(0.33, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0.67, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Bifurcation Timing (0=release, 1=catch)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, 1)

    plt.suptitle('Bifurcation Timing Distribution by Method', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(BASE_DIR / "bifurcation_timing_histograms.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: bifurcation_timing_histograms.png")

    # 2. M3 vs M6 comparison scatter
    fig, ax = plt.subplots(figsize=(8, 8))

    for outcome, color, marker in [('complete', 'green', 'o'),
                                    ('incomplete', 'red', 's'),
                                    ('interception', 'purple', '^')]:
        subset = df[df['outcome'] == outcome]
        ax.scatter(subset['bifurcation_timing_m3'],
                  subset['bifurcation_timing_m6'],
                  c=color, marker=marker, alpha=0.4, s=30, label=outcome.capitalize())

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect agreement')
    ax.set_xlabel('M3 (Confidence) Timing')
    ax.set_ylabel('M6 (Z-Score) Timing')
    ax.set_title('M3 vs M6 Bifurcation Timing Comparison')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(BASE_DIR / "bifurcation_m3_vs_m6.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: bifurcation_m3_vs_m6.png")

    # 3. Drama score distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df['drama_score'], bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0].axvline(df['drama_score'].median(), color='red', linestyle='--',
                    label=f'Median: {df["drama_score"].median():.2f}')
    axes[0].set_xlabel('Drama Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Drama Score Distribution')
    axes[0].legend()

    for outcome, color in [('complete', 'green'), ('incomplete', 'red'), ('interception', 'purple')]:
        subset = df[df['outcome'] == outcome]['drama_score']
        axes[1].hist(subset, bins=20, alpha=0.5, color=color, label=f'{outcome.capitalize()} (n={len(subset)})', density=True)

    axes[1].set_xlabel('Drama Score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Drama Score by Outcome')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(BASE_DIR / "bifurcation_drama_score.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: bifurcation_drama_score.png")

    # 4. Timing by outcome boxplot (Combo model)
    fig, ax = plt.subplots(figsize=(10, 6))

    outcome_order = ['complete', 'incomplete', 'interception']
    colors = ['green', 'red', 'purple']

    data_to_plot = [df[df['outcome'] == outcome]['bifurcation_timing_combo'] for outcome in outcome_order]
    bp = ax.boxplot(data_to_plot, labels=[o.capitalize() for o in outcome_order], patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel('Bifurcation Timing (0=release, 1=catch)')
    ax.set_title('Bifurcation Timing by Outcome (Combo Model: M3+M6)')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(BASE_DIR / "bifurcation_by_outcome_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: bifurcation_by_outcome_boxplot.png")


def create_example_trajectory_plots(df, n_examples=2):
    """Create example trajectory plots with bifurcation points marked."""

    print("\nLoading model for example plots...")
    model = lgb.Booster(model_file=str(MODEL_DIR / "completion_model.lgb"))
    with open(MODEL_DIR / "feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    outcomes = ['complete', 'incomplete', 'interception']

    for row, outcome in enumerate(outcomes):
        outcome_plays = df[df['outcome'] == outcome].sort_values('drama_score', ascending=False)

        for col in range(n_examples):
            ax = axes[row, col]

            if col == 0:
                play_row = outcome_plays.iloc[0]  # Most dramatic
            else:
                play_row = outcome_plays.iloc[len(outcome_plays)//2]  # Median

            play_path = ORGANIZED_DIR / play_row['week'] / play_row['game'] / play_row['play_folder']
            frame_probs = get_play_probabilities(play_path, model, feature_names)

            if frame_probs is None:
                ax.set_title(f"Failed to load play")
                continue

            frames = frame_probs['frame'].values

            # Plot probability trajectories
            ax.plot(frames, frame_probs['p_complete'], '-', color='green', linewidth=2, label='P(Complete)')
            ax.plot(frames, frame_probs['p_incomplete'], '-', color='red', linewidth=2, label='P(Incomplete)')
            ax.plot(frames, frame_probs['p_interception'], '-', color='purple', linewidth=2, label='P(Interception)')

            # Mark bifurcation points
            bif_points = [
                (play_row['bifurcation_frame_m3'], 'M3', 'o', 'blue'),
                (play_row['bifurcation_frame_m6'], 'M6', 's', 'orange'),
                (play_row['bifurcation_frame_combo'], 'COMBO', '^', 'black')
            ]

            y_positions = [0.92, 0.86, 0.80]
            for (bif_f, label, marker, color), y_pos in zip(bif_points, y_positions):
                bif_f = int(bif_f)
                if bif_f < len(frame_probs):
                    ax.axvline(bif_f, color=color, linestyle='--', alpha=0.5)
                    ax.scatter([bif_f], [y_pos], marker=marker, s=100, c=color,
                              edgecolors='black', linewidths=1, label=f'{label}: frame {bif_f}', zorder=5)

            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlim(-0.5, len(frames) - 0.5)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Probability')

            drama_label = "High Drama" if col == 0 else "Typical"
            ax.set_title(f'{outcome.upper()} ({drama_label})\n'
                        f'Drama: {play_row["drama_score"]:.2f}, Swing: {play_row["probability_swing"]:.2f}')

            if row == 0 and col == 0:
                ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

    plt.suptitle('Example Probability Trajectories with Bifurcation Points\n'
                 '(M3=Confidence Threshold, M6=Z-Score Breakout, COMBO=M3+M6)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(BASE_DIR / "bifurcation_example_trajectories.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: bifurcation_example_trajectories.png")


def main(max_plays=None, skip_analysis=False):
    """Main bifurcation analysis pipeline."""

    output_file = BASE_DIR / "bifurcation_results.csv"

    if skip_analysis and output_file.exists():
        print("Loading existing bifurcation results...")
        df = pd.read_csv(output_file)
    else:
        # Run analysis
        df = analyze_all_plays(max_plays=max_plays)

        # Save results
        df.to_csv(output_file, index=False)
        print(f"\nSaved results to: {output_file}")

    # Print summary
    print_summary_statistics(df)

    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    create_visualizations(df)
    create_example_trajectory_plots(df)

    return df


if __name__ == "__main__":
    import sys

    max_plays = None
    skip = False

    if "--test" in sys.argv:
        max_plays = 100

    if "--skip-analysis" in sys.argv:
        skip = True

    df = main(max_plays=max_plays, skip_analysis=skip)
