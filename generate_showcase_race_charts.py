"""
Generate race charts for all showcase bifurcation plays.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import pickle
from pathlib import Path
from scipy.signal import savgol_filter

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
ORGANIZED_DIR = BASE_DIR / "organized_plays"
OUTPUT_DIR = BASE_DIR / "showcase_race_charts"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Showcase plays to generate charts for
SHOWCASE_PLAYS = [
    {
        "play": "Week_03/PHI_vs_TB/024_Q2_048_PHI",
        "title": "Jalen Hurts INT by Devin White",
        "subtitle": "Week 3: PHI @ TB (Q2, 0:48)",
        "outcome": "interception"
    },
    {
        "play": "Week_16/BAL_vs_SF/007_Q1_1147_SF",
        "title": "Kyle Hamilton Pick on Brock Purdy",
        "subtitle": "Week 16: BAL @ SF (Q1, 11:47)",
        "outcome": "interception"
    },
    {
        "play": "Week_08/ATL_vs_TEN/027_Q3_146_TEN",
        "title": "Will Levis 61-yard TD to DeAndre Hopkins",
        "subtitle": "Week 8: ATL @ TEN (Q3, 1:46)",
        "outcome": "complete"
    },
    {
        "play": "Week_14/LA_vs_BAL/016_Q2_948_BAL",
        "title": "Lamar Jackson 46-yard TD to OBJ",
        "subtitle": "Week 14: LA @ BAL (Q2, 9:48)",
        "outcome": "complete"
    },
    {
        "play": "Week_08/NYJ_vs_NYG/008_Q1_325_NYJ",
        "title": "Zach Wilson to Garrett Wilson (Incomplete)",
        "subtitle": "Week 8: NYJ @ NYG (Q1, 3:25)",
        "outcome": "incomplete"
    },
    {
        "play": "Week_15/CHI_vs_CLE/013_Q2_1326_CLE",
        "title": "Joe Flacco INT by Eddie Jackson",
        "subtitle": "Week 15: CHI @ CLE (Q2, 13:26)",
        "outcome": "interception"
    }
]


def load_model():
    """Load the completion probability model."""
    model = lgb.Booster(model_file=str(MODEL_DIR / "completion_model.lgb"))
    with open(MODEL_DIR / "feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names


def get_play_probabilities(play_path, model, feature_names):
    """Get frame-by-frame probabilities for a play."""
    from frame_features import calculate_frame_features

    features = calculate_frame_features(play_path)
    if features is None or len(features) == 0:
        return None

    X = features[[col for col in feature_names if col in features.columns]].copy()
    X = X.fillna(X.median())

    probs = model.predict(X)

    # Smooth probabilities
    def smooth(p, window=5, poly=2):
        if len(p) < window:
            window = len(p) if len(p) % 2 == 1 else len(p) - 1
            window = max(3, window)
        poly = min(poly, window - 1)
        return np.clip(savgol_filter(p, window, poly), 0, 1)

    return pd.DataFrame({
        'frame': range(len(probs)),
        'p_incomplete': smooth(probs[:, 0]),
        'p_complete': smooth(probs[:, 1]),
        'p_interception': smooth(probs[:, 2])
    })


def find_bifurcation_frame(probs_df, actual_outcome):
    """Find the bifurcation frame using time_weighted method."""
    n_frames = len(probs_df)
    outcome_col = f'p_{actual_outcome}'
    outcome_probs = probs_df[outcome_col].values

    p_inc = probs_df['p_incomplete'].values
    p_comp = probs_df['p_complete'].values
    p_int = probs_df['p_interception'].values

    alpha = 0.5
    best_score = -1
    bif_frame = 0

    for i in range(n_frames):
        probs_at_frame = [p_inc[i], p_comp[i], p_int[i]]
        leader_idx = np.argmax(probs_at_frame)
        leader = ['incomplete', 'complete', 'interception'][leader_idx]

        if leader == actual_outcome:
            outcome_prob = outcome_probs[i]
            time_factor = (1 - i / n_frames) ** alpha
            score = outcome_prob * time_factor

            if score > best_score:
                best_score = score
                bif_frame = i

    return bif_frame


def create_race_chart(probs_df, play_info, bif_frame, output_path):
    """Create a race chart visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))

    frames = probs_df['frame'].values
    n_frames = len(frames)

    # Colors
    colors = {
        'complete': '#2ecc71',      # Green
        'incomplete': '#e74c3c',    # Red
        'interception': '#9b59b6'   # Purple
    }

    # Plot probability lines
    ax.plot(frames, probs_df['p_complete'] * 100,
            color=colors['complete'], linewidth=3, label='Complete', alpha=0.9)
    ax.plot(frames, probs_df['p_incomplete'] * 100,
            color=colors['incomplete'], linewidth=3, label='Incomplete', alpha=0.9)
    ax.plot(frames, probs_df['p_interception'] * 100,
            color=colors['interception'], linewidth=3, label='Interception', alpha=0.9)

    # Mark bifurcation point
    bif_probs = [
        probs_df['p_incomplete'].iloc[bif_frame] * 100,
        probs_df['p_complete'].iloc[bif_frame] * 100,
        probs_df['p_interception'].iloc[bif_frame] * 100
    ]

    # Vertical line at bifurcation
    ax.axvline(x=bif_frame, color='gold', linestyle='--', linewidth=2, alpha=0.8)

    # Add bifurcation marker
    outcome = play_info['outcome']
    outcome_prob_at_bif = probs_df[f'p_{outcome}'].iloc[bif_frame] * 100
    ax.scatter([bif_frame], [outcome_prob_at_bif],
               color='gold', s=200, zorder=5, edgecolors='black', linewidths=2)

    # Annotate bifurcation
    bif_pct = (bif_frame / n_frames) * 100
    ax.annotate(f'BIFURCATION\n{bif_pct:.1f}% into play\n{outcome_prob_at_bif:.1f}% {outcome}',
                xy=(bif_frame, outcome_prob_at_bif),
                xytext=(bif_frame + n_frames * 0.1, outcome_prob_at_bif + 10),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gold', lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.3))

    # Shade the actual outcome
    outcome_color = colors[outcome]
    ax.fill_between(frames, 0, probs_df[f'p_{outcome}'] * 100,
                    color=outcome_color, alpha=0.1)

    # Styling
    ax.set_xlim(0, n_frames - 1)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Frame (Ball in Air)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability (%)', fontsize=14, fontweight='bold')

    # Title
    ax.set_title(f"{play_info['title']}\n{play_info['subtitle']}",
                 fontsize=16, fontweight='bold', pad=20)

    # Add outcome badge
    outcome_label = outcome.upper()
    badge_color = colors[outcome]
    ax.text(0.98, 0.98, f"ACTUAL: {outcome_label}",
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=badge_color, alpha=0.8, edgecolor='black'),
            color='white')

    # Legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')

    # Add frame markers
    ax.set_xticks(np.linspace(0, n_frames-1, min(11, n_frames)).astype(int))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path.name}")


def main():
    print("Loading model...")
    model, feature_names = load_model()

    print(f"\nGenerating race charts for {len(SHOWCASE_PLAYS)} plays...")
    print("=" * 60)

    for i, play_info in enumerate(SHOWCASE_PLAYS, 1):
        print(f"\n{i}. {play_info['title']}")

        play_path = ORGANIZED_DIR / play_info['play']

        if not play_path.exists():
            print(f"  ERROR: Play path not found: {play_path}")
            continue

        # Get probabilities
        probs_df = get_play_probabilities(play_path, model, feature_names)
        if probs_df is None:
            print(f"  ERROR: Could not calculate probabilities")
            continue

        # Find bifurcation frame
        bif_frame = find_bifurcation_frame(probs_df, play_info['outcome'])

        # Create output filename
        safe_title = play_info['title'].replace(' ', '_').replace('/', '-').replace('(', '').replace(')', '')
        output_path = OUTPUT_DIR / f"{i:02d}_{safe_title}.png"

        # Create race chart
        create_race_chart(probs_df, play_info, bif_frame, output_path)

    print("\n" + "=" * 60)
    print(f"All race charts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
