"""
Generate complete broadcast visualization package for a play.
Includes: race chart (transparent, with THE READ), NGS dots, analysis files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import lightgbm as lgb
import pickle
from pathlib import Path
from PIL import Image
from scipy.signal import savgol_filter
import glob

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
ORGANIZED_DIR = BASE_DIR / "organized_plays"


def load_model():
    """Load the completion probability model."""
    model = lgb.Booster(model_file=str(MODEL_DIR / "completion_model.lgb"))
    with open(MODEL_DIR / "feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names


def get_play_data(play_path, model, feature_names):
    """Get frame-by-frame probabilities AND raw features for a play."""
    from frame_features import calculate_frame_features

    features = calculate_frame_features(play_path)
    if features is None or len(features) == 0:
        return None, None

    X = features[[col for col in feature_names if col in features.columns]].copy()
    X = X.fillna(X.median())

    probs = model.predict(X)

    def smooth(p, window=5, poly=2):
        if len(p) < window:
            window = len(p) if len(p) % 2 == 1 else len(p) - 1
            window = max(3, window)
        poly = min(poly, window - 1)
        return np.clip(savgol_filter(p, window, poly), 0, 1)

    probs_df = pd.DataFrame({
        'frame': range(len(probs)),
        'p_incomplete': smooth(probs[:, 0]),
        'p_complete': smooth(probs[:, 1]),
        'p_interception': smooth(probs[:, 2]),
        'p_incomplete_raw': probs[:, 0],
        'p_complete_raw': probs[:, 1],
        'p_interception_raw': probs[:, 2]
    })

    features_df = X.copy()
    features_df['frame'] = range(len(X))

    return probs_df, features_df


def find_the_read(probs_df, actual_outcome):
    """Find THE READ frame using time_weighted method."""
    n_frames = len(probs_df)
    outcome_col = f'p_{actual_outcome}'
    alpha = 0.5
    best_score = -1
    bif_frame = 0

    for i in range(n_frames):
        probs_at_frame = [probs_df['p_incomplete'].iloc[i],
                          probs_df['p_complete'].iloc[i],
                          probs_df['p_interception'].iloc[i]]
        leader_idx = np.argmax(probs_at_frame)
        leader = ['incomplete', 'complete', 'interception'][leader_idx]

        if leader == actual_outcome:
            outcome_prob = probs_df[outcome_col].iloc[i]
            time_factor = (1 - i / n_frames) ** alpha
            score = outcome_prob * time_factor

            if score > best_score:
                best_score = score
                bif_frame = i

    return bif_frame


# ============== RACE CHART GENERATION ==============

def create_race_chart_frame(probs_df, current_frame, n_frames, output_path,
                            bif_frame, actual_outcome):
    """Create a single transparent race chart frame with THE READ."""
    fig, ax = plt.subplots(figsize=(10, 6))

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    colors = {
        'complete': '#00BFFF',
        'incomplete': '#FF4444',
        'interception': '#FF00FF'
    }

    frames = probs_df['frame'].values[:current_frame + 1]

    # Plot lines
    ax.plot(frames, probs_df['p_complete'].values[:current_frame + 1] * 100,
            color=colors['complete'], linewidth=4, label='Complete', alpha=1.0)
    ax.plot(frames, probs_df['p_incomplete'].values[:current_frame + 1] * 100,
            color=colors['incomplete'], linewidth=4, label='Incomplete', alpha=1.0)
    ax.plot(frames, probs_df['p_interception'].values[:current_frame + 1] * 100,
            color=colors['interception'], linewidth=4, label='Interception', alpha=1.0)

    # Current markers
    ax.scatter([current_frame], [probs_df['p_complete'].iloc[current_frame] * 100],
               color=colors['complete'], s=150, zorder=5, edgecolors='black', linewidths=2)
    ax.scatter([current_frame], [probs_df['p_incomplete'].iloc[current_frame] * 100],
               color=colors['incomplete'], s=150, zorder=5, edgecolors='black', linewidths=2)
    ax.scatter([current_frame], [probs_df['p_interception'].iloc[current_frame] * 100],
               color=colors['interception'], s=150, zorder=5, edgecolors='black', linewidths=2)

    # THE READ marker
    if current_frame >= bif_frame:
        bif_prob = probs_df[f'p_{actual_outcome}'].iloc[bif_frame] * 100
        ax.axvline(x=bif_frame, color='gold', linestyle='--', linewidth=3, alpha=0.9)
        ax.scatter([bif_frame], [bif_prob], marker='*', s=400,
                   color='gold', edgecolors='black', linewidths=1.5, zorder=10)
        bif_pct = (bif_frame / n_frames) * 100
        ax.annotate(f'THE READ\n{bif_pct:.0f}% into play',
                    xy=(bif_frame, bif_prob),
                    xytext=(bif_frame + 2, bif_prob + 12),
                    fontsize=10, fontweight='bold', color='black',
                    ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.9, edgecolor='black'),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Styling
    ax.set_xlim(0, n_frames - 1)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Frame', fontsize=13, fontweight='bold', color='black')
    ax.set_ylabel('Probability (%)', fontsize=13, fontweight='bold', color='black')

    ax.tick_params(colors='black', labelsize=11)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
                       ncol=3, fontsize=12, framealpha=0.9, facecolor='white',
                       labelcolor='black', edgecolor='black')

    ax.grid(True, alpha=0.5, linestyle='--', color='black', linewidth=0.8)
    ax.set_xticks(np.linspace(0, n_frames-1, min(8, n_frames)).astype(int))

    # Probability labels
    y_pos = [
        probs_df['p_complete'].iloc[current_frame] * 100,
        probs_df['p_incomplete'].iloc[current_frame] * 100,
        probs_df['p_interception'].iloc[current_frame] * 100
    ]
    for y, color in zip(y_pos, [colors['complete'], colors['incomplete'], colors['interception']]):
        ax.text(n_frames + 0.5, y, f'{y:.1f}%',
                fontsize=11, fontweight='bold', color=color,
                verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=color))

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', transparent=True)
    plt.close()


# ============== NGS DOTS GENERATION ==============

def draw_football_field(ax, x_min, x_max, y_min=0, y_max=53.3):
    """Draw a football field section."""
    ax.set_facecolor('#2e7d32')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    for yard in range(0, 101, 5):
        if x_min <= yard <= x_max:
            ax.axvline(x=yard, color='white', linewidth=1, alpha=0.5)

    for yard in range(0, 101, 10):
        if x_min <= yard <= x_max:
            ax.axvline(x=yard, color='white', linewidth=2, alpha=0.7)
            yard_num = yard if yard <= 50 else 100 - yard
            if yard_num > 0:
                ax.text(yard, 5, str(yard_num), fontsize=14, color='white',
                       ha='center', va='center', fontweight='bold', alpha=0.6)
                ax.text(yard, 48.3, str(yard_num), fontsize=14, color='white',
                       ha='center', va='center', fontweight='bold', alpha=0.6)

    for yard in range(0, 101, 1):
        if x_min <= yard <= x_max:
            ax.plot([yard, yard], [22.9, 23.5], color='white', linewidth=0.5, alpha=0.3)
            ax.plot([yard, yard], [29.8, 30.4], color='white', linewidth=0.5, alpha=0.3)

    ax.axhline(y=0, color='white', linewidth=3)
    ax.axhline(y=53.3, color='white', linewidth=3)
    ax.set_aspect('equal')
    ax.axis('off')


def create_ngs_frame(tracking_df, frame_id, ball_land_x, ball_land_y, output_path,
                     probs=None, bif_frame=None, current_frame_idx=None,
                     fixed_bounds=None, actual_outcome='complete',
                     ball_start_x=None, ball_start_y=None, total_frames=None):
    """Create a single NGS-style frame."""
    frame_data = tracking_df[tracking_df['frame_id'] == frame_id]
    if len(frame_data) == 0:
        return

    if fixed_bounds:
        x_min, x_max, y_min, y_max = fixed_bounds
    else:
        all_x = np.append(frame_data['x'].values, ball_land_x)
        all_y = np.append(frame_data['y'].values, ball_land_y)
        x_center = (all_x.min() + all_x.max()) / 2
        y_center = (all_y.min() + all_y.max()) / 2
        x_range = max(all_x.max() - all_x.min() + 15, 40)
        y_range = max(all_y.max() - all_y.min() + 10, 30)
        x_min = max(0, x_center - x_range/2)
        x_max = min(120, x_center + x_range/2)
        y_min = max(0, y_center - y_range/2)
        y_max = min(53.3, y_center + y_range/2)

    fig, ax = plt.subplots(figsize=(12, 8))
    draw_football_field(ax, x_min, x_max, y_min, y_max)

    colors = {
        'Targeted Receiver': '#00BFFF',
        'Other Route Runner': '#4FC3F7',
        'Passer': '#FFD700',
        'Defensive Coverage': '#FF4444',
    }

    for _, player in frame_data.iterrows():
        role = player['player_role']
        color = colors.get(role, 'gray')
        if role == 'Targeted Receiver':
            size, edge_width, zorder = 400, 3, 10
        elif role == 'Passer':
            size, edge_width, zorder = 350, 2, 8
        else:
            size, edge_width, zorder = 250, 1.5, 5

        ax.scatter(player['x'], player['y'], s=size, c=color,
                  edgecolors='white', linewidths=edge_width, zorder=zorder)

        if role == 'Targeted Receiver' and not pd.isna(player.get('dir', np.nan)):
            dir_rad = np.radians(90 - player['dir'])
            speed = player.get('s', 5)
            dx = np.cos(dir_rad) * speed * 0.3
            dy = np.sin(dir_rad) * speed * 0.3
            ax.arrow(player['x'], player['y'], dx, dy,
                    head_width=0.8, head_length=0.4, fc=color, ec='white', linewidth=1.5, zorder=11)

    # Ball target (catch point)
    circle1 = plt.Circle((ball_land_x, ball_land_y), 1.5, fill=False,
                         color='yellow', linewidth=3, zorder=15)
    circle2 = plt.Circle((ball_land_x, ball_land_y), 0.5, fill=True,
                         color='yellow', zorder=15)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Ball in flight - interpolate position from start to landing point
    if ball_start_x is not None and ball_start_y is not None and total_frames is not None:
        # Calculate ball position based on frame progress
        progress = current_frame_idx / max(total_frames - 1, 1)
        ball_x = ball_start_x + (ball_land_x - ball_start_x) * progress
        ball_y = ball_start_y + (ball_land_y - ball_start_y) * progress

        # Draw the football as a brown ellipse
        football = plt.Circle((ball_x, ball_y), 0.8, fill=True,
                              color='#8B4513', zorder=20)  # Saddle brown
        ax.add_patch(football)
        # White laces
        ax.plot([ball_x - 0.3, ball_x + 0.3], [ball_y, ball_y],
               color='white', linewidth=2, zorder=21)

        # Trail showing ball path (faded line from start to current)
        ax.plot([ball_start_x, ball_x], [ball_start_y, ball_y],
               color='#8B4513', linestyle='-', linewidth=2, alpha=0.4, zorder=3)

    receiver = frame_data[frame_data['player_role'] == 'Targeted Receiver']
    if len(receiver) > 0:
        rx, ry = receiver['x'].iloc[0], receiver['y'].iloc[0]
        ax.plot([rx, ball_land_x], [ry, ball_land_y],
               color='yellow', linestyle='--', linewidth=2, alpha=0.7, zorder=4)

    legend_elements = [
        plt.scatter([], [], s=150, c='#00BFFF', edgecolors='white', linewidths=2, label='Receiver'),
        plt.scatter([], [], s=120, c='#FF4444', edgecolors='white', linewidths=1.5, label='Defender'),
        plt.scatter([], [], s=100, c='yellow', marker='o', label='Catch Point'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
             facecolor='black', edgecolor='white', labelcolor='white', framealpha=0.8)

    ax.text(x_min + 1, y_max - 2, f'Frame {current_frame_idx + 1}',
           fontsize=12, color='white', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # THE READ marker (no percentages)
    if probs is not None and bif_frame is not None and current_frame_idx >= bif_frame:
        ax.text(x_min + 1, y_min + 2, 'THE READ',
               fontsize=11, color='black', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()


# ============== MAIN GENERATION ==============

def generate_full_broadcast_viz(play_path, output_dir, play_name, actual_outcome):
    """Generate complete broadcast visualization package."""
    import imageio.v2 as imageio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"Generating broadcast viz for: {play_name}")
    print(f"Outcome: {actual_outcome}")
    print(f"=" * 60)

    # Load model and get probabilities
    print("\n1. Loading model and calculating probabilities...")
    model, feature_names = load_model()
    probs_df, features_df = get_play_data(play_path, model, feature_names)

    if probs_df is None:
        print("ERROR: Could not calculate probabilities")
        return

    n_frames = len(probs_df)
    print(f"   Total frames: {n_frames}")

    # Find THE READ
    bif_frame = find_the_read(probs_df, actual_outcome)
    bif_pct = (bif_frame / n_frames) * 100
    print(f"   THE READ at frame {bif_frame} ({bif_pct:.1f}% into play)")

    # Save probability data
    probs_df.to_csv(output_dir / "probabilities.csv", index=False)

    # Generate race chart frames
    print("\n2. Generating race chart frames...")
    race_dir = output_dir / "race_chart_frames"
    race_dir.mkdir(exist_ok=True)

    for i in range(n_frames):
        output_path = race_dir / f"frame_{i:03d}.png"
        create_race_chart_frame(probs_df, i, n_frames, output_path, bif_frame, actual_outcome)
    print(f"   Saved {n_frames} frames to {race_dir}")

    # Create race chart animations
    print("   Creating animations...")
    frame_files = sorted(glob.glob(str(race_dir / "frame_*.png")))
    images = [Image.open(f) for f in frame_files]

    images[0].save(output_dir / "race_chart_THE_READ.gif",
                   save_all=True, append_images=images[1:], duration=200, loop=0, disposal=2)

    writer = imageio.get_writer(str(output_dir / "race_chart_THE_READ.webm"),
                                fps=10, codec='libvpx-vp9', pixelformat='yuva420p')
    for f in frame_files:
        writer.append_data(imageio.imread(f))
    writer.close()
    print("   Saved: race_chart_THE_READ.gif, race_chart_THE_READ.webm")

    # Generate NGS dots
    print("\n3. Generating NGS dots visualization...")
    tracking = pd.read_csv(play_path / 'all_players_tracking.csv')
    ball_in_air = tracking[tracking['ball_in_air'] == True]
    frames = sorted(ball_in_air['frame_id'].unique())

    ball_land_x = tracking['ball_land_x'].iloc[0]
    ball_land_y = tracking['ball_land_y'].iloc[0]
    print(f"   Ball landing point: ({ball_land_x:.1f}, {ball_land_y:.1f})")

    # Get passer position at ball release (frame before ball_in_air or first ball_in_air frame)
    first_frame = frames[0]

    # Try to get passer from frame just before ball is in air
    pre_throw_frame = tracking[tracking['frame_id'] == first_frame - 1]
    passer_data = pre_throw_frame[pre_throw_frame['player_role'] == 'Passer']

    if len(passer_data) == 0:
        # Try the first ball_in_air frame
        first_frame_data = ball_in_air[ball_in_air['frame_id'] == first_frame]
        passer_data = first_frame_data[first_frame_data['player_role'] == 'Passer']

    if len(passer_data) == 0:
        # Last resort: search all frames for passer
        all_passer = tracking[tracking['player_role'] == 'Passer']
        if len(all_passer) > 0:
            # Get passer position closest to ball release
            passer_data = all_passer[all_passer['frame_id'] <= first_frame].tail(1)

    if len(passer_data) > 0:
        ball_start_x = passer_data['x'].iloc[0]
        ball_start_y = passer_data['y'].iloc[0]
    else:
        # Fallback: estimate from receiver's starting position
        receiver_data = ball_in_air[ball_in_air['player_role'] == 'Targeted Receiver']
        if len(receiver_data) > 0:
            first_receiver = receiver_data[receiver_data['frame_id'] == first_frame]
            if len(first_receiver) > 0:
                ball_start_x = first_receiver['x'].iloc[0] - 15
                ball_start_y = 26.65  # Middle of field
            else:
                ball_start_x = ball_land_x - 20
                ball_start_y = 26.65
        else:
            ball_start_x = ball_land_x - 20
            ball_start_y = 26.65
    print(f"   Ball start point (passer): ({ball_start_x:.1f}, {ball_start_y:.1f})")

    ngs_dir = output_dir / "ngs_frames"
    ngs_dir.mkdir(exist_ok=True)

    # Calculate fixed bounds
    all_x = np.append(ball_in_air['x'].values, ball_land_x)
    all_x = np.append(all_x, ball_start_x)
    all_y = np.append(ball_in_air['y'].values, ball_land_y)
    all_y = np.append(all_y, ball_start_y)
    x_center = (all_x.min() + all_x.max()) / 2
    x_range = max(all_x.max() - all_x.min() + 15, 45)
    x_min = max(0, x_center - x_range/2)
    x_max = min(120, x_center + x_range/2)
    fixed_bounds = (x_min, x_max, 0, 53.3)

    total_ngs_frames = len(frames)
    for i, frame_id in enumerate(frames):
        output_path = ngs_dir / f"frame_{i:03d}.png"
        create_ngs_frame(ball_in_air, frame_id, ball_land_x, ball_land_y,
                        output_path, probs_df, bif_frame, i, fixed_bounds, actual_outcome,
                        ball_start_x, ball_start_y, total_ngs_frames)
    print(f"   Saved {len(frames)} frames to {ngs_dir}")

    # Create NGS animations
    print("   Creating animations...")
    frame_files = sorted(glob.glob(str(ngs_dir / "frame_*.png")))
    images = [Image.open(f) for f in frame_files]

    images[0].save(output_dir / "ngs_dots.gif",
                   save_all=True, append_images=images[1:], duration=200, loop=0)

    writer = imageio.get_writer(str(output_dir / "ngs_dots.webm"), fps=10, codec='libvpx-vp9')
    for f in frame_files:
        writer.append_data(imageio.imread(f))
    writer.close()
    print("   Saved: ngs_dots.gif, ngs_dots.webm")

    # Generate analysis files
    print("\n4. Generating analysis files...")

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = dict(zip(feature_names, importance))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:20]]

    importance_df = pd.DataFrame(sorted_features, columns=['feature', 'importance'])
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Frame analysis
    analysis_rows = []
    for frame in range(n_frames):
        row = {
            'frame': frame,
            'p_complete': probs_df['p_complete'].iloc[frame],
            'p_incomplete': probs_df['p_incomplete'].iloc[frame],
            'p_interception': probs_df['p_interception'].iloc[frame],
            'leader': ['incomplete', 'complete', 'interception'][
                np.argmax([probs_df['p_incomplete'].iloc[frame],
                          probs_df['p_complete'].iloc[frame],
                          probs_df['p_interception'].iloc[frame]])
            ]
        }
        for feat in top_features:
            if feat in features_df.columns:
                row[feat] = features_df[feat].iloc[frame]
        if frame > 0:
            row['delta_complete'] = probs_df['p_complete'].iloc[frame] - probs_df['p_complete'].iloc[frame-1]
            row['delta_incomplete'] = probs_df['p_incomplete'].iloc[frame] - probs_df['p_incomplete'].iloc[frame-1]
            row['delta_interception'] = probs_df['p_interception'].iloc[frame] - probs_df['p_interception'].iloc[frame-1]
        else:
            row['delta_complete'] = row['delta_incomplete'] = row['delta_interception'] = 0
        analysis_rows.append(row)

    analysis_df = pd.DataFrame(analysis_rows)
    analysis_df.to_csv(output_dir / "frame_analysis.csv", index=False)

    # Commentary notes
    summary_lines = [
        f"PLAY ANALYSIS: {play_name}",
        f"=" * 60,
        f"Actual Outcome: {actual_outcome.upper()}",
        f"Total Frames: {n_frames}",
        f"",
        f"THE READ: Frame {bif_frame} ({bif_pct:.1f}% into play)",
        f"",
        f"PROBABILITY SUMMARY:",
        f"  Start: Complete={probs_df['p_complete'].iloc[0]*100:.1f}%, "
        f"Incomplete={probs_df['p_incomplete'].iloc[0]*100:.1f}%, "
        f"INT={probs_df['p_interception'].iloc[0]*100:.1f}%",
        f"  End:   Complete={probs_df['p_complete'].iloc[-1]*100:.1f}%, "
        f"Incomplete={probs_df['p_incomplete'].iloc[-1]*100:.1f}%, "
        f"INT={probs_df['p_interception'].iloc[-1]*100:.1f}%",
        f"",
        f"KEY MOMENTS (biggest probability swings):",
    ]

    for frame in range(1, n_frames):
        delta_comp = abs(probs_df['p_complete'].iloc[frame] - probs_df['p_complete'].iloc[frame-1])
        delta_inc = abs(probs_df['p_incomplete'].iloc[frame] - probs_df['p_incomplete'].iloc[frame-1])
        delta_int = abs(probs_df['p_interception'].iloc[frame] - probs_df['p_interception'].iloc[frame-1])
        max_delta = max(delta_comp, delta_inc, delta_int)
        if max_delta > 0.05:
            summary_lines.append(
                f"  Frame {frame}: {max_delta*100:.1f}% swing - "
                f"Complete={probs_df['p_complete'].iloc[frame]*100:.1f}%, "
                f"Incomplete={probs_df['p_incomplete'].iloc[frame]*100:.1f}%, "
                f"INT={probs_df['p_interception'].iloc[frame]*100:.1f}%"
            )

    summary_lines.extend([f"", f"TOP 10 MOST IMPORTANT FEATURES:"])
    for feat, imp in sorted_features[:10]:
        summary_lines.append(f"  {feat}: {imp:.2f}")

    with open(output_dir / "commentary_notes.txt", 'w') as f:
        f.write('\n'.join(summary_lines))

    print("   Saved: feature_importance.csv, frame_analysis.csv, commentary_notes.txt")

    print(f"\n{'=' * 60}")
    print(f"DONE! All files saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Zach Wilson to Garrett Wilson Incomplete
    play_path = ORGANIZED_DIR / "Week_08/NYJ_vs_NYG/008_Q1_325_NYJ"
    output_dir = BASE_DIR / "broadcast_viz" / "Zach_Wilson_Incomplete"

    generate_full_broadcast_viz(play_path, output_dir,
                                "Zach Wilson to Garrett Wilson (Incomplete)",
                                "incomplete")
