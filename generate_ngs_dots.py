"""
Generate NFL Next Gen Stats style dot visualization for plays.
Shows player positions as dots on a field with ball trajectory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from PIL import Image
import glob

BASE_DIR = Path(__file__).parent
ORGANIZED_DIR = BASE_DIR / "organized_plays"


def draw_football_field(ax, x_min, x_max, y_min=0, y_max=53.3):
    """Draw a football field section."""
    # Field background - darker green
    ax.set_facecolor('#2e7d32')

    # Field boundaries
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Yard lines (every 5 yards)
    for yard in range(0, 101, 5):
        if x_min <= yard <= x_max:
            ax.axvline(x=yard, color='white', linewidth=1, alpha=0.5)

    # Major yard lines (every 10 yards) - thicker
    for yard in range(0, 101, 10):
        if x_min <= yard <= x_max:
            ax.axvline(x=yard, color='white', linewidth=2, alpha=0.7)
            # Yard numbers
            yard_num = yard if yard <= 50 else 100 - yard
            if yard_num > 0:
                ax.text(yard, 5, str(yard_num), fontsize=14, color='white',
                       ha='center', va='center', fontweight='bold', alpha=0.6)
                ax.text(yard, 48.3, str(yard_num), fontsize=14, color='white',
                       ha='center', va='center', fontweight='bold', alpha=0.6)

    # Hash marks (simplified)
    for yard in range(0, 101, 1):
        if x_min <= yard <= x_max:
            ax.plot([yard, yard], [22.9, 23.5], color='white', linewidth=0.5, alpha=0.3)
            ax.plot([yard, yard], [29.8, 30.4], color='white', linewidth=0.5, alpha=0.3)

    # Sidelines
    ax.axhline(y=0, color='white', linewidth=3)
    ax.axhline(y=53.3, color='white', linewidth=3)

    ax.set_aspect('equal')
    ax.axis('off')


def create_ngs_frame(tracking_df, frame_id, ball_land_x, ball_land_y, output_path,
                     probs=None, bif_frame=None, current_frame_idx=None,
                     fixed_bounds=None):
    """Create a single NGS-style frame."""

    frame_data = tracking_df[tracking_df['frame_id'] == frame_id]

    if len(frame_data) == 0:
        return

    # Use fixed bounds if provided (for consistent video frame sizes)
    if fixed_bounds:
        x_min, x_max, y_min, y_max = fixed_bounds
    else:
        # Determine field view bounds (zoom to action)
        all_x = frame_data['x'].values
        all_y = frame_data['y'].values

        # Include ball landing point in bounds calculation
        all_x = np.append(all_x, ball_land_x)
        all_y = np.append(all_y, ball_land_y)

        x_center = (all_x.min() + all_x.max()) / 2
        y_center = (all_y.min() + all_y.max()) / 2

        # Set view width (at least 40 yards)
        x_range = max(all_x.max() - all_x.min() + 15, 40)
        y_range = max(all_y.max() - all_y.min() + 10, 30)

        x_min = max(0, x_center - x_range/2)
        x_max = min(120, x_center + x_range/2)
        y_min = max(0, y_center - y_range/2)
        y_max = min(53.3, y_center + y_range/2)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw field
    draw_football_field(ax, x_min, x_max, y_min, y_max)

    # Color scheme
    colors = {
        'Targeted Receiver': '#00BFFF',  # Cyan - matches race chart
        'Other Route Runner': '#4FC3F7',  # Light blue
        'Passer': '#FFD700',  # Gold
        'Defensive Coverage': '#FF4444',  # Red
    }

    # Plot players
    for _, player in frame_data.iterrows():
        role = player['player_role']
        color = colors.get(role, 'gray')

        # Dot size based on role
        if role == 'Targeted Receiver':
            size = 400
            edge_width = 3
            zorder = 10
        elif role == 'Passer':
            size = 350
            edge_width = 2
            zorder = 8
        else:
            size = 250
            edge_width = 1.5
            zorder = 5

        ax.scatter(player['x'], player['y'], s=size, c=color,
                  edgecolors='white', linewidths=edge_width, zorder=zorder)

        # Direction arrow for targeted receiver
        if role == 'Targeted Receiver' and not pd.isna(player.get('dir', np.nan)):
            dir_rad = np.radians(90 - player['dir'])  # Convert to standard math angle
            speed = player.get('s', 5)
            dx = np.cos(dir_rad) * speed * 0.3
            dy = np.sin(dir_rad) * speed * 0.3
            ax.arrow(player['x'], player['y'], dx, dy,
                    head_width=0.8, head_length=0.4, fc=color, ec='white', linewidth=1.5, zorder=11)

    # Ball trajectory / landing point
    # Draw ball landing spot as a target
    circle1 = plt.Circle((ball_land_x, ball_land_y), 1.5, fill=False,
                         color='yellow', linewidth=3, zorder=15)
    circle2 = plt.Circle((ball_land_x, ball_land_y), 0.5, fill=True,
                         color='yellow', zorder=15)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Draw line from receiver to catch point
    receiver = frame_data[frame_data['player_role'] == 'Targeted Receiver']
    if len(receiver) > 0:
        rx, ry = receiver['x'].iloc[0], receiver['y'].iloc[0]
        ax.plot([rx, ball_land_x], [ry, ball_land_y],
               color='yellow', linestyle='--', linewidth=2, alpha=0.7, zorder=4)

    # Add legend
    legend_elements = [
        plt.scatter([], [], s=150, c='#00BFFF', edgecolors='white', linewidths=2, label='Receiver'),
        plt.scatter([], [], s=120, c='#FF4444', edgecolors='white', linewidths=1.5, label='Defender'),
        plt.scatter([], [], s=100, c='yellow', marker='o', label='Catch Point'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
             facecolor='black', edgecolor='white', labelcolor='white', framealpha=0.8)

    # Frame counter
    ax.text(x_min + 1, y_max - 2, f'Frame {current_frame_idx + 1}',
           fontsize=12, color='white', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # Add probability bar if provided
    if probs is not None and current_frame_idx is not None:
        p_comp = probs['p_complete'].iloc[current_frame_idx]
        p_inc = probs['p_incomplete'].iloc[current_frame_idx]
        p_int = probs['p_interception'].iloc[current_frame_idx]

        # Mini probability display
        prob_text = f"Complete: {p_comp*100:.0f}%  |  Incomplete: {p_inc*100:.0f}%  |  INT: {p_int*100:.0f}%"
        ax.text((x_min + x_max)/2, y_min + 2, prob_text,
               fontsize=11, color='white', fontweight='bold', ha='center',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8))

        # THE READ marker
        if bif_frame is not None and current_frame_idx >= bif_frame:
            ax.text(x_min + 1, y_min + 2, 'THE READ',
                   fontsize=11, color='black', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='#1a1a1a',
                transparent=False)
    plt.close()


def generate_ngs_visualization(play_path, output_dir, play_name):
    """Generate all NGS frames for a play."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    tracking = pd.read_csv(play_path / 'all_players_tracking.csv')

    # Get ball-in-air frames only
    ball_in_air = tracking[tracking['ball_in_air'] == True]
    frames = sorted(ball_in_air['frame_id'].unique())

    ball_land_x = tracking['ball_land_x'].iloc[0]
    ball_land_y = tracking['ball_land_y'].iloc[0]

    print(f"Play: {play_name}")
    print(f"Ball in air frames: {len(frames)}")
    print(f"Ball landing point: ({ball_land_x:.1f}, {ball_land_y:.1f})")

    # Load probabilities if available
    probs_file = output_dir.parent / 'probabilities.csv'
    probs = None
    bif_frame = None
    if probs_file.exists():
        probs = pd.read_csv(probs_file)
        # Find THE READ frame
        alpha = 0.5
        best_score = -1
        n_frames = len(probs)
        for i in range(n_frames):
            p = [probs['p_incomplete'].iloc[i], probs['p_complete'].iloc[i], probs['p_interception'].iloc[i]]
            leader = ['incomplete', 'complete', 'interception'][np.argmax(p)]
            if leader == 'complete':  # Will Levis play ends in completion
                score = probs['p_complete'].iloc[i] * (1 - i/n_frames)**alpha
                if score > best_score:
                    best_score = score
                    bif_frame = i
        print(f"THE READ at frame: {bif_frame}")

    # Create frames directory
    frames_dir = output_dir / 'ngs_frames'
    frames_dir.mkdir(exist_ok=True)

    # Calculate fixed bounds across ALL frames for consistent sizing
    all_x = ball_in_air['x'].values
    all_y = ball_in_air['y'].values
    all_x = np.append(all_x, ball_land_x)
    all_y = np.append(all_y, ball_land_y)

    x_center = (all_x.min() + all_x.max()) / 2
    y_center = 53.3 / 2  # Center on field width

    x_range = max(all_x.max() - all_x.min() + 15, 45)
    y_range = 53.3  # Full field width

    x_min = max(0, x_center - x_range/2)
    x_max = min(120, x_center + x_range/2)
    y_min = 0
    y_max = 53.3

    fixed_bounds = (x_min, x_max, y_min, y_max)
    print(f"Fixed field view: x=[{x_min:.0f}, {x_max:.0f}], y=[{y_min:.0f}, {y_max:.0f}]")

    # Generate each frame
    print(f"Generating {len(frames)} NGS frames...")
    for i, frame_id in enumerate(frames):
        output_path = frames_dir / f'frame_{i:03d}.png'
        create_ngs_frame(ball_in_air, frame_id, ball_land_x, ball_land_y,
                        output_path, probs, bif_frame, i, fixed_bounds)
        if (i + 1) % 10 == 0:
            print(f"  Generated frame {i + 1}/{len(frames)}")

    print(f"Frames saved to: {frames_dir}")

    # Create GIF
    frame_files = sorted(glob.glob(str(frames_dir / 'frame_*.png')))
    images = [Image.open(f) for f in frame_files]

    gif_path = output_dir / 'ngs_dots.gif'
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0
    )
    print(f"GIF saved: {gif_path}")

    # Create WebM
    import imageio.v2 as imageio
    webm_path = output_dir / 'ngs_dots.webm'
    writer = imageio.get_writer(str(webm_path), fps=10, codec='libvpx-vp9')
    for f in frame_files:
        writer.append_data(imageio.imread(f))
    writer.close()
    print(f"WebM saved: {webm_path}")

    return len(frames)


if __name__ == "__main__":
    # Will Levis 61-yard TD to DeAndre Hopkins
    play_path = ORGANIZED_DIR / "Week_08/ATL_vs_TEN/027_Q3_146_TEN"
    output_dir = BASE_DIR / "broadcast_viz" / "Will_Levis_TD"

    generate_ngs_visualization(play_path, output_dir, "Will Levis 61-yd TD to DeAndre Hopkins")
