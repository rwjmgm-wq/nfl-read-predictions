"""
Regenerate the Love to Wicks race chart with play information at the top.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

BASE_DIR = Path(__file__).parent
VIZ_DIR = BASE_DIR / "broadcast_viz" / "Love_to_Wicks_17yd_Complete"

# Play information
TITLE = "Jordan Love to Dontayvion Wicks (17-yd)"
SUBTITLE = "Week 18: CHI @ GB - Hitch Route - Complete"

# Colors
COLORS = {
    'complete': '#00CED1',      # Cyan
    'incomplete': '#FF4444',    # Red
    'interception': '#FF00FF'   # Magenta
}

def create_race_chart_with_info():
    """Create animated race chart with play information."""

    # Load probabilities
    probs_df = pd.read_csv(VIZ_DIR / "probabilities.csv")
    n_frames = len(probs_df)

    # Find THE READ frame (frame 0 based on commentary notes)
    bif_frame = 0

    print(f"Creating race chart with info for {n_frames} frames...")

    # Create animated frames
    frames = []

    for current_frame in range(n_frames):
        fig, ax = plt.subplots(figsize=(12, 7.5))
        fig.patch.set_alpha(0.0)  # Transparent background
        ax.patch.set_alpha(0.0)   # Transparent axes background

        # Add play information at the top
        fig.text(0.5, 0.96, TITLE, ha='center', fontsize=20, fontweight='bold')
        fig.text(0.5, 0.92, SUBTITLE, ha='center', fontsize=14, color='#555555')

        # Add outcome probabilities under play info
        p_complete_current = probs_df['p_complete'].iloc[current_frame]
        p_incomplete_current = probs_df['p_incomplete'].iloc[current_frame]
        p_interception_current = probs_df['p_interception'].iloc[current_frame]

        fig.text(0.25, 0.88, f'Complete: {p_complete_current*100:.1f}%',
                ha='center', fontsize=14, fontweight='bold', color=COLORS['complete'])
        fig.text(0.5, 0.88, f'Incomplete: {p_incomplete_current*100:.1f}%',
                ha='center', fontsize=14, fontweight='bold', color=COLORS['incomplete'])
        fig.text(0.75, 0.88, f'Interception: {p_interception_current*100:.1f}%',
                ha='center', fontsize=14, fontweight='bold', color=COLORS['interception'])

        # Plot data up to current frame
        frames_so_far = probs_df.iloc[:current_frame+1]

        # Plot each probability line
        ax.plot(frames_so_far['frame'], frames_so_far['p_complete'],
                color=COLORS['complete'], linewidth=4, label='Complete', zorder=3)
        ax.plot(frames_so_far['frame'], frames_so_far['p_incomplete'],
                color=COLORS['incomplete'], linewidth=4, label='Incomplete', zorder=2)
        ax.plot(frames_so_far['frame'], frames_so_far['p_interception'],
                color=COLORS['interception'], linewidth=4, label='Interception', zorder=1)

        # Mark current frame with dots
        ax.scatter(current_frame, probs_df['p_complete'].iloc[current_frame],
                  s=250, color=COLORS['complete'], zorder=4, edgecolors='white', linewidths=3)
        ax.scatter(current_frame, probs_df['p_incomplete'].iloc[current_frame],
                  s=250, color=COLORS['incomplete'], zorder=4, edgecolors='white', linewidths=3)
        ax.scatter(current_frame, probs_df['p_interception'].iloc[current_frame],
                  s=250, color=COLORS['interception'], zorder=4, edgecolors='white', linewidths=3)

        # Mark THE READ (at frame 0) - static marker
        ax.axvline(x=bif_frame, color='gold', linestyle='--', linewidth=4, alpha=0.8, zorder=5)
        # Static gold star at frame 0
        p_complete_at_read = probs_df['p_complete'].iloc[bif_frame]
        ax.scatter(bif_frame, p_complete_at_read, s=400, marker='*', color='gold',
                  zorder=6, edgecolors='black', linewidths=2)

        if current_frame >= bif_frame:
            ax.text(bif_frame, 1.08, 'THE READ\n(0.0% into play)',
                   ha='center', fontsize=12, fontweight='bold', color='gold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9,
                            edgecolor='gold', linewidth=3))

        # Formatting
        ax.set_xlim(-0.5, n_frames - 0.5)
        ax.set_ylim(-0.05, 1.18)
        ax.set_xlabel('Frame', fontsize=15, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=15, fontweight='bold')
        ax.set_title('', fontsize=14, pad=10)  # Empty title, using fig.text instead

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)

        # Tight layout with extra space at top for title and outcome info
        plt.subplots_adjust(top=0.84, bottom=0.08, left=0.08, right=0.95)

        # Save frame to buffer (keep RGBA for transparency)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(height, width, 4)  # Keep alpha channel
        frames.append(image)

        plt.close(fig)

        if (current_frame + 1) % 5 == 0:
            print(f"  Frame {current_frame + 1}/{n_frames}")

    # Match NGS dots timing: 0.2s per frame = 5 fps (2.2s total)
    duration_per_frame = 0.2  # Match NGS dots duration
    fps = 1 / duration_per_frame  # 5 fps

    # Save as GIF (note: GIF doesn't support true transparency, use WebM for that)
    output_path = VIZ_DIR / "race_chart_with_play_info.gif"
    # Convert RGBA to RGB for GIF
    frames_rgb = [frame[:, :, :3] for frame in frames]
    # Duration needs to be in seconds for GIF writer
    imageio.mimsave(output_path, frames_rgb, duration=duration_per_frame * 1000, loop=0)  # Convert to ms
    print(f"\nSaved: {output_path} ({len(frames)} frames @ {duration_per_frame}s/frame = {len(frames) * duration_per_frame:.1f}s total)")
    print(f"  Matches NGS dots timing perfectly!")

    # Save as WebM with transparency (preferred for broadcast overlay)
    # Use same frame rate as GIF for perfect alignment with NGS frames
    try:
        webm_path = VIZ_DIR / "race_chart_with_play_info.webm"
        imageio.mimsave(webm_path, frames, fps=fps, codec='libvpx-vp9', quality=8, pixelformat='yuva420p')
        print(f"Saved: {webm_path} (with transparency, {fps:.1f} fps = {len(frames) * duration_per_frame:.1f}s total)")
    except Exception as e:
        print(f"Could not create WebM: {e}")

if __name__ == "__main__":
    create_race_chart_with_info()
    print("\nDone!")
