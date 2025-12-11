"""
Combine race chart and NGS dots GIFs into one vertical layout.
Race chart on top, NGS dots on bottom.
"""

import imageio
import numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent
VIZ_DIR = BASE_DIR / "broadcast_viz" / "Love_to_Wicks_17yd_Complete"

def combine_gifs():
    """Combine race chart and NGS dots into one vertical GIF."""

    # Load both GIFs
    print("Loading GIFs...")
    race_gif = imageio.get_reader(VIZ_DIR / "race_chart_with_play_info.gif")
    ngs_gif = imageio.get_reader(VIZ_DIR / "ngs_dots.gif")

    race_frames = list(race_gif)
    ngs_frames = list(ngs_gif)

    print(f"Race chart: {len(race_frames)} frames")
    print(f"NGS dots: {len(ngs_frames)} frames")

    # Ensure same number of frames
    n_frames = min(len(race_frames), len(ngs_frames))

    combined_frames = []

    for i in range(n_frames):
        # Convert to PIL Images for easier manipulation
        race_img = Image.fromarray(race_frames[i])
        ngs_img = Image.fromarray(ngs_frames[i])

        # Get dimensions
        race_width, race_height = race_img.size
        ngs_width, ngs_height = ngs_img.size

        # Make widths match (scale NGS to match race chart width)
        if ngs_width != race_width:
            aspect_ratio = ngs_height / ngs_width
            new_height = int(race_width * aspect_ratio)
            ngs_img = ngs_img.resize((race_width, new_height), Image.Resampling.LANCZOS)
            ngs_width, ngs_height = ngs_img.size

        # Create combined image (race on top, ngs on bottom)
        combined_height = race_height + ngs_height
        combined_img = Image.new('RGB', (race_width, combined_height))

        # Paste race chart on top
        combined_img.paste(race_img, (0, 0))

        # Paste NGS dots on bottom
        combined_img.paste(ngs_img, (0, race_height))

        combined_frames.append(np.array(combined_img))

        if (i + 1) % 5 == 0:
            print(f"  Combined frame {i + 1}/{n_frames}")

    # Save combined GIF
    output_path = VIZ_DIR / "combined_race_and_ngs.gif"
    duration = 200  # 200ms per frame to match originals

    imageio.mimsave(output_path, combined_frames, duration=duration, loop=0)
    print(f"\nSaved: {output_path}")
    print(f"  {len(combined_frames)} frames @ {duration}ms/frame = {len(combined_frames) * duration / 1000:.1f}s total")
    print(f"  Size: {combined_frames[0].shape[1]} x {combined_frames[0].shape[0]}")

if __name__ == "__main__":
    combine_gifs()
    print("\nDone!")
