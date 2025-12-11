"""
Reorganize NFL Big Data Bowl data by grouping plays into folders.

Creates structure:
    plays_by_id/
        {game_id}_{play_id}/
            input.csv   (player tracking before ball thrown)
            output.csv  (player positions while ball in air)
"""

import pandas as pd
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "114239_nfl_competition_files_published_analytics_final"
TRAIN_DIR = DATA_DIR / "train"
OUTPUT_DIR = BASE_DIR / "plays_by_id"

def reorganize_data():
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    total_plays = 0

    # Process each week
    for week in range(1, 19):
        week_str = f"{week:02d}"
        input_file = TRAIN_DIR / f"input_2023_w{week_str}.csv"
        output_file = TRAIN_DIR / f"output_2023_w{week_str}.csv"

        if not input_file.exists():
            print(f"Week {week}: Input file not found, skipping")
            continue

        print(f"Processing week {week}...")

        # Load data
        input_df = pd.read_csv(input_file)
        output_df = pd.read_csv(output_file)

        # Get unique plays from input
        plays = input_df.groupby(['game_id', 'play_id'])

        week_plays = 0
        for (game_id, play_id), input_play_df in plays:
            # Create folder for this play
            play_folder = OUTPUT_DIR / f"{game_id}_{play_id}"
            play_folder.mkdir(exist_ok=True)

            # Save input data for this play
            input_play_df.to_csv(play_folder / "input.csv", index=False)

            # Get matching output data
            output_play_df = output_df[
                (output_df['game_id'] == game_id) &
                (output_df['play_id'] == play_id)
            ]

            # Save output data for this play
            output_play_df.to_csv(play_folder / "output.csv", index=False)

            week_plays += 1

        print(f"  Week {week}: {week_plays} plays processed")
        total_plays += week_plays

    print(f"\nTotal: {total_plays} plays organized into {OUTPUT_DIR}")

if __name__ == "__main__":
    reorganize_data()
