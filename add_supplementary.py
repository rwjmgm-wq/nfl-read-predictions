"""
Add supplementary data to each play folder.

Reads supplementary_data.csv and extracts the matching row for each play,
saving it as supplementary.csv in the play folder.
"""

import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "114239_nfl_competition_files_published_analytics_final"
PLAYS_DIR = BASE_DIR / "plays_by_id"

def add_supplementary():
    # Load supplementary data
    supp_df = pd.read_csv(DATA_DIR / "supplementary_data.csv")
    print(f"Loaded supplementary data: {len(supp_df)} rows")

    # Index by game_id and play_id for fast lookup
    supp_df = supp_df.set_index(['game_id', 'play_id'])

    # Process each play folder
    play_folders = list(PLAYS_DIR.iterdir())
    total = len(play_folders)
    processed = 0
    missing = 0

    for i, play_folder in enumerate(play_folders):
        if not play_folder.is_dir():
            continue

        # Parse game_id and play_id from folder name
        folder_name = play_folder.name
        parts = folder_name.rsplit('_', 1)
        if len(parts) != 2:
            print(f"Skipping invalid folder: {folder_name}")
            continue

        game_id = int(parts[0])
        play_id = int(parts[1])

        # Get supplementary data for this play
        try:
            play_supp = supp_df.loc[(game_id, play_id)]
            # Convert to DataFrame (it's a Series when single row)
            play_supp_df = play_supp.to_frame().T
            play_supp_df.index.names = ['game_id', 'play_id']
            play_supp_df = play_supp_df.reset_index()

            # Save to play folder
            play_supp_df.to_csv(play_folder / "supplementary.csv", index=False)
            processed += 1
        except KeyError:
            missing += 1
            if missing <= 5:
                print(f"No supplementary data for {game_id}_{play_id}")

        # Progress update
        if (i + 1) % 2000 == 0:
            print(f"Progress: {i + 1}/{total} folders processed")

    print(f"\nComplete: {processed} plays updated, {missing} missing supplementary data")

if __name__ == "__main__":
    add_supplementary()
