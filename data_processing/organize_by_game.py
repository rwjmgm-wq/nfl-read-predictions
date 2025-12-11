"""
Organize plays into hierarchical structure:
    organized_plays/
        Week_01/
            DET_vs_KC/
                001_Q1_1425_DET/
                    input.csv
                    output.csv
                    supplementary.csv
                002_Q1_1006_DET/
                ...

Plays are ordered chronologically within each game.
Folder name format: {sequence}_{quarter}_{clock}_{possession_team}/
"""

import pandas as pd
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "114239_nfl_competition_files_published_analytics_final"
PLAYS_DIR = BASE_DIR / "plays_by_id"
OUTPUT_DIR = BASE_DIR / "organized_plays"

def clock_to_seconds(clock_str):
    """Convert game clock string (e.g., '14:25' or ':09') to seconds for sorting."""
    if not clock_str or pd.isna(clock_str):
        return 0
    clock_str = str(clock_str).strip()
    if clock_str.startswith(':'):
        clock_str = '0' + clock_str
    parts = clock_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0

def get_play_order_key(row):
    """
    Create a sort key for chronological ordering.
    Lower key = earlier in game.
    Quarter increases, clock decreases within quarter.
    """
    quarter = int(row['quarter']) if pd.notna(row['quarter']) else 1
    clock_seconds = clock_to_seconds(row['game_clock'])
    # Quarter * 1000 gives priority, then subtract clock (higher clock = earlier)
    return quarter * 10000 - clock_seconds

def organize_game(game_id, supp_df, test_mode=False):
    """Organize all plays for a single game."""

    # Get all plays for this game
    game_plays = supp_df[supp_df['game_id'] == game_id].copy()

    if len(game_plays) == 0:
        print(f"No plays found for game {game_id}")
        return 0

    # Get game info from first row
    first_row = game_plays.iloc[0]
    week = int(first_row['week'])
    home_team = first_row['home_team_abbr']
    visitor_team = first_row['visitor_team_abbr']

    # Create folder name: visitor_vs_home (visitor @ home)
    game_folder_name = f"{visitor_team}_vs_{home_team}"

    # Create directory structure
    week_folder = OUTPUT_DIR / f"Week_{week:02d}"
    game_folder = week_folder / game_folder_name

    if test_mode:
        print(f"Would create: {game_folder}")
    else:
        game_folder.mkdir(parents=True, exist_ok=True)

    # Sort plays chronologically
    game_plays['sort_key'] = game_plays.apply(get_play_order_key, axis=1)
    game_plays = game_plays.sort_values('sort_key')

    # Process each play
    for seq, (idx, row) in enumerate(game_plays.iterrows(), 1):
        play_id = row['play_id']
        quarter = int(row['quarter']) if pd.notna(row['quarter']) else 1
        clock = str(row['game_clock']).replace(':', '').strip()
        if clock.startswith('0'):
            clock = clock[1:] if len(clock) > 1 else '0'
        possession = row['possession_team']

        # Create play folder name
        play_folder_name = f"{seq:03d}_Q{quarter}_{clock}_{possession}"

        # Source folder
        source_folder = PLAYS_DIR / f"{game_id}_{play_id}"

        if test_mode:
            print(f"  {play_folder_name}")
            print(f"    From: {source_folder.name}")
            print(f"    Clock: {row['game_clock']}, Quarter: {quarter}")
        else:
            dest_folder = game_folder / play_folder_name
            if source_folder.exists():
                shutil.copytree(source_folder, dest_folder, dirs_exist_ok=True)

    return len(game_plays)

def main(test_game_id=None):
    # Load supplementary data
    supp_df = pd.read_csv(DATA_DIR / "supplementary_data.csv", low_memory=False)
    print(f"Loaded supplementary data: {len(supp_df)} rows")

    if test_game_id:
        # Test mode: just show what would be created for one game
        print(f"\n=== TEST MODE: Game {test_game_id} ===\n")
        count = organize_game(test_game_id, supp_df, test_mode=True)
        print(f"\nWould organize {count} plays")
    else:
        # Full mode: organize all games
        OUTPUT_DIR.mkdir(exist_ok=True)

        game_ids = supp_df['game_id'].unique()
        total_plays = 0

        for i, game_id in enumerate(game_ids):
            count = organize_game(game_id, supp_df, test_mode=False)
            total_plays += count
            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{len(game_ids)} games processed")

        print(f"\nTotal: {total_plays} plays organized across {len(game_ids)} games")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test with first game of 2023 season (KC vs DET)
        main(test_game_id=2023090700)
    else:
        main()
