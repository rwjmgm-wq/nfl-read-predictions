"""
Create a combined all_players_tracking.csv file for each play.

This file contains all players merged together with continuous frame numbering,
ball_in_air flag, and in_both_files flag - making it easy to work with for
visualizations and analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
ORGANIZED_DIR = BASE_DIR / "organized_plays"

def create_combined_tracking(play_folder, test_mode=False):
    """Create combined tracking file for a single play."""

    input_file = play_folder / "input.csv"
    output_file = play_folder / "output.csv"

    if not input_file.exists() or not output_file.exists():
        return False

    # Load data
    input_df = pd.read_csv(input_file)
    output_df = pd.read_csv(output_file)

    # Get max frame from input for offset
    max_input_frame = input_df['frame_id'].max()

    # Find players in each file
    input_players = set(input_df['nfl_id'].unique())
    output_players = set(output_df['nfl_id'].unique())
    players_in_both = input_players & output_players

    # Add ball_in_air and in_both_files to input data
    input_df['ball_in_air'] = False
    input_df['in_both_files'] = input_df['nfl_id'].isin(players_in_both)

    # Offset output frame_ids and add flags
    output_df['frame_id'] = output_df['frame_id'] + max_input_frame
    output_df['ball_in_air'] = True
    output_df['in_both_files'] = output_df['nfl_id'].isin(players_in_both)

    # Get player info from input (one row per player with static info)
    player_info_cols = ['nfl_id', 'player_name', 'player_height', 'player_weight',
                        'player_birth_date', 'player_position', 'player_side',
                        'player_role', 'player_to_predict', 'play_direction',
                        'absolute_yardline_number', 'num_frames_output',
                        'ball_land_x', 'ball_land_y']

    # Get unique player info from input (take first row for each player)
    available_cols = [col for col in player_info_cols if col in input_df.columns]
    player_info = input_df.groupby('nfl_id')[available_cols].first().reset_index(drop=True)
    player_info = input_df[available_cols].drop_duplicates(subset=['nfl_id'])

    # Merge player info into output data
    output_df = output_df.merge(player_info, on='nfl_id', how='left')

    # Add any remaining missing columns with NA
    input_cols = set(input_df.columns)
    output_cols = set(output_df.columns)

    for col in input_cols - output_cols:
        output_df[col] = pd.NA

    # Ensure same column order
    output_df = output_df[input_df.columns]

    # Combine all data
    combined = pd.concat([input_df, output_df], ignore_index=True)

    # Sort by nfl_id then frame_id for clean organization
    combined = combined.sort_values(['nfl_id', 'frame_id'])

    # Calculate velocity from position changes for frames missing s/dir/a
    # Group by player and calculate frame-to-frame differences
    fps = 10.0  # 10 frames per second

    for nfl_id in combined['nfl_id'].unique():
        player_mask = combined['nfl_id'] == nfl_id
        player_data = combined.loc[player_mask].copy()

        # Get x, y positions
        x = player_data['x'].values
        y = player_data['y'].values

        # Calculate differences
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])

        # Calculate speed (yards per second)
        calc_speed = np.sqrt(dx**2 + dy**2) * fps

        # Calculate direction (degrees, 0 = north, clockwise)
        # np.arctan2 gives angle from positive x-axis, counter-clockwise
        # NFL direction: 0 = north (positive y), 90 = east (positive x)
        calc_dir = np.degrees(np.arctan2(dx, dy))  # Note: arctan2(dx, dy) for NFL convention
        calc_dir = np.where(calc_dir < 0, calc_dir + 360, calc_dir)

        # Calculate acceleration (change in speed)
        calc_accel = np.diff(calc_speed, prepend=calc_speed[0]) * fps

        # Only fill in where values are missing (NaN)
        s_values = player_data['s'].values
        dir_values = player_data['dir'].values
        a_values = player_data['a'].values

        # Fill NaN values with calculated values
        s_filled = np.where(pd.isna(s_values), calc_speed, s_values)
        dir_filled = np.where(pd.isna(dir_values), calc_dir, dir_values)
        a_filled = np.where(pd.isna(a_values), calc_accel, a_values)

        # Update combined dataframe
        combined.loc[player_mask, 's'] = s_filled
        combined.loc[player_mask, 'dir'] = dir_filled
        combined.loc[player_mask, 'a'] = a_filled

    if test_mode:
        print(f"Play: {play_folder.name}")
        print(f"  Input frames: 1-{max_input_frame}")
        print(f"  Output frames: {max_input_frame+1}-{max_input_frame + len(output_df['frame_id'].unique())}")
        print(f"  Total players: {len(input_players | output_players)}")
        print(f"  Players in both: {len(players_in_both)}")
        print(f"  Total rows: {len(combined)}")
        print(f"\nSample (first 5 rows):")
        print(combined.head())
        return True

    # Save combined file
    combined.to_csv(play_folder / "all_players_tracking.csv", index=False)
    return True

def main(test_play=None):
    if test_play:
        # Test mode
        print(f"=== TEST MODE ===\n")
        play_folder = Path(test_play)
        if not play_folder.exists():
            print(f"Play folder not found: {test_play}")
            return
        create_combined_tracking(play_folder, test_mode=True)
    else:
        # Full mode
        play_count = 0

        for week_folder in sorted(ORGANIZED_DIR.iterdir()):
            if not week_folder.is_dir():
                continue

            for game_folder in sorted(week_folder.iterdir()):
                if not game_folder.is_dir():
                    continue

                for play_folder in sorted(game_folder.iterdir()):
                    if not play_folder.is_dir():
                        continue

                    create_combined_tracking(play_folder, test_mode=False)
                    play_count += 1

                    if play_count % 1000 == 0:
                        print(f"Progress: {play_count} plays processed")

        print(f"\nTotal: {play_count} plays with all_players_tracking.csv created")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_path = ORGANIZED_DIR / "Week_01" / "DET_vs_KC" / "001_Q1_1425_DET"
        main(test_play=str(test_path))
    else:
        main()
