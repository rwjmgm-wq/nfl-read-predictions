"""
Merge input and output tracking data into per-player files with continuous frame numbering.

Creates structure within each play folder:
    players/
        {nfl_id}_{player_name}/
            tracking.csv   (continuous frames: input frames + output frames with offset)

Adds columns:
    - ball_in_air: False for input frames, True for output frames
    - in_both_files: True if player appears in both input AND output files
"""

import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
ORGANIZED_DIR = BASE_DIR / "organized_plays"

def sanitize_name(name):
    """Make player name safe for folder names."""
    if pd.isna(name):
        return "Unknown"
    return str(name).replace(' ', '_').replace('.', '').replace("'", "")

def merge_player_tracking(play_folder, test_mode=False):
    """Merge input and output tracking for a single play."""

    input_file = play_folder / "input.csv"
    output_file = play_folder / "output.csv"

    if not input_file.exists() or not output_file.exists():
        return 0

    # Load data
    input_df = pd.read_csv(input_file)
    output_df = pd.read_csv(output_file)

    # Get max frame from input for offset
    max_input_frame = input_df['frame_id'].max()

    # Find players in each file
    input_players = set(input_df['nfl_id'].unique())
    output_players = set(output_df['nfl_id'].unique())
    players_in_both = input_players & output_players
    all_players = input_players | output_players

    # Create players directory
    players_dir = play_folder / "players"
    if not test_mode:
        players_dir.mkdir(exist_ok=True)

    player_count = 0

    for nfl_id in all_players:
        in_both = nfl_id in players_in_both

        # Get input data for this player
        player_input = input_df[input_df['nfl_id'] == nfl_id].copy()

        # Get output data for this player
        player_output = output_df[output_df['nfl_id'] == nfl_id].copy()

        # Get player name from input (if available)
        if len(player_input) > 0 and 'player_name' in player_input.columns:
            player_name = player_input['player_name'].iloc[0]
        else:
            player_name = "Unknown"

        # Add ball_in_air and in_both_files columns
        if len(player_input) > 0:
            player_input['ball_in_air'] = False
            player_input['in_both_files'] = in_both

        if len(player_output) > 0:
            # Offset output frame_ids to continue from input
            player_output['frame_id'] = player_output['frame_id'] + max_input_frame
            player_output['ball_in_air'] = True
            player_output['in_both_files'] = in_both

        # Combine input and output
        if len(player_input) > 0 and len(player_output) > 0:
            # For output, we only have x, y - need to add NAs for other columns
            input_cols = set(player_input.columns)
            output_cols = set(player_output.columns)

            # Add missing columns to output with NA
            for col in input_cols - output_cols:
                player_output[col] = pd.NA

            # Add missing columns to input with NA
            for col in output_cols - input_cols:
                player_input[col] = pd.NA

            # Ensure same column order
            all_cols = list(player_input.columns)
            player_output = player_output[all_cols]

            combined = pd.concat([player_input, player_output], ignore_index=True)
        elif len(player_input) > 0:
            combined = player_input
        else:
            combined = player_output

        # Sort by frame_id
        combined = combined.sort_values('frame_id')

        # Create player folder and save
        safe_name = sanitize_name(player_name)
        player_folder_name = f"{nfl_id}_{safe_name}"

        if test_mode:
            print(f"  Player: {player_folder_name}")
            print(f"    in_both_files: {in_both}")
            print(f"    Input frames: {len(player_input)} (1-{max_input_frame})")
            print(f"    Output frames: {len(player_output)} ({max_input_frame+1}-{max_input_frame+len(player_output)})")
            print(f"    Total frames: {len(combined)}")
        else:
            player_folder = players_dir / player_folder_name
            player_folder.mkdir(exist_ok=True)
            combined.to_csv(player_folder / "tracking.csv", index=False)

        player_count += 1

    return player_count

def main(test_play=None):
    if test_play:
        # Test mode: process single play and show output
        print(f"=== TEST MODE: {test_play} ===\n")

        # Find the play folder
        play_folder = None
        for week_folder in ORGANIZED_DIR.iterdir():
            if not week_folder.is_dir():
                continue
            for game_folder in week_folder.iterdir():
                if not game_folder.is_dir():
                    continue
                for pf in game_folder.iterdir():
                    if pf.is_dir() and test_play in pf.name:
                        play_folder = pf
                        break

        if play_folder is None:
            # Try direct path
            play_folder = Path(test_play)

        if not play_folder.exists():
            print(f"Play folder not found: {test_play}")
            return

        print(f"Play folder: {play_folder}\n")
        count = merge_player_tracking(play_folder, test_mode=True)
        print(f"\nWould create {count} player folders")
    else:
        # Full mode: process all plays
        total_players = 0
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

                    count = merge_player_tracking(play_folder, test_mode=False)
                    total_players += count
                    play_count += 1

                    if play_count % 1000 == 0:
                        print(f"Progress: {play_count} plays processed")

        print(f"\nTotal: {play_count} plays, {total_players} player folders created")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test with first play of first game
        test_path = ORGANIZED_DIR / "Week_01" / "DET_vs_KC" / "001_Q1_1425_DET"
        main(test_play=str(test_path))
    else:
        main()
