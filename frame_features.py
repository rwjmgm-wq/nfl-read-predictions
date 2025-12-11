"""
Frame-by-frame feature engineering for completion probability model.

For each frame during ball flight, calculates geometric and kinematic features
that predict completion/incompletion/interception probability.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ball_trajectory import get_ball_trajectory_for_play


def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calculate_frame_features(play_folder):
    """
    Calculate features for each frame during ball flight.

    Args:
        play_folder: Path to play folder

    Returns:
        DataFrame with one row per ball-in-air frame, containing all features
    """
    play_folder = Path(play_folder)

    # Load data
    tracking = pd.read_csv(play_folder / "all_players_tracking.csv")
    input_df = pd.read_csv(play_folder / "input.csv")

    # Get ball trajectory
    ball_traj = get_ball_trajectory_for_play(play_folder)
    if ball_traj is None:
        return None

    # Get ball landing point
    ball_land_x = input_df.iloc[0]['ball_land_x']
    ball_land_y = input_df.iloc[0]['ball_land_y']

    # Filter to ball-in-air frames only
    ball_in_air = tracking[tracking['ball_in_air'] == True].copy()

    if len(ball_in_air) == 0:
        return None

    # Identify targeted receiver
    receiver_data = ball_in_air[ball_in_air['player_role'] == 'Targeted Receiver']
    if len(receiver_data) == 0:
        # Fallback: look for player_to_predict == True on offense
        receiver_data = ball_in_air[(ball_in_air['player_to_predict'] == True) &
                                     (ball_in_air['player_side'] == 'Offense')]

    if len(receiver_data) == 0:
        return None

    receiver_id = receiver_data['nfl_id'].iloc[0]

    # Identify defenders (defensive coverage players)
    defenders = ball_in_air[ball_in_air['player_side'] == 'Defense']
    defender_ids = defenders['nfl_id'].unique()

    # Get unique frames
    frames = sorted(ball_in_air['frame_id'].unique())
    max_frame = max(frames)

    # Build features for each frame
    feature_rows = []

    prev_separation = None
    prev_ball_to_receiver = None

    for frame_id in frames:
        frame_data = ball_in_air[ball_in_air['frame_id'] == frame_id]

        # Get receiver position for this frame
        receiver_frame = frame_data[frame_data['nfl_id'] == receiver_id]
        if len(receiver_frame) == 0:
            continue

        receiver_x = receiver_frame['x'].iloc[0]
        receiver_y = receiver_frame['y'].iloc[0]
        receiver_s = receiver_frame['s'].iloc[0] if 's' in receiver_frame.columns else np.nan
        receiver_dir = receiver_frame['dir'].iloc[0] if 'dir' in receiver_frame.columns else np.nan

        # Get ball position for this frame
        ball_frame = ball_traj[ball_traj['frame_id'] == frame_id]
        if len(ball_frame) == 0:
            continue

        ball_x = ball_frame['ball_x'].iloc[0]
        ball_y = ball_frame['ball_y'].iloc[0]
        ball_z = ball_frame['ball_z'].iloc[0]

        # Get defender positions for this frame
        defenders_frame = frame_data[frame_data['nfl_id'].isin(defender_ids)]

        # Calculate distance from each defender to receiver
        defender_distances = []
        defender_info = []

        for _, def_row in defenders_frame.iterrows():
            def_x = def_row['x']
            def_y = def_row['y']
            def_s = def_row['s'] if 's' in def_row else np.nan
            def_dir = def_row['dir'] if 'dir' in def_row else np.nan

            dist_to_receiver = calculate_distance(def_x, def_y, receiver_x, receiver_y)
            dist_to_ball_land = calculate_distance(def_x, def_y, ball_land_x, ball_land_y)

            defender_distances.append(dist_to_receiver)
            defender_info.append({
                'nfl_id': def_row['nfl_id'],
                'x': def_x,
                'y': def_y,
                's': def_s,
                'dir': def_dir,
                'dist_to_receiver': dist_to_receiver,
                'dist_to_catch_point': dist_to_ball_land
            })

        # Sort defenders by distance to receiver
        defender_info = sorted(defender_info, key=lambda x: x['dist_to_receiver'])

        # Feature 1: Separation (distance to nearest defender)
        # KEY FIX: When no defender data, receiver is UNCOVERED - use large distance
        UNCOVERED_DISTANCE = 20.0  # yards - represents "wide open"

        has_defender_tracking = len(defender_info) > 0

        if has_defender_tracking:
            nearest_defender_dist = defender_info[0]['dist_to_receiver']
        else:
            nearest_defender_dist = UNCOVERED_DISTANCE  # Wide open!

        # Second nearest defender (if within 5 yards of nearest)
        if len(defender_info) >= 2:
            if defender_info[1]['dist_to_receiver'] - defender_info[0]['dist_to_receiver'] < 5:
                second_defender_dist = defender_info[1]['dist_to_receiver']
            else:
                second_defender_dist = UNCOVERED_DISTANCE
        else:
            second_defender_dist = UNCOVERED_DISTANCE

        # Feature 2: Defender closing speed (rate of change of separation)
        if prev_separation is not None:
            closing_speed = (prev_separation - nearest_defender_dist) * 10  # yards/sec (10 Hz)
        else:
            closing_speed = 0.0
        prev_separation = nearest_defender_dist

        # Feature 3: Ball-to-receiver distance
        ball_to_receiver = calculate_distance(ball_x, ball_y, receiver_x, receiver_y)

        # Feature 4: Receiver speed/direction relative to ball trajectory
        # Calculate angle from receiver to catch point
        angle_to_catch = np.degrees(np.arctan2(ball_land_y - receiver_y, ball_land_x - receiver_x))
        if not np.isnan(receiver_dir):
            # Angle difference (how aligned is receiver movement with catch point direction)
            angle_diff = abs(receiver_dir - angle_to_catch)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            receiver_alignment = np.cos(np.radians(angle_diff))  # 1 = moving toward, -1 = away
        else:
            receiver_alignment = np.nan

        receiver_to_catch_point = calculate_distance(receiver_x, receiver_y, ball_land_x, ball_land_y)

        # Feature 5: Nearest defender speed/direction relative to catch point
        # KEY FIX: When no defender data, use values that indicate "wide open"
        if has_defender_tracking:
            nearest_def = defender_info[0]
            angle_def_to_catch = np.degrees(np.arctan2(ball_land_y - nearest_def['y'],
                                                        ball_land_x - nearest_def['x']))
            if not np.isnan(nearest_def['dir']):
                angle_diff_def = abs(nearest_def['dir'] - angle_def_to_catch)
                if angle_diff_def > 180:
                    angle_diff_def = 360 - angle_diff_def
                defender_alignment = np.cos(np.radians(angle_diff_def))
            else:
                defender_alignment = 0.0  # Neutral - not closing or moving away
            defender_speed = nearest_def['s'] if not np.isnan(nearest_def['s']) else 0.0
            defender_to_catch_point = nearest_def['dist_to_catch_point']
        else:
            # No defender tracked = wide open receiver
            defender_alignment = -1.0  # Moving away from catch point (worst case for defense)
            defender_speed = 0.0  # Not closing
            defender_to_catch_point = UNCOVERED_DISTANCE  # Far from catch point

        # Feature 6: Time remaining (frames until ball arrives)
        frames_remaining = max_frame - frame_id
        time_remaining = frames_remaining / 10.0  # seconds

        # Feature 7: Contested catch likelihood
        # Project defender position at catch point based on current velocity
        if has_defender_tracking and not np.isnan(defender_info[0].get('dir', np.nan)):
            nearest_def = defender_info[0]
            # Project position after time_remaining seconds
            def_dir_rad = np.radians(nearest_def['dir'])
            projected_def_x = nearest_def['x'] + defender_speed * time_remaining * np.sin(def_dir_rad)
            projected_def_y = nearest_def['y'] + defender_speed * time_remaining * np.cos(def_dir_rad)
            projected_separation = calculate_distance(projected_def_x, projected_def_y, ball_land_x, ball_land_y)
        else:
            # No defender = not contested, use large separation
            projected_separation = UNCOVERED_DISTANCE

        # Contested if projected separation < 2 yards
        contested = projected_separation < 2.0

        # Additional useful features
        # Ball height (is it catchable this frame?)
        ball_catchable = ball_z < 3.0

        # Receiver speed
        receiver_speed = receiver_s if not np.isnan(receiver_s) else 0.0

        # Calculate additional derived features
        # Receiver vs defender speed advantage
        receiver_defender_speed_diff = receiver_speed - defender_speed

        # Will receiver arrive before defender? (simple projection)
        if receiver_speed > 0:
            time_for_receiver = receiver_to_catch_point / receiver_speed
        else:
            time_for_receiver = float('inf')

        if defender_speed > 0 and defender_to_catch_point < UNCOVERED_DISTANCE:
            time_for_defender = defender_to_catch_point / defender_speed
        else:
            time_for_defender = float('inf')

        receiver_will_arrive_first = 1.0 if time_for_receiver < time_for_defender else 0.0

        # Can defender possibly arrive in time?
        defender_can_arrive = 1.0 if time_for_defender <= time_remaining + 0.3 else 0.0

        # Ball distance to catch point
        ball_to_catch_point = calculate_distance(ball_x, ball_y, ball_land_x, ball_land_y)

        feature_rows.append({
            'frame_id': frame_id,
            'time_remaining': time_remaining,
            'frames_remaining': frames_remaining,

            # Coverage features
            'has_defender_tracking': 1.0 if has_defender_tracking else 0.0,

            # Separation features
            'separation': nearest_defender_dist,
            'second_defender_dist': second_defender_dist,
            'closing_speed': closing_speed,

            # Ball position features
            'ball_to_receiver': ball_to_receiver,
            'ball_to_catch_point': ball_to_catch_point,
            'ball_x': ball_x,
            'ball_y': ball_y,
            'ball_z': ball_z,
            'ball_catchable': ball_catchable,

            # Receiver features
            'receiver_x': receiver_x,
            'receiver_y': receiver_y,
            'receiver_speed': receiver_speed,
            'receiver_to_catch_point': receiver_to_catch_point,
            'receiver_alignment': receiver_alignment,

            # Defender features
            'defender_speed': defender_speed,
            'defender_to_catch_point': defender_to_catch_point,
            'defender_alignment': defender_alignment,

            # Race features (receiver vs defender)
            'receiver_defender_speed_diff': receiver_defender_speed_diff,
            'receiver_will_arrive_first': receiver_will_arrive_first,
            'defender_can_arrive': defender_can_arrive,

            # Projection features
            'projected_separation_at_catch': projected_separation,
            'contested': contested,
        })

    return pd.DataFrame(feature_rows)


def test_sample_play():
    """Test feature calculation on sample play."""
    BASE_DIR = Path(__file__).parent
    play_folder = BASE_DIR / "organized_plays" / "Week_01" / "DET_vs_KC" / "001_Q1_1425_DET"

    print(f"Testing play: {play_folder.name}")
    print("=" * 80)

    # Load supplementary for context
    supp = pd.read_csv(play_folder / "supplementary.csv")
    print(f"Play: {supp['play_description'].iloc[0]}")
    print(f"Result: {supp['pass_result'].iloc[0]}")
    print("=" * 80)

    features = calculate_frame_features(play_folder)

    if features is not None:
        print(f"\nFeatures for {len(features)} ball-in-air frames:\n")

        # Display key columns
        display_cols = [
            'frame_id', 'time_remaining', 'separation', 'closing_speed',
            'ball_to_receiver', 'ball_z', 'receiver_to_catch_point',
            'defender_to_catch_point', 'projected_separation_at_catch', 'contested'
        ]

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.2f}'.format)

        print(features[display_cols].to_string(index=False))

        print("\n" + "=" * 80)
        print("Feature Evolution Summary:")
        print(f"  Starting separation: {features['separation'].iloc[0]:.2f} yards")
        print(f"  Ending separation: {features['separation'].iloc[-1]:.2f} yards")
        print(f"  Max closing speed: {features['closing_speed'].max():.2f} yards/sec")
        print(f"  Projected separation at catch: {features['projected_separation_at_catch'].iloc[-1]:.2f} yards")
        print(f"  Contested: {features['contested'].iloc[-1]}")
    else:
        print("Failed to calculate features")


if __name__ == "__main__":
    test_sample_play()
