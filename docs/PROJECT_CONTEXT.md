# NFL Big Data Bowl 2026 - Analytics Broadcast Track
## Project Context & Session Memory

**Last Updated:** 2024-12-04

---

## Competition Overview

### Track: Analytics (Broadcast Visualization)
- **Goal:** Create animations, videos, or charts that effectively visualize player movement while the ball is in the air
- **Prize Pool:** $100,000 total
- **Deadline:** January 25, 2025
- **Presentation:** Finalists present at NFL Scouting Combine in Indianapolis

### Links
- [Kaggle Analytics Competition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics)
- [NFL Football Operations](https://operations.nfl.com/gameday/analytics/big-data-bowl/)
- Contact: BDBhelp@nfl.com

---

## Video Requirements

**Broadcast Visualization Track only.**

Create a dynamic, engaging, and high-quality video that demonstrates your metric or football concept in action. Participants are encouraged to show supporting documentation via Notebook if it helps a broadcast team better understand their analyses.

---

## Scoring Criteria

All entries are evaluated on four components (0-10 scale):

### Football Score (30%)
- Would NFL teams (or the league office) be able to use these results on a week-to-week basis?
- Does the analysis account for variables that make football data complex?
- Are the ideas unique?

### Data Science Score (30%)
- Is the work correct?
- Are claims backed up by data?
- Are the statistical models appropriate given the data?
- Are the analytical applications innovative?

### Writeup Score (20%)
- Is the Writeup well-written?
- Is the Writeup easy to follow?
- Is the motivation (metric, player evaluation, etc) clearly defined?

### Data Visualization Score (20%)
- Are the charts and tables provided accessible?
- Are the charts and tables accurate?
- Are the charts and tables innovative?

### Important Notes
- Participants are encouraged to show statistical code if it helps readers better understand their analyses
- Most code should be hidden in an Appendix
- **Any Writeup that doesn't use the player tracking data will not be scored**

---

## Data Structure

### Available Data
- **Training Data:** 2023 NFL season, Weeks 1-18
- **Location:** `114239_nfl_competition_files_published_analytics_final/`

### Files

#### `supplementary_data.csv` (Play-level metadata)
| Column | Description |
|--------|-------------|
| game_id | Unique game identifier |
| season | NFL season year |
| week | Week number (1-18) |
| game_date | Date of game |
| game_time_eastern | Kickoff time |
| home_team_abbr | Home team abbreviation |
| visitor_team_abbr | Visitor team abbreviation |
| play_id | Unique play identifier within game |
| play_description | Text description of play |
| quarter | Game quarter (1-4) |
| game_clock | Time remaining in quarter |
| down | Current down (1-4) |
| yards_to_go | Yards needed for first down |
| possession_team | Team with possession |
| defensive_team | Defending team |
| yardline_side | Which team's side of field |
| yardline_number | Yard line number |
| pre_snap_home_score | Home team score before play |
| pre_snap_visitor_score | Visitor team score before play |
| play_nullified_by_penalty | Whether play was nullified (Y/N) |
| pass_result | C=Complete, I=Incomplete, IN=Interception |
| pass_length | Air yards of pass |
| offense_formation | SHOTGUN, SINGLEBACK, I_FORM, PISTOL, EMPTY |
| receiver_alignment | 3x1, 2x2, 3x2, 2x1, 4x1 |
| route_of_targeted_receiver | Route type (IN, OUT, POST, CORNER, HITCH, CROSS, FLAT, SLANT, GO, ANGLE) |
| play_action | TRUE/FALSE |
| dropback_type | TRADITIONAL, DESIGNED_ROLLOUT_RIGHT/LEFT, SCRAMBLE, SCRAMBLE_ROLLOUT_RIGHT/LEFT |
| dropback_distance | QB dropback distance in yards |
| pass_location_type | INSIDE_BOX, OUTSIDE_RIGHT, OUTSIDE_LEFT |
| defenders_in_the_box | Number of defenders in box pre-snap |
| team_coverage_man_zone | MAN_COVERAGE, ZONE_COVERAGE |
| team_coverage_type | COVER_0_MAN, COVER_1_MAN, COVER_2_ZONE, COVER_3_ZONE, COVER_4_ZONE, COVER_6_ZONE |
| penalty_yards | Penalty yards (if applicable) |
| pre_penalty_yards_gained | Yards before penalty |
| yards_gained | Actual yards gained |
| expected_points | Pre-play expected points |
| expected_points_added | EPA for the play |
| pre_snap_home_team_win_probability | Win prob before play |
| pre_snap_visitor_team_win_probability | Win prob before play |
| home_team_win_probability_added | Change in home win prob |
| visitor_team_win_probility_added | Change in visitor win prob (note: typo in column name) |

#### `train/input_2023_wXX.csv` (Player tracking - BEFORE ball is thrown)
| Column | Description |
|--------|-------------|
| game_id | Unique game identifier |
| play_id | Unique play identifier |
| player_to_predict | TRUE if this player needs prediction, FALSE otherwise |
| nfl_id | Unique player identifier |
| frame_id | Frame number (1 = first frame, ~10 frames/second) |
| play_direction | Direction of offensive play (left/right) |
| absolute_yardline_number | Field position (0-100, with 0=own endzone) |
| player_name | Player's name |
| player_height | Height (e.g., "6-1") |
| player_weight | Weight in pounds |
| player_birth_date | Date of birth |
| player_position | Position (FS, SS, CB, LB, etc.) |
| player_side | Defense or Offense |
| player_role | Specific role (Defensive Coverage, etc.) |
| x | X coordinate on field (0-120, endzone to endzone) |
| y | Y coordinate on field (0-53.3, sideline to sideline) |
| s | Speed in yards/second |
| a | Acceleration in yards/second^2 |
| dir | Direction of movement (0-360 degrees) |
| o | Orientation/facing direction (0-360 degrees) |
| num_frames_output | Number of frames to predict |
| ball_land_x | X coordinate where ball will land |
| ball_land_y | Y coordinate where ball will land |

#### `train/output_2023_wXX.csv` (Player positions - WHILE ball is in air)
| Column | Description |
|--------|-------------|
| game_id | Unique game identifier |
| play_id | Unique play identifier |
| nfl_id | Player identifier |
| frame_id | Frame number during ball flight |
| x | Actual X coordinate |
| y | Actual Y coordinate |

### Key Insights About Data

1. **Frame rate:** ~10 frames per second
2. **num_frames_output:** Tells you how many frames are in output (varies by play, based on pass hang time)
3. **ball_land_x/y:** Known target location where ball will land

### Field Coordinate System
- X-axis: 0-120 yards (endzone to endzone, includes end zones)
- Y-axis: 0-53.3 yards (sideline to sideline)
- Play direction matters for interpreting coordinates

---

## Technical Decisions & Learnings

### Architecture Decisions
*(To be filled in as we make decisions)*

### Key Findings
*(To be filled in as we discover insights)*

### What Worked
*(To be filled in)*

### What Didn't Work
*(To be filled in)*

---

## Code Files

### Current Files
- `reorganize_data.py` - Script to reorganize data by play
- `add_supplementary.py` - Script to add supplementary data to each play folder
- `organize_by_game.py` - Script to organize plays hierarchically by week/game
- `merge_player_tracking.py` - Script to create per-player tracking files with continuous frames
- `create_combined_tracking.py` - Script to create combined all-players tracking file per play
- `ball_trajectory.py` - Physics-based ball trajectory interpolation (x, y, z for each frame)
- `frame_features.py` - Frame-by-frame feature engineering for completion probability model
- `train_completion_model.py` - Train LightGBM model for completion probability prediction
- `check_interception_prauc.py` - Validate interception model with PR-AUC analysis
- `test_smoothing.py` - Test Savitzky-Golay smoothing on probability trajectories
- `bifurcation_detection.py` - Detect bifurcation points using M3+M6 combo model
- `shap_bifurcation_analysis.py` - SHAP explainability for M3 vs M6 bifurcation triggers

### File Purposes
- **reorganize_data.py**: Reads all weekly input/output CSVs and creates individual folders per play in `plays_by_id/` directory
- **add_supplementary.py**: Extracts matching row from supplementary_data.csv for each play and saves as supplementary.csv
- **organize_by_game.py**: Creates hierarchical structure in `organized_plays/` with plays sorted chronologically within each game
- **merge_player_tracking.py**: Creates `players/` subfolder in each play with per-player tracking.csv files (continuous frame numbering, ball_in_air and in_both_files flags)
- **create_combined_tracking.py**: Creates `all_players_tracking.csv` in each play folder - single file with all players merged, continuous frames, ball_in_air and in_both_files flags. Also calculates velocity (s, dir, a) from position changes for output frames.
- **ball_trajectory.py**: Calculates ball position (x, y, z) for each frame during ball flight using physics-based parabolic arc. Uses QB position at release as start point, ball_land_x/y as endpoint, num_frames_output for duration.
- **frame_features.py**: Calculates frame-by-frame features for completion probability model including separation, closing speed, receiver/defender alignments, projected separation at catch point, etc.
- **train_completion_model.py**: Extracts features for all 14,105 plays, trains LightGBM multiclass model, evaluates with game-based train/test split. Saves model to `models/` folder.
- **check_interception_prauc.py**: Validates interception predictions aren't just random by computing PR-AUC and comparing to baseline.
- **bifurcation_detection.py**: Detects the "moment of truth" frame where a play's outcome becomes determined. Uses M3 (Confidence Threshold) + M6 (Z-Score Breakout) combo model. Outputs `bifurcation_results.csv` with metrics for all 14,105 plays.
- **shap_bifurcation_analysis.py**: SHAP explainability analysis comparing what features drove M3 vs M6 triggers. Creates waterfall plots and delta bar charts showing the "narrative shift" between early confidence and late anomaly detection.

---

## Reorganized Data Structure

### Flat Structure (plays_by_id/)
After running `reorganize_data.py` and `add_supplementary.py`:
```
plays_by_id/
    {game_id}_{play_id}/
        input.csv         (player tracking before ball thrown)
        output.csv        (player positions while ball in air)
        supplementary.csv (play-level metadata: formation, coverage, result, etc.)
```

### Hierarchical Structure (organized_plays/)
After running all data wrangling scripts:
```
organized_plays/
    Week_01/
        DET_vs_KC/
            001_Q1_1425_DET/    (sequence_quarter_clock_possession)
                input.csv
                output.csv
                supplementary.csv
                all_players_tracking.csv   <- MAIN FILE for visualization
                players/
                    46137_Justin_Reid/
                        tracking.csv   (frames 1-47, continuous)
                    43290_Jared_Goff/
                        tracking.csv   (frames 1-26, input only)
                    ...
            002_Q1_1256_KC/
            ...
        ARI_vs_WAS/
        ...
    Week_02/
    ...
    Week_18/
```

**Play folder naming:** `{sequence}_{quarter}_{clock}_{possession_team}/`
- Plays are ordered chronologically within each game

**all_players_tracking.csv columns:** (RECOMMENDED for visualization)
- All input columns (x, y, s, a, dir, o, player info, etc.)
- `ball_in_air`: False for pre-throw frames, True for ball-in-flight frames
- `in_both_files`: True if player tracked in both input AND output (key players)
- Sorted by nfl_id then frame_id

**Player tracking.csv columns:** (per-player files in players/ subfolder)
- Same columns as all_players_tracking.csv but for single player
- Useful for player-specific analysis

**Stats:**
- Total plays with tracking data: 14,108
- Total games: 349
- Weeks 1-18 of 2023 season
- Note: supplementary_data.csv has 18,009 rows but only 14,108 have corresponding tracking data

---

## Session Notes

### Session 1 (2024-12-04)
- Initial project setup
- Analyzed data structure
- Created this context file
- Understanding: Analytics track focuses on broadcast visualization of player movement while ball is in the air
- Created `reorganize_data.py` to split data into per-play folders
- Successfully reorganized all 14,108 plays into `plays_by_id/` directory
- Created `add_supplementary.py` to add play metadata to each folder
- Each play folder now contains: input.csv, output.csv, supplementary.csv
- Created `organize_by_game.py` to organize plays hierarchically by Week/Game
- Plays now in `organized_plays/` sorted chronologically within each game (349 games total)
- Created `merge_player_tracking.py` to create per-player tracking files with continuous frames
- 173,150 player folders created across all plays
- Created `create_combined_tracking.py` to create `all_players_tracking.csv` per play
- All 14,108 plays now have combined tracking file ready for visualization work

### Session 2 (2024-12-04)
- **Goal:** Build "outcome bifurcation points" - identify where probability swings happen during pass plays
- **Approach:** Frame-by-frame completion probability model with 3 outcomes: Completion, Incompletion, Interception
- Decided to drop PBU (Pass Breakup) as separate category due to ambiguity in classification
- Using hybrid approach: physics-based features → ML model

**Ball Trajectory Development:**
- Created `ball_trajectory.py` for physics-based ball position interpolation
- Key physics parameters:
  - z_start = 2.0 yards (QB release height ~6 ft)
  - z_end = 1.6 yards (catch height ~4.8 ft)
  - gravity = 10.71 yards/s² (9.8 m/s² converted to yards)
- Ball start: QB position at last input frame
- Ball end: ball_land_x/y from input data
- Flight duration: num_frames_output / 10 seconds (10 Hz frame rate)
- Output frame 1 is AFTER release (verified by checking player position shifts)

**Velocity Calculation:**
- Discovered output frames only have x, y (no s, dir, a velocity data)
- Updated `create_combined_tracking.py` to calculate velocity from position changes:
  - speed = sqrt(dx² + dy²) × fps
  - direction = arctan2(dx, dy) with NFL convention (0=north, 90=east, clockwise)
  - acceleration = change in speed × fps
- Re-ran on all 14,108 plays

**Frame Feature Engineering:**
- Created `frame_features.py` with features for each ball-in-air frame
- Features calculated:
  - `separation` - distance to nearest defender
  - `second_defender_dist` - distance to second nearest (if within 5 yards)
  - `closing_speed` - rate of change of separation (yards/sec)
  - `ball_to_receiver` - distance from ball to receiver
  - `ball_x`, `ball_y`, `ball_z` - ball position (from trajectory calculation)
  - `ball_catchable` - whether ball is below 3 yards (reachable)
  - `receiver_x`, `receiver_y`, `receiver_speed` - receiver position and speed
  - `receiver_to_catch_point` - distance from receiver to ball landing spot
  - `receiver_alignment` - how aligned receiver movement is with catch point (1=toward, -1=away)
  - `defender_speed`, `defender_to_catch_point`, `defender_alignment` - nearest defender metrics
  - `projected_separation_at_catch` - projected defender position at catch based on current velocity
  - `contested` - whether projected separation < 2 yards
  - `time_remaining`, `frames_remaining` - time until ball arrives

**Model Training:**
- Created `train_completion_model.py` for full ML pipeline
- Extracted features for 14,105 plays (160,318 frame-level observations)
- Train/test split BY GAME (not by play) to avoid leakage: 218 games train, 54 games test
- Model: LightGBM multiclass classifier (3 classes: Complete, Incomplete, Interception)

**Model Performance:**
- ROC-AUC (One-vs-Rest):
  - Complete: 0.8606
  - Incomplete: 0.8412
  - Interception: 0.8607
- Overall accuracy: 78%

**Top 5 Feature Importance:**
1. separation (175,279)
2. receiver_to_catch_point (128,312)
3. projected_separation_at_catch (62,856)
4. ball_to_receiver (56,518)
5. defender_to_catch_point (45,161)

**Interception Model Validation (PR-AUC):**
- PR-AUC: 0.2322
- Baseline (random): 0.0340
- Lift: 6.8x better than random
- Top 20 highest P(INT) predictions: 13/20 (65%) were actual interceptions
- Confirms model is doing real predictive work, not just guessing

**Output Files:**
- `all_frame_features.csv` - 160,318 rows of frame-level features
- `models/completion_model.lgb` - trained LightGBM model
- `models/feature_names.pkl` - feature names for model
- `models/feature_importance.csv` - feature importance ranking
- `interception_pr_curve.png` - PR curve visualization

**Probability Smoothing (for visualization):**
- Tested Savitzky-Golay filter for smoother broadcast visualization
- Settings: window_length=5, polyorder=2
- Results: 7-89% variance reduction while preserving endpoints
- Dampens frame-to-frame noise without distorting key transitions
- Output: `smoothing_comparison.png` - visual comparison of raw vs smoothed
- Function to use:
```python
from scipy.signal import savgol_filter

def smooth_probability_trajectory(probs, window_length=5, polyorder=2):
    if len(probs) < window_length:
        window_length = len(probs) if len(probs) % 2 == 1 else len(probs) - 1
        window_length = max(3, window_length)
    polyorder = min(polyorder, window_length - 1)
    smoothed = savgol_filter(probs, window_length, polyorder)
    return np.clip(smoothed, 0, 1)
```

**MODEL IS FROZEN** - No changes to completion probability model unless explicitly requested.

### Session 3 (2024-12-04) - Bifurcation Detection

**Goal:** Detect the "moment of truth" - the frame where a play's outcome becomes determined.

**Initial Approach - 9 Methods:**
Originally tested 9 different bifurcation detection methods:
- M1: Dominance Established (when winning outcome stays highest)
- M2: Max Probability Velocity (largest single-frame jump)
- M3: Confidence Threshold (crosses outcome-specific threshold)
- M4: Margin of Victory (gap exceeds threshold)
- M5: Shannon Entropy Collapse (uncertainty drops below threshold)
- M6: Z-Score Breakout (deviates from rolling baseline)
- M7: KL Divergence Surge (peak information gain)
- M8: CUSUM Shift (trend start detection)
- M9: Inflection Point (curvature zero-crossing)

**Accuracy Analysis:**
Tested which method best predicts actual outcome at bifurcation point (2000 play sample):

| Method | Overall Accuracy | INT Accuracy |
|--------|-----------------|--------------|
| M1 Dominance | 79.7% | 30.6% |
| M2 Max Velocity | 78.9% | 32.7% |
| **M3 Confidence** | **80.5%** | 34.7% |
| M4 Margin | 77.8% | 28.6% |
| M5 Shannon | 76.4% | 32.7% |
| **M6 Z-Score** | 75.9% | **42.9%** |
| M7 KL Surge | 70.9% | 30.6% |
| M8 CUSUM | 78.9% | 28.6% |
| M9 Inflection | 71.9% | 32.7% |

**Final Model - M3 + M6 Combo:**
Selected M3 (Confidence Threshold) and M6 (Z-Score Breakout) based on accuracy analysis:
- **M3**: Best overall accuracy (80.5%) - reliable for completions/incompletions
- **M6**: Best interception accuracy (42.9%) - catches sudden shifts better

**Combo Logic:**
```python
# If methods agree within 2 frames → use earlier detection
# For interceptions → prefer M6 (sudden events)
# For completions/incompletions → prefer M3 (higher reliability)
```

**M3 vs M6 Agreement (full dataset):**
- Agree within 2 frames: 44% of plays
- Agree within 5 frames: 68% of plays

**Bifurcation Timing by Outcome (Combo Model):**
| Outcome | Mean Timing | Prob at Release | Prob Swing | Lead Change % |
|---------|-------------|-----------------|------------|---------------|
| Complete | 0.206 | 0.623 | +0.029 | 13.3% |
| Incomplete | 0.381 | 0.403 | +0.110 | 35.0% |
| Interception | 0.552 | 0.069 | +0.236 | 80.0% |

**Key Insight:** Interceptions are the most dramatic plays - 80% have lead changes, bifurcation happens late (55% through flight), and require the largest probability swing (+23.6 percentage points).

**Output Files:**
- `bifurcation_results.csv` - Play-level bifurcation metrics for all 14,105 plays
- `bifurcation_timing_histograms.png` - Distribution of timing by method
- `bifurcation_m3_vs_m6.png` - M3 vs M6 comparison scatter
- `bifurcation_drama_score.png` - Drama score distribution
- `bifurcation_by_outcome_boxplot.png` - Timing by outcome
- `bifurcation_example_trajectories.png` - Example probability trajectories

**Columns in bifurcation_results.csv:**
- `bifurcation_frame_m3`, `bifurcation_frame_m6`, `bifurcation_frame_combo` - Frame indices
- `bifurcation_timing_m3`, `bifurcation_timing_m6`, `bifurcation_timing_combo` - Normalized 0-1
- `bifurcation_frame`, `bifurcation_timing` - Primary (combo) values
- `probability_at_release`, `probability_at_bifurcation`, `probability_swing`
- `margin_at_bifurcation` - Gap between winner and runner-up
- `pre_bifurcation_leader`, `had_lead_change` - Lead change analysis
- `drama_score`, `drama_integral` - Drama metrics
- `entropy_at_release`, `entropy_at_end`, `entropy_drop` - Entropy metrics

**SHAP Explainability Analysis:**
Created `shap_bifurcation_analysis.py` to explain WHY M3 and M6 trigger at different frames.

**Sample Analysis Results:**

| Play | Outcome | M3 Frame | M6 Frame | Key Narrative Shift |
|------|---------|----------|----------|---------------------|
| 040_Q4_122_BUF | Complete | 0 | 7 | Receiver slowed (-4 yds/s), separation tightened |
| 023_Q2_109_WAS | Incomplete | 2 | 5 | Separation collapsed 0.92→0.30 yds |
| 002_Q1_110_CAR | Interception | 2 | 9 | 2nd defender closed in (-4 yds), closing speed +10.5 yds/s |

**What Each Method "Sees":**
- **M3 (Confidence)**: Triggers on absolute thresholds - receiver proximity to catch point, baseline separation levels
- **M6 (Z-Score)**: Triggers on sudden anomalies - closing speed spikes, separation collapse, defender breaks on ball

**Key SHAP Insight for Interceptions:**
The M6 trigger for interceptions captures the **"defender broke on the ball"** moment that M3 misses. Top SHAP drivers at M6:
- `second_defender_dist` closing in (+0.90 SHAP)
- `closing_speed` spike (+0.86 SHAP)
- `defender_to_catch_point` shrinking (+0.76 SHAP)

**Output Files:**
- `shap_waterfall_*.png` - Side-by-side SHAP waterfalls for M3 vs M6 trigger frames
- `shap_delta_*.png` - Bar charts showing feature importance change between triggers

**Next Steps:**
- Build broadcast visualization
- Create animated probability trajectory graphics

---

## Important Reminders

1. **Submission format:** Check Kaggle for exact submission format requirements
2. **Focus on BROADCAST visualization** - make it engaging for TV viewers
3. **Ball landing position is known** - available in data as ball_land_x/y
4. **MUST use player tracking data** - writeups without tracking data will not be scored
5. **Hide most code in Appendix** - show code only when it helps understanding
6. **Video required** - must be dynamic, engaging, and high-quality

---

## Useful Code Snippets

```python
# RECOMMENDED: Load a single play's combined tracking data
import pandas as pd
from pathlib import Path

play_path = Path("organized_plays/Week_01/DET_vs_KC/001_Q1_1425_DET")
tracking = pd.read_csv(play_path / "all_players_tracking.csv")
supplementary = pd.read_csv(play_path / "supplementary.csv")

# Filter by ball_in_air to separate pre-throw vs ball-in-flight
pre_throw = tracking[tracking['ball_in_air'] == False]
ball_in_flight = tracking[tracking['ball_in_air'] == True]

# Get key players (tracked throughout entire play)
key_players = tracking[tracking['in_both_files'] == True]
```

```python
# Load raw weekly data (original format)
import pandas as pd

week = 1
input_df = pd.read_csv(f'114239_nfl_competition_files_published_analytics_final/train/input_2023_w{week:02d}.csv')
output_df = pd.read_csv(f'114239_nfl_competition_files_published_analytics_final/train/output_2023_w{week:02d}.csv')
supplementary_df = pd.read_csv('114239_nfl_competition_files_published_analytics_final/supplementary_data.csv')

# Merge play context
merged = input_df.merge(supplementary_df, on=['game_id', 'play_id'])
```
