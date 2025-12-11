# NFL Pass Completion Prediction: THE READ

Predicting pass play outcomes and identifying the critical moment when outcomes become determined.

## Project Overview

This project uses machine learning and tracking data to:
1. **Predict pass play outcomes** (Complete/Incomplete/Interception) frame-by-frame
2. **Identify THE READ** - the critical moment when a play's outcome becomes determined
3. **Visualize outcome probabilities** as dynamic race charts

## Key Findings

- **76.8% of plays are decided at the moment of release**
- **87.9% of completions** are determined when the QB throws the ball
- **THE READ detection accuracy**: 78-80% overall
- **Mean THE READ timing**: 11.06% into play (typically at release)

## Repository Structure

```
├── src/                          # Core source code
│   ├── train_completion_model.py # Train LightGBM model
│   ├── frame_features.py         # Feature extraction
│   └── bifurcation_detection.py  # THE READ detection (M3 & M6)
│
├── visualization/                # Visualization scripts
│   ├── visualize_read_accuracy.py      # THE READ accuracy charts
│   ├── analyze_read_at_release.py      # Release timing analysis
│   ├── visualize_model_performance.py  # ROC/PR curves
│   ├── generate_full_broadcast_viz.py  # Broadcast race charts
│   ├── generate_showcase_race_charts.py # Showcase plays
│   ├── add_info_to_wicks_chart.py      # Enhanced race charts
│   └── combine_race_and_ngs.py         # Combined visualizations
│
├── analysis/                     # Analysis and exploration
│   ├── find_showcase_bifurcation.py    # Find interesting READs
│   ├── find_showcase_plays.py          # Identify showcase plays
│   ├── bifurcation_optimization.py     # Method optimization
│   └── shap_bifurcation_analysis.py    # SHAP analysis
│
├── data_processing/              # Data preparation (one-time setup)
│   ├── reorganize_data.py        # Organize raw data
│   ├── organize_by_game.py       # Game-level organization
│   └── merge_player_tracking.py  # Merge tracking files
│
├── docs/                         # Documentation
│   └── BIG_DATA_BOWL_SUBMISSION_GUIDE.md
│
├── models/                       # Trained models
│   └── completion_model.lgb      # LightGBM model (2.7MB)
│
├── visualizations/               # Generated outputs
│   ├── read_metrics/            # THE READ accuracy charts
│   ├── *.png                    # Model performance charts
│
├── broadcast_viz/                # Broadcast-style outputs
│   └── [play folders]/          # Race charts & NGS frames
│
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Model
```bash
python src/train_completion_model.py
```

### 2. Generate THE READ Visualizations
```bash
python visualization/visualize_read_accuracy.py
python visualization/analyze_read_at_release.py
```

### 3. Create Broadcast-Style Race Charts
```bash
python visualization/generate_full_broadcast_viz.py
```

## Model Performance

**Completion Probability Model (LightGBM):**
- **ROC-AUC**: 0.86 (Complete), 0.84 (Incomplete), 0.86 (Interception)
- **PR-AUC**: 0.89 (Complete), 0.77 (Incomplete), 0.24 (Interception)
- **Overall Accuracy**: 78%

**Top Features:**
1. separation (175,279)
2. target_defender_distance (54,502)
3. ball_trajectory_quality (53,919)
4. qb_pocket_time (32,177)
5. defender_closing_speed (31,564)

## THE READ Detection

Two complementary methods working together:

### M3: Confidence Threshold (80.5% accuracy)
- Best for completions and incompletions
- Triggers when probability crosses outcome-specific threshold

### M6: Z-Score Breakout (42.9% interception accuracy)
- Best for detecting sudden events (tips, breaks)
- Detects when probability deviates significantly from rolling baseline

## THE READ Timing Breakdown

**At Ball Release (Frame 0):**
- 76.8% of ALL plays
- 87.9% of completions
- 55.8% of incompletions
- 6.8% of interceptions

**Within 0.2s of Release (Frames 0-2):**
- 85.0% of all plays
- 91.9% of completions
- 72.7% of incompletions
- 31.5% of interceptions

**Key Insight**: For most plays, the outcome is effectively "locked in" by the QB's decision at release.

## Data Requirements

This project requires NFL Big Data Bowl 2025 tracking data. Place data files in:
- `organized_plays/` - Organized play-by-play tracking data

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- pandas, numpy, scipy
- lightgbm, scikit-learn
- matplotlib, seaborn
- imageio, Pillow

## License

MIT License

## Acknowledgments

NFL Big Data Bowl 2025 - Tracking data provided by NFL Next Gen Stats
