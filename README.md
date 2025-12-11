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
├── visualizations/               # Generated plots and charts
│   └── read_metrics/            # THE READ accuracy visualizations
├── broadcast_viz/                # Broadcast-style race chart outputs
├── models/                       # Trained model files (gitignored)
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Core Files

### Model Training & Features
- `train_completion_model.py` - Train LightGBM completion probability model
- `frame_features.py` - Extract features from tracking data
- `bifurcation_detection.py` - THE READ detection algorithms (M3 & M6)

### Visualization Scripts
- `visualize_read_accuracy.py` - Generate THE READ accuracy visualizations
- `analyze_read_at_release.py` - Analyze timing relative to ball release
- `visualize_model_performance.py` - ROC-AUC and PR-AUC comparisons
- `generate_full_broadcast_viz.py` - Create broadcast-style race charts
- `add_info_to_wicks_chart.py` - Enhanced race charts with play information

### Analysis Scripts
- `find_showcase_bifurcation.py` - Find plays with interesting READ moments
- `find_showcase_plays.py` - Identify showcase plays for visualization

## Usage

### Train the Model
```bash
python train_completion_model.py
```

### Generate THE READ Visualizations
```bash
python visualize_read_accuracy.py
python analyze_read_at_release.py
```

### Create Broadcast-Style Race Charts
```bash
python generate_full_broadcast_viz.py
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
