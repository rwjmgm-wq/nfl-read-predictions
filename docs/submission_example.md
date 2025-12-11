# THE READ: Real-Time Pass Outcome Prediction for Broadcast

**NFL Big Data Bowl 2026 - Broadcast Visualization Track**

---

## Introduction

Every pass play contains a "moment of truth" — the instant when the outcome becomes inevitable. We call this **THE READ**, borrowing from quarterback terminology. Our system predicts pass outcomes (completion, incompletion, interception) in real-time as the ball travels through the air, identifying the exact frame when the play's fate is sealed.

This creates a powerful broadcast tool: an animated probability race chart that viewers can follow alongside the play, building tension until THE READ reveals the outcome.

## Methodology

### Data & Feature Engineering

We analyzed **14,108 pass plays** from the 2023 NFL season, tracking receiver-defender dynamics frame-by-frame while the ball is in the air.

**Key features extracted per frame:**
- **Separation**: Current distance between receiver and nearest defender
- **Projected Separation at Catch**: Estimated separation when ball arrives (accounting for player trajectories)
- **Closing Speed**: Rate at which defender is gaining/losing ground
- **Receiver/Defender Speed Differential**: Who has the athletic advantage
- **Distance to Catch Point**: How far each player must travel to the ball's landing spot
- **Defender Can Arrive**: Binary flag if defender geometry allows a play on the ball

For uncovered receivers (no defender within 20 yards), we assign default values reflecting an open target.

### Model Architecture

We trained a **LightGBM multiclass classifier** predicting three outcomes:
- Complete (68.8% of plays)
- Incomplete (28.2% of plays)
- Interception (3.0% of plays)

The model outputs frame-by-frame probability distributions, updated 10 times per second as the ball travels.

### THE READ Detection

To identify when the outcome becomes determined, we use a **time-weighted scoring method**:

```
score = P(actual_outcome) × (1 - normalized_time)^α
```

Where α=0.5 balances early detection with confidence. THE READ occurs at the frame maximizing this score while the correct outcome leads all probabilities.

**Performance:**
- 87.8% accuracy at THE READ
- Average timing: 6.4% into play (very early detection)
- Predictive Value: 0.821

| Outcome | Accuracy | Avg Timing |
|---------|----------|------------|
| Complete | 98.5% | 5.0% |
| Incomplete | 65.5% | 11.2% |
| Interception | 58.3% | 8.9% |

## Broadcast Visualization

### The Race Chart

Our primary visualization is an animated "race chart" showing all three outcome probabilities competing in real-time:

![Race Chart Example](race_chart_example.gif)

**Design choices for broadcast:**
- **Cyan for Complete** (stands out against green grass)
- **Red for Incomplete**
- **Magenta for Interception**
- Transparent background for overlay on game footage
- Current percentages displayed with high-contrast labels

### THE READ Marker

When THE READ occurs, a gold star and vertical line mark the moment, with annotation showing how early in the play the outcome was determined.

[See animated visualization: race_chart_THE_READ.gif]

## Case Study: Will Levis 61-Yard TD to DeAndre Hopkins

**Week 8: ATL @ TEN, Q3, 1:46 remaining**

This play demonstrates the model's ability to read developing situations:

| Frame | Complete | Incomplete | INT | Key Factor |
|-------|----------|------------|-----|------------|
| 0 | 27.6% | 70.0% | 2.4% | Deep throw, defender in position |
| 3 | **80.4%** | 18.1% | 1.5% | **THE READ** - Hopkins separation clear |
| 6 | 70.8% | 25.1% | 4.1% | Defender closing, uncertainty rises |
| 27 | 95.7% | 4.2% | 0.1% | Catch imminent |

**What caused the dip from frame 3-6?**

Analysis of feature changes reveals the defender made a recovery run:
- Projected separation at catch dropped from 9.87 to 7.09 yards
- Closing speed increased (defender gaining ground)
- Receiver-defender speed differential narrowed

The model correctly identified this threat before Hopkins ultimately won the race.

## Route-Level Performance

The model excels at different route types:

| Route | Accuracy | Notes |
|-------|----------|-------|
| FLAT | 88.1% | Short, quick, predictable |
| SCREEN | 88.0% | Clear completion geometry |
| ANGLE | 85.1% | Timing routes |
| GO | 78.0% | Deep uncertainty |
| IN | 75.0% | Traffic over middle |
| POST | 76.2% | Contested deep balls |

**Insight**: Short routes with clear separation are easiest to predict. Deep routes and crossing patterns over the middle retain uncertainty longer.

## Broadcast Applications

### 1. Live Probability Overlay
Display the race chart in corner of screen during replay, synced frame-by-frame with game footage.

### 2. THE READ Highlight
Pause replay at THE READ moment: "Right here — the model sees Hopkins has won. 80% completion probability with 90% of the play still remaining."

### 3. Upset Plays
Feature plays where the model was wrong for dramatic effect:
- Herbert INT vs KC: Model said 92% complete, intercepted
- Carr 51-yard bomb: Model said 89% incomplete, completed

### 4. Route Analysis Segments
"On GO routes, our model struggles more — these deep balls are inherently unpredictable. But watch this FLAT route — the model knows within 2 frames."

## Files & Resources

**Visualization Assets:**
- `race_chart_THE_READ.gif` - Animated race chart with THE READ marker
- `race_chart_THE_READ.webm` - Transparent video for overlay
- `frames_transparent_v4/` - Individual PNGs for custom editing

**Analysis Files:**
- `frame_analysis.csv` - Per-frame features and probabilities
- `commentary_notes.txt` - Key moments and feature importance
- `feature_importance.csv` - Model feature rankings

**Model:**
- `completion_model.lgb` - Trained LightGBM model
- `feature_names.pkl` - Feature configuration

## Conclusion

THE READ transforms complex probability modeling into an intuitive broadcast element. Viewers can watch the race chart and feel the tension build — will completion pull ahead? Is the defender closing? — until THE READ locks in the outcome, often before the ball arrives.

The system performs best on short, quick routes (88% accuracy) while appropriately reflecting uncertainty on deep balls and contested catches. This matches viewer intuition: a screen pass feels "safe" while a deep shot is anyone's guess.

Future work could incorporate:
- Pre-snap features (coverage type, formation)
- Weather conditions
- Quarterback tendencies
- Real-time integration with broadcast systems

---

*Code and visualizations available at: [GitHub Repository Link]*

*Contact: [Your Email]*
