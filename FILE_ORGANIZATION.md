# File Organization Guide

## Quick Reference: Where to Find Things

### 🎯 Want to train the model?
**Location:** `src/train_completion_model.py`

### 📊 Want to create visualizations?
**Location:** `visualization/` folder
- **THE READ accuracy charts:** `visualize_read_accuracy.py`
- **Release timing analysis:** `analyze_read_at_release.py`
- **Model performance (ROC/PR):** `visualize_model_performance.py`
- **Broadcast race charts:** `generate_full_broadcast_viz.py`
- **Showcase plays:** `generate_showcase_race_charts.py`

### 🔍 Want to analyze plays?
**Location:** `analysis/` folder
- **Find interesting READs:** `find_showcase_bifurcation.py`
- **Optimize detection methods:** `bifurcation_optimization.py`
- **SHAP analysis:** `shap_bifurcation_analysis.py`

### 🗂️ Want to process raw data?
**Location:** `data_processing/` folder
- **Organize raw data:** `reorganize_data.py`
- **Merge tracking files:** `merge_player_tracking.py`

### 📚 Want documentation?
**Location:** `docs/` folder
- **Submission guide:** `BIG_DATA_BOWL_SUBMISSION_GUIDE.md`
- **Project context:** `PROJECT_CONTEXT.md`

---

## Directory Structure

```
📦 nfl-read-predictions/
│
├── 📂 src/                        # Core ML code (3 files)
│   ├── train_completion_model.py  # ⭐ Start here to train
│   ├── frame_features.py          # Feature extraction logic
│   └── bifurcation_detection.py   # THE READ detection (M3 & M6)
│
├── 📂 visualization/              # Make charts & videos (10 files)
│   ├── visualize_read_accuracy.py      # ⭐ THE READ accuracy
│   ├── analyze_read_at_release.py      # ⭐ Release timing stats
│   ├── visualize_model_performance.py  # ROC/PR curves
│   ├── generate_full_broadcast_viz.py  # Full broadcast package
│   ├── generate_showcase_race_charts.py # Top plays
│   ├── generate_pr_curves.py           # PR curves only
│   ├── generate_roc_curves.py          # ROC curves only
│   ├── generate_ngs_dots.py            # NGS tracking viz
│   ├── add_info_to_wicks_chart.py      # Enhanced race chart
│   └── combine_race_and_ngs.py         # Combine GIFs
│
├── 📂 analysis/                   # Research & exploration (7 files)
│   ├── find_showcase_bifurcation.py    # Find dramatic READs
│   ├── find_showcase_plays.py          # Identify top plays
│   ├── find_upset_plays.py             # Find unexpected outcomes
│   ├── explain_showcase_plays.py       # Generate explanations
│   ├── bifurcation_optimization.py     # Optimize M3/M6
│   ├── check_interception_prauc.py     # Interception metrics
│   └── shap_bifurcation_analysis.py    # Feature importance
│
├── 📂 data_processing/            # One-time setup (5 files)
│   ├── reorganize_data.py         # Organize raw data
│   ├── organize_by_game.py        # Game-level structure
│   ├── merge_player_tracking.py   # Merge tracking CSVs
│   ├── add_supplementary.py       # Add supplementary data
│   └── create_combined_tracking.py # Combine all tracking
│
├── 📂 docs/                       # Documentation (5 files)
│   ├── BIG_DATA_BOWL_SUBMISSION_GUIDE.md # ⭐ How to submit
│   ├── PROJECT_CONTEXT.md          # Full project history
│   ├── CLEANUP_SUMMARY.md          # What we cleaned
│   ├── submission_example.md       # Example writeup
│   └── cleanup_for_git.py          # Cleanup script
│
├── 📂 models/                     # Trained models
│   ├── completion_model.lgb       # ⭐ Main model (2.7MB)
│   └── feature_importance.csv     # Feature rankings
│
├── 📂 visualizations/             # Generated outputs
│   ├── read_metrics/              # THE READ charts
│   └── *.png                      # Model performance
│
├── 📂 broadcast_viz/              # Race chart outputs
│   ├── Love_to_Wicks_17yd_Complete/
│   ├── Will_Levis_TD/
│   ├── Carr_51yd_Bomb_to_Shaheed/
│   └── Zach_Wilson_Incomplete/
│
├── 📂 showcase_race_charts/       # Top play charts
├── 📂 shap_plots/                 # SHAP visualizations
│
├── 📄 README.md                   # ⭐ Start here!
├── 📄 requirements.txt            # Python dependencies
├── 📄 .gitignore                  # What's excluded
└── 📄 FILE_ORGANIZATION.md        # This file!
```

---

## Common Tasks

### 1. **I want to run the whole pipeline**
```bash
# Step 1: Train model
python src/train_completion_model.py

# Step 2: Generate THE READ analysis
python visualization/visualize_read_accuracy.py
python visualization/analyze_read_at_release.py

# Step 3: Create broadcast visualizations
python visualization/generate_full_broadcast_viz.py
```

### 2. **I want to find interesting plays**
```bash
python analysis/find_showcase_bifurcation.py
python analysis/find_showcase_plays.py
```

### 3. **I want to understand model performance**
```bash
python visualization/visualize_model_performance.py
python visualization/generate_pr_curves.py
python visualization/generate_roc_curves.py
```

### 4. **I want to create a race chart for one play**
```bash
python visualization/generate_showcase_race_charts.py
```

### 5. **I want to process raw tracking data** (one-time setup)
```bash
python data_processing/reorganize_data.py
python data_processing/organize_by_game.py
```

---

## File Count by Category

- **Core Source:** 3 files (src/)
- **Visualization:** 10 files (visualization/)
- **Analysis:** 7 files (analysis/)
- **Data Processing:** 5 files (data_processing/)
- **Documentation:** 5 files (docs/)
- **Total Python files:** 30 files

---

## Most Important Files (⭐ Start Here!)

1. **README.md** - Project overview
2. **src/train_completion_model.py** - Train the model
3. **visualization/visualize_read_accuracy.py** - THE READ metrics
4. **visualization/analyze_read_at_release.py** - Release timing
5. **docs/BIG_DATA_BOWL_SUBMISSION_GUIDE.md** - Submission guide

---

## For Kaggle Notebook

When creating your Kaggle submission, you'll primarily use code from:
- **src/** - Model training and THE READ detection
- **visualization/** - Charts and analysis
- **analysis/** - Finding showcase plays

You can ignore:
- **data_processing/** - One-time data setup
- **docs/** - Supporting documentation
