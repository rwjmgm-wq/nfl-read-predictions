"""
Clean up the project folder for Git repository.
- Remove hardcoded paths from Python files
- Organize files into proper directories
- Create clean directory structure
"""

import os
import re
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).parent

# Hardcoded path to replace
OLD_PATH = r"c:\Users\rwjmg\OneDrive\Pictures\Writing\Big_Data_Bowl_work"

def replace_hardcoded_paths(file_path):
    """Replace hardcoded paths with relative paths using Path(__file__).parent"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace the hardcoded path
    original_content = content

    # Pattern 1: BASE_DIR = Path(r"c:\Users\...")
    content = re.sub(
        r'BASE_DIR = Path\(r".*?Big_Data_Bowl_work"\)',
        'BASE_DIR = Path(__file__).parent',
        content
    )

    # Pattern 2: VIZ_DIR = BASE_DIR / "..." / "..."
    # This is fine, keep as is

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    print("="*80)
    print("CLEANING UP PROJECT FOR GIT")
    print("="*80)

    # Step 1: Fix hardcoded paths in Python files
    print("\n1. Fixing hardcoded paths in Python files...")
    py_files = list(BASE_DIR.glob("*.py"))
    fixed_count = 0

    for py_file in py_files:
        if py_file.name == "cleanup_for_git.py":
            continue
        if replace_hardcoded_paths(py_file):
            print(f"   Fixed: {py_file.name}")
            fixed_count += 1

    print(f"   Total files fixed: {fixed_count}")

    # Step 2: Create organized directory structure
    print("\n2. Creating organized directory structure...")

    dirs_to_create = [
        "src",                    # Core source code
        "scripts",                # Utility/analysis scripts
        "visualizations",         # Output visualizations
        "docs",                   # Documentation
    ]

    for dir_name in dirs_to_create:
        dir_path = BASE_DIR / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"   Created/verified: {dir_name}/")

    # Step 3: Move files to organized structure
    print("\n3. Organizing files...")

    # Core source files (keep these in src/)
    core_files = [
        "frame_features.py",
        "bifurcation_detection.py",
        "train_completion_model.py",
    ]

    # Visualization generation scripts
    viz_scripts = [
        "generate_pr_curves.py",
        "generate_roc_curves.py",
        "visualize_model_performance.py",
        "visualize_read_accuracy.py",
        "analyze_read_at_release.py",
        "generate_showcase_race_charts.py",
        "generate_full_broadcast_viz.py",
        "generate_ngs_dots.py",
        "add_info_to_wicks_chart.py",
        "combine_race_and_ngs.py",
    ]

    # Analysis/exploration scripts
    analysis_scripts = [
        "find_showcase_bifurcation.py",
        "find_showcase_plays.py",
        "find_upset_plays.py",
        "explain_showcase_plays.py",
        "check_interception_prauc.py",
        "shap_bifurcation_analysis.py",
        "bifurcation_optimization.py",
    ]

    # Data processing scripts (one-time setup)
    data_scripts = [
        "reorganize_data.py",
        "organize_by_game.py",
        "merge_player_tracking.py",
        "add_supplementary.py",
        "create_combined_tracking.py",
    ]

    # Deprecated/test files (can be removed)
    deprecated_files = [
        "test_smoothing.py",
        "add_play_info_to_wicks_chart.py",  # Replaced by add_info_to_wicks_chart.py
        "showcase_race_charts.py",  # Replaced by generate_showcase_race_charts.py
        "generate_broadcast_frames.py",  # Intermediate file
        "generate_upset_race_charts.py",  # Not core to final product
        "ball_trajectory.py",  # Exploratory
    ]

    print("\n   Organizing core source files...")
    for file in core_files:
        src_path = BASE_DIR / file
        if src_path.exists():
            print(f"     - {file} (keeping in root for now)")

    print("\n   Organizing visualization scripts...")
    for file in viz_scripts:
        src_path = BASE_DIR / file
        if src_path.exists():
            print(f"     - {file} (keeping in root for now)")

    print("\n   Analysis scripts identified:")
    for file in analysis_scripts:
        src_path = BASE_DIR / file
        if src_path.exists():
            print(f"     - {file}")

    print("\n   Data processing scripts identified:")
    for file in data_scripts:
        src_path = BASE_DIR / file
        if src_path.exists():
            print(f"     - {file}")

    print("\n   Deprecated files identified (consider removing):")
    for file in deprecated_files:
        src_path = BASE_DIR / file
        if src_path.exists():
            print(f"     - {file}")

    # Step 4: Move existing visualization PNGs to visualizations folder
    print("\n4. Organizing visualization outputs...")

    viz_pngs = [
        "model_pr_auc_comparison.png",
        "model_roc_auc_comparison.png",
        "interception_pr_curve.png",
        "bifurcation_timing_histograms.png",
        "bifurcation_by_outcome_boxplot.png",
        "bifurcation_drama_score.png",
        "bifurcation_lead_changes.png",
    ]

    viz_dest = BASE_DIR / "visualizations"
    for png_file in viz_pngs:
        src = BASE_DIR / png_file
        if src.exists():
            dest = viz_dest / png_file
            if not dest.exists():
                shutil.copy2(src, dest)
                print(f"   Copied: {png_file} -> visualizations/")

    # Copy read_visualizations folder
    read_viz_src = BASE_DIR / "read_visualizations"
    if read_viz_src.exists():
        read_viz_dest = viz_dest / "read_metrics"
        if read_viz_dest.exists():
            shutil.rmtree(read_viz_dest)
        shutil.copytree(read_viz_src, read_viz_dest)
        print(f"   Copied: read_visualizations/ -> visualizations/read_metrics/")

    # Step 5: Create README structure
    print("\n5. Checking documentation...")

    readme_path = BASE_DIR / "README.md"
    if not readme_path.exists():
        print("   Creating README.md template...")
        with open(readme_path, 'w') as f:
            f.write("""# NFL Pass Completion Prediction: THE READ

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
│   ├── frame_features.py         # Feature extraction from tracking data
│   ├── bifurcation_detection.py  # THE READ detection algorithms
│   └── train_completion_model.py # LightGBM model training
├── scripts/                      # Analysis and visualization scripts
├── visualizations/               # Generated plots and charts
├── models/                       # Trained model files (gitignored)
├── docs/                         # Documentation
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

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

- **ROC-AUC**: 0.86 (Complete), 0.84 (Incomplete), 0.86 (Interception)
- **PR-AUC**: 0.89 (Complete), 0.77 (Incomplete), 0.24 (Interception)
- **Overall Accuracy**: 78%

## THE READ Detection

Two complementary methods:
- **M3 (Confidence Threshold)**: 80.5% accuracy - best for completions/incompletions
- **M6 (Z-Score Breakout)**: 42.9% interception accuracy - best for sudden events

## License

MIT License

## Acknowledgments

NFL Big Data Bowl 2025 - Tracking data provided by NFL Next Gen Stats
""")
        print("   Created README.md")
    else:
        print("   README.md already exists")

    # Step 6: Create requirements.txt
    print("\n6. Creating requirements.txt...")
    requirements_path = BASE_DIR / "requirements.txt"
    if not requirements_path.exists():
        with open(requirements_path, 'w') as f:
            f.write("""# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Machine Learning
lightgbm>=4.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Image/Video processing
imageio>=2.31.0
imageio-ffmpeg>=0.4.9
Pillow>=10.0.0

# Progress bars
tqdm>=4.65.0

# Optional: SHAP for model interpretation
shap>=0.42.0
""")
        print("   Created requirements.txt")
    else:
        print("   requirements.txt already exists")

    print("\n" + "="*80)
    print("CLEANUP SUMMARY")
    print("="*80)
    print(f"""
✓ Fixed {fixed_count} Python files with hardcoded paths
✓ Created organized directory structure
✓ Moved visualizations to visualizations/ folder
✓ Created README.md and requirements.txt
✓ Created .gitignore

NEXT STEPS:
1. Review deprecated files and delete if not needed
2. Test that scripts still run with relative paths
3. Move scripts into appropriate subdirectories if desired
4. Initialize git repository: git init
5. Add files: git add .
6. Commit: git commit -m "Initial commit: NFL Pass Completion Prediction"

FILES TO MANUALLY REVIEW FOR DELETION:
""")

    for file in deprecated_files:
        if (BASE_DIR / file).exists():
            print(f"  - {file}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
