"""
Create visualizations of model performance metrics.
Separate charts for ROC-AUC and PR-AUC across all three outcomes.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Performance metrics from check_interception_prauc.py output
roc_auc_scores = {
    'Complete': 0.8606,
    'Incomplete': 0.8412,
    'Interception': 0.8607
}

pr_auc_scores = {
    'Complete': 0.8941,
    'Incomplete': 0.7660,
    'Interception': 0.2356
}

pr_baselines = {
    'Complete': 0.6306,
    'Incomplete': 0.3354,
    'Interception': 0.0340
}

# Calculate lifts for PR-AUC
pr_lifts = {
    'Complete': 1.4,
    'Incomplete': 2.3,
    'Interception': 6.9
}

# Colors for each outcome
colors = {
    'Complete': '#00CED1',      # Cyan
    'Incomplete': '#FF4444',    # Red
    'Interception': '#FF00FF'   # Magenta
}

# =============================================================================
# ROC-AUC VISUALIZATION
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

outcomes = list(roc_auc_scores.keys())
scores = list(roc_auc_scores.values())
outcome_colors = [colors[o] for o in outcomes]

# Create bar chart
bars = ax.bar(outcomes, scores, color=outcome_colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add random baseline line
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Random Baseline (0.50)', zorder=1)

# Add value labels on bars
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{score:.4f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Formatting
ax.set_ylim(0, 1.0)
ax.set_ylabel('ROC-AUC Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Outcome', fontsize=14, fontweight='bold')
ax.set_title('Model Performance: ROC-AUC by Outcome\n(One-vs-Rest Classification)',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(BASE_DIR / "model_roc_auc_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {BASE_DIR / 'model_roc_auc_comparison.png'}")
plt.close()

# =============================================================================
# PR-AUC VISUALIZATION
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(outcomes))
width = 0.35

# Create grouped bars
bars1 = ax.bar(x - width/2, [pr_auc_scores[o] for o in outcomes], width,
               label='Model PR-AUC', color=outcome_colors, alpha=0.8,
               edgecolor='black', linewidth=2)
bars2 = ax.bar(x + width/2, [pr_baselines[o] for o in outcomes], width,
               label='Baseline (Random)', color='gray', alpha=0.5,
               edgecolor='black', linewidth=2)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add lift annotations
for i, outcome in enumerate(outcomes):
    lift = pr_lifts[outcome]
    model_score = pr_auc_scores[outcome]
    baseline = pr_baselines[outcome]

    # Arrow from baseline to model
    ax.annotate('', xy=(i, model_score), xytext=(i, baseline),
                arrowprops=dict(arrowstyle='<->', lw=2, color='green', alpha=0.7))

    # Lift text
    mid_y = (model_score + baseline) / 2
    ax.text(i + 0.4, mid_y, f'{lift:.1f}x\nlift',
            ha='left', va='center', fontsize=10, fontweight='bold',
            color='green',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='green'))

# Formatting
ax.set_ylim(0, 1.0)
ax.set_ylabel('PR-AUC Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Outcome', fontsize=14, fontweight='bold')
ax.set_title('Model Performance: PR-AUC by Outcome\n(Precision-Recall Analysis)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(outcomes, fontsize=12)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(BASE_DIR / "model_pr_auc_comparison.png", dpi=300, bbox_inches='tight')
print(f"Saved: {BASE_DIR / 'model_pr_auc_comparison.png'}")
plt.close()

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "="*70)
print("MODEL PERFORMANCE SUMMARY")
print("="*70)
print(f"\n{'Outcome':<15} {'ROC-AUC':<12} {'PR-AUC':<12} {'Baseline':<12} {'Lift':<8}")
print("-"*70)
for outcome in outcomes:
    print(f"{outcome:<15} {roc_auc_scores[outcome]:<12.4f} {pr_auc_scores[outcome]:<12.4f} "
          f"{pr_baselines[outcome]:<12.4f} {pr_lifts[outcome]:<8.1f}x")

print("\n" + "="*70)
print("KEY INSIGHTS:")
print("="*70)
print("✓ ROC-AUC ~0.86 across all outcomes = EXCELLENT discrimination")
print("✓ Interception PR-AUC 6.9x better than random despite 3.4% prevalence")
print("✓ Complete PR-AUC 0.89 = strong precision-recall for majority class")
print("✓ Model performs well across all three outcomes, handling class imbalance")
print("="*70)
