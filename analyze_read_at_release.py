"""
Analyze THE READ timing relative to ball release.
What percentage of plays are "decided" at the moment the ball is thrown?
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Load bifurcation results
print("Loading bifurcation results...")
df = pd.read_csv(BASE_DIR / "bifurcation_results.csv")
print(f"Loaded {len(df)} plays\n")

# Frame 0 is typically the ball release moment (or very close to it)
# THE READ at frame 0 means the outcome was determined at the moment of release

print("="*80)
print("THE READ TIMING RELATIVE TO BALL RELEASE")
print("="*80)

# Count READs by frame timing
frame_0_reads = (df['bifurcation_frame_combo'] == 0).sum()
frame_1_reads = (df['bifurcation_frame_combo'] == 1).sum()
frame_2_reads = (df['bifurcation_frame_combo'] == 2).sum()
frame_3_reads = (df['bifurcation_frame_combo'] == 3).sum()

# Frames 0-2 = "at release" (within 0.2s of release)
at_release = (df['bifurcation_frame_combo'] <= 2).sum()
early_flight = (df['bifurcation_frame_combo'] > 2) & (df['bifurcation_frame_combo'] <= 5)
early_flight = early_flight.sum()
mid_flight = (df['bifurcation_frame_combo'] > 5) & (df['bifurcation_frame_combo'] <= 10)
mid_flight = mid_flight.sum()
late_flight = (df['bifurcation_frame_combo'] > 10).sum()

total = len(df)

print("\n--- BY FRAME NUMBER ---")
print(f"Frame 0 (ball release):     {frame_0_reads:6,} plays ({frame_0_reads/total:6.1%})")
print(f"Frame 1 (0.1s after):       {frame_1_reads:6,} plays ({frame_1_reads/total:6.1%})")
print(f"Frame 2 (0.2s after):       {frame_2_reads:6,} plays ({frame_2_reads/total:6.1%})")
print(f"Frame 3 (0.3s after):       {frame_3_reads:6,} plays ({frame_3_reads/total:6.1%})")

print("\n--- BY FLIGHT PHASE ---")
print(f"At Release (frames 0-2):    {at_release:6,} plays ({at_release/total:6.1%})")
print(f"Early Flight (frames 3-5):  {early_flight:6,} plays ({early_flight/total:6.1%})")
print(f"Mid Flight (frames 6-10):   {mid_flight:6,} plays ({mid_flight/total:6.1%})")
print(f"Late Flight (frames 11+):   {late_flight:6,} plays ({late_flight/total:6.1%})")

# Breakdown by outcome type
print("\n" + "="*80)
print("BREAKDOWN BY OUTCOME TYPE")
print("="*80)

for outcome in ['complete', 'incomplete', 'interception']:
    outcome_df = df[df['outcome'] == outcome]
    n = len(outcome_df)

    f0 = (outcome_df['bifurcation_frame_combo'] == 0).sum()
    f1 = (outcome_df['bifurcation_frame_combo'] == 1).sum()
    f2 = (outcome_df['bifurcation_frame_combo'] == 2).sum()

    at_rel = (outcome_df['bifurcation_frame_combo'] <= 2).sum()
    early = ((outcome_df['bifurcation_frame_combo'] > 2) &
             (outcome_df['bifurcation_frame_combo'] <= 5)).sum()
    mid = ((outcome_df['bifurcation_frame_combo'] > 5) &
           (outcome_df['bifurcation_frame_combo'] <= 10)).sum()
    late = (outcome_df['bifurcation_frame_combo'] > 10).sum()

    print(f"\n{outcome.upper()} ({n:,} plays):")
    print(f"  Frame 0 (release):        {f0:6,} ({f0/n:6.1%})")
    print(f"  Frame 1:                  {f1:6,} ({f1/n:6.1%})")
    print(f"  Frame 2:                  {f2:6,} ({f2/n:6.1%})")
    print(f"  ---")
    print(f"  At Release (0-2):         {at_rel:6,} ({at_rel/n:6.1%})")
    print(f"  Early Flight (3-5):       {early:6,} ({early/n:6.1%})")
    print(f"  Mid Flight (6-10):        {mid:6,} ({mid/n:6.1%})")
    print(f"  Late Flight (11+):        {late:6,} ({late/n:6.1%})")

# Create a more detailed breakdown
print("\n" + "="*80)
print("DETAILED FRAME BREAKDOWN (All Outcomes)")
print("="*80)

frame_counts = df['bifurcation_frame_combo'].value_counts().sort_index()
cumulative_pct = 0

print(f"\n{'Frame':<8} {'Count':>8} {'Percent':>10} {'Cumulative':>12}")
print("-" * 40)

for frame in sorted(frame_counts.index[:20]):  # First 20 frames
    count = frame_counts[frame]
    pct = count / total * 100
    cumulative_pct += pct
    print(f"{frame:<8} {count:>8,} {pct:>9.1f}% {cumulative_pct:>11.1f}%")

# Summary statistics
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print(f"""
1. MOMENT OF RELEASE DECISIONS:
   - {frame_0_reads/total:.1%} of plays are decided AT THE EXACT MOMENT of release (frame 0)
   - {at_release/total:.1%} of plays are decided within 0.2s of release (frames 0-2)

2. COMPLETIONS vs INCOMPLETIONS vs INTERCEPTIONS:
   - Completions: {(df[df['outcome']=='complete']['bifurcation_frame_combo']==0).sum()/(df['outcome']=='complete').sum():.1%} decided at release
   - Incompletions: {(df[df['outcome']=='incomplete']['bifurcation_frame_combo']==0).sum()/(df['outcome']=='incomplete').sum():.1%} decided at release
   - Interceptions: {(df[df['outcome']=='interception']['bifurcation_frame_combo']==0).sum()/(df['outcome']=='interception').sum():.1%} decided at release

3. THE VAST MAJORITY OF READS:
   - {(df['bifurcation_frame_combo'] <= 5).sum()/total:.1%} of plays have THE READ within 0.5s of release
   - {(df['bifurcation_frame_combo'] <= 10).sum()/total:.1%} of plays have THE READ within 1.0s of release

4. THIS MEANS:
   - For most plays, the outcome is effectively "locked in" by the QB's decision
   - The throw quality, target selection, and timing at release determine the result
   - Late changes (great catches, tipped balls, etc.) only affect ~{late_flight/total:.1%} of plays
""")

print("="*80)
