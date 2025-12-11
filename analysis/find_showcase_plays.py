"""
Find the best plays to showcase in a 3-minute video demonstration.

Criteria for compelling showcase plays:
1. High drama (lead changes, big probability swings)
2. Clear narrative arc (model correctly predicts outcome)
3. Variety of outcomes (Complete, Incomplete, Interception)
4. Interesting bifurcation patterns (late determinations, M3/M6 differences)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent


def analyze_showcase_candidates():
    """Find the best plays for video showcase."""

    # Load bifurcation results
    df = pd.read_csv(BASE_DIR / "bifurcation_results.csv")

    print("=" * 70)
    print("FINDING BEST PLAYS FOR 3-MINUTE VIDEO SHOWCASE")
    print("=" * 70)
    print(f"\nTotal plays analyzed: {len(df)}")

    # Basic stats
    print(f"\nOutcome distribution:")
    print(df['outcome'].value_counts())

    # Standardize outcome names for display
    outcome_display = {
        'complete': 'Complete',
        'incomplete': 'Incomplete',
        'interception': 'Interception',
        'C': 'Complete',
        'I': 'Incomplete',
        'IN': 'Interception'
    }
    df['outcome_display'] = df['outcome'].map(outcome_display).fillna(df['outcome'])

    showcase_categories = {}

    # ========================================
    # CATEGORY 1: High Drama Plays
    # ========================================
    print("\n" + "=" * 70)
    print("CATEGORY 1: HIGH DRAMA PLAYS (High drama score + big swings)")
    print("=" * 70)

    # High drama score plays
    high_drama = df.nlargest(20, 'drama_score').copy()

    print(f"\nTop 10 highest drama score plays:")
    for i, (idx, row) in enumerate(high_drama.head(10).iterrows()):
        print(f"\n  {i+1}. {row['play_folder']} ({row['game']})")
        print(f"     Outcome: {row['outcome_display']}, Drama Score: {row['drama_score']:.2f}")
        print(f"     Prob Swing: {row['probability_swing']:.1%}, Lead Change: {row['had_lead_change']}")
        print(f"     Frames: {row['total_frames']}")

    showcase_categories['high_drama'] = high_drama.head(5)

    # ========================================
    # CATEGORY 2: Lead Change Plays
    # ========================================
    print("\n" + "=" * 70)
    print("CATEGORY 2: LEAD CHANGE PLAYS")
    print("=" * 70)

    lead_changes = df[df['had_lead_change'] == True].copy()
    lead_changes = lead_changes.sort_values('drama_score', ascending=False)

    print(f"\nFound {len(lead_changes)} plays with lead changes")

    if len(lead_changes) > 0:
        print("\nTop 5 lead change plays by drama:")
        for i, (idx, row) in enumerate(lead_changes.head(5).iterrows()):
            print(f"\n  {i+1}. {row['play_folder']} ({row['game']})")
            print(f"     Outcome: {row['outcome_display']}, Drama: {row['drama_score']:.2f}")
            print(f"     Prob at release: {row['probability_at_release']:.1%}")
            print(f"     Pre-bif leader: {row['pre_bifurcation_leader']}")

        showcase_categories['lead_changes'] = lead_changes.head(3)

    # ========================================
    # CATEGORY 3: Late Bifurcation (Suspenseful)
    # ========================================
    print("\n" + "=" * 70)
    print("CATEGORY 3: LATE BIFURCATION (Suspenseful plays)")
    print("=" * 70)

    # Bifurcation in last 25% of play (using combo timing)
    late_bif = df[df['bifurcation_timing_combo'] > 0.75].copy()
    late_bif = late_bif.sort_values('bifurcation_timing_combo', ascending=False)

    print(f"\nFound {len(late_bif)} plays with late bifurcation (>75% through)")

    if len(late_bif) > 0:
        print("\nTop 5 late bifurcation plays:")
        for i, (idx, row) in enumerate(late_bif.head(5).iterrows()):
            print(f"\n  {i+1}. {row['play_folder']} ({row['game']})")
            print(f"     Outcome: {row['outcome_display']}")
            print(f"     Bifurcation at: {row['bifurcation_timing_combo']:.1%} through")
            print(f"     Frames: {row['total_frames']}, Drama: {row['drama_score']:.2f}")

        showcase_categories['late_bifurcation'] = late_bif.head(3)

    # ========================================
    # CATEGORY 4: Best Play for Each Outcome Type
    # ========================================
    print("\n" + "=" * 70)
    print("CATEGORY 4: BEST PLAY FOR EACH OUTCOME TYPE")
    print("=" * 70)

    for outcome in ['complete', 'incomplete', 'interception']:
        outcome_df = df[df['outcome'] == outcome].copy()

        if len(outcome_df) == 0:
            print(f"\n{outcome.upper()}: No plays found")
            continue

        # Rank by drama score
        best = outcome_df.nlargest(3, 'drama_score')

        print(f"\n{outcome.upper()} - Best showcase plays (by drama):")
        for i, (idx, row) in enumerate(best.iterrows()):
            print(f"  {i+1}. {row['play_folder']} ({row['game']})")
            print(f"     Drama: {row['drama_score']:.2f}, Swing: {row['probability_swing']:.1%}")
            print(f"     Frames: {row['total_frames']}")

        showcase_categories[f'best_{outcome}'] = best.head(1)

    # ========================================
    # CATEGORY 5: Dramatic Interceptions
    # ========================================
    print("\n" + "=" * 70)
    print("CATEGORY 5: DRAMATIC INTERCEPTIONS (Most compelling)")
    print("=" * 70)

    int_plays = df[df['outcome'] == 'interception'].copy()
    int_drama = int_plays.nlargest(10, 'drama_score')

    print(f"\nFound {len(int_plays)} total interceptions")

    if len(int_drama) > 0:
        print("\nTop 5 dramatic interceptions:")
        for i, (idx, row) in enumerate(int_drama.head(5).iterrows()):
            print(f"\n  {i+1}. {row['play_folder']} ({row['game']})")
            print(f"     Drama: {row['drama_score']:.2f}, Lead Change: {row['had_lead_change']}")
            print(f"     Prob at release: {row['probability_at_release']:.1%}")
            print(f"     Entropy drop: {row['entropy_drop']:.3f}")

        showcase_categories['dramatic_interceptions'] = int_drama.head(2)

    # ========================================
    # CATEGORY 6: M3 vs M6 Timing Differences
    # ========================================
    print("\n" + "=" * 70)
    print("CATEGORY 6: M3/M6 TIMING DIFFERENCES (Shows combo model value)")
    print("=" * 70)

    # Plays where M3 and M6 trigger at very different times
    df['m3_m6_frame_gap'] = abs(df['bifurcation_frame_m3'] - df['bifurcation_frame_m6'])
    big_gap = df[df['m3_m6_frame_gap'] >= 5].sort_values('m3_m6_frame_gap', ascending=False)

    print(f"\nFound {len(big_gap)} plays with 5+ frame gap between M3 and M6")

    if len(big_gap) > 0:
        print("\nTop 5 plays with M3/M6 timing difference:")
        for i, (idx, row) in enumerate(big_gap.head(5).iterrows()):
            print(f"\n  {i+1}. {row['play_folder']} ({row['game']})")
            print(f"     M3 frame: {row['bifurcation_frame_m3']}, M6 frame: {row['bifurcation_frame_m6']}")
            print(f"     Gap: {row['m3_m6_frame_gap']} frames")
            print(f"     Outcome: {row['outcome_display']}")

        showcase_categories['m3_m6_difference'] = big_gap.head(2)

    # ========================================
    # CATEGORY 7: Early Confident Reads
    # ========================================
    print("\n" + "=" * 70)
    print("CATEGORY 7: EARLY CONFIDENT READS (Model reliability)")
    print("=" * 70)

    # Early bifurcation with high margin
    early_conf = df[
        (df['bifurcation_timing_combo'] < 0.3) &
        (df['margin_at_bifurcation'] > 0.5)
    ].copy()
    early_conf = early_conf.sort_values('bifurcation_timing_combo')

    print(f"\nFound {len(early_conf)} plays with early confident bifurcation")

    if len(early_conf) > 0:
        print("\nTop 5 early confident reads:")
        for i, (idx, row) in enumerate(early_conf.head(5).iterrows()):
            print(f"\n  {i+1}. {row['play_folder']} ({row['game']})")
            print(f"     Bifurcation at: {row['bifurcation_timing_combo']:.1%} through")
            print(f"     Margin: {row['margin_at_bifurcation']:.1%}")
            print(f"     Outcome: {row['outcome_display']}")

        showcase_categories['early_confident'] = early_conf.head(2)

    # ========================================
    # CATEGORY 8: Highest Entropy Drop
    # ========================================
    print("\n" + "=" * 70)
    print("CATEGORY 8: HIGHEST ENTROPY DROP (Uncertainty -> Certainty)")
    print("=" * 70)

    high_entropy = df.nlargest(10, 'entropy_drop')

    print("\nTop 5 plays by entropy reduction:")
    for i, (idx, row) in enumerate(high_entropy.head(5).iterrows()):
        print(f"\n  {i+1}. {row['play_folder']} ({row['game']})")
        print(f"     Entropy: {row['entropy_at_release']:.3f} -> {row['entropy_at_end']:.3f}")
        print(f"     Drop: {row['entropy_drop']:.3f}")
        print(f"     Outcome: {row['outcome_display']}")

    showcase_categories['high_entropy_drop'] = high_entropy.head(2)

    # ========================================
    # FINAL RECOMMENDATIONS
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL VIDEO SHOWCASE RECOMMENDATIONS")
    print("=" * 70)

    print("""
For a 3-minute video, I recommend showing 4-6 plays that demonstrate:

1. OPEN PLAY: Start with a high-drama play that stays uncertain
   (demonstrates real-time probability tracking)

2. COMPLETION: Show a clean completion with clear narrative
   (model shows increasing completion probability)

3. INTERCEPTION: Include a dramatic interception
   (most visually compelling, rare event detection)

4. LATE DETERMINATION: Show a play that stays uncertain until the end
   (demonstrates suspense and moment-of-truth detection)

5. LEAD CHANGE: If available, show a play where predicted outcome
   flips during ball flight (demonstrates model responsiveness)
""")

    # Compile final list - pick diverse, compelling plays
    final_plays = []

    # 1. High drama opener
    if 'high_drama' in showcase_categories and len(showcase_categories['high_drama']) > 0:
        play = showcase_categories['high_drama'].iloc[0]
        final_plays.append(('1. High Drama Opener', play))

    # 2. Best completion
    if 'best_complete' in showcase_categories and len(showcase_categories['best_complete']) > 0:
        play = showcase_categories['best_complete'].iloc[0]
        final_plays.append(('2. Dramatic Completion', play))

    # 3. Dramatic interception
    if 'dramatic_interceptions' in showcase_categories and len(showcase_categories['dramatic_interceptions']) > 0:
        play = showcase_categories['dramatic_interceptions'].iloc[0]
        final_plays.append(('3. Dramatic Interception', play))

    # 4. Late bifurcation
    if 'late_bifurcation' in showcase_categories and len(showcase_categories['late_bifurcation']) > 0:
        play = showcase_categories['late_bifurcation'].iloc[0]
        final_plays.append(('4. Late Suspense', play))

    # 5. Lead change if available
    if 'lead_changes' in showcase_categories and len(showcase_categories['lead_changes']) > 0:
        play = showcase_categories['lead_changes'].iloc[0]
        # Don't duplicate
        if play['play_folder'] not in [p['play_folder'] for _, p in final_plays]:
            final_plays.append(('5. Lead Change Drama', play))

    # 6. High entropy drop (uncertainty resolved)
    if 'high_entropy_drop' in showcase_categories and len(showcase_categories['high_entropy_drop']) > 0:
        for i in range(len(showcase_categories['high_entropy_drop'])):
            play = showcase_categories['high_entropy_drop'].iloc[i]
            if play['play_folder'] not in [p['play_folder'] for _, p in final_plays]:
                final_plays.append(('6. Uncertainty Resolved', play))
                break

    print("\n" + "=" * 70)
    print("RECOMMENDED SHOWCASE PLAYS (in suggested order)")
    print("=" * 70)

    for category, row in final_plays:
        print(f"\n{category}")
        print(f"   Play: {row['play_folder']}")
        print(f"   Game: {row['game']} (Week {row['week'].replace('Week_', '')})")
        print(f"   Outcome: {row['outcome_display']}")
        print(f"   Drama Score: {row['drama_score']:.2f}")
        print(f"   Probability Swing: {row['probability_swing']:.1%}")
        print(f"   Lead Change: {row['had_lead_change']}")
        print(f"   Bifurcation Timing: {row['bifurcation_timing_combo']:.1%} through play")
        print(f"   Total Frames: {row['total_frames']}")

    # Save recommendations to CSV
    if final_plays:
        recs_data = []
        for cat, row in final_plays:
            recs_data.append({
                'category': cat,
                'play_folder': row['play_folder'],
                'week': row['week'],
                'game': row['game'],
                'game_id': row['game_id'],
                'play_id': row['play_id'],
                'outcome': row['outcome_display'],
                'drama_score': row['drama_score'],
                'probability_swing': row['probability_swing'],
                'had_lead_change': row['had_lead_change'],
                'bifurcation_timing': row['bifurcation_timing_combo'],
                'total_frames': row['total_frames'],
                'entropy_drop': row['entropy_drop']
            })

        recs_df = pd.DataFrame(recs_data)
        recs_df.to_csv(BASE_DIR / "showcase_recommendations.csv", index=False)
        print(f"\n\nRecommendations saved to: {BASE_DIR / 'showcase_recommendations.csv'}")

    # Also print full paths for easy access
    print("\n" + "=" * 70)
    print("FULL PLAY PATHS FOR VISUALIZATION")
    print("=" * 70)

    organized_dir = BASE_DIR / "organized_plays"
    for category, row in final_plays:
        play_path = organized_dir / row['week'] / row['game'] / row['play_folder']
        print(f"\n{category}:")
        print(f"   {play_path}")

    return final_plays


if __name__ == "__main__":
    analyze_showcase_candidates()
