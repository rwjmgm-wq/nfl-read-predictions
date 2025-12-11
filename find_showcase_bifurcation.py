"""
Find plays that best showcase bifurcation detection for video.

Criteria for great showcase plays:
1. Early bifurcation (model commits early)
2. Correct prediction at bifurcation
3. Dramatic probability swings (interesting story)
4. Clear "moment of truth" visible in the data
5. Mix of outcomes (completions, incompletes, interceptions)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from pathlib import Path
from tqdm import tqdm
import json

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
ORGANIZED_DIR = BASE_DIR / "organized_plays"


def load_model():
    """Load the completion probability model."""
    model = lgb.Booster(model_file=str(MODEL_DIR / "completion_model.lgb"))
    with open(MODEL_DIR / "feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names


def get_play_probabilities(play_path, model, feature_names):
    """Get frame-by-frame probabilities for a play."""
    from frame_features import calculate_frame_features
    from scipy.signal import savgol_filter

    features = calculate_frame_features(play_path)
    if features is None or len(features) == 0:
        return None

    X = features[[col for col in feature_names if col in features.columns]].copy()
    X = X.fillna(X.median())

    probs = model.predict(X)

    # Smooth probabilities
    def smooth(p, window=5, poly=2):
        if len(p) < window:
            window = len(p) if len(p) % 2 == 1 else len(p) - 1
            window = max(3, window)
        poly = min(poly, window - 1)
        return np.clip(savgol_filter(p, window, poly), 0, 1)

    return pd.DataFrame({
        'frame': range(len(probs)),
        'p_incomplete': smooth(probs[:, 0]),
        'p_complete': smooth(probs[:, 1]),
        'p_interception': smooth(probs[:, 2])
    })


def analyze_play_for_showcase(probs_df, actual_outcome):
    """
    Analyze a play for showcase potential.

    Returns metrics that indicate how good the play is for demonstration.
    """
    n_frames = len(probs_df)
    if n_frames < 5:
        return None

    p_inc = probs_df['p_incomplete'].values
    p_comp = probs_df['p_complete'].values
    p_int = probs_df['p_interception'].values

    outcome_col = f'p_{actual_outcome}'
    outcome_probs = probs_df[outcome_col].values

    # Find bifurcation frame (time_weighted method)
    alpha = 0.5
    best_score = -1
    bif_frame = 0

    for i in range(n_frames):
        probs_at_frame = [p_inc[i], p_comp[i], p_int[i]]
        leader_idx = np.argmax(probs_at_frame)
        leader = ['incomplete', 'complete', 'interception'][leader_idx]

        if leader == actual_outcome:
            outcome_prob = outcome_probs[i]
            time_factor = (1 - i / n_frames) ** alpha
            score = outcome_prob * time_factor

            if score > best_score:
                best_score = score
                bif_frame = i

    # Check if prediction at bifurcation is correct
    probs_at_bif = [p_inc[bif_frame], p_comp[bif_frame], p_int[bif_frame]]
    pred_idx = np.argmax(probs_at_bif)
    predicted = ['incomplete', 'complete', 'interception'][pred_idx]
    correct = predicted == actual_outcome

    if not correct:
        return None  # Only showcase correct predictions

    # Calculate showcase metrics
    bif_timing = bif_frame / n_frames

    # Probability swing: how much did the outcome probability change?
    prob_start = outcome_probs[0]
    prob_at_bif = outcome_probs[bif_frame]
    prob_end = outcome_probs[-1]

    # Drama score: measures how interesting the probability trajectory is
    # High drama = big swings, close races, clear resolution

    # 1. Early confidence: how confident at bifurcation?
    confidence_at_bif = max(probs_at_bif)

    # 2. Swing magnitude: how much did probabilities change?
    max_swing = max(outcome_probs) - min(outcome_probs)

    # 3. Competition: was there a close race between outcomes?
    sorted_probs_at_bif = sorted(probs_at_bif, reverse=True)
    margin_at_bif = sorted_probs_at_bif[0] - sorted_probs_at_bif[1]

    # 4. Trajectory interest: did the outcome have to "come from behind"?
    came_from_behind = prob_start < 0.4 and prob_at_bif > 0.5

    # 5. Clear resolution: how dominant is the outcome at the end?
    final_confidence = outcome_probs[-1]

    # Calculate overall showcase score
    # We want: early bifurcation, high confidence, dramatic swings
    showcase_score = (
        (1 - bif_timing) * 0.3 +  # Early is better
        confidence_at_bif * 0.2 +  # Confident prediction
        max_swing * 0.2 +  # Big swings are dramatic
        (1 - margin_at_bif) * 0.1 +  # Close races are interesting
        (0.2 if came_from_behind else 0) +  # Comeback stories
        final_confidence * 0.1  # Clear resolution
    )

    # Find the "story" of the play
    # What was leading at start?
    start_probs = [p_inc[0], p_comp[0], p_int[0]]
    start_leader_idx = np.argmax(start_probs)
    start_leader = ['incomplete', 'complete', 'interception'][start_leader_idx]

    # Was there a lead change?
    lead_changes = 0
    prev_leader = start_leader
    for i in range(1, n_frames):
        probs_at_frame = [p_inc[i], p_comp[i], p_int[i]]
        leader_idx = np.argmax(probs_at_frame)
        leader = ['incomplete', 'complete', 'interception'][leader_idx]
        if leader != prev_leader:
            lead_changes += 1
            prev_leader = leader

    return {
        'bif_frame': bif_frame,
        'bif_timing': bif_timing,
        'n_frames': n_frames,
        'confidence_at_bif': confidence_at_bif,
        'max_swing': max_swing,
        'margin_at_bif': margin_at_bif,
        'came_from_behind': came_from_behind,
        'start_leader': start_leader,
        'lead_changes': lead_changes,
        'showcase_score': showcase_score,
        'prob_trajectory': {
            'start': prob_start,
            'at_bif': prob_at_bif,
            'end': prob_end
        }
    }


def build_play_index():
    """Build play index from folder structure."""
    play_index = []

    for week_dir in sorted(ORGANIZED_DIR.glob("Week_*")):
        week = week_dir.name.replace("Week_", "")

        for game_dir in sorted(week_dir.glob("*_vs_*")):
            game = game_dir.name

            # Plays are named like "001_Q1_1430_WAS", not "play_*"
            for play_dir in sorted(game_dir.iterdir()):
                if not play_dir.is_dir():
                    continue

                # Get outcome from supplementary.csv
                supp_file = play_dir / "supplementary.csv"
                if supp_file.exists():
                    try:
                        df = pd.read_csv(supp_file)
                        if 'pass_result' in df.columns:
                            result = df['pass_result'].iloc[0]
                            if result == 'C':
                                outcome = 'complete'
                            elif result == 'I':
                                outcome = 'incomplete'
                            elif result == 'IN':
                                outcome = 'interception'
                            else:
                                continue
                        else:
                            continue
                    except:
                        continue
                else:
                    continue

                # Get play description for context
                play_desc = ""
                if 'play_description' in df.columns:
                    play_desc = df['play_description'].iloc[0]

                play_index.append({
                    'play': f"{week_dir.name}/{game}/{play_dir.name}",
                    'outcome': outcome,
                    'game': game,
                    'week': week,
                    'description': play_desc
                })

    return play_index


def main():
    print("Loading model...")
    model, feature_names = load_model()

    # Build play index from folder structure
    print("Building play index...")
    play_index = build_play_index()

    print(f"Analyzing {len(play_index)} plays for showcase potential...")

    # Analyze all plays
    showcase_candidates = {
        'complete': [],
        'incomplete': [],
        'interception': []
    }

    for play_info in tqdm(play_index):
        play_path = ORGANIZED_DIR / play_info['play']

        probs_df = get_play_probabilities(play_path, model, feature_names)
        if probs_df is None:
            continue

        metrics = analyze_play_for_showcase(probs_df, play_info['outcome'])
        if metrics is None:
            continue

        # Add play info
        metrics['play'] = play_info['play']
        metrics['outcome'] = play_info['outcome']
        metrics['game'] = play_info.get('game', '')
        metrics['week'] = play_info.get('week', '')
        metrics['description'] = play_info.get('description', '')

        showcase_candidates[play_info['outcome']].append(metrics)

    # Sort by showcase score and get top candidates
    print("\n" + "="*80)
    print("TOP SHOWCASE PLAYS BY OUTCOME")
    print("="*80)

    all_top_plays = []

    for outcome in ['complete', 'incomplete', 'interception']:
        candidates = showcase_candidates[outcome]
        if not candidates:
            print(f"\nNo {outcome} plays found")
            continue

        # Sort by showcase score
        candidates.sort(key=lambda x: x['showcase_score'], reverse=True)

        print(f"\n{'='*60}")
        print(f"TOP {outcome.upper()} PLAYS ({len(candidates)} total)")
        print(f"{'='*60}")

        # Show top 5
        for i, play in enumerate(candidates[:5]):
            print(f"\n{i+1}. {play['play']}")
            print(f"   Game: {play['game']}, Week: {play['week']}")
            print(f"   Showcase Score: {play['showcase_score']:.3f}")
            print(f"   Bifurcation: frame {play['bif_frame']}/{play['n_frames']} ({play['bif_timing']*100:.1f}% into play)")
            print(f"   Confidence at bifurcation: {play['confidence_at_bif']*100:.1f}%")
            print(f"   Probability swing: {play['max_swing']*100:.1f}%")
            print(f"   Lead changes: {play['lead_changes']}")
            print(f"   Started as: {play['start_leader']} -> ended as: {play['outcome']}")
            if play['came_from_behind']:
                print(f"   ** COMEBACK: Started at {play['prob_trajectory']['start']*100:.1f}%, rose to {play['prob_trajectory']['at_bif']*100:.1f}%")

            all_top_plays.append(play)

    # Overall top plays (mix of outcomes)
    print("\n" + "="*80)
    print("OVERALL TOP 10 SHOWCASE PLAYS (RECOMMENDED FOR VIDEO)")
    print("="*80)

    # Weight interceptions higher since they're rare and dramatic
    for play in all_top_plays:
        if play['outcome'] == 'interception':
            play['adjusted_score'] = play['showcase_score'] * 1.5
        elif play['outcome'] == 'incomplete':
            play['adjusted_score'] = play['showcase_score'] * 1.2
        else:
            play['adjusted_score'] = play['showcase_score']

    all_top_plays.sort(key=lambda x: x['adjusted_score'], reverse=True)

    for i, play in enumerate(all_top_plays[:10]):
        print(f"\n{i+1}. [{play['outcome'].upper()}] {play['play']}")
        print(f"   Week {play['week']}: {play['game']}")
        print(f"   Bifurcation at {play['bif_timing']*100:.1f}% with {play['confidence_at_bif']*100:.1f}% confidence")
        print(f"   Story: {play['start_leader']} -> {play['outcome']} ({play['lead_changes']} lead changes)")
        if play['came_from_behind']:
            print(f"   ** Comeback story!")
        if play.get('description'):
            # Truncate long descriptions
            desc = play['description'][:100] + "..." if len(play['description']) > 100 else play['description']
            print(f"   Play: {desc}")

    # Save results
    results = {
        'top_completions': showcase_candidates['complete'][:10],
        'top_incompletes': showcase_candidates['incomplete'][:10],
        'top_interceptions': showcase_candidates['interception'][:10],
        'overall_top_10': all_top_plays[:10]
    }

    with open(BASE_DIR / "showcase_bifurcation_plays.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to showcase_bifurcation_plays.json")

    return all_top_plays[:10]


if __name__ == "__main__":
    main()
