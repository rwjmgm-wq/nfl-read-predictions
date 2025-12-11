"""
Find plays where the model was WRONG at bifurcation -
plays that looked like one outcome but ended up another.

These "upset" plays are great for showcasing the uncertainty in football.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from pathlib import Path
from tqdm import tqdm
import json
from scipy.signal import savgol_filter

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


def analyze_upset_play(probs_df, actual_outcome):
    """
    Find plays where the model was confidently wrong.

    Returns metrics for "upset" plays where:
    - Model predicted one outcome confidently
    - Actual outcome was different
    """
    n_frames = len(probs_df)
    if n_frames < 5:
        return None

    p_inc = probs_df['p_incomplete'].values
    p_comp = probs_df['p_complete'].values
    p_int = probs_df['p_interception'].values

    outcome_col = f'p_{actual_outcome}'
    outcome_probs = probs_df[outcome_col].values

    # Find the frame where the WRONG outcome was most confident
    # This is where the model was most "fooled"

    outcomes = ['incomplete', 'complete', 'interception']
    wrong_outcomes = [o for o in outcomes if o != actual_outcome]

    best_wrong_confidence = 0
    best_wrong_frame = 0
    best_wrong_outcome = None

    for i in range(n_frames):
        probs_at_frame = [p_inc[i], p_comp[i], p_int[i]]

        for wrong_outcome in wrong_outcomes:
            wrong_idx = outcomes.index(wrong_outcome)
            wrong_prob = probs_at_frame[wrong_idx]

            # Check if this wrong outcome was leading AND confident
            if wrong_prob == max(probs_at_frame) and wrong_prob > best_wrong_confidence:
                best_wrong_confidence = wrong_prob
                best_wrong_frame = i
                best_wrong_outcome = wrong_outcome

    # Only consider plays where model was confidently wrong (>50% for wrong outcome)
    if best_wrong_confidence < 0.50:
        return None

    # Calculate how dramatic the swing was
    final_correct_prob = outcome_probs[-1]
    prob_swing = final_correct_prob - outcome_probs[best_wrong_frame]

    # The "upset score" - how wrong was the model and how dramatic was the correction
    upset_score = best_wrong_confidence * prob_swing

    # Find when the correct outcome finally took the lead
    correct_takeover_frame = n_frames - 1
    for i in range(best_wrong_frame, n_frames):
        probs_at_frame = [p_inc[i], p_comp[i], p_int[i]]
        leader_idx = np.argmax(probs_at_frame)
        leader = outcomes[leader_idx]
        if leader == actual_outcome:
            correct_takeover_frame = i
            break

    takeover_timing = correct_takeover_frame / n_frames

    return {
        'wrong_prediction': best_wrong_outcome,
        'wrong_confidence': best_wrong_confidence,
        'wrong_frame': best_wrong_frame,
        'wrong_timing': best_wrong_frame / n_frames,
        'correct_takeover_frame': correct_takeover_frame,
        'correct_takeover_timing': takeover_timing,
        'final_correct_prob': final_correct_prob,
        'prob_swing': prob_swing,
        'upset_score': upset_score,
        'n_frames': n_frames
    }


def build_play_index():
    """Build play index from folder structure."""
    play_index = []

    for week_dir in sorted(ORGANIZED_DIR.glob("Week_*")):
        week = week_dir.name.replace("Week_", "")

        for game_dir in sorted(week_dir.glob("*_vs_*")):
            game = game_dir.name

            for play_dir in sorted(game_dir.iterdir()):
                if not play_dir.is_dir():
                    continue

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

    print("Building play index...")
    play_index = build_play_index()

    print(f"Analyzing {len(play_index)} plays for upsets...")

    # Analyze all plays for upsets
    upset_plays = {
        'complete': [],      # Plays that ended complete but model thought otherwise
        'incomplete': [],    # Plays that ended incomplete but model thought otherwise
        'interception': []   # Plays that ended INT but model thought otherwise
    }

    for play_info in tqdm(play_index):
        play_path = ORGANIZED_DIR / play_info['play']

        probs_df = get_play_probabilities(play_path, model, feature_names)
        if probs_df is None:
            continue

        metrics = analyze_upset_play(probs_df, play_info['outcome'])
        if metrics is None:
            continue

        # Add play info
        metrics['play'] = play_info['play']
        metrics['outcome'] = play_info['outcome']
        metrics['game'] = play_info.get('game', '')
        metrics['week'] = play_info.get('week', '')
        metrics['description'] = play_info.get('description', '')

        upset_plays[play_info['outcome']].append(metrics)

    # Sort by upset score and show top plays
    print("\n" + "="*80)
    print("TOP UPSET PLAYS - MODEL WAS WRONG!")
    print("="*80)

    all_upsets = []

    for outcome in ['complete', 'incomplete', 'interception']:
        candidates = upset_plays[outcome]
        if not candidates:
            continue

        # Sort by upset score
        candidates.sort(key=lambda x: x['upset_score'], reverse=True)

        print(f"\n{'='*60}")
        print(f"ACTUAL {outcome.upper()} - Model predicted wrong ({len(candidates)} plays)")
        print(f"{'='*60}")

        for i, play in enumerate(candidates[:5]):
            print(f"\n{i+1}. {play['play']}")
            print(f"   Game: {play['game']}, Week: {play['week']}")
            print(f"   Model predicted: {play['wrong_prediction'].upper()} at {play['wrong_confidence']*100:.1f}% confidence")
            print(f"   Actual outcome: {play['outcome'].upper()}")
            print(f"   Wrong prediction at: {play['wrong_timing']*100:.1f}% into play")
            print(f"   Correct outcome took lead at: {play['correct_takeover_timing']*100:.1f}% into play")
            print(f"   Probability swing: {play['prob_swing']*100:.1f}%")
            print(f"   Upset score: {play['upset_score']:.3f}")
            if play.get('description'):
                desc = play['description'][:100] + "..." if len(play['description']) > 100 else play['description']
                print(f"   Play: {desc}")

            all_upsets.append(play)

    # Overall top upsets
    print("\n" + "="*80)
    print("TOP 10 UPSETS OVERALL (Best for video)")
    print("="*80)

    # Weight interceptions higher (rare and dramatic)
    for play in all_upsets:
        if play['outcome'] == 'interception':
            play['adjusted_upset'] = play['upset_score'] * 1.5
        else:
            play['adjusted_upset'] = play['upset_score']

    all_upsets.sort(key=lambda x: x['adjusted_upset'], reverse=True)

    for i, play in enumerate(all_upsets[:10]):
        print(f"\n{i+1}. [{play['outcome'].upper()}] {play['play']}")
        print(f"   Week {play['week']}: {play['game']}")
        print(f"   Model said {play['wrong_prediction'].upper()} ({play['wrong_confidence']*100:.1f}%) -> Actually {play['outcome'].upper()}")
        print(f"   Swing: {play['prob_swing']*100:.1f}% | Upset Score: {play['upset_score']:.3f}")
        if play.get('description'):
            desc = play['description'][:100] + "..." if len(play['description']) > 100 else play['description']
            print(f"   Play: {desc}")

    # Save results
    results = {
        'upset_completions': upset_plays['complete'][:10],
        'upset_incompletes': upset_plays['incomplete'][:10],
        'upset_interceptions': upset_plays['interception'][:10],
        'overall_top_10': all_upsets[:10]
    }

    with open(BASE_DIR / "upset_plays.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to upset_plays.json")

    return all_upsets[:10]


if __name__ == "__main__":
    main()
