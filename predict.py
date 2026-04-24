"""
UFC Fight Predictor — The Final Product

Takes two fighter names, builds their profiles from the internet,
feeds them into the trained ML model, outputs winner/method/round.

Usage:
    # Predict a single fight
    python3 predict.py "Tommy McMillen" "Manolo Zecchini"

    # Predict full UFC 327 card
    python3 predict.py --card ufc327

    # Use manual stats instead of scraping
    python3 predict.py --manual
"""

import sys
from ml_model import load_models, predict
from profile_builder import build_live_profile, build_profile_manual


def predict_fight(name_a: str, name_b: str) -> dict:
    """Build live profiles and predict."""
    print(f"\n  Building profile: {name_a}...")
    profile_a = build_live_profile(name_a)
    if not profile_a:
        print(f"  ERROR: Could not build profile for {name_a}")
        return None

    print(f"  Building profile: {name_b}...")
    profile_b = build_live_profile(name_b)
    if not profile_b:
        print(f"  ERROR: Could not build profile for {name_b}")
        return None

    print(f"\n  {name_a}: Elo {profile_a['elo']:.0f} | {profile_a['career_wins']}-{profile_a['losses']} | "
          f"Finish rate {profile_a['finish_rate']:.0%} | Form {profile_a['recent_form_3']:.0%}")
    print(f"  {name_b}: Elo {profile_b['elo']:.0f} | {profile_b['career_wins']}-{profile_b['losses']} | "
          f"Finish rate {profile_b['finish_rate']:.0%} | Form {profile_b['recent_form_3']:.0%}")

    result = predict(profile_a, profile_b)

    winner_name = name_a if result["winner"] == "A" else name_b
    result["winner_name"] = winner_name
    result["fighter_a"] = name_a
    result["fighter_b"] = name_b
    return result


def print_prediction(result: dict):
    """Pretty print a prediction."""
    if not result:
        return
    r2 = result.get("reaches_r2_prob", 0.5)
    print(f"\n  {'─' * 60}")
    print(f"  PREDICTION: {result['winner_name']}")
    print(f"  Method: {result['method']} | Round: {result['round']} | Confidence: {result['win_prob']:.0%}")
    print(f"  Reaches R2: {'YES' if r2 >= 0.5 else 'NO'} ({r2:.0%})")
    print(f"  Method probs: {' | '.join(f'{k}: {v:.0%}' for k, v in sorted(result['method_probs'].items(), key=lambda x: -x[1]))}")
    print(f"  {'─' * 60}")


def predict_card():
    """Predict the full UFC 327 card."""
    card = [
        # Main Card
        ("Jiri Prochazka", "Carlos Ulberg", True),
        ("Azamat Murzakanov", "Paulo Costa", False),
        ("Curtis Blaydes", "Josh Hokit", False),
        ("Dominick Reyes", "Johnny Walker", False),
        ("Cub Swanson", "Nate Landwehr", False),
        # Prelims
        ("Patricio Pitbull", "Aaron Pico", False),
        ("Kevin Holland", "Randy Brown", False),
        ("Mateusz Gamrot", "Esteban Ribovics", False),
        ("Tatiana Suarez", "Loopy Godinez", False),
        # Early Prelims
        ("Chris Padilla", "MarQuel Mederos", False),
        ("Kelvin Gastelum", "Vicente Luque", False),
        ("Charles Radtke", "Francisco Prado", False),
    ]

    print("=" * 80)
    print("  UFC 327: PROCHAZKA vs ULBERG — ML MODEL PREDICTIONS")
    print("  Kaseya Center, Miami FL — April 11, 2026")
    print("  Trained on 8,500+ fights | Elo + Competition Level + Rolling Stats")
    print("=" * 80)

    results = []
    for name_a, name_b, is_main in card:
        result = predict_fight(name_a, name_b)
        if result:
            result["is_main"] = is_main
            results.append(result)

    # Summary table
    print(f"\n\n{'=' * 80}")
    print(f"  FULL CARD PREDICTIONS")
    print(f"{'=' * 80}\n")

    print(f"  {'#':<3} {'Fight':<45} {'Winner':<20} {'Method':<8} {'Rd':>3} {'Conf':>6} {'Reaches R2':>11}")
    print(f"  {'─' * 100}")

    for i, r in enumerate(results, 1):
        tag = " *" if r.get("is_main") else ""
        fight = f"{r['fighter_a']} vs {r['fighter_b']}{tag}"
        r2 = r.get("reaches_r2_prob", 0.5)
        r2_label = f"{'YES' if r2 >= 0.5 else 'NO'} ({r2:.0%})"
        print(f"  {i:<3} {fight:<45} {r['winner_name']:<20} {r['method']:<8} R{r['round']:>1} {r['win_prob']:>5.0%} {r2_label:>11}")

    print(f"\n{'=' * 80}")


def main():
    if "--card" in sys.argv:
        predict_card()
    elif "--manual" in sys.argv:
        print("\n  Manual mode — enter fighter stats")
        print("  (This uses build_profile_manual from profile_builder.py)")
        # Could build interactive input here
        print("  Not yet implemented. Use predict.py 'Fighter A' 'Fighter B' instead.")
    elif len(sys.argv) >= 3:
        name_a = sys.argv[1]
        name_b = sys.argv[2]

        print("=" * 70)
        print(f"  PREDICTING: {name_a} vs {name_b}")
        print("=" * 70)

        result = predict_fight(name_a, name_b)
        if result:
            print_prediction(result)
    else:
        print("\n  Usage:")
        print("    python3 predict.py 'Fighter A' 'Fighter B'")
        print("    python3 predict.py --card vegas115")
        print("    python3 predict.py --manual")


if __name__ == "__main__":
    main()
