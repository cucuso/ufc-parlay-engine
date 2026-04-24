"""
UFC Fight Night: Burns vs. Malott — April 18, 2026
Full card predictions via the calibrated ML model.
"""
from prediction_store import run_card_and_save

EVENT_NAME = "UFC Fight Night: Burns vs. Malott"
EVENT_DATE = "2026-04-18"

CARD = [
    # (fighter_a, fighter_b, weight_class, is_main_event)
    ("Gilbert Burns",      "Mike Malott",          "Welterweight",         True),
    ("Kyler Phillips",     "Charles Jourdain",     "Bantamweight",         False),
    ("Mandel Nallo",       "Jai Herbert",          "Lightweight",          False),
    ("Jasmine Jasudavicius", "Karine Silva",       "W-Flyweight",          False),
    ("Thiago Moises",      "Gauge Young",          "Lightweight",          False),
    ("Dennis Buzukja",     "Marcio Barbosa",       "Featherweight",        False),
    ("Julien Leblanc",     "Robert Valentin",      "Middleweight",         False),
    ("Tanner Boser",       "Gokhan Saricam",       "Heavyweight",          False),
    ("Melissa Croden",     "Daria Zhelezniakova",  "W-Bantamweight",       False),
    ("JJ Aldrich",         "Jamey-Lyn Horth",      "W-Flyweight",          False),
    ("John Castaneda",     "Mark Vologdin",        "Bantamweight",         False),
]


def main():
    print("=" * 95)
    print(f"  {EVENT_NAME.upper()} — {EVENT_DATE}")
    print("  Full card predictions (calibrated ML model)")
    print("=" * 95)
    results, _ = run_card_and_save(EVENT_NAME, EVENT_DATE, CARD)

    # -------- Summary table --------
    print("\n\n" + "=" * 95)
    print("  SUMMARY TABLE")
    print("=" * 95)
    print(f"  {'Fight':<42} {'Winner':<20} {'Conf':>5} {'Method':>8} {'R':>2} "
          f"{'KO%':>5} {'SUB%':>5} {'Dist%':>6}")
    print("  " + "-" * 93)
    for na, nb, wc, is_main, r in results:
        tag = "*" if is_main else " "
        fight = f"{na} vs {nb}"[:40]
        print(f"  {tag}{fight:<41} {r['winner_name']:<20} {r['win_prob']*100:>4.0f}% "
              f"{r['method']:>8} R{r['round']:>1} "
              f"{r['method_probs']['KO/TKO']*100:>4.0f}% {r['method_probs']['SUB']*100:>4.0f}% "
              f"{r['goes_distance_prob']*100:>5.0f}%")

    # -------- Sorted by Elo delta (biggest mismatches first) --------
    print("\n  FIGHTS RANKED BY ELO DELTA (biggest mismatches first):")
    print(f"  {'Δ Elo':>7}  {'Favorite':<22} {'Elo':>5}  vs  {'Underdog':<22} {'Elo':>5}  Pick @ Conf")
    print("  " + "-" * 93)
    ranked = []
    for na, nb, wc, is_main, r in results:
        delta = abs(r["elo_a"] - r["elo_b"])
        if r["elo_a"] >= r["elo_b"]:
            fav, fav_elo = na, r["elo_a"]
            dog, dog_elo = nb, r["elo_b"]
        else:
            fav, fav_elo = nb, r["elo_b"]
            dog, dog_elo = na, r["elo_a"]
        ranked.append((delta, fav, fav_elo, dog, dog_elo, r))
    for delta, fav, fav_elo, dog, dog_elo, r in sorted(ranked, key=lambda x: -x[0]):
        flags = []
        if r["debut_a"]:
            flags.append(f"{r['fighter_a']}=DEBUT")
        if r["debut_b"]:
            flags.append(f"{r['fighter_b']}=DEBUT")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  {delta:>5.0f}   {fav:<22} {fav_elo:>5.0f}  vs  {dog:<22} {dog_elo:>5.0f}  "
              f"{r['winner_name']} @ {r['win_prob']:.0%}{flag_str}")

    # -------- Highest-confidence picks --------
    print("\n  HIGHEST-CONFIDENCE WINNER PICKS (usable band 65–85%):")
    for na, nb, wc, is_main, r in sorted(results, key=lambda x: -x[4]["win_prob"]):
        if 0.65 <= r["win_prob"] <= 0.85:
            print(f"    {r['winner_name']:<25} @ {r['win_prob']:.0%}  "
                  f"({na} vs {nb})")

    # -------- Strongest method/round prop leans --------
    print("\n  STRONGEST PROP LEANS:")
    for na, nb, wc, is_main, r in results:
        leans = []
        if r["goes_distance_prob"] >= 0.60:
            leans.append(f"goes dist {r['goes_distance_prob']:.0%}")
        elif r["goes_distance_prob"] <= 0.30:
            leans.append(f"finish {1-r['goes_distance_prob']:.0%}")
        if r["ends_r1_prob"] >= 0.35:
            leans.append(f"ends R1 {r['ends_r1_prob']:.0%}")
        if r["method_probs"]["KO/TKO"] >= 0.45:
            leans.append(f"KO/TKO {r['method_probs']['KO/TKO']:.0%}")
        if r["method_probs"]["SUB"] >= 0.30:
            leans.append(f"SUB {r['method_probs']['SUB']:.0%}")
        if leans:
            print(f"    {na} vs {nb}: " + " | ".join(leans))

    print("\n" + "=" * 95)


if __name__ == "__main__":
    main()
