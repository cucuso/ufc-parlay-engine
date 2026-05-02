"""
UFC Fight Night: Della Maddalena vs. Prates — May 2, 2026
RAC Arena, Perth, Australia
Full card predictions via the calibrated ML model.
"""
from prediction_store import run_card_and_save, print_bettable_card

EVENT_NAME = "UFC Fight Night: Della Maddalena vs. Prates"
EVENT_DATE = "2026-05-02"

CARD = [
    # (fighter_a, fighter_b, weight_class, is_main_event_card)
    # Main Card
    ("Jack Della Maddalena", "Carlos Prates",         "Welterweight",     True),
    ("Beneil Dariush",       "Quillan Salkilld",      "Lightweight",      True),
    ("Tim Elliott",          "Steve Erceg",           "Flyweight",        True),
    ("Marwan Rahiki",        "Ollie Schmid",          "Featherweight",    True),
    ("Shamil Gaziev",        "Brando Pericic",        "Heavyweight",      True),
    ("Tai Tuivasa",          "Louie Sutherland",      "Heavyweight",      True),
    # Prelims
    ("Cam Rowston",          "Robert Bryczek",        "Middleweight",     False),
    ("Junior Tafa",          "Kevin Christian",       "Light Heavyweight",False),
    ("Jacob Malkoun",        "Gerald Meerschaert",    "Middleweight",     False),
    ("Colby Thicknesse",     "Vince Morales",         "Bantamweight",     False),
    ("Ben Johnston",         "Wes Schultz",           "Middleweight",     False),
    ("Jonathan Micallef",    "Themba Gorimbo",        "Welterweight",     False),
    ("Dom Mar Fan",          "Kody Steele",           "Lightweight",      False),
]


def main():
    print("=" * 95)
    print(f"  {EVENT_NAME.upper()} — {EVENT_DATE}")
    print("  RAC Arena, Perth, Australia")
    print("  Full card predictions (calibrated ML model)")
    print("=" * 95)
    results, snap_path = run_card_and_save(EVENT_NAME, EVENT_DATE, CARD)

    print("\n\n" + "=" * 95)
    print("  SUMMARY TABLE")
    print("=" * 95)
    print(f"  {'Fight':<42} {'Winner':<22} {'Conf':>5} {'Method':>8} {'R':>2} "
          f"{'KO%':>5} {'SUB%':>5} {'Dist%':>6}")
    print("  " + "-" * 95)
    for na, nb, wc, is_main, r in results:
        tag = "*" if is_main else " "
        fight = f"{na} vs {nb}"[:40]
        print(f"  {tag}{fight:<41} {r['winner_name']:<22} {r['win_prob']*100:>4.0f}% "
              f"{r['method']:>8} R{r['round']:>1} "
              f"{r['method_probs']['KO/TKO']*100:>4.0f}% {r['method_probs']['SUB']*100:>4.0f}% "
              f"{r['goes_distance_prob']*100:>5.0f}%")

    print("\n  FIGHTS RANKED BY ELO DELTA (biggest mismatches first):")
    print(f"  {'Δ Elo':>7}  {'Favorite':<22} {'Elo':>5}  vs  {'Underdog':<22} {'Elo':>5}  Pick @ Conf")
    print("  " + "-" * 95)
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

    print("\n  HIGHEST-CONFIDENCE WINNER PICKS (usable band 65-85%):")
    for na, nb, wc, is_main, r in sorted(results, key=lambda x: -x[4]["win_prob"]):
        if 0.65 <= r["win_prob"] <= 0.85:
            print(f"    {r['winner_name']:<25} @ {r['win_prob']:.0%}  "
                  f"({na} vs {nb})")

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
    # Bettable-card view (gate_picks output) so the bettor sees the
    # post-gate buckets — same final layer used for prior cards.
    print_bettable_card(snap_path)


if __name__ == "__main__":
    main()
