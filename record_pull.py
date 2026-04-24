"""
Pull raw records for every fighter on UFC Fight Night: Burns vs Malott.
NO model, NO predictions. Just records for expert-eye analysis.
"""
from profile_builder import find_espn_id, fetch_espn_profile, parse_espn_data


FIGHTS = [
    ("Gilbert Burns",        "Mike Malott",           "Welterweight",         True),
    ("Kyler Phillips",       "Charles Jourdain",      "Bantamweight",         False),
    ("Mandel Nallo",         "Jai Herbert",           "Lightweight",          False),
    ("Jasmine Jasudavicius", "Karine Silva",          "W-Flyweight",          False),
    ("Thiago Moises",        "Gauge Young",           "Lightweight",          False),
    ("Dennis Buzukja",       "Marcio Barbosa",        "Featherweight",        False),
    ("Julien Leblanc",       "Robert Valentin",       "Middleweight",         False),
    ("Tanner Boser",         "Gokhan Saricam",        "Heavyweight",          False),
    ("Melissa Croden",       "Daria Zhelezniakova",   "W-Bantamweight",       False),
    ("JJ Aldrich",           "Jamey-Lyn Horth",       "W-Flyweight",          False),
    ("John Castaneda",       "Mark Vologdin",         "Bantamweight",         False),
]


def fetch_record(name: str):
    eid = find_espn_id(name)
    if not eid:
        return None
    data = fetch_espn_profile(eid)
    if not data or "athlete" not in data:
        return None
    p = parse_espn_data(data)
    w, l = p["total_wins"], p["total_losses"]
    ko_w, sub_w, dec_w = p["ko_wins"], p["sub_wins"], p["dec_wins"]
    ko_l, sub_l, dec_l = p["ko_losses"], p["sub_losses"], p["dec_losses"]
    # Recent streak from fight history
    streak = 0
    for f in reversed(p["fights"]):
        r = f["result"]
        if r == "W":
            streak = streak + 1 if streak >= 0 else 1
        elif r == "L":
            if streak <= 0:
                streak -= 1
            else:
                streak = -1
            break
        else:
            break
    last5 = "".join(f["result"] or "?" for f in p["fights"][-5:])
    return {
        "name": p["name"],
        "age": p["age"],
        "w": w, "l": l,
        "ko_w": ko_w, "ko_l": ko_l,
        "sub_w": sub_w, "sub_l": sub_l,
        "dec_w": dec_w, "dec_l": dec_l,
        "finish_rate_wins": (ko_w + sub_w) / max(w, 1),
        "been_finished_rate": (ko_l + sub_l) / max(w + l, 1),
        "streak": streak,
        "last5": last5,
        "num_fights": len(p["fights"]),
    }


def main():
    print(f"{'Fighter':<24} {'Rec':>7} {'Age':>3} {'KO':>5} {'SUB':>5} {'DEC':>5}  "
          f"{'Finish%':>7}  {'Finished%':>9}  {'Streak':>6}  last5")
    print("-" * 110)
    for a, b, wc, main_evt in FIGHTS:
        print(f"\n[{wc}{' — MAIN' if main_evt else ''}]")
        for name in (a, b):
            r = fetch_record(name)
            if r is None:
                print(f"  {name:<24}  (could not fetch)")
                continue
            print(f"  {r['name']:<24} {r['w']:>3}-{r['l']:<3} {r['age']:>3}  "
                  f"{r['ko_w']:>2}-{r['ko_l']:<2}  {r['sub_w']:>2}-{r['sub_l']:<2}  "
                  f"{r['dec_w']:>2}-{r['dec_l']:<2}  "
                  f"{r['finish_rate_wins']*100:>6.0f}%  {r['been_finished_rate']*100:>8.0f}%  "
                  f"{r['streak']:>+6}  {r['last5']}")


if __name__ == "__main__":
    main()
