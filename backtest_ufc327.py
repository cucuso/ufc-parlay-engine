"""
UFC 327 Backtest — score ml_model predictions against actual results.

Leak-free: uses each fighter's latest snapshot in enriched_fighters.csv
with date < 2026-04-11 as the pre-fight profile.
"""
import pandas as pd
from ml_model import predict
from pathlib import Path

BASE = Path(__file__).parent
CUTOFF = pd.Timestamp("2026-04-11")

NAME_ALIAS = {"Patricio Pitbull": "Patricio Freire"}

CARD = [
    ("Jiri Prochazka",    "Carlos Ulberg",    "Ulberg",    "KO/TKO", 1),
    ("Azamat Murzakanov", "Paulo Costa",      "Costa",     "KO/TKO", 3),
    ("Curtis Blaydes",    "Josh Hokit",       "Hokit",     "DEC",    3),
    ("Dominick Reyes",    "Johnny Walker",    "Reyes",     "DEC",    3),
    ("Cub Swanson",       "Nate Landwehr",    "Swanson",   "KO/TKO", 1),
    ("Patricio Pitbull",  "Aaron Pico",       "Pico",      "DEC",    3),
    ("Kevin Holland",     "Randy Brown",      "Holland",   "DEC",    3),
    ("Mateusz Gamrot",    "Esteban Ribovics", "Gamrot",    "SUB",    2),
    ("Tatiana Suarez",    "Loopy Godinez",    "Suarez",    "SUB",    2),
    ("Chris Padilla",     "MarQuel Mederos",  "DRAW",      "DEC",    3),
    ("Kelvin Gastelum",   "Vicente Luque",    "Luque",     "SUB",    1),
    ("Charles Radtke",    "Francisco Prado",  "Radtke",    "DEC",    3),
]


def load_profile(df: pd.DataFrame, name: str) -> dict | None:
    lookup = NAME_ALIAS.get(name, name)
    sub = df[(df["fighter"] == lookup) & (df["date"] < CUTOFF)].sort_values("date")
    if sub.empty:
        return None
    return sub.iloc[-1].to_dict()


def main():
    df = pd.read_csv(BASE / "data" / "enriched_fighters.csv")
    df["date"] = pd.to_datetime(df["date"])

    winner_hits = 0
    method_hits = 0
    round_hits = 0
    round_within1 = 0
    scored = 0
    rows = []

    for name_a, name_b, actual_winner_last, actual_method, actual_round in CARD:
        pa = load_profile(df, name_a)
        pb = load_profile(df, name_b)
        if pa is None or pb is None:
            print(f"  SKIP (no profile): {name_a} vs {name_b}")
            continue

        result = predict(pa, pb)
        pred_name = name_a if result["winner"] == "A" else name_b
        pred_last = pred_name.split()[-1]

        is_draw = actual_winner_last == "DRAW"
        w_ok = (not is_draw) and (pred_last == actual_winner_last)
        m_ok = result["method"] == actual_method
        r_ok = result["round"] == actual_round

        if not is_draw:
            scored += 1
            winner_hits += int(w_ok)
            method_hits += int(m_ok)
            round_hits += int(r_ok)
            round_within1 += int(abs(result["round"] - actual_round) <= 1)

        rows.append({
            "fight": f"{name_a} vs {name_b}",
            "pred": pred_last,
            "pred_conf": result["win_prob"],
            "pred_method": result["method"],
            "pred_round": result["round"],
            "actual": actual_winner_last,
            "actual_method": actual_method,
            "actual_round": actual_round,
            "W": "Y" if w_ok else ("—" if is_draw else "N"),
            "M": "Y" if m_ok else "N",
            "R": "Y" if r_ok else "N",
            "a_elo": pa["elo"],
            "b_elo": pb["elo"],
        })

    print(f"\n{'Fight':<40} {'Pred':<10} {'Conf':>5} {'Method':<7} {'R':>1} | "
          f"{'Actual':<10} {'Method':<7} {'R':>1} | W M R")
    print("─" * 115)
    for r in rows:
        print(f"{r['fight']:<40} {r['pred']:<10} {r['pred_conf']*100:>4.0f}% "
              f"{r['pred_method']:<7} {r['pred_round']} | "
              f"{r['actual']:<10} {r['actual_method']:<7} {r['actual_round']} | "
              f"{r['W']} {r['M']} {r['R']}")

    print("─" * 115)
    if scored:
        print(f"  Winner: {winner_hits}/{scored} ({winner_hits/scored:.0%})   "
              f"Method: {method_hits}/{scored} ({method_hits/scored:.0%})   "
              f"Round:  {round_hits}/{scored} ({round_hits/scored:.0%}) exact, "
              f"{round_within1}/{scored} ({round_within1/scored:.0%}) within ±1")

    # Elo-favorite analysis
    elo_fav_right = 0
    elo_fav_n = 0
    for r in rows:
        if r["actual"] == "DRAW":
            continue
        elo_fav_n += 1
        fav_last = (r["fight"].split(" vs ")[0] if r["a_elo"] >= r["b_elo"]
                    else r["fight"].split(" vs ")[1]).split()[-1]
        if fav_last == r["actual"]:
            elo_fav_right += 1
    print(f"  Elo favorite won: {elo_fav_right}/{elo_fav_n} ({elo_fav_right/elo_fav_n:.0%}) "
          f"— this is the ceiling of any 'pick higher Elo' strategy")


if __name__ == "__main__":
    main()
