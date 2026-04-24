"""UFC Fight Night: Sterling vs. Zalal — April 25, 2026"""
from predict import predict_fight

CARD = [
    ("Aljamain Sterling", "Youssef Zalal", True),
    ("Norma Dumont", "Joselyne Edwards", True),
    ("Rafa Garcia", "Alexander Hernandez", True),
    ("Davey Grant", "Adrian Luna Martinetti", True),
    ("Montel Jackson", "Raoni Barcelos", True),
    ("Marcus Buchecha", "Ryan Spann", False),
    ("Rodolfo Vieira", "Eric McConico", False),
    ("Jackson McVey", "Sedriques Dumas", False),
    ("Mayra Bueno Silva", "Michelle Montague", False),
    ("Jafel Filho", "Cody Durden", False),
    ("Francis Marshall", "Lucas Brennan", False),
    ("Max Griffin", "Victor Valenzuela", False),
    ("Talita Alencar", "Julia Polastri", False),
]

print("=" * 80)
print("  UFC FIGHT NIGHT: STERLING vs. ZALAL — ML MODEL PREDICTIONS")
print("  April 25, 2026")
print("=" * 80)

results = []
for a, b, is_main in CARD:
    r = predict_fight(a, b)
    if r:
        r["is_main"] = is_main
        results.append(r)

print(f"\n\n{'=' * 100}")
print(f"  FULL CARD PREDICTIONS")
print(f"{'=' * 100}\n")
print(f"  {'#':<3} {'Fight':<50} {'Winner':<22} {'Method':<8} {'Rd':>3} {'Conf':>6} {'Reaches R2':>12}")
print(f"  {'-' * 105}")
for i, r in enumerate(results, 1):
    tag = " *" if r.get("is_main") else ""
    fight = f"{r['fighter_a']} vs {r['fighter_b']}{tag}"
    r2 = r.get("reaches_r2_prob", 0.5)
    r2_label = f"{'YES' if r2 >= 0.5 else 'NO'} ({r2:.0%})"
    print(f"  {i:<3} {fight:<50} {r['winner_name']:<22} {r['method']:<8} R{r['round']:>1} {r['win_prob']:>5.0%} {r2_label:>12}")
print(f"\n{'=' * 100}")
