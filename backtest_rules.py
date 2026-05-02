"""
Backtest the rules-first engine against the same test set used by backtest_all.py
(last 20% of enriched_fights.csv) and compare to ML accuracy.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from rules_engine import predict_rules, row_to_profiles


def normalize_method(m: str) -> str:
    m = str(m).upper()
    if "KO" in m or "TKO" in m: return "KO/TKO"
    if "SUB" in m: return "SUB"
    if "DEC" in m: return "DEC"
    return "OTHER"


def main():
    df = pd.read_csv("data/enriched_fights.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    test = df.iloc[int(n * 0.80):].copy().reset_index(drop=True)
    print(f"Test set: {len(test):,} fights  ({test['date'].min().date()} → {test['date'].max().date()})")

    rule_predictions = []
    rule_used = []
    for _, row in test.iterrows():
        a, b = row_to_profiles(row)
        pred = predict_rules(a, b)
        rule_predictions.append(pred)
        rule_used.append(pred.rule)

    test["rule_winner"] = ["A" if p.winner == "A" else "B" for p in rule_predictions]
    test["rule_method"] = [p.method for p in rule_predictions]
    test["rule_round"]  = [p.round  for p in rule_predictions]
    test["rule"] = rule_used

    # Actuals
    test["actual_winner"] = test["a_wins"].apply(lambda x: "A" if x == 1 else "B")
    test["actual_method"] = test["method"].apply(normalize_method)
    test["actual_round"]  = test["round_finished"].fillna(3).astype(int).clip(upper=5)

    # Hits
    test["winner_hit"] = test["rule_winner"] == test["actual_winner"]
    test["method_hit"] = test["rule_method"] == test["actual_method"]
    test["round_exact"]   = test["rule_round"] == test["actual_round"]
    test["round_within1"] = (test["rule_round"] - test["actual_round"]).abs() <= 1

    print()
    print("=" * 76)
    print(f"{'METRIC':<28} {'RULES':>10} {'ML (current)':>14} {'baseline':>15}")
    print("=" * 76)
    print(f"{'Winner accuracy':<28} {test['winner_hit'].mean()*100:>9.1f}%  "
          f"{'62.4%':>13}  {'Elo 58.0%':>15}")
    print(f"{'Method accuracy':<28} {test['method_hit'].mean()*100:>9.1f}%  "
          f"{'52.9%':>13}  {'DEC-only 51.3%':>15}")
    print(f"{'Round exact':<28} {test['round_exact'].mean()*100:>9.1f}%  "
          f"{'50.2%':>13}  {'-':>15}")
    print(f"{'Round within ±1':<28} {test['round_within1'].mean()*100:>9.1f}%  "
          f"{'85.4%':>13}  {'-':>15}")

    print()
    print("RULES FIRED — distribution + per-rule accuracy:")
    print(f"{'Rule':<32} {'N':>5} {'%':>5}  {'Winner':>8} {'Method':>8} {'Rd ±1':>8}")
    print("-" * 70)
    for rule in test["rule"].value_counts().index:
        sub = test[test["rule"] == rule]
        print(f"{rule:<32} {len(sub):>5} {len(sub)/len(test)*100:>4.1f}%  "
              f"{sub['winner_hit'].mean()*100:>7.1f}% "
              f"{sub['method_hit'].mean()*100:>7.1f}% "
              f"{sub['round_within1'].mean()*100:>7.1f}%")

    print()
    print("ACCURACY BY YEAR:")
    print(f"{'Year':<6} {'N':>5} {'Winner':>10} {'Method':>10} {'Rd ±1':>10}")
    test["year"] = test["date"].dt.year
    for yr, g in test.groupby("year"):
        if len(g) < 30: continue
        print(f"{yr:<6} {len(g):>5} "
              f"{g['winner_hit'].mean()*100:>9.1f}% "
              f"{g['method_hit'].mean()*100:>9.1f}% "
              f"{g['round_within1'].mean()*100:>9.1f}%")

    # Save side-by-side comparison
    print()
    print("SAMPLE: Rules vs actual on 10 random test fights:")
    sample = test.sample(min(10, len(test)), random_state=42)
    for _, r in sample.iterrows():
        wh = "✓" if r["winner_hit"] else "✗"
        mh = "✓" if r["method_hit"] else "✗"
        rh = "✓" if r["round_within1"] else "✗"
        print(f"  [{r['rule']:<24}] pick: {r['rule_winner']} {r['rule_method']:<7} R{r['rule_round']}  "
              f"actual: {r['actual_winner']} {r['actual_method']:<7} R{r['actual_round']}  "
              f"({wh}{mh}{rh})")


if __name__ == "__main__":
    main()
