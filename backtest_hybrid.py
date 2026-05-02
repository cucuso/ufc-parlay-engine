"""
Backtest the hybrid (rules-override-ML) vs pure ML on the same held-out
test set used by backtest_all.py / backtest_rules.py.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ml_model import load_and_prepare, load_models, ELO_BLEND_WEIGHT
from rules_engine import predict_with_override, row_to_profiles


def main():
    df, feat_cols, _, _ = load_and_prepare()
    n = len(df)
    test = df.iloc[int(n * 0.80):].copy().reset_index(drop=True)
    print(f"Test set: {len(test):,} fights  ({test['date'].min().date()} → {test['date'].max().date()})")

    models = load_models()
    wm = models["winner_model"]
    wcols = models["winner_feature_columns"]

    # ---- Pure ML winner predictions (mirrors backtest_all.py) ----
    X = test[wcols].fillna(0).astype(np.float32)
    p_model = wm.predict_proba(X)[:, 1]
    ea, eb = test["a_elo"].values, test["b_elo"].values
    p_elo = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))
    p_a_ml = ELO_BLEND_WEIGHT * p_elo + (1 - ELO_BLEND_WEIGHT) * p_model
    pred_a_ml = (p_a_ml >= 0.5).astype(int)

    # ---- Hybrid: rule override when fired, else ML ----
    pred_a_hybrid = pred_a_ml.copy()
    rule_fired = []
    rule_overrides = 0

    for i, row in test.iterrows():
        a, b = row_to_profiles(row)
        pred = predict_with_override(a, b)
        if pred is not None:
            rule_overrides += 1
            pred_a_hybrid[i] = 1 if pred.winner == "A" else 0
            rule_fired.append(pred.rule)
        else:
            rule_fired.append(None)

    test["rule"] = rule_fired
    y = test["a_wins"].astype(int).values

    ml_acc = (pred_a_ml == y).mean()
    hyb_acc = (pred_a_hybrid == y).mean()
    elo_acc = ((p_elo >= 0.5).astype(int) == y).mean()

    print()
    print("=" * 70)
    print("WINNER ACCURACY — pure ML  vs  hybrid (rules override ML)")
    print("=" * 70)
    print(f"  Pure ML       : {ml_acc*100:>5.1f}%   (baseline)")
    print(f"  Hybrid        : {hyb_acc*100:>5.1f}%   ({(hyb_acc-ml_acc)*100:+.1f}pp vs ML)")
    print(f"  Elo-only      : {elo_acc*100:>5.1f}%")
    print(f"  Rule overrides fired: {rule_overrides}/{len(test)} = {rule_overrides/len(test)*100:.1f}%")

    # Per-rule breakdown
    print()
    print("Per-rule lift (only fights where rule fired):")
    print(f"  {'Rule':<26} {'N':>5} {'ML acc':>9} {'Rule acc':>11} {'Lift':>8}")
    for rule in sorted({r for r in rule_fired if r}):
        mask = np.array([r == rule for r in rule_fired])
        if mask.sum() < 5:
            continue
        ml_sub = (pred_a_ml[mask] == y[mask]).mean()
        hyb_sub = (pred_a_hybrid[mask] == y[mask]).mean()
        print(f"  {rule:<26} {mask.sum():>5} {ml_sub*100:>8.1f}% {hyb_sub*100:>10.1f}% "
              f"{(hyb_sub-ml_sub)*100:>+7.1f}pp")

    # Year breakdown
    print()
    print("Hybrid vs ML by year:")
    print(f"  {'Year':<6} {'N':>5} {'ML':>8} {'Hybrid':>9} {'Δ':>7}")
    test["year"] = test["date"].dt.year
    for yr, g in test.groupby("year"):
        if len(g) < 30: continue
        ix = g.index.values
        m = (pred_a_ml[ix] == y[ix]).mean()
        h = (pred_a_hybrid[ix] == y[ix]).mean()
        print(f"  {yr:<6} {len(g):>5} {m*100:>7.1f}% {h*100:>8.1f}% {(h-m)*100:>+6.1f}pp")

    # Disagreement analysis
    disagree = pred_a_ml != pred_a_hybrid
    n_d = disagree.sum()
    if n_d > 0:
        ml_right = (pred_a_ml[disagree] == y[disagree]).sum()
        hyb_right = (pred_a_hybrid[disagree] == y[disagree]).sum()
        print()
        print(f"Disagreement zone — {n_d} fights where rules and ML pick different winners:")
        print(f"  ML correct on these:    {ml_right}/{n_d} ({ml_right/n_d*100:.1f}%)")
        print(f"  Hybrid correct on these: {hyb_right}/{n_d} ({hyb_right/n_d*100:.1f}%)")
        print(f"  Net: rules-override {'WINS' if hyb_right > ml_right else 'LOSES'} by {abs(hyb_right-ml_right)} fights")


if __name__ == "__main__":
    main()
