"""
Full backtest: apply current trained models to every held-out test fight
(most recent 20% of enriched_fights.csv) and report aggregate performance.

This is the honest generalization number — the model never saw these fights
during training, pruning, or calibration.
"""
import numpy as np
import pandas as pd
from ml_model import (
    load_and_prepare, load_models, ELO_BLEND_WEIGHT,
)


def main():
    df, feat_cols, _, _ = load_and_prepare()
    n = len(df)
    i_calib = int(n * 0.80)  # test set starts after calibration slice

    test = df.iloc[i_calib:].copy().reset_index(drop=True)
    print(f"Test set: {len(test):,} fights  ({test['date'].min().date()} → {test['date'].max().date()})")

    models = load_models()
    wm = models["winner_model"]
    fm = models["finish_model"]
    kom = models["ko_vs_sub_model"]
    r2m = models["reaches_r2_model"]
    r3m = models["reaches_r3_model"]
    r1em = models["ends_r1_model"]
    wcols = models["winner_feature_columns"]
    fcols = models["finish_feature_columns"]
    kocols = models["ko_feature_columns"]
    r2cols = models["r2_feature_columns"]
    r3cols = models["r3_feature_columns"]
    r1ecols = models["r1end_feature_columns"]

    def sl(cols):
        return test[cols].fillna(0).astype(np.float32)

    # ---- Winner (with Elo blend, mirrors predict()) ----
    p_model = wm.predict_proba(sl(wcols))[:, 1]
    ea, eb = test["a_elo"].values, test["b_elo"].values
    p_elo = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))
    p_a = ELO_BLEND_WEIGHT * p_elo + (1 - ELO_BLEND_WEIGHT) * p_model
    pred_a_wins = (p_a >= 0.5).astype(int)
    y_w = test["a_wins"].astype(int).values
    winner_acc = (pred_a_wins == y_w).mean()
    elo_acc = ((p_elo >= 0.5).astype(int) == y_w).mean()

    # ---- Method via decomposition: DEC = 1-P(finish); KO = P(finish)*P(KO|f); SUB = rest ----
    p_finish = fm.predict_proba(sl(fcols))[:, 1]
    p_ko_given = kom.predict_proba(sl(kocols))[:, 1]
    method_probs = np.column_stack([
        1 - p_finish,                     # DEC
        p_finish * p_ko_given,            # KO/TKO
        p_finish * (1 - p_ko_given),      # SUB
    ])
    labels = np.array(["DEC", "KO/TKO", "SUB"])
    m_pred = labels[method_probs.argmax(axis=1)]

    def norm(m):
        m = str(m).upper()
        if "KO" in m or "TKO" in m:
            return "KO/TKO"
        if "SUB" in m:
            return "SUB"
        if "DEC" in m:
            return "DEC"
        return "OTHER"
    m_true = test["method"].apply(norm).values
    method_acc = (m_pred == m_true).mean()

    # ---- Prop markets (calibrated binaries) ----
    p_goes_dist = 1 - p_finish
    p_ends_r1 = r1em.predict_proba(sl(r1ecols))[:, 1]
    y_goes_dist = (m_true == "DEC").astype(int)
    y_ends_r1 = ((test["round_finished"].fillna(3).astype(int) == 1) & (m_true != "DEC")).astype(int).values
    gd_acc = ((p_goes_dist >= 0.5).astype(int) == y_goes_dist).mean()
    gd_brier = ((p_goes_dist - y_goes_dist) ** 2).mean()
    r1_acc = ((p_ends_r1 >= 0.5).astype(int) == y_ends_r1).mean()
    r1_brier = ((p_ends_r1 - y_ends_r1) ** 2).mean()

    # ---- Round cascade ----
    r2_p = r2m.predict_proba(sl(r2cols))[:, 1]
    r3_p = r3m.predict_proba(sl(r3cols))[:, 1]
    round_pred = np.where(r2_p < 0.5, 1, np.where(r3_p < 0.5, 2, 3))
    round_true = test["round_finished"].fillna(3).astype(int).clip(upper=3).values
    round_exact = (round_pred == round_true).mean()
    round_within1 = (np.abs(round_pred - round_true) <= 1).mean()

    # ---- Confidence-bucket calibration for WINNER ----
    # Pick the confidence of whichever side was predicted
    conf = np.where(p_a >= 0.5, p_a, 1 - p_a)
    hit = (pred_a_wins == y_w).astype(int)
    print()
    print("=" * 66)
    print(f"{'METRIC':<30} {'VALUE':>15} {'vs. BASELINE':>18}")
    print("=" * 66)
    print(f"{'Winner accuracy':<30} {winner_acc*100:>13.1f}%   {'Elo ' + f'{elo_acc*100:.1f}%':>18}")
    print(f"{'Method accuracy':<30} {method_acc*100:>13.1f}%   {'DEC-only ' + f'{(m_true==chr(68)+chr(69)+chr(67)).mean()*100:.1f}%':>18}")
    print(f"{'Round exact':<30} {round_exact*100:>13.1f}%")
    print(f"{'Round within ±1':<30} {round_within1*100:>13.1f}%")
    print(f"{'Goes-distance acc':<30} {gd_acc*100:>13.1f}%   brier={gd_brier:.4f}")
    print(f"{'Ends-R1 acc':<30} {r1_acc*100:>13.1f}%   brier={r1_brier:.4f}")

    # Method-class calibration tables (the money markets) — check each class
    def calib_table(name, p, y):
        print(f"\n{name} CALIBRATION:")
        print(f"{'P bucket':<15} {'N':>6} {'predicted':>12} {'actual':>10} {'diff':>8}")
        for lo, hi in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
                       (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]:
            mask = (p >= lo) & (p < hi)
            if mask.sum() < 15:
                continue
            print(f"{lo:.2f}–{hi:.2f}    {mask.sum():>6} "
                  f"{p[mask].mean()*100:>10.1f}%  {y[mask].mean()*100:>8.1f}%  "
                  f"{(y[mask].mean() - p[mask].mean())*100:>+6.1f}%")

    y_ko = (m_true == "KO/TKO").astype(int)
    y_sub = (m_true == "SUB").astype(int)
    p_ko = method_probs[:, 1]
    p_sub = method_probs[:, 2]
    calib_table("P(KO/TKO)", p_ko, y_ko)
    calib_table("P(SUB)", p_sub, y_sub)
    calib_table("P(goes distance)", p_goes_dist, y_goes_dist)
    calib_table("P(ends in R1)", p_ends_r1, y_ends_r1)

    # Winner calibration table
    print()
    print("WINNER CALIBRATION — does X% confidence actually win X% of the time?")
    print(f"{'Conf bucket':<15} {'N':>6} {'predicted':>12} {'actual':>10} {'diff':>8}")
    for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
                   (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 1.01)]:
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() < 10:
            continue
        pred_mean = conf[mask].mean()
        actual_mean = hit[mask].mean()
        diff = actual_mean - pred_mean
        print(f"{lo:.2f}–{hi:.2f}    {mask.sum():>6} {pred_mean*100:>10.1f}%  {actual_mean*100:>8.1f}%  {diff*100:>+6.1f}%")

    # Per-weight-class breakdown (winner only)
    print()
    print("BY WEIGHT CLASS:")
    print(f"{'Division':<25} {'N':>5} {'Winner':>10} {'Method':>10} {'Round ±1':>10}")
    test["_pred_a"] = pred_a_wins
    test["_m_pred"] = m_pred
    test["_m_true"] = m_true
    test["_r_pred"] = round_pred
    test["_r_true"] = round_true
    for wc, g in test.groupby("weight_class"):
        if len(g) < 30:
            continue
        ix = g.index
        w = (pred_a_wins[ix] == y_w[ix]).mean()
        mm_ = (m_pred[ix] == m_true[ix]).mean()
        r_ = (np.abs(round_pred[ix] - round_true[ix]) <= 1).mean()
        print(f"{wc:<25} {len(g):>5} {w*100:>9.1f}% {mm_*100:>9.1f}% {r_*100:>9.1f}%")

    # Per-year trend
    print()
    print("BY YEAR (is the model stable over time?):")
    print(f"{'Year':<6} {'N':>5} {'Winner':>10} {'Elo':>8} {'Method':>10} {'Round ±1':>10}")
    test["_year"] = test["date"].dt.year
    for yr, g in test.groupby("_year"):
        if len(g) < 30:
            continue
        ix = g.index
        w = (pred_a_wins[ix] == y_w[ix]).mean()
        e = ((p_elo[ix] >= 0.5).astype(int) == y_w[ix]).mean()
        mm_ = (m_pred[ix] == m_true[ix]).mean()
        r_ = (np.abs(round_pred[ix] - round_true[ix]) <= 1).mean()
        print(f"{yr:<6} {len(g):>5} {w*100:>9.1f}% {e*100:>7.1f}% {mm_*100:>9.1f}% {r_*100:>9.1f}%")


if __name__ == "__main__":
    main()
