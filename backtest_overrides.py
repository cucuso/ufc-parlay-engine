"""
Backtest the pattern override layer (predict's R1/SUB/KO/GRINDER overrides)
against the raw ML output. Compare on the same 1,359 held-out test fights.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ml_model import load_and_prepare, load_models, ELO_BLEND_WEIGHT


def normalize_method(m: str) -> str:
    m = str(m).upper()
    if "KO" in m or "TKO" in m: return "KO/TKO"
    if "SUB" in m: return "SUB"
    if "DEC" in m: return "DEC"
    return "OTHER"


def main():
    df, feat_cols, _, _ = load_and_prepare()
    n = len(df)
    test = df.iloc[int(n * 0.80):].copy().reset_index(drop=True)
    print(f"Test set: {len(test):,} fights  ({test['date'].min().date()} → {test['date'].max().date()})")

    models = load_models()
    wm, fm, kom = models["winner_model"], models["finish_model"], models["ko_vs_sub_model"]
    r2m, r3m, r1em = models["reaches_r2_model"], models["reaches_r3_model"], models["ends_r1_model"]
    wcols = models["winner_feature_columns"]
    fcols = models["finish_feature_columns"]
    kocols = models["ko_feature_columns"]
    r2cols = models["r2_feature_columns"]
    r3cols = models["r3_feature_columns"]
    r1ecols = models["r1end_feature_columns"]

    def sl(cols):
        return test[cols].fillna(0).astype(np.float32)

    # Raw ML predictions
    p_model = wm.predict_proba(sl(wcols))[:, 1]
    ea, eb = test["a_elo"].values, test["b_elo"].values
    p_elo = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))
    p_a = ELO_BLEND_WEIGHT * p_elo + (1 - ELO_BLEND_WEIGHT) * p_model
    pred_a_wins = (p_a >= 0.5).astype(int)
    p_finish = fm.predict_proba(sl(fcols))[:, 1]
    p_ko_given = kom.predict_proba(sl(kocols))[:, 1]
    method_dec = 1 - p_finish
    method_ko = p_finish * p_ko_given
    method_sub = p_finish * (1 - p_ko_given)
    raw_method = np.where(method_dec >= np.maximum(method_ko, method_sub), "DEC",
                          np.where(method_ko >= method_sub, "KO/TKO", "SUB"))
    r2_p = r2m.predict_proba(sl(r2cols))[:, 1]
    r3_p = r3m.predict_proba(sl(r3cols))[:, 1]
    raw_round = np.where(r2_p < 0.5, 1, np.where(r3_p < 0.5, 2, 3))

    # Apply overrides per-row
    over_method = raw_method.copy()
    over_round = raw_round.copy()
    overrides_fired = []

    for i, row in test.iterrows():
        # winner per row (after potential A/B swap convention)
        w_is_a = pred_a_wins[i] == 1
        prefix_w = "a_" if w_is_a else "b_"
        prefix_l = "b_" if w_is_a else "a_"

        w_r1_5 = float(row.get(f"{prefix_w}r1_ending_rate_5", row.get(f"{prefix_w}r1_ending_rate", 0)) or 0)
        l_r1_5 = float(row.get(f"{prefix_l}r1_ending_rate_5", row.get(f"{prefix_l}r1_ending_rate", 0)) or 0)
        w_dec_5 = float(row.get(f"{prefix_w}decision_rate_5", row.get(f"{prefix_w}decision_rate", 0)) or 0)
        l_dec_5 = float(row.get(f"{prefix_l}decision_rate_5", row.get(f"{prefix_l}decision_rate", 0)) or 0)
        w_sub_5 = float(row.get(f"{prefix_w}sub_win_rate_5", row.get(f"{prefix_w}sub_win_rate", 0)) or 0)
        w_ko_5 = float(row.get(f"{prefix_w}ko_win_rate_5", row.get(f"{prefix_w}ko_win_rate", 0)) or 0)
        w_sub_c = float(row.get(f"{prefix_w}sub_win_rate", 0) or 0)
        w_ko_c = float(row.get(f"{prefix_w}ko_win_rate", 0) or 0)

        grinder = (w_dec_5 * l_dec_5 >= 0.30 and min(w_dec_5, l_dec_5) >= 0.50)
        r1_spec = (min(w_r1_5, l_r1_5) >= 0.50 and (w_r1_5 * l_r1_5) >= 0.35)
        winner_dec_spec = (w_dec_5 >= 0.75)

        # Round override
        if grinder:
            over_round[i] = 3
            overrides_fired.append("grinder")
        elif winner_dec_spec and not r1_spec:
            over_round[i] = 3
            overrides_fired.append("winner_dec_spec")
        elif r1_spec:
            over_round[i] = 1
            overrides_fired.append("r1_spec")
        else:
            overrides_fired.append("")

        # Method override
        if grinder:
            over_method[i] = "DEC"
        elif winner_dec_spec and not r1_spec:
            over_method[i] = "DEC"
        elif r1_spec and over_method[i] == "DEC":
            over_method[i] = "KO/TKO" if method_ko[i] >= method_sub[i] else "SUB"

    test["override"] = overrides_fired
    actual_method = test["method"].apply(normalize_method).values
    actual_round = test["round_finished"].fillna(3).astype(int).clip(upper=5).values

    print()
    print("=" * 70)
    print("METHOD ACCURACY: raw ML  vs  with overrides")
    print("=" * 70)
    raw_method_acc = (raw_method == actual_method).mean()
    over_method_acc = (over_method == actual_method).mean()
    print(f"  Raw ML:        {raw_method_acc*100:.1f}%")
    print(f"  With overrides: {over_method_acc*100:.1f}%   ({(over_method_acc-raw_method_acc)*100:+.1f}pp)")

    print()
    print("=" * 70)
    print("ROUND-EXACT ACCURACY: raw ML  vs  with overrides")
    print("=" * 70)
    raw_round_acc = (raw_round == actual_round).mean()
    over_round_acc = (over_round == actual_round).mean()
    print(f"  Raw ML:        {raw_round_acc*100:.1f}%")
    print(f"  With overrides: {over_round_acc*100:.1f}%   ({(over_round_acc-raw_round_acc)*100:+.1f}pp)")

    print()
    print("=" * 70)
    print("ROUND ±1 ACCURACY: raw ML  vs  with overrides")
    print("=" * 70)
    raw_rw1 = (np.abs(raw_round - actual_round) <= 1).mean()
    over_rw1 = (np.abs(over_round - actual_round) <= 1).mean()
    print(f"  Raw ML:        {raw_rw1*100:.1f}%")
    print(f"  With overrides: {over_rw1*100:.1f}%   ({(over_rw1-raw_rw1)*100:+.1f}pp)")

    # Analyze per-override-type
    print()
    print("Per-override-type lift:")
    print(f"  {'Override':<14} {'N':>5} {'raw method':>11} {'over method':>13} {'raw round':>11} {'over round':>13}")
    for ov in ["", "grinder", "winner_dec_spec", "r1_spec"]:
        mask = np.array([o == ov for o in overrides_fired])
        if mask.sum() == 0: continue
        rm = (raw_method[mask] == actual_method[mask]).mean()
        om = (over_method[mask] == actual_method[mask]).mean()
        rr = (raw_round[mask] == actual_round[mask]).mean()
        orr = (over_round[mask] == actual_round[mask]).mean()
        label = ov or "(no override)"
        print(f"  {label:<14} {mask.sum():>5} {rm*100:>10.1f}% {om*100:>12.1f}% {rr*100:>10.1f}% {orr*100:>12.1f}%")


if __name__ == "__main__":
    main()
