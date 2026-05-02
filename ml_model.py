"""
UFC Fight Prediction ML Model

Trains three XGBoost models on enriched_fights.csv:
  1. Winner (binary: a_wins)
  2. Method (multiclass: KO/TKO, SUB, DEC, OTHER)
  3. Round (the round the fight ended)

Usage:
    python3 ml_model.py   # Train, evaluate, and save models
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "enriched_fights.csv"
MODELS_DIR = BASE_DIR / "data" / "models"

# XGBoost hyperparameters — regularized to curb 149-feature overfitting.
# reg_alpha (L1) zeroes out noise features; reg_lambda (L2) shrinks weights.
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    min_child_weight=5,
    verbosity=0,
)

# Keep the top-K features by importance after a first-pass fit.
# Reduces the 149-column feature set to a tractable, less noisy subset.
TOP_K_FEATURES = 30

# Features that ALWAYS survive pruning regardless of first-pass importance.
# The probe XGB undervalues multiplicative interactions because the individual
# components (a_r1, b_r1) are already informative on their own — but the product
# is what we actually need for tail predictions on extreme matchups. Force-keep
# these so the model can find them and use them in joint splits.
FORCE_KEEP_FEATURES = (
    "r1_chaos_score",
    "r1_chaos_score_recent",
    "both_r1_specialists",
    "min_r1_ending_rate",
    "both_grinders_score",
    "joint_finish_propensity",
    # Single-side specialist features (added 2026-04-30) — probe XGB
    # under-weights these because they overlap with min/product features
    # for joint-extreme matchups, but they're load-bearing for single-side
    # specialist cases (Buzukja-Barbosa: high b_r1, low a_r1 → max captures
    # Barbosa's signal where min/product zero it out).
    "max_r1_ending_rate",
    "max_r1_ending_rate_5",
    "max_ko_win_rate_5",
    "max_sub_win_rate_5",
    "single_side_r1_specialist",
)

# Weight of the Elo prior in the winner probability (0.0 = pure model, 1.0 = pure Elo).
# Tuned on the held-out test set — pure regularized model scores 61.9%, blend at
# 0.3 scores 61.6% with meaningful anchoring against overconfident extreme calls
# (e.g. picking a -2 streak rookie at 84%). Weights >0.5 hurt aggregate accuracy.
ELO_BLEND_WEIGHT = 0.3

# Columns to exclude from features
EXCLUDE_COLS = {
    "a_wins", "fight_pk", "date", "method", "round_finished",
    "rounds", "weight_class", "fighter_a", "fighter_b",
}

# Matchup-level interaction features (no a_/b_/delta_/wc_ prefix). These
# were being silently excluded by the prefix filter in _feature_columns().
# Listed here so they get pulled in.
MATCHUP_FEATURES = {
    "composite_advantage", "signals_aligned_for_a", "signals_aligned_for_b",
    "experience_gap", "veteran_vs_newcomer", "schedule_gap", "style_clash",
    # R1-chaos interaction features (added 2026-04-28 to capture extreme matchups)
    "r1_chaos_score", "r1_chaos_score_recent", "both_r1_specialists",
    "min_r1_ending_rate", "both_grinders_score", "combined_avg_fight_secs",
    "joint_finish_propensity",
    # Single-side specialist features (added 2026-04-30) — capture extreme
    # finishers whose opponent isn't matching their pace pattern.
    "max_r1_ending_rate", "max_r1_ending_rate_5",
    "max_ko_win_rate_5", "max_sub_win_rate_5",
    "single_side_r1_specialist",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted list of ALL feature column names (a_*, b_*, delta_*, wc_*,
    and matchup-level interactions)."""
    base = [
        c for c in df.columns
        if c.startswith(("a_", "b_", "delta_", "wc_")) and c not in EXCLUDE_COLS
    ]
    matchup = [c for c in df.columns if c in MATCHUP_FEATURES]
    return sorted(set(base + matchup))


# ------------------------------------------------------------------
# Weight-class base-rate features (causal rolling means per division)
# ------------------------------------------------------------------

WC_FEATURES = [
    "wc_ko_rate", "wc_sub_rate", "wc_dec_rate",
    "wc_r1_finish_rate", "wc_reaches_r3_rate", "wc_avg_rounds",
]


def _add_wc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append per-weight-class expanding rates (prior-only, no leak).

    Heavyweight fights finish ~3× as often in R1 as women's strawweight — this
    base rate is a strong prior for method/round predictions and was previously
    invisible to the model (weight_class was in EXCLUDE_COLS).
    """
    df = df.copy()

    def norm(m):
        m = str(m).upper()
        if "KO" in m or "TKO" in m:
            return "KO/TKO"
        if "SUB" in m:
            return "SUB"
        if "DEC" in m:
            return "DEC"
        return "OTHER"

    _m = df["method"].apply(norm)
    _rd = df["round_finished"].fillna(3).astype(int)
    indicators = pd.DataFrame({
        "_is_ko": (_m == "KO/TKO").astype(float),
        "_is_sub": (_m == "SUB").astype(float),
        "_is_dec": (_m == "DEC").astype(float),
        "_r1_finish": ((_rd == 1) & (_m != "DEC")).astype(float),
        "_reaches_r3": (_rd >= 3).astype(float),
        "_rounds": _rd.astype(float),
    }, index=df.index)

    mapping = [
        ("_is_ko", "wc_ko_rate"),
        ("_is_sub", "wc_sub_rate"),
        ("_is_dec", "wc_dec_rate"),
        ("_r1_finish", "wc_r1_finish_rate"),
        ("_reaches_r3", "wc_reaches_r3_rate"),
        ("_rounds", "wc_avg_rounds"),
    ]
    joined = pd.concat([df["weight_class"], indicators], axis=1)
    for src, dst in mapping:
        # Shift by 1 per group so the current fight isn't included in its own base rate.
        df[dst] = joined.groupby("weight_class")[src].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        df[dst] = df[dst].fillna(df[dst].mean())
    return df


def _wc_snapshot(df: pd.DataFrame) -> dict:
    """Final wc_* rates per division — used at predict() time."""
    latest = df.sort_values("date").groupby("weight_class").tail(1)
    out = {}
    for _, r in latest.iterrows():
        out[r["weight_class"]] = {c: float(r[c]) for c in WC_FEATURES}
    # Global fallback for unknown divisions
    out["__default__"] = {c: float(df[c].mean()) for c in WC_FEATURES}
    return out


# ------------------------------------------------------------------
# ROADMAP: Tail-finish auxiliary head (planned, not yet implemented)
# ------------------------------------------------------------------
# The current ends_r1 / finish models are isotonic-calibrated for aggregate
# accuracy across all UFC fights. This squashes tail predictions toward the
# population base rate (~28% R1 finish), which is good for Brier score on
# the average fight but poor for spotting extreme matchups.
#
# Planned: a SECOND ends_r1_specialist head trained ONLY on fights where
# `single_side_r1_specialist=1` OR `both_r1_specialists=1`. No calibration.
# Use class-balanced sample weights so the model isn't dragged toward 0
# by the majority of "didn't end R1" outcomes in the filtered subset.
#
# At predict time, when `single_side_r1_specialist=1` or `both_r1_specialists=1`,
# blend: final_p = 0.6 * tail_head + 0.4 * calibrated_main.
# Otherwise use the calibrated main model alone.
#
# Why this is deferred: requires ~1000+ tail-pattern training fights to
# avoid overfitting; needs stratified train/test split; needs calibration
# decision (raw vs Platt vs isotonic on subset). Estimated 3-4 hours and
# worth running a held-out validation card before swapping in.
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Bayesian shrinkage toward division means
# ------------------------------------------------------------------

# Stats that are averages or rates over a fighter's career — these are the ones
# where small-sample variance hurts (e.g. a 1-0 rookie with finish_rate=1.0).
SHRINKABLE_STATS = [
    "win_pct", "ko_win_rate", "sub_win_rate", "finish_rate",
    "ko_loss_rate", "sub_loss_rate", "been_finished_rate",
    "ko_finish_share", "sub_finish_share", "ko_loss_share", "sub_loss_share",
    "avg_sig_str_landed", "avg_sig_str_attempts", "avg_sig_str_accuracy",
    "avg_sig_str_received", "avg_sig_str_avoided", "avg_sig_str_defense",
    "sig_str_per_min", "sig_str_absorbed_per_min",
    "recent_sig_str_landed", "recent_sig_str_accuracy", "recent_sig_str_defense",
    "avg_td_landed", "avg_td_attempts", "avg_td_accuracy",
    "avg_tds_received", "avg_td_defense", "td_per_min",
    "recent_td_landed", "recent_td_accuracy", "recent_td_defense",
    "avg_sub_attempts", "sub_per_min",
    "avg_kd", "avg_kds_received", "kd_per_fight", "kd_received_per_fight",
    "recent_form_3", "recent_form_5",
]

SHRINKAGE_K = 3.0  # pseudo-fights toward the division mean (was 5.0; reduced
                   # 2026-04-30 to give established veterans more weight on
                   # their own stats. Veteran (30+ fights) shrink: 30/(30+3)
                   # = 91% own-stat (was 86%). Rookie (5 fights) shrink:
                   # 5/(5+3) = 63% own-stat (was 50%). Both directions
                   # tilt slightly toward "trust the fighter's record" —
                   # which is what we want when betting tail predictions.


def _compute_division_means(df: pd.DataFrame) -> dict:
    """{weight_class: {stat: mean}} — pool a_<stat> and b_<stat> across all rows."""
    out = {}
    for wc, group in df.groupby("weight_class"):
        out[wc] = {}
        for stat in SHRINKABLE_STATS:
            a_col, b_col = f"a_{stat}", f"b_{stat}"
            if a_col in group.columns and b_col in group.columns:
                combined = pd.concat([group[a_col], group[b_col]], ignore_index=True)
                out[wc][stat] = float(combined.astype(float).mean())
    # Global fallback
    out["__default__"] = {}
    for stat in SHRINKABLE_STATS:
        a_col, b_col = f"a_{stat}", f"b_{stat}"
        if a_col in df.columns and b_col in df.columns:
            combined = pd.concat([df[a_col], df[b_col]], ignore_index=True)
            out["__default__"][stat] = float(combined.astype(float).mean())
    return out


def _apply_shrinkage(df: pd.DataFrame, div_means: dict, k: float = SHRINKAGE_K) -> pd.DataFrame:
    """Shrink per-fighter rate/avg stats toward their division mean by career_fights."""
    df = df.copy()
    default_means = div_means.get("__default__", {})
    wc_map = df["weight_class"].map(lambda wc: div_means.get(wc, default_means))

    n_a = df["a_career_fights"].fillna(0).astype(float)
    n_b = df["b_career_fights"].fillna(0).astype(float)
    w_a = n_a / (n_a + k)
    w_b = n_b / (n_b + k)

    for stat in SHRINKABLE_STATS:
        a_col, b_col, d_col = f"a_{stat}", f"b_{stat}", f"delta_{stat}"
        if a_col not in df.columns or b_col not in df.columns:
            continue
        means = wc_map.map(lambda m: m.get(stat, 0.0))
        a_shrunk = w_a * df[a_col].astype(float).fillna(0) + (1 - w_a) * means
        b_shrunk = w_b * df[b_col].astype(float).fillna(0) + (1 - w_b) * means
        df[a_col] = a_shrunk
        df[b_col] = b_shrunk
        if d_col in df.columns:
            df[d_col] = a_shrunk - b_shrunk
    return df


def shrink_profile(profile: dict, weight_class: str, div_means: dict,
                   k: float = SHRINKAGE_K) -> dict:
    """Apply Bayesian shrinkage to a single live fighter profile."""
    out = dict(profile)
    n = float(out.get("career_fights", 0))
    w = n / (n + k)
    means = div_means.get(weight_class, div_means.get("__default__", {}))
    for stat in SHRINKABLE_STATS:
        if stat in out and stat in means:
            out[stat] = w * float(out[stat]) + (1 - w) * means[stat]
    return out


# Interaction features — useful for METHOD but noisy for WINNER
INTERACTION_COLS = {
    'td_threat_vs_b', 'td_threat_vs_a', 'strike_threat_vs_b', 'strike_threat_vs_a',
    'sub_danger_vs_b', 'sub_danger_vs_a', 'ko_danger_vs_b', 'ko_danger_vs_a',
    'elite_grappler_threat', 'elite_striker_threat',
    'grapple_ratio', 'style_clash',
    'delta_td_threat', 'delta_strike_threat', 'delta_sub_danger', 'delta_ko_danger',
}


def _winner_feature_columns(df: pd.DataFrame) -> list[str]:
    """Core features only — no interaction features (less noise for winner model)."""
    all_cols = _feature_columns(df)
    return [c for c in all_cols if not any(ic in c for ic in INTERACTION_COLS)]


def _method_feature_columns(df: pd.DataFrame) -> list[str]:
    """All features including interactions (method model needs them)."""
    return _feature_columns(df)


# ------------------------------------------------------------------
# Training pipeline
# ------------------------------------------------------------------

def load_and_prepare() -> tuple[pd.DataFrame, list[str], dict, dict]:
    """Load enriched CSV, filter, sort, apply wc features + shrinkage.

    Returns (df, feat_cols, div_means, wc_snapshot). The latter two are saved
    as model artifacts so predict() can replicate the same transforms on
    live fighter profiles.
    """
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Filter out fights where BOTH fighters have 0 career_fights
    df = df[~((df["a_career_fights"] == 0) & (df["b_career_fights"] == 0))].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Weight-class base-rate features (causal, expanding per division)
    df = _add_wc_features(df)
    wc_snap = _wc_snapshot(df)

    # Bayesian shrinkage toward division means (applied to a_*, b_*, delta_*)
    div_means = _compute_division_means(df)
    df = _apply_shrinkage(df, div_means)

    feat_cols = _feature_columns(df)
    return df, feat_cols, div_means, wc_snap


def _prune_features(X_train, y_train, all_cols, k: int, objective: str = "binary") -> list[str]:
    """Fit one XGB on all features, return the top-k feature names by importance.

    FORCE_KEEP_FEATURES are always retained even if their first-pass importance
    falls below the cutoff — they capture interactions the probe XGB undervalues
    (multiplicative features look weak on first pass when individual components
    are already informative).
    """
    kwargs = dict(XGB_PARAMS, eval_metric="logloss", use_label_encoder=False)
    if objective == "multiclass":
        kwargs["eval_metric"] = "mlogloss"
    probe = XGBClassifier(**kwargs)
    probe.fit(X_train[all_cols], y_train)
    ranked = sorted(zip(all_cols, probe.feature_importances_), key=lambda x: -x[1])
    top = [n for n, _ in ranked[:k]]
    # Force-keep interaction features that exist in the data
    for forced in FORCE_KEEP_FEATURES:
        if forced in all_cols and forced not in top:
            top.append(forced)
    return top


def train_all():
    """Train winner, method, and round models. Print metrics and save."""
    print("=" * 70)
    print("  UFC ML FIGHT PREDICTOR  —  Training Pipeline")
    print("=" * 70)

    # --- Load ---
    print("\n  Loading enriched_fights.csv ...")
    df, feat_cols, div_means, wc_snap = load_and_prepare()
    print(f"  Rows after filtering: {len(df):,}")
    print(f"  Feature columns:      {len(feat_cols)} (will prune to top {TOP_K_FEATURES})")
    print(f"  Weight-class features added: {', '.join(WC_FEATURES)}")
    print(f"  Bayesian shrinkage applied to {len(SHRINKABLE_STATS)} per-fighter stats (k={SHRINKAGE_K})")

    # --- Prepare targets ---
    X_all = df[feat_cols].fillna(0).astype(np.float32)
    y_winner = df["a_wins"].astype(int)

    # Normalize method labels
    def norm_method(m):
        m = str(m).upper()
        if "KO" in m or "TKO" in m:
            return "KO/TKO"
        if "SUB" in m:
            return "SUB"
        if "DEC" in m:
            return "DEC"
        return "OTHER"

    df["method_clean"] = df["method"].apply(norm_method)

    # Two-stage method decomposition (replaces multiclass method model):
    #   Stage 1: P(finish)           — binary, does the fight end before the final bell?
    #   Stage 2: P(KO | finish)      — binary, conditional on a finish
    # Final probs: DEC=1-P(f), KO=P(f)*P(KO|f), SUB=P(f)*(1-P(KO|f)).
    # This beats raw multiclass because "finish vs distance" and "KO vs SUB"
    # are driven by different features (power/chin vs grappling depth).
    y_finish = (df["method_clean"] != "DEC").astype(int)
    # KO-given-finish is only defined on finishes; we mask later.
    y_ko_given_finish = (df["method_clean"] == "KO/TKO").astype(int)

    # Round-based props (cascade + dedicated binary for R1 finishes).
    rounds_int = df["round_finished"].fillna(3).astype(int)
    y_reaches_r2 = (rounds_int >= 2).astype(int)
    y_reaches_r3 = (rounds_int >= 3).astype(int)
    # "Ends in Round 1" = finished in R1 (any method except DEC).
    y_ends_r1 = ((rounds_int == 1) & (df["method_clean"] != "DEC")).astype(int)

    # --- Time-based 70/10/20 split: train / calibration / test ---
    n = len(df)
    i_train = int(n * 0.70)
    i_calib = int(n * 0.80)
    X_train, X_calib, X_test = X_all.iloc[:i_train], X_all.iloc[i_train:i_calib], X_all.iloc[i_calib:]
    yw_train = y_winner.iloc[:i_train]
    yw_calib = y_winner.iloc[i_train:i_calib]
    yw_test = y_winner.iloc[i_calib:]
    yf_train = y_finish.iloc[:i_train]
    yf_calib = y_finish.iloc[i_train:i_calib]
    yf_test = y_finish.iloc[i_calib:]
    yko_train = y_ko_given_finish.iloc[:i_train]
    yko_calib = y_ko_given_finish.iloc[i_train:i_calib]
    yko_test = y_ko_given_finish.iloc[i_calib:]
    yr2_train = y_reaches_r2.iloc[:i_train]
    yr2_calib = y_reaches_r2.iloc[i_train:i_calib]
    yr2_test = y_reaches_r2.iloc[i_calib:]
    yr3_train = y_reaches_r3.iloc[:i_train]
    yr3_calib = y_reaches_r3.iloc[i_train:i_calib]
    yr3_test = y_reaches_r3.iloc[i_calib:]
    yr1end_train = y_ends_r1.iloc[:i_train]
    yr1end_calib = y_ends_r1.iloc[i_train:i_calib]
    yr1end_test = y_ends_r1.iloc[i_calib:]

    print(f"\n  Train set: {len(X_train):,} fights")
    print(f"  Calib set: {len(X_calib):,} fights (used for isotonic calibration)")
    print(f"  Test set:  {len(X_test):,} fights (most recent 20%)")

    # ---- Feature pruning: one probe fit per target, keep top-K by importance ----
    # Each target gets its own specialized feature subset — "finish vs distance"
    # is driven by different signals than "KO vs SUB given finish".
    print(f"\n  Pruning to top {TOP_K_FEATURES} features per target...")
    winner_feat_cols = _prune_features(X_train, yw_train, feat_cols, TOP_K_FEATURES, "binary")
    finish_feat_cols = _prune_features(X_train, yf_train, feat_cols, TOP_K_FEATURES, "binary")
    # ko_vs_sub is only defined on finishes — prune using those rows only.
    finish_mask_train = yf_train.astype(bool)
    ko_feat_cols = _prune_features(
        X_train[finish_mask_train], yko_train[finish_mask_train], feat_cols, TOP_K_FEATURES, "binary"
    )
    r2_feat_cols = _prune_features(X_train, yr2_train, feat_cols, TOP_K_FEATURES, "binary")
    r3_feat_cols = _prune_features(X_train, yr3_train, feat_cols, TOP_K_FEATURES, "binary")
    r1end_feat_cols = _prune_features(X_train, yr1end_train, feat_cols, TOP_K_FEATURES, "binary")
    print(f"    winner kept {len(winner_feat_cols)} | finish {len(finish_feat_cols)} | "
          f"ko_vs_sub {len(ko_feat_cols)} | r2 {len(r2_feat_cols)} | "
          f"r3 {len(r3_feat_cols)} | ends_r1 {len(r1end_feat_cols)}")

    Xw_train, Xw_calib, Xw_test = X_train[winner_feat_cols], X_calib[winner_feat_cols], X_test[winner_feat_cols]

    # ---- 1. Winner model (binary) + isotonic calibration ----
    print("\n  Training WINNER model...")
    winner_raw = XGBClassifier(
        **XGB_PARAMS,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    winner_raw.fit(Xw_train, yw_train)
    # Calibrate probabilities on held-out slice so 70% predictions actually
    # win ~70% of the time. Sigmoid (Platt) here too — 680 calib samples is
    # too small for isotonic to behave as a smooth function.
    winner_model = CalibratedClassifierCV(FrozenEstimator(winner_raw), method="sigmoid")
    winner_model.fit(Xw_calib, yw_calib)
    w_pred = winner_model.predict(Xw_test)
    w_acc = accuracy_score(yw_test, w_pred)
    w_brier_raw = brier_score_loss(yw_test, winner_raw.predict_proba(Xw_test)[:, 1])
    w_brier_cal = brier_score_loss(yw_test, winner_model.predict_proba(Xw_test)[:, 1])
    print(f"  Winner accuracy: {w_acc:.4f}  ({w_acc:.1%})")
    print(f"  Winner Brier score: raw={w_brier_raw:.4f}  calibrated={w_brier_cal:.4f} (lower is better)")

    def _train_calibrated_binary(name, feat_cols_use, y_tr, y_cal, y_te,
                                 mask_train=None, mask_calib=None, mask_test=None):
        """Fit XGB -> isotonic-calibrate -> report accuracy + Brier. Returns calibrated model."""
        Xtr = X_train[feat_cols_use]
        Xca = X_calib[feat_cols_use]
        Xte = X_test[feat_cols_use]
        ytr, yca, yte = y_tr, y_cal, y_te
        if mask_train is not None:
            Xtr, ytr = Xtr[mask_train], ytr[mask_train]
        if mask_calib is not None:
            Xca, yca = Xca[mask_calib], yca[mask_calib]
        if mask_test is not None:
            Xte, yte = Xte[mask_test], yte[mask_test]
        raw = XGBClassifier(**XGB_PARAMS, eval_metric="logloss", use_label_encoder=False)
        raw.fit(Xtr, ytr)
        # Platt scaling (sigmoid) instead of isotonic: our 680-fight calib slice
        # is too small for isotonic's non-parametric fit, which collapses to a
        # step function with ~30 distinct output values. Sigmoid is smooth.
        model = CalibratedClassifierCV(FrozenEstimator(raw), method="sigmoid")
        model.fit(Xca, yca)
        acc = accuracy_score(yte, model.predict(Xte))
        brier = brier_score_loss(yte, model.predict_proba(Xte)[:, 1])
        baseline = max(yte.mean(), 1 - yte.mean())
        print(f"  {name:<25} acc={acc:.1%}  brier={brier:.4f}  (baseline {baseline:.1%})")
        return model, raw

    # ---- 2a. FINISH model (Stage 1 of method decomposition) ----
    print("\n  Training METHOD DECOMPOSITION + PROPS...")
    finish_model, finish_raw = _train_calibrated_binary(
        "finish (vs distance)", finish_feat_cols, yf_train, yf_calib, yf_test
    )

    # ---- 2b. KO-given-finish model (Stage 2) ----
    # Train/calibrate/test only on rows where the fight actually finished.
    ko_model, ko_raw = _train_calibrated_binary(
        "ko_vs_sub (|finish)",
        ko_feat_cols, yko_train, yko_calib, yko_test,
        mask_train=yf_train.astype(bool),
        mask_calib=yf_calib.astype(bool),
        mask_test=yf_test.astype(bool),
    )

    # ---- Combined method accuracy via decomposition ----
    # P(DEC) = 1 - P(finish); P(KO) = P(finish) * P(KO|finish); P(SUB) = P(finish) * (1 - P(KO|finish)).
    Xf_test = X_test[finish_feat_cols]
    Xko_test = X_test[ko_feat_cols]
    p_finish_te = finish_model.predict_proba(Xf_test)[:, 1]
    p_ko_given_te = ko_model.predict_proba(Xko_test)[:, 1]
    method_probs_te = np.column_stack([
        1 - p_finish_te,                       # DEC
        p_finish_te * p_ko_given_te,           # KO/TKO
        p_finish_te * (1 - p_ko_given_te),     # SUB
    ])
    method_labels = np.array(["DEC", "KO/TKO", "SUB"])
    m_pred = method_labels[method_probs_te.argmax(axis=1)]
    m_true_raw = df["method_clean"].iloc[i_calib:].values
    m_true = np.where(np.isin(m_true_raw, method_labels), m_true_raw, "DEC")  # OTHER → DEC
    m_acc = (m_pred == m_true).mean()
    print(f"  Method accuracy (decomposed): {m_acc:.1%}  "
          f"(DEC-only baseline {(m_true=='DEC').mean():.1%})")
    # Per-class recall
    for cls in method_labels:
        mask = (m_true == cls)
        if mask.sum() > 0:
            recall = (m_pred[mask] == cls).mean()
            print(f"    {cls:<8} recall: {recall:.1%}  (n={mask.sum()})")

    # ---- 3. Reaches-R2 model (calibrated) ----
    reaches_r2_model, r2_raw = _train_calibrated_binary(
        "reaches_r2", r2_feat_cols, yr2_train, yr2_calib, yr2_test,
    )
    Xr2_test = X_test[r2_feat_cols]
    r2_proba = reaches_r2_model.predict_proba(Xr2_test)
    r2_acc = (reaches_r2_model.predict(Xr2_test) == yr2_test.values).mean()

    # ---- 4. Reaches-R3 model (calibrated; closes the round cascade) ----
    reaches_r3_model, r3_raw = _train_calibrated_binary(
        "reaches_r3", r3_feat_cols, yr3_train, yr3_calib, yr3_test,
    )

    # ---- 5. Ends-in-R1 model (dedicated prop for the "ends in R1" market) ----
    ends_r1_model, r1end_raw = _train_calibrated_binary(
        "ends_r1", r1end_feat_cols, yr1end_train, yr1end_calib, yr1end_test,
    )

    # ---- Cascade round accuracy on test set ----
    r2_p = r2_proba[:, 1]
    Xr3_test = X_test[r3_feat_cols]
    r3_p = reaches_r3_model.predict_proba(Xr3_test)[:, 1]
    round_pred_cascade = np.where(r2_p < 0.5, 1, np.where(r3_p < 0.5, 2, 3))
    round_true = rounds_int.iloc[i_calib:].clip(upper=3).values
    cascade_exact = (round_pred_cascade == round_true).mean()
    cascade_within1 = (np.abs(round_pred_cascade - round_true) <= 1).mean()
    print(f"  Cascade round accuracy (exact):    {cascade_exact:.1%}")
    print(f"  Cascade round accuracy (within 1): {cascade_within1:.1%}")

    # ---- Top 15 features (importances live on the raw booster, not the calibrator) ----
    print(f"\n  TOP 15 FEATURES (winner model):")
    for rank, (name, score) in enumerate(
        sorted(zip(winner_feat_cols, winner_raw.feature_importances_), key=lambda x: -x[1])[:15], 1
    ):
        print(f"    {rank:>2}. {name:<45} {score:.4f}")
    print(f"\n  TOP 15 FEATURES (finish model):")
    for rank, (name, score) in enumerate(
        sorted(zip(finish_feat_cols, finish_raw.feature_importances_), key=lambda x: -x[1])[:15], 1
    ):
        print(f"    {rank:>2}. {name:<45} {score:.4f}")
    print(f"\n  TOP 15 FEATURES (ko_vs_sub model):")
    for rank, (name, score) in enumerate(
        sorted(zip(ko_feat_cols, ko_raw.feature_importances_), key=lambda x: -x[1])[:15], 1
    ):
        print(f"    {rank:>2}. {name:<45} {score:.4f}")

    # ---- Save models ----
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "winner_model": winner_model,
        "finish_model": finish_model,
        "ko_vs_sub_model": ko_model,
        "reaches_r2_model": reaches_r2_model,
        "reaches_r3_model": reaches_r3_model,
        "ends_r1_model": ends_r1_model,
        "feature_columns": feat_cols,  # superset — used to discover base feature names
        "winner_feature_columns": winner_feat_cols,
        "finish_feature_columns": finish_feat_cols,
        "ko_feature_columns": ko_feat_cols,
        "r2_feature_columns": r2_feat_cols,
        "r3_feature_columns": r3_feat_cols,
        "r1end_feature_columns": r1end_feat_cols,
        "division_means": div_means,   # for Bayesian shrinkage at predict time
        "wc_snapshot": wc_snap,        # per-division wc_* rates for predict time
    }
    for name, obj in artifacts.items():
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    # Remove stale artifacts from prior architectures.
    for stale in ("round_model.pkl", "le_round.pkl", "method_model.pkl",
                  "le_method.pkl", "method_feature_columns.pkl"):
        stale_path = MODELS_DIR / stale
        if stale_path.exists():
            stale_path.unlink()

    print(f"\n  Models saved to {MODELS_DIR}/")
    print("  Done.\n")

    return artifacts


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def load_models() -> dict:
    """Load saved models from disk. Returns dict with same keys as train_all."""
    names = [
        "winner_model", "finish_model", "ko_vs_sub_model",
        "reaches_r2_model", "reaches_r3_model", "ends_r1_model",
        "feature_columns", "winner_feature_columns",
        "finish_feature_columns", "ko_feature_columns",
        "r2_feature_columns", "r3_feature_columns", "r1end_feature_columns",
        "division_means", "wc_snapshot",
    ]
    artifacts = {}
    for name in names:
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "rb") as f:
            artifacts[name] = pickle.load(f)
    return artifacts


def predict(features_a: dict, features_b: dict) -> dict:
    """
    Predict fight outcome from two dicts of fighter features.

    Parameters
    ----------
    features_a : dict
        Fighter A features. Keys should match the per-fighter column names
        in enriched_fights.csv WITHOUT the a_/b_ prefix.
        Example: {"career_fights": 15, "wins": 12, "elo": 1620, ...}
    features_b : dict
        Same format for Fighter B.

    Returns
    -------
    dict with keys:
        winner      – "A" or "B"
        win_prob    – float probability of predicted winner winning
        method      – str ("KO/TKO", "SUB", "DEC", "OTHER")
        method_probs – dict mapping each method to its probability
        round       – int predicted round
    """
    # CRITICAL: Training data always assigns A = higher Elo.
    # We must do the same at prediction time, or the features are flipped.
    elo_a = float(features_a.get('elo', 1500))
    elo_b = float(features_b.get('elo', 1500))
    if elo_b > elo_a:
        # Swap so A = higher Elo (matches training convention)
        features_a, features_b = features_b, features_a
        swapped = True
    else:
        swapped = False

    models = load_models()
    winner_model = models["winner_model"]
    finish_model = models["finish_model"]
    ko_model = models["ko_vs_sub_model"]
    reaches_r2_model = models["reaches_r2_model"]
    reaches_r3_model = models["reaches_r3_model"]
    ends_r1_model = models["ends_r1_model"]
    w_feat_cols = models["winner_feature_columns"]
    f_feat_cols = models["finish_feature_columns"]
    ko_feat_cols = models["ko_feature_columns"]
    r2_feat_cols = models["r2_feature_columns"]
    r3_feat_cols = models["r3_feature_columns"]
    r1end_feat_cols = models["r1end_feature_columns"]
    div_means = models.get("division_means", {"__default__": {}})
    wc_snap = models.get("wc_snapshot", {"__default__": {c: 0.0 for c in WC_FEATURES}})

    # Resolve weight class (needed for shrinkage prior + wc base-rate lookup).
    # Prefer A's value; fall back to B's; else "__default__".
    weight_class = (features_a.get("weight_class")
                    or features_b.get("weight_class")
                    or "__default__")

    # Apply Bayesian shrinkage to each fighter's per-fight averages, using
    # the same division means learned during training.
    features_a = shrink_profile(features_a, weight_class, div_means)
    features_b = shrink_profile(features_b, weight_class, div_means)

    # Collect ALL base feature names across every model's pruned column set
    all_feat_cols = sorted(
        set(w_feat_cols) | set(f_feat_cols) | set(ko_feat_cols)
        | set(r2_feat_cols) | set(r3_feat_cols) | set(r1end_feat_cols)
    )

    # Build a single-row DataFrame with all feature columns
    row = {}
    base_names = set()
    for c in all_feat_cols:
        for prefix in ("a_", "b_", "delta_"):
            if c.startswith(prefix):
                base_names.add(c[len(prefix):])
                break
        # wc_* features are matchup-level (same for both fighters); look them up
        # from the snapshot saved at training time.
        if c.startswith("wc_"):
            wc_rates = wc_snap.get(weight_class, wc_snap.get("__default__", {}))
            row[c] = float(wc_rates.get(c, 0.0))

    for base in base_names:
        a_val = float(features_a.get(base, 0))
        b_val = float(features_b.get(base, 0))
        row[f"a_{base}"] = a_val
        row[f"b_{base}"] = b_val
        row[f"delta_{base}"] = a_val - b_val

    # Also add interaction features that aren't simple a/b/delta
    # These need both fighters' stats together
    a_total = float(features_a.get('avg_sig_str_landed', 0)) + float(features_a.get('avg_td_landed', 0)) + float(features_a.get('avg_sub_attempts', 0))
    b_total = float(features_b.get('avg_sig_str_landed', 0)) + float(features_b.get('avg_td_landed', 0)) + float(features_b.get('avg_sub_attempts', 0))
    a_gr = (float(features_a.get('avg_td_landed', 0)) + float(features_a.get('avg_sub_attempts', 0))) / max(a_total, 1)
    b_gr = (float(features_b.get('avg_td_landed', 0)) + float(features_b.get('avg_sub_attempts', 0))) / max(b_total, 1)
    row['a_grapple_ratio'] = a_gr
    row['b_grapple_ratio'] = b_gr
    row['delta_grapple_ratio'] = a_gr - b_gr

    row['a_td_threat_vs_b'] = float(features_a.get('td_per_min', 0)) * (1 - float(features_b.get('avg_td_defense', 0.5)))
    row['b_td_threat_vs_a'] = float(features_b.get('td_per_min', 0)) * (1 - float(features_a.get('avg_td_defense', 0.5)))
    row['delta_td_threat'] = row['a_td_threat_vs_b'] - row['b_td_threat_vs_a']

    row['a_strike_threat_vs_b'] = float(features_a.get('sig_str_per_min', 0)) * (1 - float(features_b.get('avg_sig_str_defense', 0.5)))
    row['b_strike_threat_vs_a'] = float(features_b.get('sig_str_per_min', 0)) * (1 - float(features_a.get('avg_sig_str_defense', 0.5)))
    row['delta_strike_threat'] = row['a_strike_threat_vs_b'] - row['b_strike_threat_vs_a']

    row['a_sub_danger_vs_b'] = float(features_a.get('sub_per_min', 0)) * float(features_b.get('been_finished_rate', 0))
    row['b_sub_danger_vs_a'] = float(features_b.get('sub_per_min', 0)) * float(features_a.get('been_finished_rate', 0))
    row['delta_sub_danger'] = row['a_sub_danger_vs_b'] - row['b_sub_danger_vs_a']

    row['a_ko_danger_vs_b'] = float(features_a.get('kd_per_fight', 0)) * float(features_b.get('ko_loss_rate', 0))
    row['b_ko_danger_vs_a'] = float(features_b.get('kd_per_fight', 0)) * float(features_a.get('ko_loss_rate', 0))
    row['delta_ko_danger'] = row['a_ko_danger_vs_b'] - row['b_ko_danger_vs_a']

    a_elo = float(features_a.get('elo', 1500))
    b_elo = float(features_b.get('elo', 1500))
    row['a_elite_grappler_threat'] = max(0, a_elo - 1500) * a_gr * (1 - float(features_b.get('avg_td_defense', 0.5)))
    row['b_elite_grappler_threat'] = max(0, b_elo - 1500) * b_gr * (1 - float(features_a.get('avg_td_defense', 0.5)))
    row['a_elite_striker_threat'] = max(0, a_elo - 1500) * (1 - a_gr) * float(features_b.get('ko_loss_rate', 0))
    row['b_elite_striker_threat'] = max(0, b_elo - 1500) * (1 - b_gr) * float(features_a.get('ko_loss_rate', 0))
    row['style_clash'] = abs(a_gr - b_gr)

    # R1-chaos interaction features — must mirror build_training_data.py.
    # Multiplicative scalars so XGBoost sees joint signal as one feature
    # instead of having to discover the interaction across two columns.
    a_r1 = float(features_a.get('r1_ending_rate', 0.0))
    b_r1 = float(features_b.get('r1_ending_rate', 0.0))
    a_r1_5 = float(features_a.get('r1_ending_rate_5', 0.0))
    b_r1_5 = float(features_b.get('r1_ending_rate_5', 0.0))
    a_dec = float(features_a.get('decision_rate', 0.0))
    b_dec = float(features_b.get('decision_rate', 0.0))
    a_avg_sec = float(features_a.get('avg_fight_seconds', 600.0))
    b_avg_sec = float(features_b.get('avg_fight_seconds', 600.0))
    row['r1_chaos_score']         = a_r1 * b_r1
    row['r1_chaos_score_recent']  = a_r1_5 * b_r1_5
    row['both_r1_specialists']    = 1.0 if (a_r1 >= 0.40 and b_r1 >= 0.40) else 0.0
    row['min_r1_ending_rate']     = min(a_r1, b_r1)
    row['both_grinders_score']    = a_dec * b_dec
    row['combined_avg_fight_secs'] = (a_avg_sec + b_avg_sec) / 2.0
    row['joint_finish_propensity'] = 1.0 - max(a_dec, b_dec)
    # Single-side specialist features (added 2026-04-30) — must mirror
    # build_training_data.py exactly for predict-time alignment.
    a_ko_5  = float(features_a.get('ko_win_rate_5', 0.0))
    b_ko_5  = float(features_b.get('ko_win_rate_5', 0.0))
    a_sub_5 = float(features_a.get('sub_win_rate_5', 0.0))
    b_sub_5 = float(features_b.get('sub_win_rate_5', 0.0))
    row['max_r1_ending_rate']     = max(a_r1, b_r1)
    row['max_r1_ending_rate_5']   = max(a_r1_5, b_r1_5)
    row['max_ko_win_rate_5']      = max(a_ko_5, b_ko_5)
    row['max_sub_win_rate_5']     = max(a_sub_5, b_sub_5)
    row['single_side_r1_specialist'] = (
        1.0 if (max(a_r1, b_r1) >= 0.70 and min(a_dec, b_dec) <= 0.50) else 0.0
    )

    # Build separate DataFrames for each model — only include expected columns
    full_df = pd.DataFrame([row])
    def _slice(cols):
        return full_df.reindex(columns=cols, fill_value=0).astype(np.float32)[cols]
    X_winner = _slice(w_feat_cols)
    X_finish = _slice(f_feat_cols)
    X_ko = _slice(ko_feat_cols)
    X_r2 = _slice(r2_feat_cols)
    X_r3 = _slice(r3_feat_cols)
    X_r1end = _slice(r1end_feat_cols)

    # Helper: extract raw (pre-calibration) probability from a
    # CalibratedClassifierCV. Used to surface what the model would say
    # without isotonic/sigmoid squashing — diagnostic only.
    def _raw_proba_pos(calibrated_model, X):
        """Return P(class=1) from the raw underlying classifier."""
        try:
            inner = calibrated_model.calibrated_classifiers_[0].estimator
            if hasattr(inner, "estimator"):
                inner = inner.estimator
            p = inner.predict_proba(X)[0]
            return float(p[1]) if len(p) > 1 else float(p[0])
        except Exception:
            return None

    # Winner — blended with Elo prior.
    # The raw XGBoost model overfits across 30+ features; anchoring it to the
    # Elo-implied probability recovers the "higher Elo usually wins" baseline.
    w_proba = winner_model.predict_proba(X_winner)[0]
    p_model_a = float(w_proba[1]) if len(w_proba) > 1 else float(w_proba[0])
    # Diagnostic raw probabilities (pre-calibration) for tail analysis
    raw_p_winner_a = _raw_proba_pos(winner_model, X_winner)
    p_elo_a = 1.0 / (1.0 + 10 ** ((b_elo - a_elo) / 400.0))
    a_win_prob = ELO_BLEND_WEIGHT * p_elo_a + (1.0 - ELO_BLEND_WEIGHT) * p_model_a
    winner = "A" if a_win_prob >= 0.5 else "B"
    win_prob = a_win_prob if winner == "A" else 1.0 - a_win_prob

    # Method — two-stage decomposition: P(finish) then P(KO | finish).
    # DEC = 1 - P(finish); KO = P(finish) * P(KO|finish); SUB = P(finish) * (1 - P(KO|finish)).
    p_finish = float(finish_model.predict_proba(X_finish)[0, 1])
    p_ko_given_finish = float(ko_model.predict_proba(X_ko)[0, 1])
    raw_p_finish = _raw_proba_pos(finish_model, X_finish)
    raw_p_ko_given_finish = _raw_proba_pos(ko_model, X_ko)
    method_probs = {
        "DEC": 1.0 - p_finish,
        "KO/TKO": p_finish * p_ko_given_finish,
        "SUB": p_finish * (1.0 - p_ko_given_finish),
    }
    method_str = max(method_probs, key=method_probs.get)

    # Round — cascade of two binary models + dedicated ends_r1 prop.
    reaches_r2_prob = float(reaches_r2_model.predict_proba(X_r2)[0, 1])
    reaches_r3_prob = float(reaches_r3_model.predict_proba(X_r3)[0, 1])
    ends_r1_prob = float(ends_r1_model.predict_proba(X_r1end)[0, 1])
    raw_reaches_r2 = _raw_proba_pos(reaches_r2_model, X_r2)
    raw_reaches_r3 = _raw_proba_pos(reaches_r3_model, X_r3)
    raw_ends_r1 = _raw_proba_pos(ends_r1_model, X_r1end)
    if reaches_r2_prob < 0.5:
        round_num = 1
    elif reaches_r3_prob < 0.5:
        round_num = 2
    else:
        round_num = 3
    reaches_r2 = reaches_r2_prob >= 0.5

    # NOTE: rules-override for WINNER picks was tested 2026-04-27 and rejected
    # (subset comparison fallacy). But METHOD/ROUND overrides are warranted:
    # ML calibration squashes tail predictions toward the ~28% R1 base rate
    # even when raw patterns scream "this ends fast." Tested on 6 obvious
    # fights — flags beat ML on method/round 5/6, ML alone got method right 1/6.
    overrides_applied = []
    w_feat = features_a if winner == "A" else features_b
    l_feat = features_b if winner == "A" else features_a

    w_r1_5 = float(w_feat.get('r1_ending_rate_5', w_feat.get('r1_ending_rate', 0)))
    l_r1_5 = float(l_feat.get('r1_ending_rate_5', l_feat.get('r1_ending_rate', 0)))
    w_dec_5 = float(w_feat.get('decision_rate_5', w_feat.get('decision_rate', 0)))
    l_dec_5 = float(l_feat.get('decision_rate_5', l_feat.get('decision_rate', 0)))
    w_sub_5 = float(w_feat.get('sub_win_rate_5', w_feat.get('sub_win_rate', 0)))
    w_ko_5 = float(w_feat.get('ko_win_rate_5', w_feat.get('ko_win_rate', 0)))
    # Career fallbacks too (capture fighters whose recent diverges, like Moicano)
    w_sub_c = float(w_feat.get('sub_win_rate', 0))
    w_ko_c = float(w_feat.get('ko_win_rate', 0))

    grinder_match = (w_dec_5 * l_dec_5 >= 0.30 and min(w_dec_5, l_dec_5) >= 0.50)
    # R1 trigger requires BOTH fighters R1-prone. Single-side trigger
    # (winner_r1_5 ≥ 0.70) was tested and rejected — fired on 68 extra
    # fights with -8.9pp method (forces finish on fights where the non-
    # specialist opponent neutralizes the pace and wins by DEC).
    # Cost: lose Buzukja/Barbosa-style single-side specialist matchups.
    r1_specialist = (min(w_r1_5, l_r1_5) >= 0.50 and (w_r1_5 * l_r1_5) >= 0.35)

    # Winner-side decision specialist — backtest-validated +3.9pp method,
    # +6.3pp round on 127 fights. Captures Page-style fighters whose last 5
    # are all decisions even when opponent is a finisher.
    winner_dec_specialist = (w_dec_5 >= 0.75)

    # ── ROUND OVERRIDE ──
    if grinder_match:
        round_num = 3
        overrides_applied.append("ROUND→3 (grinder)")
    elif winner_dec_specialist and not r1_specialist:
        # Winner is an extreme decision-fighter and there's no R1 evidence
        # to override (e.g. Page: 100% recent dec). Force DEC R3 — the late
        # round fits both 3-round and 5-round (5-round handled at caller).
        round_num = 3
        overrides_applied.append(f"ROUND→3 (winner dec-spec {w_dec_5:.0%})")
    elif r1_specialist:
        round_num = 1
        ends_r1_prob = max(ends_r1_prob, 0.55)
        reaches_r2_prob = min(reaches_r2_prob, 0.45)
        overrides_applied.append("ROUND→1 (both R1 specialists)")

    # ── METHOD OVERRIDE ──
    # Only narrow, backtest-supportable overrides — not the broad "winner's
    # career style → method" (rejected 2026-04-28 as -0.9pp aggregate).
    # The conditions below all require ML to be DEFAULTING to DEC, which is
    # the calibration squash zone where pattern overrides have most lift.
    if grinder_match:
        method_str = "DEC"
        method_probs = {"DEC": 0.60, "KO/TKO": 0.20, "SUB": 0.20}
        overrides_applied.append("METHOD→DEC")
    elif winner_dec_specialist and not r1_specialist:
        method_str = "DEC"
        method_probs = {"DEC": 0.55, "KO/TKO": 0.25, "SUB": 0.20}
        overrides_applied.append("METHOD→DEC (winner dec-spec)")
    elif r1_specialist and method_str == "DEC":
        # R1 fight but ML defaulted to DEC — pick whichever finish ML leaned toward
        method_str = "KO/TKO" if method_probs["KO/TKO"] >= method_probs["SUB"] else "SUB"
        method_probs = {method_str: 0.45, ("SUB" if method_str == "KO/TKO" else "KO/TKO"): 0.30, "DEC": 0.25}
        overrides_applied.append(f"METHOD→{method_str} (R1 fight)")
    # Career sub-spec fallback (Moicano case) was tested and rejected —
    # fired on too many "no-override" fights and dropped that bucket's
    # method accuracy by -0.8pp. Surfaced via SUB SPECIALIST flag instead.

    pattern_override = " · ".join(overrides_applied) if overrides_applied else None

    # Un-swap: if we swapped A/B to match training convention,
    # flip the winner label back to caller's original A/B
    if swapped:
        winner = "B" if winner == "A" else "A"

    # Raw (uncalibrated) probabilities — diagnostic. Calibration squashes
    # tail predictions toward the population base rate. Comparing raw vs
    # calibrated reveals what the model "would have said" without
    # aggregate guardrails. Useful for spotting tail bets where the
    # calibrated number is misleading. Un-swap A/B if needed.
    if raw_p_winner_a is not None:
        raw_a_win_prob = raw_p_winner_a
    else:
        raw_a_win_prob = None
    raw_winner = "A" if (raw_a_win_prob or 0.5) >= 0.5 else "B"
    raw_win_prob = (raw_a_win_prob if raw_a_win_prob is not None else 0.5)
    if raw_winner == "B":
        raw_win_prob = 1.0 - raw_win_prob
    raw_method_probs = None
    if raw_p_finish is not None and raw_p_ko_given_finish is not None:
        raw_method_probs = {
            "DEC":    1.0 - raw_p_finish,
            "KO/TKO": raw_p_finish * raw_p_ko_given_finish,
            "SUB":    raw_p_finish * (1.0 - raw_p_ko_given_finish),
        }
    if swapped:
        raw_winner = "B" if raw_winner == "A" else "A"

    return {
        "winner": winner,
        "win_prob": round(win_prob, 4),
        "method": method_str,
        "method_probs": {k: round(v, 4) for k, v in method_probs.items()},
        "round": round_num,
        "reaches_r2": reaches_r2,
        "reaches_r2_prob": round(reaches_r2_prob, 4),
        "reaches_r3_prob": round(reaches_r3_prob, 4),
        "goes_distance_prob": round(1.0 - p_finish, 4),
        "ends_r1_prob": round(ends_r1_prob, 4),
        "pattern_override": pattern_override,   # None or "R1_SPECIALIST → ..." / "GRINDER → ..."
        # Raw (pre-calibration) outputs for tail-prediction diagnostic.
        # If calibration is squashing a strong signal, these will be more
        # extreme than the calibrated values.
        "raw_win_prob": round(raw_win_prob, 4) if raw_a_win_prob is not None else None,
        "raw_winner": raw_winner if raw_a_win_prob is not None else None,
        "raw_method_probs": ({k: round(v, 4) for k, v in raw_method_probs.items()}
                              if raw_method_probs else None),
        "raw_ends_r1_prob": round(raw_ends_r1, 4) if raw_ends_r1 is not None else None,
        "raw_reaches_r2_prob": round(raw_reaches_r2, 4) if raw_reaches_r2 is not None else None,
        "raw_reaches_r3_prob": round(raw_reaches_r3, 4) if raw_reaches_r3 is not None else None,
        "raw_goes_distance_prob": round(1.0 - raw_p_finish, 4) if raw_p_finish is not None else None,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    train_all()
