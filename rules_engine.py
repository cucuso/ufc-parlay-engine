"""
Rules-first prediction engine — interpretable baseline that mirrors how a human
handicapper reads fight history.

Each rule fires with a confidence and produces a Prediction. The cascade picks
the highest-confidence prediction; if no rule fires, falls back to "higher Elo
wins, decision in R3" as the structural default.

Trained ML is optional and only invoked when explicitly asked. The point of
this module is to measure how much of the signal is captured by transparent
rules alone.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ────────────────────────────────────────────────────────────────────
# Thresholds (calibrated against UFC base rates from training data:
# R1-ending rate ≈ 22% average, decision rate ≈ 51%, avg fight ≈ 700s)
# ────────────────────────────────────────────────────────────────────

R1_SPECIALIST_THRESHOLD       = 0.35    # career r1_ending_rate (vs 22% base)
GRINDER_DECISION_THRESHOLD    = 0.55    # decision_rate above 55% = grinder
GRINDER_R1_CEILING            = 0.20    # AND r1_ending_rate below 20%
LOPSIDED_ELO_GAP              = 250
SKID_FINISH_LOSS_THRESHOLD    = 0.66    # 2+ of last 3 ended in finish loss
DECLINING_VET_AGE             = 35
DECLINING_VET_STREAK          = -2      # 2+ losses in a row


@dataclass
class Prediction:
    winner: str                   # 'A' or 'B'
    method: str                   # 'KO/TKO' | 'SUB' | 'DEC'
    round: int                    # 1, 2, 3, or 5 (championship)
    win_prob: float
    rule: str                     # which rule fired (or 'fallback_elo')


# ────────────────────────────────────────────────────────────────────
# RULES — ordered by specificity. Most-specific patterns first.
# ────────────────────────────────────────────────────────────────────

def rule_both_r1_specialists(a: dict, b: dict) -> Optional[Prediction]:
    """Two fighters whose careers are dominated by R1 endings → R1 finish.
    Winner = whoever has the better KO win rate (offensive R1 specialist
    beats defensive R1 specialist who just gets caught)."""
    if (a["r1_ending_rate"] >= R1_SPECIALIST_THRESHOLD
            and b["r1_ending_rate"] >= R1_SPECIALIST_THRESHOLD):
        # Whoever has higher KO win rate is the offensive finisher
        if a["ko_win_rate"] >= b["ko_win_rate"]:
            winner, p_ko = "A", a["ko_win_rate"]
        else:
            winner, p_ko = "B", b["ko_win_rate"]
        # If sub rate dominates, predict SUB instead
        a_sub_dom = a["sub_win_rate"] > a["ko_win_rate"]
        b_sub_dom = b["sub_win_rate"] > b["ko_win_rate"]
        if (winner == "A" and a_sub_dom) or (winner == "B" and b_sub_dom):
            method = "SUB"
        else:
            method = "KO/TKO"
        return Prediction(
            winner=winner, method=method, round=1,
            win_prob=0.65,
            rule="both_r1_specialists",
        )
    return None


def rule_both_grinders(a: dict, b: dict) -> Optional[Prediction]:
    """Two grinders → DEC, higher Elo wins."""
    a_grinder = (a["decision_rate"] >= GRINDER_DECISION_THRESHOLD
                 and a["r1_ending_rate"] <= GRINDER_R1_CEILING)
    b_grinder = (b["decision_rate"] >= GRINDER_DECISION_THRESHOLD
                 and b["r1_ending_rate"] <= GRINDER_R1_CEILING)
    if a_grinder and b_grinder:
        winner = "A" if a["elo"] >= b["elo"] else "B"
        return Prediction(
            winner=winner, method="DEC", round=3,
            win_prob=0.65,
            rule="both_grinders",
        )
    return None


def rule_finish_loss_skid(a: dict, b: dict) -> Optional[Prediction]:
    """Fighter on a finish-loss skid → opponent wins by finish.
    'Skid' = 2+ of last 3 fights ended in finish loss."""
    if a["recent_3_finish_loss_rate"] >= SKID_FINISH_LOSS_THRESHOLD:
        # B wins; method based on B's finish style
        method = "KO/TKO" if b["ko_win_rate"] >= b["sub_win_rate"] else "SUB"
        return Prediction(
            winner="B", method=method, round=2,
            win_prob=0.62,
            rule="a_on_finish_skid",
        )
    if b["recent_3_finish_loss_rate"] >= SKID_FINISH_LOSS_THRESHOLD:
        method = "KO/TKO" if a["ko_win_rate"] >= a["sub_win_rate"] else "SUB"
        return Prediction(
            winner="A", method=method, round=2,
            win_prob=0.62,
            rule="b_on_finish_skid",
        )
    return None


def rule_declining_vet(a: dict, b: dict) -> Optional[Prediction]:
    """Old fighter on a losing streak vs younger opponent → opponent wins."""
    a_vet = (a["age"] >= DECLINING_VET_AGE
             and a["streak"] <= DECLINING_VET_STREAK
             and b["age"] < a["age"] - 2)
    b_vet = (b["age"] >= DECLINING_VET_AGE
             and b["streak"] <= DECLINING_VET_STREAK
             and a["age"] < b["age"] - 2)
    if a_vet and not b_vet:
        return Prediction(
            winner="B", method="DEC", round=3, win_prob=0.62,
            rule="a_declining_vet",
        )
    if b_vet and not a_vet:
        return Prediction(
            winner="A", method="DEC", round=3, win_prob=0.62,
            rule="b_declining_vet",
        )
    return None


def rule_lopsided_elo(a: dict, b: dict) -> Optional[Prediction]:
    """Big Elo gap → higher Elo wins. Method = whichever finish style they
    prefer (KO vs SUB) or DEC if they grind."""
    gap = abs(a["elo"] - b["elo"])
    if gap < LOPSIDED_ELO_GAP:
        return None
    fav = a if a["elo"] > b["elo"] else b
    winner = "A" if fav is a else "B"

    if fav["decision_rate"] >= GRINDER_DECISION_THRESHOLD:
        method, rd = "DEC", 3
    elif fav["sub_win_rate"] > fav["ko_win_rate"]:
        method, rd = "SUB", 2
    else:
        method, rd = "KO/TKO", 2

    # Confidence scales with Elo gap (250 → 0.62, 400 → 0.72, 500+ → 0.78)
    conf = min(0.78, 0.50 + gap / 1000)
    return Prediction(
        winner=winner, method=method, round=rd, win_prob=conf,
        rule="lopsided_elo",
    )


def rule_grinder_vs_finisher(a: dict, b: dict) -> Optional[Prediction]:
    """One R1 specialist vs one grinder. The R1 specialist either lands
    early or gets ground out — go with whoever has higher Elo."""
    a_finisher = a["r1_ending_rate"] >= R1_SPECIALIST_THRESHOLD
    b_finisher = b["r1_ending_rate"] >= R1_SPECIALIST_THRESHOLD
    a_grinder = (a["decision_rate"] >= GRINDER_DECISION_THRESHOLD
                 and a["r1_ending_rate"] <= GRINDER_R1_CEILING)
    b_grinder = (b["decision_rate"] >= GRINDER_DECISION_THRESHOLD
                 and b["r1_ending_rate"] <= GRINDER_R1_CEILING)

    if (a_finisher and b_grinder) or (b_finisher and a_grinder):
        # Higher Elo wins. If finisher wins → R1 finish. If grinder wins → DEC.
        winner = "A" if a["elo"] >= b["elo"] else "B"
        is_finisher_winning = (
            (winner == "A" and a_finisher) or (winner == "B" and b_finisher)
        )
        if is_finisher_winning:
            return Prediction(
                winner=winner, method="KO/TKO", round=1,
                win_prob=0.60, rule="finisher_beats_grinder",
            )
        else:
            return Prediction(
                winner=winner, method="DEC", round=3,
                win_prob=0.60, rule="grinder_beats_finisher",
            )
    return None


# ────────────────────────────────────────────────────────────────────
# CASCADE — try rules in order of specificity, return first match.
# Falls back to "higher Elo wins, DEC R3" if nothing fires.
# ────────────────────────────────────────────────────────────────────

RULE_ORDER = [
    rule_both_r1_specialists,
    rule_both_grinders,
    rule_finish_loss_skid,
    rule_declining_vet,
    rule_lopsided_elo,
    rule_grinder_vs_finisher,
]


def fallback_elo(a: dict, b: dict) -> Prediction:
    """Default when no rule fires: higher Elo wins, fight goes to decision."""
    winner = "A" if a["elo"] >= b["elo"] else "B"
    fav = a if winner == "A" else b
    # Pick method based on the favorite's career pattern
    if fav["decision_rate"] >= 0.50:
        method, rd = "DEC", 3
    elif fav["sub_win_rate"] > fav["ko_win_rate"]:
        method, rd = "SUB", 3
    else:
        method, rd = "KO/TKO", 3
    return Prediction(
        winner=winner, method=method, round=rd, win_prob=0.55,
        rule="fallback_elo",
    )


def predict_rules(a: dict, b: dict) -> Prediction:
    """Run the rule cascade. a, b are profile dicts (must contain all fields
    referenced by rules above)."""
    for rule_fn in RULE_ORDER:
        result = rule_fn(a, b)
        if result is not None:
            return result
    return fallback_elo(a, b)


# ────────────────────────────────────────────────────────────────────
# OVERRIDE WHITELIST — rules that empirically beat ML on the held-out
# test set (backtest_rules.py output, 1,446 fights):
#
#   Rule                     N    Winner acc   ML baseline   Lift
#   b_declining_vet         19      73.7%        62.4%      +11.3pp
#   b_on_finish_skid       182      67.6%        62.4%       +5.2pp
#   lopsided_elo           133      64.7%        62.4%       +2.3pp
#
# These rules override ML in `predict_hybrid()`. Rules outside this list
# either don't fire often enough or hit below ML — defer to ML for those.
# ────────────────────────────────────────────────────────────────────

WINNING_RULES = frozenset({
    "b_on_finish_skid",
    "b_declining_vet",
    "lopsided_elo",
})


def predict_with_override(a: dict, b: dict) -> Optional[Prediction]:
    """Run the rule cascade; return the rule's prediction ONLY if it's a
    member of WINNING_RULES. Otherwise return None (caller should fall
    back to ML)."""
    for rule_fn in RULE_ORDER:
        result = rule_fn(a, b)
        if result is not None and result.rule in WINNING_RULES:
            return result
        if result is not None:
            # A rule fired but it's not in the winning whitelist — stop
            # trying further rules and defer to ML. (Rule cascade is ordered
            # by specificity, so the first match is the most informative.)
            return None
    return None


# ────────────────────────────────────────────────────────────────────
# Adapter — convert a row from enriched_fights.csv (a_*/b_* columns) into
# the {a:..., b:...} dict shape that the rules expect.
# ────────────────────────────────────────────────────────────────────

PROFILE_KEYS = (
    "elo", "r1_ending_rate", "r1_ending_rate_5", "decision_rate",
    "avg_fight_seconds", "ko_win_rate", "sub_win_rate", "finish_rate",
    "been_finished_rate", "streak", "recent_form_3",
    "recent_3_finish_loss_rate", "recent_5_r1_loss_rate", "age",
)


def row_to_profiles(row) -> tuple[dict, dict]:
    a = {k: float(row.get(f"a_{k}", 0.0)) for k in PROFILE_KEYS}
    b = {k: float(row.get(f"b_{k}", 0.0)) for k in PROFILE_KEYS}
    return a, b


# ────────────────────────────────────────────────────────────────────
# CLI sanity check
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test: McVey/Dumas should fire 'both_r1_specialists'
    mcvey = {"elo": 1415, "r1_ending_rate": 0.89, "r1_ending_rate_5": 0.80,
             "decision_rate": 0.0, "avg_fight_seconds": 138,
             "ko_win_rate": 0.50, "sub_win_rate": 0.50, "finish_rate": 1.0,
             "been_finished_rate": 0.22, "streak": 1, "recent_form_3": 0.33,
             "recent_3_finish_loss_rate": 0.67, "recent_5_r1_loss_rate": 0.20,
             "age": 27}
    dumas = {"elo": 1420, "r1_ending_rate": 0.50, "r1_ending_rate_5": 0.60,
             "decision_rate": 0.25, "avg_fight_seconds": 395,
             "ko_win_rate": 0.40, "sub_win_rate": 0.20, "finish_rate": 0.60,
             "been_finished_rate": 0.31, "streak": -1, "recent_form_3": 0.17,
             "recent_3_finish_loss_rate": 0.67, "recent_5_r1_loss_rate": 0.40,
             "age": 30}
    pred = predict_rules(mcvey, dumas)
    print(f"McVey vs Dumas → {pred.rule}")
    print(f"  Winner: {pred.winner}  Method: {pred.method}  Round: {pred.round}  "
          f"Conf: {pred.win_prob:.0%}")
