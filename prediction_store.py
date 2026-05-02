"""
Generic persistence + scoring for UFC fight predictions.

Predict scripts call save() to dump a JSON snapshot of their output to
predictions/<slug>_<event_date>.json. Later, score() compares a stored
snapshot against a dict of actual results to produce a hit/miss table.

This decouples prediction records from the live model, Elo, and training
data — all of which drift over time and silently rewrite history if you
only re-run the script after the fact.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).parent
PREDICTIONS_DIR = BASE_DIR / "predictions"

METHOD_CLASSES = ("KO/TKO", "SUB", "DEC", "DRAW", "NC")


def _slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _fight_key(fighter_a: str, fighter_b: str) -> str:
    return "|".join(sorted([fighter_a.strip(), fighter_b.strip()]))


def save(
    event_name: str,
    event_date: str,
    fights: Iterable[dict],
    *,
    notes: str | None = None,
) -> Path:
    """Write a prediction snapshot to predictions/<slug>_<event_date>.json.

    Each fight dict should contain at minimum:
        fighter_a, fighter_b, winner, win_prob, method, round
    Any additional keys (elo, age, method_probs, round_probs, etc.) are
    preserved verbatim.
    """
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    slug = _slug(event_name)
    path = PREDICTIONS_DIR / f"{slug}_{event_date}.json"

    normalized = []
    for f in fights:
        entry = dict(f)
        entry["fight_key"] = _fight_key(entry["fighter_a"], entry["fighter_b"])
        normalized.append(entry)

    payload = {
        "event": event_name,
        "event_date": event_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
        "fights": normalized,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def score(predictions: str | Path | dict, actuals: dict[str, dict]) -> dict:
    """Compare a predictions snapshot against actual results.

    actuals is keyed by fight_key ("fighter1|fighter2", sorted alphabetically
    — use the _fight_key helper or score_by_names() below) and each value is:
        {"winner": str, "method": "KO/TKO"|"SUB"|"DEC"|"DRAW"|"NC",
         "round": int | None}

    Returns a dict with rows (per-fight verdict) and aggregate counts.
    """
    snap = predictions if isinstance(predictions, dict) else load(predictions)

    rows = []
    winner_hits = method_hits = round_hits = graded = 0

    for f in snap["fights"]:
        key = f.get("fight_key") or _fight_key(f["fighter_a"], f["fighter_b"])
        actual = actuals.get(key)
        if actual is None:
            rows.append({**f, "_verdict": "NO_RESULT"})
            continue

        graded += 1
        winner_hit = f["winner"] == actual["winner"]
        method_hit = f["method"] == actual["method"]
        round_hit = (
            actual.get("round") is not None
            and f.get("round") == actual["round"]
        )

        winner_hits += int(winner_hit)
        method_hits += int(method_hit)
        round_hits += int(round_hit)

        rows.append({
            **f,
            "actual_winner": actual["winner"],
            "actual_method": actual["method"],
            "actual_round": actual.get("round"),
            "winner_hit": winner_hit,
            "method_hit": method_hit,
            "round_hit": round_hit,
        })

    return {
        "event": snap["event"],
        "event_date": snap["event_date"],
        "graded": graded,
        "winner_hits": winner_hits,
        "method_hits": method_hits,
        "round_hits": round_hits,
        "rows": rows,
    }


def score_by_names(
    predictions: str | Path | dict,
    actuals_by_pair: dict[tuple[str, str], dict],
) -> dict:
    """Convenience wrapper: accept {(a, b): {...}} with any name order."""
    keyed = {_fight_key(a, b): v for (a, b), v in actuals_by_pair.items()}
    return score(predictions, keyed)


def run_card_and_save(
    event_name: str,
    event_date: str,
    card: Iterable[tuple],
    *,
    notes: str | None = None,
    verbose: bool = True,
) -> tuple[list, Path]:
    """Run every fight in a card, print per-fight output, and persist a snapshot.

    card is an iterable of (fighter_a, fighter_b, weight_class, is_main_event).
    Returns (results, snapshot_path) where results is the list of
    (na, nb, wc, is_main, r) tuples — the same shape callers already use
    to build summary tables.
    """
    from profile_builder import build_live_profile
    from ml_model import predict

    results = []
    for na, nb, wc, is_main in card:
        if verbose:
            print(f"\n  {na} vs {nb} ({wc}){' *MAIN*' if is_main else ''}")
        pa = build_live_profile(na)
        pb = build_live_profile(nb)
        if not pa or not pb:
            missing = na if not pa else nb
            if verbose:
                print(f"    SKIPPED — could not build profile for {missing}")
            continue
        r = predict(pa, pb)
        # Main events go 5 rounds — if model predicts DEC, output R5 instead of R3.
        # The round cascade in ml_model only knows about 3-round structure;
        # championship/main-event rounds 4-5 are added here at the caller level.
        if is_main and r["method"] == "DEC" and r["round"] == 3:
            r["round"] = 5
        winner = na if r["winner"] == "A" else nb
        r["winner_name"] = winner
        r["loser_name"] = nb if winner == na else na
        r["fighter_a"] = na
        r["fighter_b"] = nb
        r["elo_a"] = pa["elo"]
        r["elo_b"] = pb["elo"]
        r["age_a"] = pa["age"]
        r["age_b"] = pb["age"]
        r["debut_a"] = pa.get("is_ufc_debutant", False)
        r["debut_b"] = pb.get("is_ufc_debutant", False)
        # Pace snapshot per fighter for extreme-matchup gate flag
        # Career rates kept for context; recent-5 rates are what the flags use
        # (capture current style, not lifetime average).
        r["r1_ending_rate_a"]    = pa.get("r1_ending_rate", 0)
        r["r1_ending_rate_b"]    = pb.get("r1_ending_rate", 0)
        r["r1_ending_rate_5_a"]  = pa.get("r1_ending_rate_5", 0)
        r["r1_ending_rate_5_b"]  = pb.get("r1_ending_rate_5", 0)
        r["decision_rate_a"]     = pa.get("decision_rate", 0)
        r["decision_rate_b"]     = pb.get("decision_rate", 0)
        r["decision_rate_5_a"]   = pa.get("decision_rate_5", 0)
        r["decision_rate_5_b"]   = pb.get("decision_rate_5", 0)
        r["avg_fight_seconds_a"] = pa.get("avg_fight_seconds", 600)
        r["avg_fight_seconds_b"] = pb.get("avg_fight_seconds", 600)
        r["sub_win_rate_a"]      = pa.get("sub_win_rate", 0)
        r["sub_win_rate_b"]      = pb.get("sub_win_rate", 0)
        r["sub_win_rate_5_a"]    = pa.get("sub_win_rate_5", 0)
        r["sub_win_rate_5_b"]    = pb.get("sub_win_rate_5", 0)
        r["ko_win_rate_a"]       = pa.get("ko_win_rate", 0)
        r["ko_win_rate_b"]       = pb.get("ko_win_rate", 0)
        r["ko_win_rate_5_a"]     = pa.get("ko_win_rate_5", 0)
        r["ko_win_rate_5_b"]     = pb.get("ko_win_rate_5", 0)
        r["finish_rate_a"]       = pa.get("finish_rate", 0)
        r["finish_rate_b"]       = pb.get("finish_rate", 0)
        # Fight-denominator pace stats (added 2026-04-30) — used by gate
        # flags instead of the win-conditional sub_win_rate_5 / ko_win_rate_5.
        r["recent_5_sub_per_fight_a"]   = pa.get("recent_5_sub_per_fight", 0)
        r["recent_5_sub_per_fight_b"]   = pb.get("recent_5_sub_per_fight", 0)
        r["recent_5_ko_per_fight_a"]    = pa.get("recent_5_ko_per_fight", 0)
        r["recent_5_ko_per_fight_b"]    = pb.get("recent_5_ko_per_fight", 0)
        r["recent_5_finish_per_fight_a"] = pa.get("recent_5_finish_per_fight", 0)
        r["recent_5_finish_per_fight_b"] = pb.get("recent_5_finish_per_fight", 0)
        r["recent_5_r1_finish_per_fight_a"] = pa.get("recent_5_r1_finish_per_fight", 0)
        r["recent_5_r1_finish_per_fight_b"] = pb.get("recent_5_r1_finish_per_fight", 0)
        r["wins_in_last_5_a"]    = pa.get("wins_in_last_5", 0)
        r["wins_in_last_5_b"]    = pb.get("wins_in_last_5", 0)
        # Staleness + method-history fields (raw counts, not just rates).
        # These let gate_picks() detect when (5)-rates are computed off
        # last-5 WINS that go back years for a fighter on a long losing
        # streak (Meerschaert case: 4 straight losses, "100% sub" was
        # from 2023 wins) — and detect when the model picks a fighter
        # by a method they've never finished by (Gorimbo by KO with 0
        # career KOs).
        r["streak_a"]            = pa.get("streak", 0)
        r["streak_b"]            = pb.get("streak", 0)
        r["recent_form_5_a"]     = pa.get("recent_form_5", 0.5)
        r["recent_form_5_b"]     = pb.get("recent_form_5", 0.5)
        r["ko_wins_a"]           = int(pa.get("ko_wins", 0))
        r["ko_wins_b"]           = int(pb.get("ko_wins", 0))
        r["sub_wins_a"]          = int(pa.get("sub_wins", 0))
        r["sub_wins_b"]          = int(pb.get("sub_wins", 0))
        r["dec_wins_a"]          = int(pa.get("dec_wins", 0))
        r["dec_wins_b"]          = int(pb.get("dec_wins", 0))
        r["career_wins_a"]       = int(pa.get("career_wins", 0))
        r["career_wins_b"]       = int(pb.get("career_wins", 0))
        if verbose:
            print(f"    {na}: Elo {r['elo_a']:.0f} age {r['age_a']:.0f}"
                  f" | {nb}: Elo {r['elo_b']:.0f} age {r['age_b']:.0f}")
            print(f"    → {r['winner_name']} @ {r['win_prob']:.0%}  "
                  f"{r['method']} R{r['round']}  "
                  f"(KO {r['method_probs']['KO/TKO']:.0%} / "
                  f"SUB {r['method_probs']['SUB']:.0%} / "
                  f"DEC {r['method_probs']['DEC']:.0%})")
            print(f"    Round: ends-R1 {r['ends_r1_prob']:.0%}  "
                  f"reaches-R2 {r['reaches_r2_prob']:.0%}  "
                  f"reaches-R3 {r['reaches_r3_prob']:.0%}  "
                  f"goes-dist {r['goes_distance_prob']:.0%}")
        results.append((na, nb, wc, is_main, r))

    payload = [_card_row_to_payload(na, nb, wc, is_main, r)
               for na, nb, wc, is_main, r in results]
    path = save(event_name, event_date, payload, notes=notes)
    if verbose:
        print(f"\n  Saved predictions snapshot → {path}")
    return results, path


def _card_row_to_payload(na, nb, wc, is_main, r) -> dict:
    return {
        "fighter_a": na,
        "fighter_b": nb,
        "weight_class": wc,
        "is_main_event": is_main,
        "elo_a": r["elo_a"],
        "elo_b": r["elo_b"],
        "age_a": r["age_a"],
        "age_b": r["age_b"],
        "debut_a": r["debut_a"],
        "debut_b": r["debut_b"],
        "winner": r["winner_name"],
        "loser": r["loser_name"],
        "win_prob": r["win_prob"],
        "method": r["method"],
        "round": r["round"],
        "method_probs": r["method_probs"],
        "round_probs": {
            "ends_r1": r["ends_r1_prob"],
            "reaches_r2": r["reaches_r2_prob"],
            "reaches_r3": r["reaches_r3_prob"],
            "goes_distance": r["goes_distance_prob"],
        },
        # Pace-signature snapshot per fighter — used by extreme-matchup flag in gates.
        # Both career and recent-5 versions stored. Flags use recent-5 (current
        # style) but career is shown for context.
        "pace_a": {
            "r1_ending_rate": r.get("r1_ending_rate_a", 0),
            "r1_ending_rate_5": r.get("r1_ending_rate_5_a", 0),
            "decision_rate": r.get("decision_rate_a", 0),
            "decision_rate_5": r.get("decision_rate_5_a", 0),
            "avg_fight_seconds": r.get("avg_fight_seconds_a", 0),
            "sub_win_rate": r.get("sub_win_rate_a", 0),
            "sub_win_rate_5": r.get("sub_win_rate_5_a", 0),
            "ko_win_rate": r.get("ko_win_rate_a", 0),
            "ko_win_rate_5": r.get("ko_win_rate_5_a", 0),
            "finish_rate": r.get("finish_rate_a", 0),
            # Staleness + method-history sanity check fields
            "streak": r.get("streak_a", 0),
            "recent_form_5": r.get("recent_form_5_a", 0.5),
            "ko_wins": r.get("ko_wins_a", 0),
            "sub_wins": r.get("sub_wins_a", 0),
            "dec_wins": r.get("dec_wins_a", 0),
            "career_wins": r.get("career_wins_a", 0),
            # Fight-denominator pace stats (use these for gate flags)
            "recent_5_sub_per_fight": r.get("recent_5_sub_per_fight_a", 0),
            "recent_5_ko_per_fight": r.get("recent_5_ko_per_fight_a", 0),
            "recent_5_finish_per_fight": r.get("recent_5_finish_per_fight_a", 0),
            "recent_5_r1_finish_per_fight": r.get("recent_5_r1_finish_per_fight_a", 0),
            "wins_in_last_5": r.get("wins_in_last_5_a", 0),
        },
        "pace_b": {
            "r1_ending_rate": r.get("r1_ending_rate_b", 0),
            "r1_ending_rate_5": r.get("r1_ending_rate_5_b", 0),
            "decision_rate": r.get("decision_rate_b", 0),
            "decision_rate_5": r.get("decision_rate_5_b", 0),
            "avg_fight_seconds": r.get("avg_fight_seconds_b", 0),
            "sub_win_rate": r.get("sub_win_rate_b", 0),
            "sub_win_rate_5": r.get("sub_win_rate_5_b", 0),
            "ko_win_rate": r.get("ko_win_rate_b", 0),
            "ko_win_rate_5": r.get("ko_win_rate_5_b", 0),
            "finish_rate": r.get("finish_rate_b", 0),
            "streak": r.get("streak_b", 0),
            "recent_form_5": r.get("recent_form_5_b", 0.5),
            "ko_wins": r.get("ko_wins_b", 0),
            "sub_wins": r.get("sub_wins_b", 0),
            "dec_wins": r.get("dec_wins_b", 0),
            "career_wins": r.get("career_wins_b", 0),
            "recent_5_sub_per_fight": r.get("recent_5_sub_per_fight_b", 0),
            "recent_5_ko_per_fight": r.get("recent_5_ko_per_fight_b", 0),
            "recent_5_finish_per_fight": r.get("recent_5_finish_per_fight_b", 0),
            "recent_5_r1_finish_per_fight": r.get("recent_5_r1_finish_per_fight_b", 0),
            "wins_in_last_5": r.get("wins_in_last_5_b", 0),
        },
    }


# ────────────────────────────────────────────────────────────────────
# BETTING GATES — derived from backtest_all.py calibration analysis.
# Recalibrated 2026-04-27 after pace-feature retrain (winner acc 60.3%
# → 62.4%, round±1 83.7% → 85.4%, goes-dist 55.2% → 56.4%).
# Rules encode where the model has demonstrable edge vs. where it's
# noise or miscalibrated.
# ────────────────────────────────────────────────────────────────────

# Conf bands on winner — post-retrain calibration table:
#   0.50-0.55  N=295  pred 52% / actual 55%  +2.6pp  (calibrated, no longer fade)
#   0.55-0.60  N=273  pred 58% / actual 54%  -3.7pp  (slight overconf)
#   0.60-0.65  N=237  pred 62% / actual 60%  -3.0pp  (slight overconf)
#   0.65-0.70  N=256  pred 67% / actual 71%  +4.1pp  (calibrated, edge zone)
#   0.70-0.75  N=179  pred 72% / actual 67%  -5.3pp  (overconf)
#   0.75-0.80  N=98   pred 77% / actual 81%  +3.4pp  (calibrated, edge zone)
#   0.80-0.90  N=21   pred 82% / actual 76%  -5.6pp  (overconf, small N)
# Sweet spots: 65-70 and 75-80. Avoid 70-75 (overconf) and 80+ (sparse).
ML_MIN_CONF  = 0.60      # below here: too noisy to bet
ML_GOOD_CONF = 0.65      # core band starts here
ML_MAX_CONF  = 0.80      # above 0.80 sample size collapses (N=21), drop from 0.85

# Featherweight: still worst division but materially better post-retrain
# (51.8% → 56.0% winner acc). Keep haircut but lower it slightly.
FEATHERWEIGHT_TAGS = {"Featherweight", "Men_Featherweight", "FW"}
FW_MIN_CONF = 0.65       # FW picks need core-band conf, not borderline

# IMPORTANT — book-graded "Over X.5 Rounds" is TIME-based, not bell-based.
# Books grade Over 1.5 as "fight still going past 2:30 of R2" (halfway through
# the next round), NOT "did the fight reach R2." The model's reaches_r2 = 1 if
# the fight enters R2 at all — so it overstates the book-graded Over 1.5 prob
# every time a fight ends in the first half of R2. Empirically, across 8,604
# fights:
#
#   Market    Book-graded   Model reaches_r(X+1)   Gap (model overstates)
#   Over 0.5    86.7%         100.0%                +13.3pp  ← skip, no model
#   Over 1.5    64.7%          72.0%                 +7.3pp
#   Over 2.5    51.7%          55.5%                 +3.7pp
#
# Conditional: 10.2% of fights that reach R2 end in first half of R2;
#              6.7% of fights that reach R3 end in first half of R3.
#
# Fix: apply a haircut when surfacing Over X.5 plays. Multiplicative form is
# more principled than additive (works at any prob level).
OVER_1_5_HAIRCUT = 1.0 - 0.102   # multiply reaches_r2 by 0.898 to get true Over 1.5
OVER_2_5_HAIRCUT = 1.0 - 0.067   # multiply reaches_r3 by 0.933 to get true Over 2.5

# Round/distance edge — thresholds gate for edge over book vig AFTER haircut.
# Post-retrain GOES-DISTANCE calibration (no haircut needed; 'goes_distance'
# market = method == DEC, which the model directly outputs):
#   0.30-0.40  pred 36% / actual 38%   +2.3pp
#   0.40-0.50  pred 45% / actual 52%   +7.3pp
#   0.50-0.60  pred 55% / actual 58%   +3.8pp
#   0.60-0.70  pred 63% / actual 71%   +8.5pp  ← strong edge zone
#
#   Prop                Typical line   Implied   Threshold (post-haircut)
#   Over 1.5 Rounds     -250 to -400   71-80%    over_1_5_prob ≥ 0.78
#   Over 2.5 Rounds     -150 to +110   47-60%    over_2_5_prob ≥ 0.60
#   Goes Distance       -130 to +110   47-57%    goes_distance ≥ 0.50
#   Goes Distance ★     same           same      goes_distance ≥ 0.60
#   Under 2.5 / Finish  +120 to +180   36-46%    finish_prob   ≥ 0.65

OVER_1_5_ROUNDS_MIN     = 0.78   # applied to haircut-adjusted prob
OVER_2_5_ROUNDS_MIN     = 0.60   # applied to haircut-adjusted prob
GOES_DISTANCE_MIN       = 0.50   # no haircut (market = DEC outcome)
GOES_DISTANCE_STRONG    = 0.60
FINISH_INSIDE_MIN       = 0.65

# Extreme-matchup flags — ML is structurally bad at tail predictions because
# isotonic calibration drags high-confidence predictions toward the population
# mean (UFC R1 finish base rate ~28%). The flag below is an explicit override
# layer: when raw fighter pattern unambiguously points to early finish, surface
# it as a high-conviction tier the bettor can see independently of ML.
#
# Two trigger conditions:
#   1. EITHER fighter has r1_ending_rate >= EXTREME_R1_MAX (single specialist
#      makes the fight likely to end early regardless of opponent — Marcio
#      Barbosa at 85% career R1 with a 5-fight R1 streak is the canonical case)
#   2. BOTH fighters have r1_ending_rate above EXTREME_R1_BOTH_MIN (mutual
#      R1-specialist matchup — McMillen 90% × Zecchini 75% = 0.68)
EXTREME_R1_MAX = 0.70           # ANY fighter at 70%+ R1 rate (fires single-side)
EXTREME_R1_BOTH_MIN = 0.45      # BOTH fighters above this level (fires joint)
EXTREME_GRINDER_MIN_PRODUCT = 0.30   # lowered from 0.40 — captures Evloev (65%) × Murphy (53%) = 0.34
EXTREME_GRINDER_MIN_INDIVIDUAL = 0.50  # OR either fighter at 50%+ decisions

# Submission specialist threshold — switched 2026-04-30 from per-WIN
# rate to FIGHT-DENOMINATOR rate (recent_5_sub_per_fight). The per-win
# rate hit 100% for any fighter whose wins were sub-flavored, regardless
# of how many losses sat in between (Morales canonical case: 3 UFC DEC
# losses + 2 regional sub wins → 100% per-win sub but 40% per-fight sub).
# Per-fight rate is the honest "is this fighter generating subs at high
# frequency" signal. Threshold of 0.40 means fighter subs in ≥2 of last 5
# fights — which is genuinely sub-specialist territory.
EXTREME_SUB_SPECIALIST_MIN = 0.40   # subs/last-5-fights threshold

# KO specialist — same denominator switch. 0.40 = KOs in 2+ of last 5 fights.
# Buzukja-style fighters (5-fight R1-KO streak) hit 100% on this. Generic
# punchers with 1 KO in 5 don't.
EXTREME_KO_SPECIALIST_MIN = 0.40   # KOs/last-5-fights threshold

# R1 finish props are calibration-broken (predicted 50%+, actual 39%).
# Hard-block, do not surface.
BLOCK_R1_PROPS = True

# Staleness guard for (5)-rate flags ─────────────────────────────────
# Sub/KO/R1 specialist flags are computed off last-5 WINS. For a fighter
# on a long losing streak, "last-5 wins" reaches back years and stops
# representing current form. Meerschaert canonical case: 4-fight skid
# entering this card, "100% sub last-5" was on wins from 2023; flag
# fired but he hasn't won since.
#
# Threshold updated 2026-04-30 after Morales miss: he was at exactly 40%
# form-5 (2 wins of last 5 fights), which is the borderline-stale case
# "<" missed. Changed to "won FEWER THAN HALF of last 5" — clearer
# semantic: if the fighter loses majority of his recent fights, his
# last-5-wins-based stats are by definition reaching back to a different
# era. Streak still triggers at -3 since profile_builder's streak field
# is fragile (sometimes resets after consecutive same-result counts).
STALE_LOSING_STREAK = -3        # streak <= -3 (3+ losses in a row) → stale
STALE_RECENT_FORM_5 = 0.50      # won fewer than half of last 5 → stale
                                 # (Morales 2025 case: 2W-3L UFC skid = 40%, MUST flag)

# Method-history sanity ──────────────────────────────────────────────
# A method-specialist flag (or method-pick recommendation) requires the
# fighter to have ACTUALLY finished by that method at least once.
# Gorimbo canonical case: model output "Gorimbo by KO/TKO R2" and pace
# rate said 14% career KO — but raw count rounds to ~0 actual KOs in his
# fight log. Block any method-flag where raw count == 0; warn at == 1.
METHOD_HISTORY_BLOCK_AT = 0     # zero finishes by this method → block flag entirely
METHOD_HISTORY_WARN_AT  = 1     # one finish by this method → keep flag but warn


def _pace_is_stale(pace: dict) -> bool:
    """True when (5)-rates aren't representative of current form.

    Triggers on extended losing streak OR low last-5 win rate. Either
    case means the fighter's "last-5 wins" stat reaches back too far
    to represent who they are now (durability decline, age, etc.).
    """
    streak = pace.get("streak", 0)
    form5 = pace.get("recent_form_5", 0.5)
    return streak <= STALE_LOSING_STREAK or form5 < STALE_RECENT_FORM_5
    # Note: form5 strictly LESS THAN 0.50 means "lost more than half of last 5"
    # — won 0/5, 1/5, or 2/5 fights all trigger. 3/5 (won majority) is fine.


def _eff_rate(pace: dict, key_recent: str, key_career: str) -> float:
    """Resolve the right (5) rate, falling back to career when stale.

    If the fighter is on a skid, the (5) rate is computed off old wins
    and overstates their current style. Use career rate instead — it
    averages over more fights and is less prone to small-sample drift.
    """
    if _pace_is_stale(pace):
        return float(pace.get(key_career, 0))
    recent = pace.get(key_recent, None)
    return float(recent if recent is not None else pace.get(key_career, 0))


def _method_history_ok(pace: dict, method: str) -> tuple[bool, int]:
    """Verify fighter has actually won by this method before.

    Returns (ok, count). ok=False means the flag/recommendation should
    be blocked entirely — the fighter has zero career wins by this
    method, so any pace-rate above zero is scraper noise (doctor
    stoppages mis-tagged as TKO, name collisions, etc.).
    """
    key = {"KO/TKO": "ko_wins", "SUB": "sub_wins", "DEC": "dec_wins"}.get(method)
    if key is None:
        return True, -1
    count = int(pace.get(key, 0))
    return count > METHOD_HISTORY_BLOCK_AT, count


# Pace-blended tail estimate ────────────────────────────────────────
# The ML R1/finish heads under-fire on extreme matchups because they're
# trained for aggregate accuracy (~28% base rate). The flag layer captures
# tail patterns the model misses. This helper produces a rule-based
# estimate of P(fight ends R1) that combines pace data signals more
# aggressively than the calibrated model. Surfaced ALONGSIDE the model's
# number so the bettor can compare and choose. Not used to replace the
# model; used as a second opinion for tail props.
def _pace_blended_r1_prob(pace_a: dict, pace_b: dict, model_r1_prob: float) -> dict:
    """Compute pace-rule R1 estimate alongside model prob; return both.

    Strong patterns (single-side specialist + opp not pure decision; or both
    fighters R1-prone) push the rule-based estimate toward 60-75%. Soft
    patterns push toward 35-45%. No pattern stays at the model's number.
    """
    a_r1 = _eff_rate(pace_a, "r1_ending_rate_5", "r1_ending_rate")
    b_r1 = _eff_rate(pace_b, "r1_ending_rate_5", "r1_ending_rate")
    a_dec = _eff_rate(pace_a, "decision_rate_5", "decision_rate")
    b_dec = _eff_rate(pace_b, "decision_rate_5", "decision_rate")
    max_r1 = max(a_r1, b_r1)
    min_r1 = min(a_r1, b_r1)
    min_dec = min(a_dec, b_dec)

    # Strong: single-side extreme (≥0.70) AND opponent not a pure decision-grinder
    if max_r1 >= 0.70 and min_dec <= 0.50:
        rule_estimate = 0.60 + 0.15 * (max_r1 - 0.70) / 0.30
    # Strong: both fighters R1-prone (≥0.45 each)
    elif min_r1 >= 0.45:
        rule_estimate = 0.55 + 0.15 * (min_r1 - 0.45) / 0.55
    # Soft: one side R1-prone (≥0.50) but opp not extreme
    elif max_r1 >= 0.50:
        rule_estimate = 0.40
    # No pattern — use model's number directly
    else:
        return {
            "model_r1": model_r1_prob,
            "rule_r1": model_r1_prob,
            "blended_r1": model_r1_prob,
            "pattern": None,
        }

    # Blend: 60% rule + 40% model. Tunable; rule weighted higher because
    # backtest shows model under-fires on tail patterns specifically.
    blended = 0.60 * rule_estimate + 0.40 * model_r1_prob
    pattern = (
        f"single-side R1-spec (max={max_r1:.0%})" if max_r1 >= 0.70 and min_dec <= 0.50
        else f"both R1-prone (min={min_r1:.0%})" if min_r1 >= 0.45
        else f"soft R1 (max={max_r1:.0%})"
    )
    return {
        "model_r1": model_r1_prob,
        "rule_r1": rule_estimate,
        "blended_r1": blended,
        "pattern": pattern,
    }


def _pace_blended_sub_prob(pace_a: dict, pace_b: dict, model_sub_prob: float,
                            fighter_a: str, fighter_b: str) -> dict:
    """Pace-rule estimate for P(fight ends by submission).

    Uses FIGHT-denominator recent_5_sub_per_fight (subs in last 5 fights / 5),
    not the per-WIN rate (which inflates for fighters on losing skids).
    """
    a_sub = pace_a.get("recent_5_sub_per_fight", 0)
    b_sub = pace_b.get("recent_5_sub_per_fight", 0)
    max_sub = max(a_sub, b_sub)
    sub_specialist = fighter_a if a_sub >= b_sub else fighter_b
    sub_count = pace_a.get("sub_wins", 0) if a_sub >= b_sub else pace_b.get("sub_wins", 0)

    # Strong: 40%+ of recent fights are sub finishes (2+ of last 5)
    # AND ≥3 career sub wins to verify it's not regional fluke.
    if max_sub >= 0.40 and sub_count >= 3:
        rule_estimate = 0.45 + 0.20 * min(1.0, (max_sub - 0.40) / 0.40)
    # Soft: 20%+ recent sub finishes (1 of last 5) + verified ≥3 career
    elif max_sub >= 0.20 and sub_count >= 3:
        rule_estimate = 0.32
    else:
        return {
            "model_sub": model_sub_prob,
            "rule_sub": model_sub_prob,
            "blended_sub": model_sub_prob,
            "pattern": None,
            "specialist": None,
        }

    blended = 0.60 * rule_estimate + 0.40 * model_sub_prob
    return {
        "model_sub": model_sub_prob,
        "rule_sub": rule_estimate,
        "blended_sub": blended,
        "pattern": f"{sub_specialist} sub-spec (last-5 sub-rate={max_sub:.0%}, {sub_count} career)",
        "specialist": sub_specialist,
    }


def _pace_blended_finish_inside_prob(pace_a: dict, pace_b: dict,
                                       model_finish_prob: float) -> dict:
    """Pace-rule estimate for P(fight ends inside the distance, any round).

    The model's `1 - goes_distance_prob` is the calibrated version. This
    rule-based estimate fires when both fighters have high career finish
    rates (Moicano-Duncan, McMillen-Zecchini cases). Captures fights that
    end fast but aren't necessarily R1.
    """
    a_fin = pace_a.get("finish_rate", 0.5)
    b_fin = pace_b.get("finish_rate", 0.5)
    a_dec = _eff_rate(pace_a, "decision_rate_5", "decision_rate")
    b_dec = _eff_rate(pace_b, "decision_rate_5", "decision_rate")
    min_fin = min(a_fin, b_fin)
    max_dec = max(a_dec, b_dec)

    # Strong: both fighters finish ≥75% of wins, neither is a decision-grinder
    if min_fin >= 0.75 and max_dec <= 0.40:
        rule_estimate = 0.78
    elif min_fin >= 0.65 and max_dec <= 0.50:
        rule_estimate = 0.68
    else:
        return {
            "model_finish": model_finish_prob,
            "rule_finish": model_finish_prob,
            "blended_finish": model_finish_prob,
            "pattern": None,
        }

    blended = 0.55 * rule_estimate + 0.45 * model_finish_prob
    return {
        "model_finish": model_finish_prob,
        "rule_finish": rule_estimate,
        "blended_finish": blended,
        "pattern": f"both finishers (min finish-rate={min_fin:.0%}, no grinder)",
    }


def _is_featherweight(wc: str) -> bool:
    if not wc:
        return False
    return any(tag.lower() in wc.lower() for tag in FEATHERWEIGHT_TAGS)


def gate_picks(snapshot: str | Path | dict) -> dict:
    """Categorize a card snapshot into bettable / soft-edge / avoid buckets.

    Returns: {"ml_plays": [...], "round_plays": [...], "avoid": [...]}
    Each entry includes fight identifiers, the play, the model probability,
    and a short reason. Backtest-derived thresholds (see constants above).
    """
    snap = snapshot if isinstance(snapshot, dict) else load(snapshot)

    ml_plays   = []
    round_plays = []
    extreme_flags = []   # human-pattern overrides ML can't access
    tail_estimates = []  # pace-blended R1/SUB/finish probs that override the model
    avoid       = []

    for f in snap["fights"]:
        fight = f"{f['fighter_a']} vs {f['fighter_b']}"
        wc = f.get("weight_class", "")
        is_fw = _is_featherweight(wc)
        conf = f["win_prob"]
        winner = f["winner"]
        rp = f.get("round_probs", {})
        ends_r1 = rp.get("ends_r1", 0)
        reaches_r2 = rp.get("reaches_r2", 0)
        reaches_r3 = rp.get("reaches_r3", 0)
        goes_dist = rp.get("goes_distance", 0)
        pace_a = f.get("pace_a", {})
        pace_b = f.get("pace_b", {})

        # ── Extreme-matchup pattern detection ──
        # ML calibration squashes tail predictions toward base rate (~28% R1
        # finish). When raw fighter pattern unambiguously points to R1 chaos
        # or grinder grind, surface a flag the bettor can act on independent
        # of ML's calibrated probability.
        # Use _eff_rate() for stale-aware resolution — falls back to career
        # rate when fighter is on a long skid (Meerschaert case: 4-fight L
        # streak made his "100% sub last-5" stat reach back to 2023 wins).
        a_r1 = _eff_rate(pace_a, "r1_ending_rate_5", "r1_ending_rate")
        b_r1 = _eff_rate(pace_b, "r1_ending_rate_5", "r1_ending_rate")
        a_dec = _eff_rate(pace_a, "decision_rate_5", "decision_rate")
        b_dec = _eff_rate(pace_b, "decision_rate_5", "decision_rate")
        # Specialist flags now use FIGHT-denominator rates (per-fight, not
        # per-win) — the honest "is this fighter currently generating these
        # finishes" signal. _eff_rate's stale-fallback to career still
        # applies but degrades gracefully because career rates are also
        # win-conditional and tend to overstate. For specialists we want
        # current activity, not lifetime style.
        a_sub = pace_a.get("recent_5_sub_per_fight", 0)
        b_sub = pace_b.get("recent_5_sub_per_fight", 0)
        a_ko = pace_a.get("recent_5_ko_per_fight", 0)
        b_ko = pace_b.get("recent_5_ko_per_fight", 0)
        r1_product = a_r1 * b_r1
        dec_product = a_dec * b_dec

        # ── SUB SPECIALIST flag (single-fighter trigger) ──
        # Fires when a fighter's wins are dominantly by submission. ML's
        # method head averages across all training fights and dilutes this
        # individual-style signal. Blocked when fighter has 0 career sub
        # wins (rate is scraper noise) and downgraded when stale.
        if max(a_sub, b_sub) >= EXTREME_SUB_SPECIALIST_MIN:
            if a_sub >= b_sub:
                sub_specialist_name = f["fighter_a"]
                sub_pace, sub_rate = pace_a, a_sub
            else:
                sub_specialist_name = f["fighter_b"]
                sub_pace, sub_rate = pace_b, b_sub
            ok, sub_count = _method_history_ok(sub_pace, "SUB")
            stale = _pace_is_stale(sub_pace)
            method_probs = f.get("method_probs", {})
            sub_prob = method_probs.get("SUB", 0)
            if not ok:
                # 0 career sub wins — flag is scraper noise, suppress.
                avoid.append({
                    "fight": fight, "play": f"{sub_specialist_name} by SUB",
                    "prob": sub_prob,
                    "reason": (
                        f"SUB-specialist flag suppressed: {sub_specialist_name} "
                        f"has {sub_count} career sub wins (rate {sub_rate:.0%} "
                        f"is scraper noise — likely doctor-stoppage or name collision)"
                    ),
                })
            else:
                detail = f"{sub_specialist_name} wins {sub_rate:.0%} by submission ({sub_count} career)"
                implication = (
                    f"ML says SUB method prob={sub_prob:.0%} — "
                    f"likely understated for sub-style finishers. "
                    f"Consider {sub_specialist_name} by SUB or fight-ends-by-SUB props."
                )
                if stale:
                    detail += "  ⚠️ STALE"
                    implication = (
                        f"⚠️  Fighter on losing skid (streak {sub_pace.get('streak',0)}, "
                        f"form-5 {sub_pace.get('recent_form_5',0):.0%}) — (5)-rate built "
                        f"on old wins. Career SUB rate = {sub_pace.get('sub_win_rate',0):.0%}. "
                        f"Flag DOWNGRADED — be skeptical."
                    )
                if sub_count == METHOD_HISTORY_WARN_AT:
                    detail += f"  ⚠️ thin sample ({sub_count} sub win)"
                extreme_flags.append({
                    "fight": fight,
                    "flag": "🐍 SUB SPECIALIST" + (" (STALE)" if stale else ""),
                    "detail": detail,
                    "implication": implication,
                })

        # ── KO SPECIALIST flag (single-fighter trigger) ──
        # Fighter wins dominantly by KO/TKO. UFC base rate ~35% so 55%+ is
        # extreme. Barbosa (83%) and Zecchini (82%) canonical cases.
        # Same suppression rules as SUB flag: zero KOs blocks, stale downgrades.
        if max(a_ko, b_ko) >= EXTREME_KO_SPECIALIST_MIN:
            if a_ko >= b_ko:
                ko_specialist_name = f["fighter_a"]
                ko_pace, ko_rate = pace_a, a_ko
            else:
                ko_specialist_name = f["fighter_b"]
                ko_pace, ko_rate = pace_b, b_ko
            ok, ko_count = _method_history_ok(ko_pace, "KO/TKO")
            stale = _pace_is_stale(ko_pace)
            method_probs = f.get("method_probs", {})
            ko_prob = method_probs.get("KO/TKO", 0)
            if not ok:
                avoid.append({
                    "fight": fight, "play": f"{ko_specialist_name} by KO/TKO",
                    "prob": ko_prob,
                    "reason": (
                        f"KO-specialist flag suppressed: {ko_specialist_name} "
                        f"has {ko_count} career KO wins (rate {ko_rate:.0%} "
                        f"is scraper noise — likely TKO mis-classification)"
                    ),
                })
            else:
                detail = f"{ko_specialist_name} wins {ko_rate:.0%} by KO/TKO ({ko_count} career)"
                implication = (
                    f"ML says KO/TKO method prob={ko_prob:.0%} — "
                    f"check for understatement on KO-style finishers. "
                    f"Consider {ko_specialist_name} by KO/TKO props."
                )
                if stale:
                    detail += "  ⚠️ STALE"
                    implication = (
                        f"⚠️  Fighter on losing skid (streak {ko_pace.get('streak',0)}, "
                        f"form-5 {ko_pace.get('recent_form_5',0):.0%}) — (5)-rate built "
                        f"on old wins. Career KO rate = {ko_pace.get('ko_win_rate',0):.0%}. "
                        f"Flag DOWNGRADED — be skeptical."
                    )
                if ko_count == METHOD_HISTORY_WARN_AT:
                    detail += f"  ⚠️ thin sample ({ko_count} KO win)"
                extreme_flags.append({
                    "fight": fight,
                    "flag": "💥 KO SPECIALIST" + (" (STALE)" if stale else ""),
                    "detail": detail,
                    "implication": implication,
                })

        # ── Model method-pick sanity check ──
        # The ML method head can pick a fighter "by KO/TKO R2" even when
        # that fighter has zero career KOs — it's regressing off weight-class
        # base rates and opponent vulnerability, not the fighter's actual
        # finishing toolkit. Surface a warning when this happens so the
        # method prop doesn't get bet on phantom data.
        # Two tiers: hard-block at 0 finishes (Malkoun-by-SUB case), soft-warn
        # at 1-2 finishes (Gorimbo-by-KO case where scraper may have
        # mis-classified a doctor stoppage as KO).
        winner_pace = pace_a if winner == f["fighter_a"] else pace_b
        method_pick = f.get("method", "")
        m_ok, m_count = _method_history_ok(winner_pace, method_pick)
        if method_pick in ("KO/TKO", "SUB"):
            if not m_ok:
                avoid.append({
                    "fight": fight,
                    "play": f"{winner} by {method_pick}",
                    "prob": f.get("method_probs", {}).get(method_pick, 0),
                    "reason": (
                        f"Model picked {winner} by {method_pick} but their career "
                        f"has {m_count} wins by this method — method pick is regressing "
                        f"off weight-class base rates, not actual finishing history. "
                        f"Bet ML or pick a different method prop."
                    ),
                })
            elif m_count <= 2:
                avoid.append({
                    "fight": fight,
                    "play": f"{winner} by {method_pick} (thin sample)",
                    "prob": f.get("method_probs", {}).get(method_pick, 0),
                    "reason": (
                        f"Model picked {winner} by {method_pick} but their career "
                        f"has only {m_count} wins by this method. Rate may be "
                        f"scraper noise (TKO-vs-doctor-stoppage classification). "
                        f"Verify against actual fight log before betting."
                    ),
                })

        if max(a_r1, b_r1) >= EXTREME_R1_MAX:
            # One fighter is an extreme R1 specialist — fight skews early
            # regardless of opponent
            specialist = (f["fighter_a"] if a_r1 >= b_r1 else f["fighter_b"])
            specialist_rate = max(a_r1, b_r1)
            extreme_flags.append({
                "fight": fight,
                "flag": "🔥 R1 SPECIALIST",
                "detail": (f"{specialist} ends fights in R1 {specialist_rate:.0%} of the time "
                           f"(opponent {min(a_r1, b_r1):.0%}, product {r1_product:.2f})"),
                "implication": (
                    f"ML says R1 finish prob={ends_r1:.0%} — "
                    f"likely understated (calibration squashes tails). "
                    f"Consider R1 finish props or Under 1.5 Rounds beyond ML's number."
                ),
            })
        elif min(a_r1, b_r1) >= EXTREME_R1_BOTH_MIN:
            # Both fighters above threshold — joint R1 lean
            extreme_flags.append({
                "fight": fight,
                "flag": "🔥 BOTH R1 SPECIALISTS",
                "detail": f"a_r1={a_r1:.0%}  b_r1={b_r1:.0%}  product={r1_product:.2f}",
                "implication": (
                    f"ML says R1 finish prob={ends_r1:.0%} — "
                    f"likely understated. Consider Under 1.5 / R1 finish props."
                ),
            })
        elif (dec_product >= EXTREME_GRINDER_MIN_PRODUCT
              and min(a_dec, b_dec) >= EXTREME_GRINDER_MIN_INDIVIDUAL):
            extreme_flags.append({
                "fight": fight,
                "flag": "⏳ GRINDER MATCHUP",
                "detail": f"a_dec={a_dec:.0%}  b_dec={b_dec:.0%}  product={dec_product:.2f}",
                "implication": (
                    f"ML says goes-distance={goes_dist:.0%}. "
                    f"Both fighters grind to decision — strong Over 2.5 Rounds / Goes Distance lean."
                ),
            })

        # ── ML pick gate ──
        min_conf = FW_MIN_CONF if is_fw else ML_MIN_CONF
        if conf < min_conf:
            avoid.append({
                "fight": fight, "play": f"ML {winner}", "prob": conf,
                "reason": f"conf {conf:.0%} below {min_conf:.0%} gate"
                          + (" (FW haircut)" if is_fw else ""),
            })
        elif conf > ML_MAX_CONF:
            avoid.append({
                "fight": fight, "play": f"ML {winner}", "prob": conf,
                "reason": f"conf {conf:.0%} above 85% — sparse backtest data",
            })
        else:
            tier = "core" if conf >= ML_GOOD_CONF else "borderline"
            ml_plays.append({
                "fight": fight, "play": f"ML {winner}", "prob": conf,
                "tier": tier,
                "reason": (f"conf {conf:.0%} in {tier} band"
                           + (" (FW gated up)" if is_fw else "")),
            })

        # ── Pace-blended TAIL ESTIMATES (R1, SUB, finish-inside) ──
        # Surface rule-based estimates alongside the model's calibrated probs
        # whenever a tail pattern fires. Backtest shows model under-fires on
        # tail patterns specifically (Buzukja-Barbosa: model 20% R1 vs rule
        # 75% — fight ended R1 KO). Blend is 60% rule / 40% model so the
        # bettor sees a number that doesn't get squashed by aggregate calibration.
        method_probs_local = f.get("method_probs", {})
        sub_prob_model = method_probs_local.get("SUB", 0)
        finish_prob_model = 1.0 - goes_dist
        r1_blend     = _pace_blended_r1_prob(pace_a, pace_b, ends_r1)
        sub_blend    = _pace_blended_sub_prob(pace_a, pace_b, sub_prob_model,
                                                f["fighter_a"], f["fighter_b"])
        finish_blend = _pace_blended_finish_inside_prob(pace_a, pace_b, finish_prob_model)

        if r1_blend["pattern"]:
            tail_estimates.append({
                "fight": fight, "type": "P(ends R1)",
                "model": r1_blend["model_r1"],
                "rule": r1_blend["rule_r1"],
                "blended": r1_blend["blended_r1"],
                "pattern": r1_blend["pattern"],
                "delta_pp": (r1_blend["blended_r1"] - r1_blend["model_r1"]) * 100,
            })
        if sub_blend["pattern"]:
            tail_estimates.append({
                "fight": fight, "type": "P(fight ends by SUB)",
                "model": sub_blend["model_sub"],
                "rule": sub_blend["rule_sub"],
                "blended": sub_blend["blended_sub"],
                "pattern": sub_blend["pattern"],
                "delta_pp": (sub_blend["blended_sub"] - sub_blend["model_sub"]) * 100,
            })
        if finish_blend["pattern"]:
            tail_estimates.append({
                "fight": fight, "type": "P(finish inside dist)",
                "model": finish_blend["model_finish"],
                "rule": finish_blend["rule_finish"],
                "blended": finish_blend["blended_finish"],
                "pattern": finish_blend["pattern"],
                "delta_pp": (finish_blend["blended_finish"] - finish_blend["model_finish"]) * 100,
            })

        # ── Round / distance plays (primary product) ──
        # Apply book-vig haircut to convert model 'reaches_rN' (bell-based)
        # to true book-graded 'Over X.5' (time-based, halfway through next
        # round). See constants block: ~10% of R2-reaching fights end in the
        # first half of R2 → model overstates Over 1.5 by ~7pp uncorrected.
        over_1_5_prob = reaches_r2 * OVER_1_5_HAIRCUT
        over_2_5_prob = reaches_r3 * OVER_2_5_HAIRCUT
        finish_inside_prob = 1 - goes_dist

        if over_1_5_prob >= OVER_1_5_ROUNDS_MIN:
            round_plays.append({
                "fight": fight, "play": "Over 1.5 Rounds",
                "prob": over_1_5_prob,
                "reason": f"book-graded Over 1.5 = {over_1_5_prob:.0%} "
                          f"(model reaches_r2 {reaches_r2:.0%} × {OVER_1_5_HAIRCUT:.2f} haircut)",
            })
        if over_2_5_prob >= OVER_2_5_ROUNDS_MIN:
            round_plays.append({
                "fight": fight, "play": "Over 2.5 Rounds",
                "prob": over_2_5_prob,
                "reason": f"book-graded Over 2.5 = {over_2_5_prob:.0%} "
                          f"(model reaches_r3 {reaches_r3:.0%} × {OVER_2_5_HAIRCUT:.2f} haircut)",
            })
        if goes_dist >= GOES_DISTANCE_STRONG:
            round_plays.append({
                "fight": fight, "play": "Goes the Distance ★",
                "prob": goes_dist,
                "reason": f"distance {goes_dist:.0%} — STRONG edge zone "
                          f"(60-70% predicted → 71% actual, +8.5pp)",
            })
        elif goes_dist >= GOES_DISTANCE_MIN:
            round_plays.append({
                "fight": fight, "play": "Goes the Distance",
                "prob": goes_dist,
                "reason": f"distance {goes_dist:.0%} — calibrated edge zone "
                          f"(model under-prices by 5-8pp post-retrain)",
            })
        if finish_inside_prob >= FINISH_INSIDE_MIN:
            round_plays.append({
                "fight": fight, "play": "Under 2.5 / Finish Inside",
                "prob": finish_inside_prob,
                "reason": f"P(finish) {finish_inside_prob:.0%} — typically priced +130/+150",
            })

        # ── Blocked: R1 finish props (calibration broken) ──
        if BLOCK_R1_PROPS and ends_r1 >= 0.40:
            avoid.append({
                "fight": fight, "play": f"Ends R1",
                "prob": ends_r1,
                "reason": f"R1 prop calibration broken: predicted {ends_r1:.0%}, "
                          f"backtest actual ~{ends_r1*0.7:.0%}",
            })

    return {
        "event": snap.get("event"),
        "event_date": snap.get("event_date"),
        "ml_plays": sorted(ml_plays, key=lambda x: -x["prob"]),
        "round_plays": sorted(round_plays, key=lambda x: -x["prob"]),
        "extreme_flags": extreme_flags,
        # Tail estimates sorted by largest model-vs-rule delta first — those
        # are the fights where calibration is hiding the most signal.
        "tail_estimates": sorted(tail_estimates, key=lambda x: -x["delta_pp"]),
        "avoid": sorted(avoid, key=lambda x: -x["prob"]),
    }


def print_bettable_card(snapshot: str | Path | dict) -> None:
    """Pretty print of gate_picks() output."""
    g = gate_picks(snapshot)
    print()
    print("=" * 88)
    print(f"  BETTABLE CARD — {g['event']} ({g['event_date']})")
    print("  Gates derived from 1,359-fight backtest calibration")
    print("=" * 88)

    # Surface tail estimates FIRST — they're where calibration was hiding signal,
    # so they're the highest-information lines for spotting bets the model missed.
    if g.get("tail_estimates"):
        print(f"\n  🎯 TAIL ESTIMATES (pace-blended — where the model under-fires):")
        print(f"  {'Δ vs model':>10}  {'Type':<22} {'Model':>6} {'Rule':>6} {'Blend':>6}  Pattern / Fight")
        print(f"  {'-' * 90}")
        for t in g["tail_estimates"]:
            sign = "+" if t["delta_pp"] >= 0 else ""
            print(f"  {sign}{t['delta_pp']:>5.0f}pp     {t['type']:<22} "
                  f"{t['model']*100:>5.0f}% {t['rule']*100:>5.0f}% {t['blended']*100:>5.0f}%  "
                  f"{t['pattern']}")
            print(f"  {'':>10}      → {t['fight']}")

    print(f"\n  CORE ML PLAYS (conf 65-85%, well-calibrated band):")
    core = [p for p in g["ml_plays"] if p["tier"] == "core"]
    if not core:
        print("    (none — no fight clears the 65% gate this card)")
    for p in core:
        print(f"    {p['prob']*100:>4.0f}%  {p['play']:<35}  {p['fight']}")

    print(f"\n  BORDERLINE ML (conf 60-65%, bet small or skip):")
    border = [p for p in g["ml_plays"] if p["tier"] == "borderline"]
    if not border:
        print("    (none)")
    for p in border:
        print(f"    {p['prob']*100:>4.0f}%  {p['play']:<35}  {p['fight']}")

    print(f"\n  ROUND / DISTANCE PLAYS (model's strongest signal):")
    if not g["round_plays"]:
        print("    (none clear threshold)")
    for p in g["round_plays"]:
        print(f"    {p['prob']*100:>4.0f}%  {p['play']:<20}  {p['fight']}")

    print(f"\n  EXTREME MATCHUP FLAGS (raw-pattern overrides — ML can't see these):")
    if not g["extreme_flags"]:
        print("    (no extreme matchups on this card)")
    for fl in g["extreme_flags"]:
        print(f"    {fl['flag']}   {fl['fight']}")
        print(f"      {fl['detail']}")
        print(f"      → {fl['implication']}")

    print(f"\n  AVOID — flagged by gates:")
    if not g["avoid"]:
        print("    (none)")
    for p in g["avoid"]:
        print(f"    {p['prob']*100:>4.0f}%  {p['play']:<25}  {p['fight']}")
        print(f"          ↳ {p['reason']}")
    print()
    print("=" * 88)


def print_scorecard(result: dict) -> None:
    n = result["graded"]
    print(f"\n{result['event']} ({result['event_date']}) — {n} fights graded")
    if n == 0:
        print("  (no matching actuals)")
        return
    print(f"  Winner : {result['winner_hits']}/{n} "
          f"({result['winner_hits']/n:.0%})")
    print(f"  Method : {result['method_hits']}/{n} "
          f"({result['method_hits']/n:.0%})")
    print(f"  Round  : {result['round_hits']}/{n} "
          f"({result['round_hits']/n:.0%})")
    print()
    print(f"  {'Fight':<44} {'Pick':<22} {'Actual':<22} W M R")
    print("  " + "-" * 96)
    for r in result["rows"]:
        fight = f"{r['fighter_a']} vs {r['fighter_b']}"[:42]
        if r.get("_verdict") == "NO_RESULT":
            print(f"  {fight:<44} {r['winner']:<22} {'—':<22} - - -")
            continue
        pick = f"{r['winner']} ({r['method']} R{r.get('round','?')})"[:20]
        actual = (f"{r['actual_winner']} ({r['actual_method']}"
                  f" R{r.get('actual_round','?')})")[:20]
        w = "Y" if r["winner_hit"] else "N"
        m = "Y" if r["method_hit"] else "N"
        rnd = "Y" if r["round_hit"] else "N"
        print(f"  {fight:<44} {pick:<22} {actual:<22} {w} {m} {rnd}")
