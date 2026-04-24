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
    }


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
