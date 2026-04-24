"""
UFC Edge Finder — Over/Under Mispricing Detection Engine

The books set over/under round totals based on their models.
We cross-reference THREE independent signals to catch them slipping:

  Signal 1: Fight-Duration Model (fighter history + style matchup)
  Signal 2: Method Prop Consistency (method odds must square with distance odds)
  Signal 3: Simulation Output (Monte Carlo finish distribution)

When 2+ signals agree the book is wrong, we have a high-confidence edge.

Usage:
    # Pull live odds from The Odds API (best — uses real prices)
    python3 edge_finder.py --live

    # Pull live + check SGP prices after
    python3 edge_finder.py --live --sgp

    # Use hardcoded defaults (no API call)
    python3 edge_finder.py --defaults

    # Manual entry
    python3 edge_finder.py --manual
"""

import sys
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import everything from v2 engine
from ufc_vegas_115_v2 import (
    fighters, matchups,
    book_odds as DEFAULT_BOOK_ODDS,
    round_props as DEFAULT_ROUND_PROPS,
    Fighter, calculate_elo, simulate, analyze, stat_prob,
    american_to_implied, american_to_decimal, kelly,
    fighter_finish_rate, mismatch_multiplier, elo_wp,
)

# Import Fight IQ for matchup intelligence signal
from fight_iq import analyze_matchup, build_profile

# ============================================================
# LIVE ODDS FROM THE ODDS API
# ============================================================

def _match_fighter_key(api_name: str) -> Optional[str]:
    """Match an API fighter name like 'Landon Vannata' to our key like 'Vannata'."""
    api_lower = api_name.lower()
    for key in fighters:
        full_name = fighters[key].name.lower()
        # Exact match
        if api_lower == full_name:
            return key
        # API name contains our full name or vice versa
        if api_lower in full_name or full_name in api_lower:
            return key
        # Last name match
        api_last = api_lower.split()[-1]
        key_last = full_name.split()[-1]
        if api_last == key_last:
            return key
    return None


def _find_matchup_key(fkey_a: str, fkey_b: str) -> Optional[Tuple[str, str]]:
    """Find which matchup tuple matches these two fighter keys."""
    for a_key, b_key, nr, is_main in matchups:
        if (fkey_a == a_key and fkey_b == b_key) or (fkey_a == b_key and fkey_b == a_key):
            return (a_key, b_key)
    return None


def fetch_live_odds(event_date: str = None, preferred_book: str = None) -> Tuple[Dict, Dict]:
    """
    Pull live odds from The Odds API and map them to our fighter keys.
    Returns (book_odds, round_props) in the same format the engine expects.

    Uses consensus (average) across all books for best estimate,
    and tracks best available price per fighter.
    """
    from odds_scanner import OddsClient, PREFERRED_BOOKS

    print("\n  Connecting to The Odds API...")
    client = OddsClient()

    raw = client.get_odds(markets="h2h,totals")
    print(f"  Got {len(raw)} events | API quota remaining: {client.remaining_requests}")

    # Filter to our card's date if specified
    if event_date:
        from datetime import datetime, timedelta
        base = datetime.strptime(event_date, "%Y-%m-%d")
        next_day = (base + timedelta(days=1)).strftime("%Y-%m-%d")
        raw = [e for e in raw if e.get("commence_time", "")[:10] in (event_date, next_day)]
        print(f"  Filtered to {event_date}: {len(raw)} events")

    # Start from hardcoded defaults (method props aren't in the API)
    book_odds = deepcopy(DEFAULT_BOOK_ODDS)
    round_props = deepcopy(DEFAULT_ROUND_PROPS)

    matched = 0
    unmatched = []

    # Collect ALL lines per fighter across books for consensus
    ml_lines = {}   # fighter_key -> [american_odds, ...]
    ou_lines = {}   # (a_key, b_key) -> {"over": [(point, odds)], "under": [(point, odds)]}

    for event in raw:
        home = event.get("home_team", "")
        away = event.get("away_team", "")

        home_key = _match_fighter_key(home)
        away_key = _match_fighter_key(away)

        if not home_key or not away_key:
            unmatched.append(f"{home} vs {away}")
            continue

        matchup_key = _find_matchup_key(home_key, away_key)
        if not matchup_key:
            continue

        matched += 1

        for bookmaker in event.get("bookmakers", []):
            book = bookmaker["key"]
            # Use all US books, not just preferred
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        fkey = _match_fighter_key(outcome["name"])
                        if fkey:
                            ml_lines.setdefault(fkey, []).append(outcome["price"])

                elif market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        side = "over" if outcome["name"] == "Over" else "under"
                        point = outcome.get("point", 2.5)
                        price = outcome["price"]
                        ou_lines.setdefault(matchup_key, {"over": [], "under": [], "points": []})
                        ou_lines[matchup_key][side].append(price)
                        if point:
                            ou_lines[matchup_key]["points"].append(float(point))

    # Build consensus ML odds (median across books — more robust than mean)
    print(f"\n  Matched {matched} fights to our card")
    if unmatched:
        print(f"  Unmatched API events: {', '.join(unmatched[:5])}")

    print(f"\n  {'Fighter':<28} {'Books':>5} {'Best':>8} {'Worst':>8} {'Consensus':>10}")
    print(f"  {'─' * 63}")

    for fkey, lines in sorted(ml_lines.items()):
        if not lines:
            continue
        lines_sorted = sorted(lines)
        # Use median for consensus
        mid = len(lines_sorted) // 2
        consensus = lines_sorted[mid]
        best = max(lines)  # best price for bettor (highest)
        worst = min(lines)

        # Update book_odds with live consensus ML
        if isinstance(book_odds.get(fkey), dict):
            book_odds[fkey]["ml"] = consensus
        else:
            book_odds[fkey] = {"ml": consensus}

        print(f"  {fkey:<28} {len(lines):>5} {best:>+8d} {worst:>+8d} {consensus:>+10d}")

    # Build consensus O/U
    print(f"\n  {'Fight':<38} {'Line':>5} {'Over':>8} {'Under':>8} {'Books':>6}")
    print(f"  {'─' * 69}")

    for matchup_key, data in ou_lines.items():
        overs = data["over"]
        unders = data["under"]
        points = data["points"]

        if not overs or not unders:
            continue

        # Determine the line (most common point)
        if points:
            from collections import Counter
            line = Counter(points).most_common(1)[0][0]
        else:
            line = round_props.get(matchup_key, {}).get("line", 2.5)

        # Median for consensus
        overs_sorted = sorted(overs)
        unders_sorted = sorted(unders)
        over_consensus = overs_sorted[len(overs_sorted) // 2]
        under_consensus = unders_sorted[len(unders_sorted) // 2]

        round_props[matchup_key] = {
            "line": line,
            "over": over_consensus,
            "under": under_consensus,
        }

        a_key, b_key = matchup_key
        fight_name = f"{fighters[a_key].name} vs {fighters[b_key].name}"
        print(f"  {fight_name:<38} {line:>5} {over_consensus:>+8d} {under_consensus:>+8d} {len(overs):>6}")

    print(f"\n  NOTE: Method props (KO/SUB/DEC) are not available from The Odds API.")
    print(f"  Using hardcoded method props from v2. To update them, use --manual mode.\n")

    # Save for reuse
    save_odds(book_odds, round_props)

    return book_odds, round_props


# ============================================================
# MANUAL ODDS INPUT
# ============================================================

ODDS_FILE = Path(__file__).parent / "live_odds.json"


def load_saved_odds() -> Tuple[Dict, Dict]:
    """Load previously saved odds from live_odds.json."""
    if ODDS_FILE.exists():
        data = json.loads(ODDS_FILE.read_text())
        return data.get("book_odds", {}), data.get("round_props", {})
    return {}, {}


def save_odds(book_odds: Dict, round_props: Dict):
    """Save current odds to live_odds.json for reuse."""
    # Convert round_props tuple keys to strings for JSON
    rp_serializable = {}
    for key, val in round_props.items():
        if isinstance(key, tuple):
            str_key = f"{key[0]}|{key[1]}"
        else:
            str_key = key
        rp_serializable[str_key] = val

    data = {"book_odds": book_odds, "round_props": rp_serializable}
    ODDS_FILE.write_text(json.dumps(data, indent=2))
    print(f"  Odds saved to {ODDS_FILE}")


def deserialize_round_props(rp_raw: Dict) -> Dict:
    """Convert string keys back to tuple keys."""
    result = {}
    for key, val in rp_raw.items():
        if "|" in key:
            parts = key.split("|")
            result[(parts[0], parts[1])] = val
        else:
            result[key] = val
    return result


def prompt_odds(prompt_text: str, default: int = None) -> Optional[int]:
    """Prompt user for American odds. Returns None to skip."""
    default_str = f" [{default:+d}]" if default is not None else ""
    raw = input(f"    {prompt_text}{default_str}: ").strip()
    if raw == "" and default is not None:
        return default
    if raw.lower() in ("s", "skip", ""):
        return None
    try:
        val = int(raw.replace("+", ""))
        return val
    except ValueError:
        print(f"      Invalid — enter American odds like +155 or -200. Skipping.")
        return None


def collect_live_odds(update_only: bool = False) -> Tuple[Dict, Dict]:
    """
    Interactive odds collection. Walks through each fight and asks
    for the user's book prices. Press Enter to keep defaults, 's' to skip.
    """
    # Start from defaults
    book_odds = deepcopy(DEFAULT_BOOK_ODDS)
    round_props = deepcopy(DEFAULT_ROUND_PROPS)

    # Check for saved odds
    saved_book, saved_rp_raw = load_saved_odds()
    saved_rp = deserialize_round_props(saved_rp_raw) if saved_rp_raw else {}
    if saved_book:
        print(f"\n  Found saved odds from {ODDS_FILE}")
        use_saved = input("  Use saved odds as starting point? (y/n) [y]: ").strip().lower()
        if use_saved != "n":
            # Merge saved into defaults
            for k, v in saved_book.items():
                if k in book_odds:
                    if isinstance(v, dict):
                        book_odds[k].update(v)
                    else:
                        book_odds[k] = v
            for k, v in saved_rp.items():
                if k in round_props:
                    round_props[k].update(v)
            print("  Loaded saved odds. Press Enter to keep, or type new values.\n")

    print(f"\n{'─' * 70}")
    print(f"  ENTER YOUR BOOK'S ODDS")
    print(f"  Press Enter to keep default | Type 's' to skip a fight entirely")
    print(f"{'─' * 70}\n")

    fights_updated = 0

    for a_key, b_key, nr, is_main in matchups:
        a, b = fighters[a_key], fighters[b_key]
        tag = " [MAIN EVENT]" if is_main else ""
        rp_key = (a_key, b_key)

        current_rp = round_props.get(rp_key, {})
        current_line = current_rp.get("line", 2.5)

        print(f"  {a.name} vs {b.name}{tag} ({nr}R, O/U {current_line})")

        if update_only:
            update = input(f"    Update this fight? (y/n) [n]: ").strip().lower()
            if update != "y":
                print()
                continue

        changed = False

        # Over/Under
        print(f"    Over/Under (line: {current_line}):")
        over_val = prompt_odds(f"Over {current_line}", current_rp.get("over"))
        if over_val is not None:
            round_props[rp_key]["over"] = over_val
            changed = True
        under_val = prompt_odds(f"Under {current_line}", current_rp.get("under"))
        if under_val is not None:
            round_props[rp_key]["under"] = under_val
            changed = True

        # Method props for fighter A
        a_odds = book_odds.get(a_key, {})
        print(f"    {a.name}:")
        for label, odds_key in [("ML", "ml"), ("by KO", "by_ko"), ("by SUB", "by_sub"), ("by DEC", "by_dec")]:
            val = prompt_odds(f"{label}", a_odds.get(odds_key))
            if val is not None:
                if isinstance(book_odds[a_key], dict):
                    book_odds[a_key][odds_key] = val
                else:
                    book_odds[a_key] = {"ml": book_odds[a_key], odds_key: val}
                changed = True

        # Method props for fighter B
        b_odds = book_odds.get(b_key, {})
        print(f"    {b.name}:")
        for label, odds_key in [("ML", "ml"), ("by KO", "by_ko"), ("by SUB", "by_sub"), ("by DEC", "by_dec")]:
            val = prompt_odds(f"{label}", b_odds.get(odds_key))
            if val is not None:
                if isinstance(book_odds[b_key], dict):
                    book_odds[b_key][odds_key] = val
                else:
                    book_odds[b_key] = {"ml": book_odds[b_key], odds_key: val}
                changed = True

        if changed:
            fights_updated += 1
        print()

    print(f"  Updated {fights_updated} fights.\n")

    # Offer to save
    if fights_updated > 0:
        save = input("  Save these odds for next time? (y/n) [y]: ").strip().lower()
        if save != "n":
            save_odds(book_odds, round_props)

    return book_odds, round_props


def quick_update_single_fight(book_odds: Dict, round_props: Dict) -> Tuple[Dict, Dict]:
    """Quick mode: show a numbered list, user picks which fights to update."""
    print(f"\n  Select fights to update (comma-separated, or 'all'):\n")
    for i, (a_key, b_key, nr, is_main) in enumerate(matchups, 1):
        a, b = fighters[a_key], fighters[b_key]
        tag = " *" if is_main else ""
        rp = round_props.get((a_key, b_key), {})
        print(f"    {i:>2}. {a.name} vs {b.name}{tag}  O/U {rp.get('line', '?')}: "
              f"o{rp.get('over', '?'):+d} / u{rp.get('under', '?'):+d}")

    choice = input(f"\n  Enter fight numbers: ").strip()
    if choice.lower() == "all":
        indices = list(range(len(matchups)))
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
        except ValueError:
            print("  Invalid input. Running with current odds.")
            return book_odds, round_props

    for idx in indices:
        if idx < 0 or idx >= len(matchups):
            continue
        a_key, b_key, nr, is_main = matchups[idx]
        a, b = fighters[a_key], fighters[b_key]
        rp_key = (a_key, b_key)
        rp = round_props.get(rp_key, {})

        print(f"\n  {a.name} vs {b.name}:")

        over_val = prompt_odds(f"Over {rp.get('line', 2.5)}", rp.get("over"))
        if over_val is not None:
            round_props[rp_key]["over"] = over_val
        under_val = prompt_odds(f"Under {rp.get('line', 2.5)}", rp.get("under"))
        if under_val is not None:
            round_props[rp_key]["under"] = under_val

        for fkey in [a_key, b_key]:
            fn = fighters[fkey].name
            odds = book_odds.get(fkey, {})
            print(f"    {fn}:")
            for label, odds_key in [("ML", "ml"), ("by KO", "by_ko"), ("by SUB", "by_sub"), ("by DEC", "by_dec")]:
                val = prompt_odds(f"{label}", odds.get(odds_key) if isinstance(odds, dict) else (odds if odds_key == "ml" else None))
                if val is not None:
                    if isinstance(book_odds[fkey], dict):
                        book_odds[fkey][odds_key] = val
                    else:
                        book_odds[fkey] = {"ml": val} if odds_key == "ml" else {"ml": book_odds[fkey], odds_key: val}

    save = input("\n  Save these odds? (y/n) [y]: ").strip().lower()
    if save != "n":
        save_odds(book_odds, round_props)

    return book_odds, round_props


# ============================================================
# ODDS MATH
# ============================================================

def remove_vig(prob_a: float, prob_b: float) -> Tuple[float, float]:
    """Strip the juice to get true probabilities."""
    total = prob_a + prob_b
    if total == 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total


def implied_to_american(prob: float) -> int:
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    return round(100 * (1 - prob) / prob)


def ev_per_dollar(true_prob: float, decimal_odds: float) -> float:
    """Expected value per $1 wagered."""
    return true_prob * decimal_odds - 1.0


# ============================================================
# SIGNAL 1: FIGHT-DURATION MODEL
# ============================================================

def historical_finish_rate(f: Fighter) -> float:
    total = f.wins + f.losses
    if total == 0:
        return 0.5
    finishes_by = f.ko_wins + f.sub_wins
    finished_by = f.ko_losses + f.sub_losses
    return (finishes_by + finished_by) / total


def fighter_distance_rate(f: Fighter) -> float:
    return 1.0 - historical_finish_rate(f)


def pace_score(f: Fighter) -> float:
    striking_output = f.sig_strikes_per_min * f.sig_strike_accuracy
    grappling_output = f.takedowns_per_15 * f.takedown_accuracy + f.sub_attempts_per_15
    return min(1.0, (striking_output + grappling_output) / 5.0)


def defensive_quality(f: Fighter) -> float:
    total = f.wins + f.losses
    if total == 0:
        return 0.5
    times_finished = f.ko_losses + f.sub_losses
    durability = 1.0 - (times_finished / total)
    return (durability * 0.4 +
            f.sig_strike_defense * 0.3 +
            f.takedown_defense * 0.3)


STYLE_DURATION_MATRIX = {
    ("striker", "striker"):    -0.08,
    ("striker", "grappler"):   +0.02,
    ("striker", "wrestler"):   +0.05,
    ("striker", "balanced"):   -0.03,
    ("grappler", "grappler"):  +0.06,
    ("grappler", "wrestler"):  +0.04,
    ("grappler", "balanced"):  +0.01,
    ("wrestler", "wrestler"):  +0.08,
    ("wrestler", "balanced"):  +0.03,
    ("balanced", "balanced"):   0.00,
}


def style_duration_adj(a: Fighter, b: Fighter) -> float:
    key = (a.style, b.style)
    if key in STYLE_DURATION_MATRIX:
        return STYLE_DURATION_MATRIX[key]
    rev_key = (b.style, a.style)
    if rev_key in STYLE_DURATION_MATRIX:
        return STYLE_DURATION_MATRIX[rev_key]
    return 0.0


def duration_model(a: Fighter, b: Fighter, num_rounds: int) -> Dict[str, float]:
    """Signal 1: Independent fight-duration estimate."""
    a_dist = fighter_distance_rate(a)
    b_dist = fighter_distance_rate(b)
    base_distance = (a_dist + b_dist) / 2

    style_adj = style_duration_adj(a, b)
    combined_pace = (pace_score(a) + pace_score(b)) / 2
    pace_adj = -(combined_pace - 0.5) * 0.15
    combined_defense = (defensive_quality(a) + defensive_quality(b)) / 2
    defense_adj = (combined_defense - 0.5) * 0.20

    elo_gap = abs(a.elo - b.elo)
    mismatch_adj = -min(0.15, max(0, elo_gap - 100) / 2000)

    older_age = max(a.age, b.age)
    age_adj = 0.0
    if older_age >= 35:
        age_adj = -0.04 * (older_age - 34)
    age_adj = max(-0.12, age_adj)

    avg_form = (a.recent_form + b.recent_form) / 2
    form_adj = (avg_form - 0.5) * 0.08

    distance_prob = base_distance + style_adj + pace_adj + defense_adj + mismatch_adj + age_adj + form_adj
    distance_prob = max(0.10, min(0.90, distance_prob))

    if num_rounds == 5:
        distance_prob_for_line = min(0.90, distance_prob * 1.15)
    else:
        distance_prob_for_line = distance_prob

    return {
        "distance_prob": distance_prob,
        "over_prob": distance_prob_for_line,
        "under_prob": 1.0 - distance_prob_for_line,
        "base_distance": base_distance,
        "style_adj": style_adj,
        "pace_adj": pace_adj,
        "defense_adj": defense_adj,
        "mismatch_adj": mismatch_adj,
        "age_adj": age_adj,
        "form_adj": form_adj,
        "a_hist_dist": a_dist,
        "b_hist_dist": b_dist,
        "combined_pace": combined_pace,
        "combined_defense": combined_defense,
    }


# ============================================================
# SIGNAL 2: METHOD PROP CONSISTENCY CHECK
# ============================================================

def method_prop_analysis(a_key: str, b_key: str, book_odds: Dict, round_props: Dict) -> Dict:
    """
    Sum up all method prop implied probabilities for both fighters.
    Compare the implied finish rate to the over/under line.
    """
    a_odds = book_odds.get(a_key, {})
    b_odds = book_odds.get(b_key, {})

    if not a_odds or not b_odds:
        return {}
    if not isinstance(a_odds, dict) or not isinstance(b_odds, dict):
        return {}

    a_ko_imp = american_to_implied(a_odds.get("by_ko", 5000))
    a_sub_imp = american_to_implied(a_odds.get("by_sub", 5000))
    a_dec_imp = american_to_implied(a_odds.get("by_dec", 5000))

    b_ko_imp = american_to_implied(b_odds.get("by_ko", 5000))
    b_sub_imp = american_to_implied(b_odds.get("by_sub", 5000))
    b_dec_imp = american_to_implied(b_odds.get("by_dec", 5000))

    total_implied = a_ko_imp + a_sub_imp + a_dec_imp + b_ko_imp + b_sub_imp + b_dec_imp
    overround = total_implied - 1.0

    if total_implied > 0:
        a_ko_true = a_ko_imp / total_implied
        a_sub_true = a_sub_imp / total_implied
        a_dec_true = a_dec_imp / total_implied
        b_ko_true = b_ko_imp / total_implied
        b_sub_true = b_sub_imp / total_implied
        b_dec_true = b_dec_imp / total_implied
    else:
        return {}

    finish_from_methods = a_ko_true + a_sub_true + b_ko_true + b_sub_true
    distance_from_methods = a_dec_true + b_dec_true

    rp = round_props.get((a_key, b_key), {})
    if not rp:
        return {}

    over_imp = american_to_implied(rp["over"])
    under_imp = american_to_implied(rp["under"])
    over_true, under_true = remove_vig(over_imp, under_imp)

    line = rp["line"]

    if line == 2.5:
        distance_from_ou = over_true
        finish_from_ou = under_true
    elif line == 1.5:
        distance_from_ou = None
        finish_from_ou = None
    elif line == 3.5:
        distance_from_ou = over_true * 0.85
        finish_from_ou = under_true
    else:
        distance_from_ou = over_true
        finish_from_ou = under_true

    gap = None
    if distance_from_ou is not None:
        gap = distance_from_methods - distance_from_ou

    return {
        "a_key": a_key, "b_key": b_key,
        "method_probs": {
            f"{a_key}_ko": a_ko_true, f"{a_key}_sub": a_sub_true, f"{a_key}_dec": a_dec_true,
            f"{b_key}_ko": b_ko_true, f"{b_key}_sub": b_sub_true, f"{b_key}_dec": b_dec_true,
        },
        "total_implied": total_implied,
        "overround": overround,
        "finish_from_methods": finish_from_methods,
        "distance_from_methods": distance_from_methods,
        "over_true": over_true,
        "under_true": under_true,
        "distance_from_ou": distance_from_ou,
        "finish_from_ou": finish_from_ou,
        "consistency_gap": gap,
        "line": line,
    }


# ============================================================
# SIGNAL 3: MONTE CARLO SIMULATION OUTPUT
# ============================================================

def simulation_signal(a_key: str, b_key: str, nr: int, round_props: Dict) -> Dict:
    a, b = fighters[a_key], fighters[b_key]
    res = simulate(a, b, nr, ns=20000)
    ana = analyze(res, a.name, b.name, nr)

    cumulative_finish = 0.0
    round_cum = {}
    for rd in range(1, nr + 1):
        cumulative_finish += ana.get(f"r{rd}_finish", 0)
        round_cum[rd] = cumulative_finish

    rp = round_props.get((a_key, b_key), {})
    line = rp.get("line", 2.5)

    if line == 1.5:
        sim_under = ana.get("r1_finish", 0)
        sim_over = 1.0 - sim_under
    elif line == 2.5:
        sim_under = round_cum.get(2, 0)
        sim_over = 1.0 - sim_under
    elif line == 3.5:
        sim_under = round_cum.get(3, 0)
        sim_over = 1.0 - sim_under
    else:
        sim_under = ana["finish"]
        sim_over = ana["distance"]

    return {
        "distance_prob": ana["distance"],
        "finish_prob": ana["finish"],
        "sim_over": sim_over,
        "sim_under": sim_under,
        "per_round_finish": {rd: ana.get(f"r{rd}_finish", 0) for rd in range(1, nr + 1)},
        "round_cumulative": round_cum,
        "line": line,
        "full_analysis": ana,
    }


# ============================================================
# EDGE DATA STRUCTURES
# ============================================================

@dataclass
class Edge:
    fight: str
    side: str
    line: float
    book_odds_american: int
    book_implied: float
    signal_duration: float
    signal_methods: float
    signal_sim: float
    consensus_prob: float
    edge: float
    ev_per_dollar: float
    kelly_pct: float
    confidence: str
    agreement: int
    explanation: str
    components: Dict = field(default_factory=dict)


@dataclass
class CorrelatedParlay:
    fight: str
    thesis: str
    legs: List[str]
    leg_odds: List[int]        # american odds per leg
    sgp_odds: Optional[int]    # if user entered an SGP price
    combined_prob: float
    naive_payout: float        # what a naive parlay pays
    sgp_payout: Optional[float]  # what the SGP actually pays
    ev_naive: float
    ev_sgp: Optional[float]
    explanation: str


# ============================================================
# THE EDGE FINDER
# ============================================================

def find_edges(book_odds: Dict, round_props: Dict) -> List[Edge]:
    """Run all 3 signals for each fight and find O/U mispricings."""
    for k, f in fighters.items():
        f.elo = calculate_elo(f)

    edges = []

    for a_key, b_key, nr, is_main in matchups:
        a, b = fighters[a_key], fighters[b_key]
        fight_name = f"{a.name} vs {b.name}"

        rp = round_props.get((a_key, b_key))
        if not rp:
            continue

        line = rp["line"]
        over_american = rp["over"]
        under_american = rp["under"]
        over_imp = american_to_implied(over_american)
        under_imp = american_to_implied(under_american)
        over_true_book, under_true_book = remove_vig(over_imp, under_imp)

        # --- Signal 1: Duration Model (uses fighter stats) ---
        dur = duration_model(a, b, nr)
        if line == 1.5:
            s1_over = dur["distance_prob"] + (1.0 - dur["distance_prob"]) * 0.55
            s1_under = 1.0 - s1_over
        elif line == 2.5:
            s1_over = dur["over_prob"]
            s1_under = dur["under_prob"]
        elif line == 3.5:
            s1_over = dur["over_prob"]
            s1_under = dur["under_prob"]
        else:
            s1_over = dur["over_prob"]
            s1_under = dur["under_prob"]

        # --- Signal 2: Method Props (uses BOOK pricing — independent source) ---
        mpa = method_prop_analysis(a_key, b_key, book_odds, round_props)
        if mpa and mpa.get("distance_from_ou") is not None:
            s2_over = mpa["distance_from_methods"]
            s2_under = mpa["finish_from_methods"]
            if line == 1.5:
                s2_under = mpa["finish_from_methods"] * 0.40
                s2_over = 1.0 - s2_under
            s2_is_real = True
        else:
            s2_over = 0.5
            s2_under = 0.5
            s2_is_real = False

        # --- Signal 3: Monte Carlo Sim (uses fighter stats — CORRELATED with S1) ---
        sim = simulation_signal(a_key, b_key, nr, round_props)
        s3_over = sim["sim_over"]
        s3_under = sim["sim_under"]

        # --- Signal 4: Fight IQ — Matchup Intelligence (INDEPENDENT) ---
        # Reasons from matchup archetypes and vulnerability mapping,
        # NOT from the same stat formulas as S1/S3. Genuinely independent.
        fp = analyze_matchup(a_key, b_key, nr)
        if line == 1.5:
            s4_under = fp.p_finish_r1
            s4_over = 1.0 - s4_under
        elif line == 2.5:
            s4_under = fp.p_finish_r1 + fp.p_finish_r2
            s4_over = 1.0 - s4_under
        elif line == 3.5:
            s4_under = fp.p_finish_r1 + fp.p_finish_r2 + fp.p_finish_r3
            s4_over = 1.0 - s4_under
        else:
            s4_under = fp.p_finish
            s4_over = fp.p_distance
        s4_confidence = fp.read_confidence  # 0-1, how clear-cut the matchup read is

        # --- Consensus: Correlation-Aware Blending ---
        #
        # We have 4 signals but only 3 INDEPENDENT voices:
        #   Voice 1 (OUR MODEL): S1 + S3 merged — they share inputs
        #   Voice 2 (BOOK):      S2 (method props from the book's own pricing)
        #   Voice 3 (FIGHT IQ):  S4 (matchup archetype reasoning — independent)
        #   Prior:               The book's O/U line itself
        #
        # The book's O/U line is the prior we shrink toward.
        # Each independent voice that disagrees with the prior pulls us away.
        #
        model_over = (s1_over + s3_over) / 2
        model_under = (s1_under + s3_under) / 2

        model_spread = abs(s1_over - s3_over)
        model_confidence = max(0.3, 1.0 - model_spread * 2)

        # Weight allocation:
        #   Book prior: 25% (respect market efficiency)
        #   Our model (S1+S3 merged): 30% * model_confidence
        #   Method props (S2): 15% * reliability (independent but noisy)
        #   Fight IQ (S4): 30% * s4_confidence (independent, matchup-specific)
        s2_reliability = 0.7 if line == 2.5 else (0.5 if line == 3.5 else 0.3)
        s2_reliability *= (1.0 if s2_is_real else 0.0)

        w_book = 0.25
        w_model = 0.30 * model_confidence
        w_s2 = 0.15 * s2_reliability
        w_s4 = 0.30 * s4_confidence

        total_w = w_book + w_model + w_s2 + w_s4
        w_book /= total_w
        w_model /= total_w
        w_s2 /= total_w
        w_s4 /= total_w

        consensus_over = w_book * over_true_book + w_model * model_over + w_s2 * s2_over + w_s4 * s4_over
        consensus_under = w_book * under_true_book + w_model * model_under + w_s2 * s2_under + w_s4 * s4_under

        for side, book_american, book_imp_raw, consensus_p, signals in [
            ("over",  over_american,  over_true_book,  consensus_over,  [s1_over, s2_over, s3_over, s4_over]),
            ("under", under_american, under_true_book, consensus_under, [s1_under, s2_under, s3_under, s4_under]),
        ]:
            edge_val = consensus_p - book_imp_raw
            # Raw agreement count (for display)
            agreement = sum(1 for s in signals if s > book_imp_raw)

            if edge_val <= 0.02:
                continue

            dec_odds = american_to_decimal(book_american)
            ev = ev_per_dollar(consensus_p, dec_odds)
            k = kelly(consensus_p, dec_odds)

            # Confidence scoring — correlation-aware
            #
            # Three independent voices:
            #   Voice 1 (model): S1+S3 — correlated, counts as ONE voice
            #   Voice 2 (book):  S2 — method props, independent
            #   Voice 3 (IQ):   S4 — matchup archetype, independent
            #
            # "high" = 2+ independent voices agree + meaningful edge
            # "medium" = model + 1 independent agrees, OR strong edge
            # "low" = only model says edge
            model_agrees = (signals[0] > book_imp_raw) and (signals[2] > book_imp_raw)
            s2_agrees = signals[1] > book_imp_raw
            s4_agrees = signals[3] > book_imp_raw
            s2_meaningful = s2_is_real and (line != 1.5)

            independent_agree = sum([
                s2_meaningful and s2_agrees,
                s4_agrees and s4_confidence > 0.4,
            ])

            if model_agrees and independent_agree >= 2 and edge_val > 0.05:
                confidence = "high"
            elif model_agrees and independent_agree >= 1 and edge_val > 0.05:
                confidence = "high"
            elif model_agrees and edge_val > 0.05:
                confidence = "medium"
            elif independent_agree >= 1 and edge_val > 0.04:
                confidence = "medium"
            elif edge_val > 0.03:
                confidence = "low"
            else:
                confidence = "low"

            explanations = []
            if abs(signals[0] - book_imp_raw) > 0.03:
                dur_note = (f"Duration model: {a.name} goes distance "
                            f"{dur['a_hist_dist']:.0%}, {b.name} {dur['b_hist_dist']:.0%}")
                if dur["mismatch_adj"] < -0.05:
                    dur_note += f" | Elo gap ({abs(a.elo - b.elo):.0f}) pushes toward finish"
                if dur["style_adj"] > 0.03:
                    dur_note += f" | {a.style} vs {b.style} favors distance"
                elif dur["style_adj"] < -0.03:
                    dur_note += f" | {a.style} vs {b.style} favors finish"
                explanations.append(dur_note)

            if mpa and mpa.get("consistency_gap") is not None and abs(mpa["consistency_gap"]) > 0.05:
                gap_dir = "more" if mpa["consistency_gap"] > 0 else "less"
                explanations.append(
                    f"Method props imply {gap_dir} distance ({mpa['distance_from_methods']:.0%}) "
                    f"than O/U implies ({mpa['distance_from_ou']:.0%}) -- {abs(mpa['consistency_gap']):.0%} gap"
                )

            sim_edge = (signals[2] - book_imp_raw)
            if abs(sim_edge) > 0.03:
                explanations.append(
                    f"20K sim: {side} hits {signals[2]:.1%} vs book's {book_imp_raw:.1%}"
                )

            explanation = " || ".join(explanations) if explanations else "Marginal edge across signals"

            edges.append(Edge(
                fight=fight_name, side=side, line=line,
                book_odds_american=book_american, book_implied=book_imp_raw,
                signal_duration=signals[0], signal_methods=signals[1], signal_sim=signals[2],
                consensus_prob=consensus_p, edge=edge_val,
                ev_per_dollar=ev, kelly_pct=k, confidence=confidence,
                agreement=agreement, explanation=explanation,
                components={"duration_model": dur, "method_analysis": mpa,
                            "simulation": {"sim_over": sim["sim_over"], "sim_under": sim["sim_under"],
                                           "per_round": sim["per_round_finish"]},
                            "fight_iq_signal": signals[3],
                            "fight_iq_archetype": fp.archetype_desc,
                            "fight_iq_confidence": s4_confidence},
            ))

    tier_order = {"high": 0, "medium": 1, "low": 2}
    edges.sort(key=lambda e: (tier_order[e.confidence], -e.edge))
    return edges


# ============================================================
# CORRELATED PARLAY FINDER
# ============================================================

def find_correlated_parlays(book_odds: Dict, round_props: Dict) -> List[CorrelatedParlay]:
    """
    Find same-fight parlays where legs are positively correlated.
    Books price legs independently — correlation = free edge.
    """
    for k, f in fighters.items():
        f.elo = calculate_elo(f)

    parlays = []

    for a_key, b_key, nr, is_main in matchups:
        a, b = fighters[a_key], fighters[b_key]
        fight_name = f"{a.name} vs {b.name}"

        rp = round_props.get((a_key, b_key))
        if not rp:
            continue

        res = simulate(a, b, nr, ns=20000)
        total = len(res)
        ana = analyze(res, a.name, b.name, nr)
        line = rp["line"]

        for fkey, fn in [(a_key, a.name), (b_key, b.name)]:
            odds = book_odds.get(fkey, {})
            if not isinstance(odds, dict):
                continue

            # --- Correlation 1: Fighter wins by KO/SUB + Under ---
            for method, method_label, odds_key in [
                ("ko", "KO/TKO", "by_ko"),
                ("sub", "SUB", "by_sub"),
            ]:
                if odds_key not in odds:
                    continue
                method_prob = ana.get(f"{fn}_{method}", 0)
                if method_prob < 0.05:
                    continue

                method_imp = american_to_implied(odds[odds_key])
                method_dec = american_to_decimal(odds[odds_key])

                if line == 2.5:
                    method_and_under = sum(1 for w, m, r, dd in res
                                           if w == fn and m == method and r <= 2) / total
                elif line == 1.5:
                    method_and_under = sum(1 for w, m, r, dd in res
                                           if w == fn and m == method and r <= 1) / total
                elif line == 3.5:
                    method_and_under = sum(1 for w, m, r, dd in res
                                           if w == fn and m == method and r <= 3) / total
                else:
                    continue

                if method_and_under < 0.03:
                    continue

                under_imp = american_to_implied(rp["under"])
                under_dec = american_to_decimal(rp["under"])
                naive_joint = method_imp * under_imp
                true_joint = method_and_under
                naive_payout = method_dec * under_dec
                naive_ev = true_joint * naive_payout - 1.0

                if naive_ev > 0.05:
                    parlays.append(CorrelatedParlay(
                        fight=fight_name,
                        thesis=f"{fn} by {method_label} + Under {line}",
                        legs=[f"{fn} by {method_label} ({odds[odds_key]:+d})",
                              f"Under {line} ({rp['under']:+d})"],
                        leg_odds=[odds[odds_key], rp["under"]],
                        sgp_odds=None, combined_prob=true_joint,
                        naive_payout=naive_payout, sgp_payout=None,
                        ev_naive=naive_ev, ev_sgp=None,
                        explanation=(
                            f"Correlated: {method_label} finish = under hits. "
                            f"True joint P={true_joint:.1%}, books price as {naive_joint:.1%}. "
                            f"Naive parlay {naive_payout:.1f}x, EV: {naive_ev:+.1%}."
                        ),
                    ))

            # --- Correlation 2: Fighter wins by DEC + Over ---
            if "by_dec" in odds:
                dec_prob = ana.get(f"{fn}_dec", 0)
                if dec_prob > 0.05:
                    dec_imp = american_to_implied(odds["by_dec"])
                    dec_dec_odds = american_to_decimal(odds["by_dec"])
                    over_imp = american_to_implied(rp["over"])
                    over_dec_odds = american_to_decimal(rp["over"])

                    true_joint = dec_prob
                    naive_joint = dec_imp * over_imp
                    naive_payout = dec_dec_odds * over_dec_odds
                    naive_ev = true_joint * naive_payout - 1.0

                    if naive_ev > 0.05:
                        parlays.append(CorrelatedParlay(
                            fight=fight_name,
                            thesis=f"{fn} by DEC + Over {line}",
                            legs=[f"{fn} by DEC ({odds['by_dec']:+d})",
                                  f"Over {line} ({rp['over']:+d})"],
                            leg_odds=[odds["by_dec"], rp["over"]],
                            sgp_odds=None, combined_prob=true_joint,
                            naive_payout=naive_payout, sgp_payout=None,
                            ev_naive=naive_ev, ev_sgp=None,
                            explanation=(
                                f"DEC = over (near-perfect correlation). "
                                f"True P={true_joint:.1%}, books price as {naive_joint:.1%}. "
                                f"Naive {naive_payout:.1f}x, EV: {naive_ev:+.1%}."
                            ),
                        ))

            # --- Correlation 3: Underdog ML + Over ---
            ml_prob = ana.get(f"{fn}_ml", 0)
            ml_imp = american_to_implied(odds["ml"])
            if ml_imp < 0.45 and ml_prob > ml_imp:
                dog_and_over = sum(1 for w, m, r, dd in res if w == fn and dd) / total
                if line == 2.5:
                    dog_and_over += sum(1 for w, m, r, dd in res
                                        if w == fn and not dd and r >= 3) / total
                elif line == 3.5:
                    dog_and_over += sum(1 for w, m, r, dd in res
                                        if w == fn and not dd and r >= 4) / total

                over_imp = american_to_implied(rp["over"])
                naive_joint = ml_imp * over_imp
                naive_payout = american_to_decimal(odds["ml"]) * american_to_decimal(rp["over"])
                naive_ev = dog_and_over * naive_payout - 1.0

                if naive_ev > 0.05 and dog_and_over > naive_joint * 1.1:
                    parlays.append(CorrelatedParlay(
                        fight=fight_name,
                        thesis=f"{fn} (dog) ML + Over {line}",
                        legs=[f"{fn} ML ({odds['ml']:+d})",
                              f"Over {line} ({rp['over']:+d})"],
                        leg_odds=[odds["ml"], rp["over"]],
                        sgp_odds=None, combined_prob=dog_and_over,
                        naive_payout=naive_payout, sgp_payout=None,
                        ev_naive=naive_ev, ev_sgp=None,
                        explanation=(
                            f"Underdog survival: if {fn} wins, likely by decision. "
                            f"Joint P={dog_and_over:.1%} vs naive {naive_joint:.1%}. "
                            f"EV: {naive_ev:+.1%}."
                        ),
                    ))

    parlays.sort(key=lambda p: -p.ev_naive)
    return parlays


# ============================================================
# SGP EVALUATOR — Check if your book's SGP price is still +EV
# ============================================================

def evaluate_sgp(parlay: CorrelatedParlay, sgp_american: int) -> Dict:
    """
    Your book gave you an SGP price (like +166 for Jandiroba DEC + Over 2.5).
    Is it still +EV after the book's correlation discount?
    """
    sgp_decimal = american_to_decimal(sgp_american)
    sgp_implied = american_to_implied(sgp_american)
    ev = parlay.combined_prob * sgp_decimal - 1.0
    k = kelly(parlay.combined_prob, sgp_decimal)

    # How much did the book discount vs naive?
    naive_implied = 1.0 / parlay.naive_payout if parlay.naive_payout > 0 else 0
    discount = sgp_implied - naive_implied

    return {
        "sgp_american": sgp_american,
        "sgp_decimal": sgp_decimal,
        "sgp_implied": sgp_implied,
        "naive_payout": parlay.naive_payout,
        "naive_implied": naive_implied,
        "book_discount": discount,
        "model_prob": parlay.combined_prob,
        "ev": ev,
        "kelly": k,
        "is_plus_ev": ev > 0,
        "verdict": (
            f"SGP at {sgp_american:+d} (implied {sgp_implied:.1%}) vs "
            f"model {parlay.combined_prob:.1%} = "
            f"{'STILL +EV ({:+.1%})'.format(ev) if ev > 0 else 'NEGATIVE EV ({:+.1%})'.format(ev)}. "
            f"Book charged {discount:.1%} correlation tax."
        ),
    }


# ============================================================
# MAIN OUTPUT
# ============================================================

def run_analysis(book_odds: Dict, round_props: Dict):
    """Run the full analysis with given odds."""
    print("\n" + "=" * 100)
    print("  UFC EDGE FINDER — Over/Under Mispricing Detection")
    print("  3 Independent Signals | Correlated Parlays | Kelly Sizing")
    print("=" * 100)

    # ── SECTION 1: Over/Under Edges ──
    print(f"\n{'━' * 100}")
    print(f"  OVER/UNDER EDGES — Ranked by Confidence + Edge Size")
    print(f"{'━' * 100}\n")

    edges = find_edges(book_odds, round_props)

    if not edges:
        print("  No edges found. Lines look efficient today.")
    else:
        for tier in ["high", "medium", "low"]:
            tier_edges = [e for e in edges if e.confidence == tier]
            if not tier_edges:
                continue

            print(f"\n  -- {tier.upper()} CONFIDENCE --\n")
            print(f"  {'Fight':<38} {'Side':<10} {'Book':>6} {'Sim':>6} {'Meth':>6} {'IQ':>6} {'Cons':>6} {'Edge':>7} {'EV/$':>7} {'Kelly':>7}")
            print(f"  {'─' * 107}")

            for e in tier_edges:
                # Show S3 (sim), S2 (methods), S4 (fight IQ) — skip S1 since it's correlated with S3
                s_sim = (e.signal_duration + e.signal_sim) / 2  # merged model signal
                s_meth = e.signal_methods
                s_iq = e.components.get("fight_iq_signal", 0.5)
                print(f"  {e.fight:<38} {e.side} {e.line:<4} "
                      f"{e.book_implied:>5.0%} "
                      f"{s_sim:>5.0%} "
                      f"{s_meth:>5.0%} "
                      f"{s_iq:>5.0%} "
                      f"{e.consensus_prob:>5.0%} "
                      f"{e.edge:>+6.1%} "
                      f"{e.ev_per_dollar:>+6.1%} "
                      f"{e.kelly_pct:>6.1%} ")

            if tier == "high":
                print(f"\n  WHY THESE HIT:")
                for e in tier_edges:
                    print(f"\n  * {e.fight} -- {e.side.upper()} {e.line} ({e.book_odds_american:+d})")
                    for part in e.explanation.split(" || "):
                        print(f"    {part}")

    # ── SECTION 2: Consistency Check ──
    print(f"\n\n{'━' * 100}")
    print(f"  METHOD PROP vs OVER/UNDER CONSISTENCY CHECK")
    print(f"{'━' * 100}\n")

    print(f"  {'Fight':<38} {'Meth->Dist':>10} {'O/U->Dist':>10} {'Gap':>8} {'Verdict':<25}")
    print(f"  {'─' * 95}")

    for a_key, b_key, nr, _ in matchups:
        a, b = fighters[a_key], fighters[b_key]
        mpa = method_prop_analysis(a_key, b_key, book_odds, round_props)
        if not mpa or mpa.get("distance_from_ou") is None:
            continue

        gap = mpa["consistency_gap"]
        fight_name = f"{a.name} vs {b.name}"

        if abs(gap) > 0.10:
            verdict = "MISPRICED -- big gap"
        elif abs(gap) > 0.05:
            verdict = "Slight inconsistency"
        else:
            verdict = "Consistent"

        print(f"  {fight_name:<38} "
              f"{mpa['distance_from_methods']:>9.0%} "
              f"{mpa['distance_from_ou']:>9.0%} "
              f"{gap:>+7.0%}  "
              f"{verdict}")

    # ── SECTION 3: Correlated Parlays ──
    print(f"\n\n{'━' * 100}")
    print(f"  CORRELATED PARLAYS — Same-Fight Legs with Hidden Edge")
    print(f"  NOTE: EV assumes naive parlay (legs priced independently).")
    print(f"  If your book offers SGPs, use the SGP evaluator to check the real price.")
    print(f"{'━' * 100}")

    corr_parlays = find_correlated_parlays(book_odds, round_props)

    if not corr_parlays:
        print("\n  No high-EV correlated parlays found.")
    else:
        for rank, cp in enumerate(corr_parlays[:12], 1):
            print(f"\n  #{rank}  {cp.fight}")
            print(f"  Thesis: {cp.thesis}")
            for leg in cp.legs:
                print(f"    Leg: {leg}")
            print(f"  True P: {cp.combined_prob:.1%} | Naive payout: {cp.naive_payout:.1f}x | "
                  f"Naive EV: {cp.ev_naive:+.1%} | $10 -> ${10 * cp.naive_payout:.0f}")
            print(f"  {cp.explanation}")

    # ── SECTION 4: Cross-Fight O/U Parlays ──
    print(f"\n\n{'━' * 100}")
    print(f"  BEST CROSS-FIGHT O/U PARLAY COMBOS")
    print(f"{'━' * 100}")

    pos_edges = [e for e in edges if e.edge > 0.03]

    if len(pos_edges) >= 2:
        import itertools

        for n_legs in [2, 3]:
            if len(pos_edges) < n_legs:
                continue

            print(f"\n  -- {n_legs}-Leg O/U Parlays --\n")

            combos = []
            for combo in itertools.combinations(pos_edges, n_legs):
                fights = [e.fight for e in combo]
                if len(set(fights)) != len(fights):
                    continue
                cp = 1.0
                co = 1.0
                for e in combo:
                    cp *= e.consensus_prob
                    co *= american_to_decimal(e.book_odds_american)
                ev = cp * co - 1.0
                combos.append((ev, cp, co, combo))

            combos.sort(key=lambda x: x[0], reverse=True)

            for rank, (ev, prob, odds, combo) in enumerate(combos[:3], 1):
                print(f"  #{rank}  EV: {ev:+.1%} | Hit Rate: {prob:.1%} | Payout: {odds:.1f}x | $10 -> ${10 * odds:.0f}")
                for e in combo:
                    print(f"    {e.fight}: {e.side.upper()} {e.line} ({e.book_odds_american:+d}) "
                          f"-- Edge: {e.edge:+.1%} [{e.confidence}]")
                print()

    # ── SECTION 5: Action Sheet ──
    print(f"\n{'=' * 100}")
    print(f"  QUICK ACTION SHEET")
    print(f"{'=' * 100}\n")

    high_edges = [e for e in edges if e.confidence == "high"]
    if high_edges:
        print(f"  HIGH CONFIDENCE PLAYS:")
        for e in high_edges:
            fair_line = implied_to_american(e.consensus_prob)
            print(f"    {e.side.upper()} {e.line} in {e.fight}")
            print(f"      Book: {e.book_odds_american:+d} | Fair: {fair_line:+d} | Edge: {e.edge:+.1%} | Kelly: {e.kelly_pct:.1%}")
        print()

    if corr_parlays:
        print(f"  TOP CORRELATED PARLAY (naive pricing):")
        cp = corr_parlays[0]
        print(f"    {cp.thesis}")
        for leg in cp.legs:
            print(f"      {leg}")
        print(f"    Naive: $10 -> ${10 * cp.naive_payout:.0f} | EV: {cp.ev_naive:+.1%}")
        print(f"\n    >> Check your book's SGP price for this. If they offer it,")
        print(f"       re-run with --sgp to evaluate whether it's still +EV.")
        print()

    total_edges = len([e for e in edges if e.edge > 0.03])
    high_count = len(high_edges)
    corr_count = len([cp for cp in corr_parlays if cp.ev_naive > 0.10])
    print(f"  Total O/U edges found: {total_edges}")
    print(f"  High confidence: {high_count}")
    print(f"  High-EV correlated parlays: {corr_count}")

    print(f"\n{'=' * 100}")
    print(f"  DISCLAIMER: Entertainment only. Gamble responsibly.")
    print(f"  Three signals agree > two signals > one.")
    print(f"{'=' * 100}\n")

    return edges, corr_parlays


# ============================================================
# SGP CHECKER MODE
# ============================================================

def sgp_checker(corr_parlays: List[CorrelatedParlay]):
    """Interactive: user enters their book's SGP price for each parlay."""
    print(f"\n{'━' * 100}")
    print(f"  SGP PRICE CHECKER")
    print(f"  Enter your book's actual SGP odds. We'll tell you if it's still +EV.")
    print(f"{'━' * 100}\n")

    for i, cp in enumerate(corr_parlays[:10], 1):
        print(f"  #{i}  {cp.thesis}")
        for leg in cp.legs:
            print(f"    {leg}")
        print(f"    Model prob: {cp.combined_prob:.1%} | Naive parlay: {cp.naive_payout:.1f}x")

        raw = input(f"    Your book's SGP odds (American, or 's' to skip): ").strip()
        if raw.lower() in ("s", "skip", ""):
            print()
            continue

        try:
            sgp_american = int(raw.replace("+", ""))
        except ValueError:
            print("    Invalid odds, skipping.\n")
            continue

        result = evaluate_sgp(cp, sgp_american)
        print(f"\n    VERDICT: {result['verdict']}")
        if result["is_plus_ev"]:
            print(f"    Payout: {result['sgp_decimal']:.2f}x | EV: {result['ev']:+.1%} | Kelly: {result['kelly']:.1%}")
            print(f"    On $100 bankroll, quarter-Kelly = ${100 * result['kelly'] / 4:.2f}")
        else:
            print(f"    Book's correlation tax killed the edge. Pass on this one.")
        print()


# ============================================================
# MAIN
# ============================================================

def main():
    live_mode = "--live" in sys.argv
    use_defaults = "--defaults" in sys.argv
    manual_mode = "--manual" in sys.argv
    update_mode = "--update" in sys.argv
    sgp_mode = "--sgp" in sys.argv

    # Parse optional date: --date 2026-04-04
    event_date = None
    for i, arg in enumerate(sys.argv):
        if arg == "--date" and i + 1 < len(sys.argv):
            event_date = sys.argv[i + 1]

    if live_mode:
        # Pull real-time odds from The Odds API
        if not event_date:
            event_date = "2026-04-04"  # default to Vegas 115
        book_odds, round_props = fetch_live_odds(event_date=event_date)

    elif use_defaults:
        print("\n  Running with hardcoded default odds...")
        book_odds = deepcopy(DEFAULT_BOOK_ODDS)
        round_props = deepcopy(DEFAULT_ROUND_PROPS)

    elif update_mode:
        book_odds = deepcopy(DEFAULT_BOOK_ODDS)
        round_props = deepcopy(DEFAULT_ROUND_PROPS)
        saved_book, saved_rp_raw = load_saved_odds()
        if saved_book:
            saved_rp = deserialize_round_props(saved_rp_raw)
            for k, v in saved_book.items():
                if k in book_odds and isinstance(v, dict):
                    book_odds[k].update(v)
            for k, v in saved_rp.items():
                if k in round_props:
                    round_props[k].update(v)
        book_odds, round_props = quick_update_single_fight(book_odds, round_props)

    elif manual_mode:
        book_odds, round_props = collect_live_odds()

    else:
        # Default: show menu
        print("\n  Modes:")
        print("    1. LIVE — Pull real odds from The Odds API (recommended)")
        print("    2. Manual — Enter your book's odds by hand")
        print("    3. Quick update — Pick specific fights to update")
        print("    4. Defaults — Use hardcoded odds (fast, no API call)")
        print()
        try:
            choice = input("  Choose [1/2/3/4]: ").strip()
        except EOFError:
            choice = "4"

        if choice == "4":
            book_odds = deepcopy(DEFAULT_BOOK_ODDS)
            round_props = deepcopy(DEFAULT_ROUND_PROPS)
        elif choice == "3":
            book_odds = deepcopy(DEFAULT_BOOK_ODDS)
            round_props = deepcopy(DEFAULT_ROUND_PROPS)
            saved_book, saved_rp_raw = load_saved_odds()
            if saved_book:
                saved_rp = deserialize_round_props(saved_rp_raw)
                for k, v in saved_book.items():
                    if k in book_odds and isinstance(v, dict):
                        book_odds[k].update(v)
                for k, v in saved_rp.items():
                    if k in round_props:
                        round_props[k].update(v)
            book_odds, round_props = quick_update_single_fight(book_odds, round_props)
        elif choice == "2":
            book_odds, round_props = collect_live_odds()
        else:
            # Default to live
            if not event_date:
                event_date = "2026-04-04"
            book_odds, round_props = fetch_live_odds(event_date=event_date)

    edges, corr_parlays = run_analysis(book_odds, round_props)

    # Offer SGP check (skip prompt in non-interactive modes)
    if sgp_mode and corr_parlays:
        sgp_checker(corr_parlays)
    elif corr_parlays and not use_defaults and not live_mode:
        try:
            check_sgp = input("\n  Check your book's SGP prices? (y/n) [n]: ").strip().lower()
            if check_sgp == "y":
                sgp_checker(corr_parlays)
        except EOFError:
            pass


if __name__ == "__main__":
    main()
