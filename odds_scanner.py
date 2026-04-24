"""
UFC Odds Scanner — Find Mispriced Lines Hiding in Plain Sight

Pulls live odds from The Odds API across multiple sportsbooks,
checks for:
  1. Cross-market inconsistencies (method props vs moneyline vs distance)
  2. Cross-book divergences (same prop, different book = someone's wrong)
  3. Model vs market gaps (your sim says X, books say Y)

Usage:
  # Set your API key (free at https://the-odds-api.com)
  export ODDS_API_KEY="your_key_here"

  # Scan upcoming UFC odds
  python odds_scanner.py

  # Or import and use with your engine
  from odds_scanner import OddsScanner
  scanner = OddsScanner()
  scanner.scan_event("UFC Vegas 115")
"""

import os
import json
import math
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

def _load_env():
    """Load .env file from project root."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

_load_env()

API_KEY = os.environ.get("ODDS_API_KEY", "")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "mma_mixed_martial_arts"

# Books to compare — add/remove as you like
PREFERRED_BOOKS = [
    "draftkings", "fanduel", "betmgm", "caesars",
    "bovada", "betonlineag", "mybookieag", "betrivers",
    "pointsbetus", "unibet_us", "williamhill_us",
]

# Thresholds
EDGE_THRESHOLD = 0.05        # 5% edge = flag it
CROSS_BOOK_SPREAD = 0.08     # 8% implied prob spread across books = divergence
CONSISTENCY_GAP = 0.10        # 10% gap between related props = inconsistency


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class OddsLine:
    book: str
    fighter: str
    market: str          # "ml", "by_ko", "by_sub", "by_dec", "over", "under"
    american: int
    implied_prob: float
    decimal_odds: float
    timestamp: str = ""


@dataclass
class FightOdds:
    fighter_a: str
    fighter_b: str
    commence_time: str
    lines: List[OddsLine] = field(default_factory=list)

    def get_lines(self, market: str, fighter: str = None) -> List[OddsLine]:
        return [l for l in self.lines
                if l.market == market and (fighter is None or l.fighter == fighter)]

    def best_odds(self, market: str, fighter: str) -> Optional[OddsLine]:
        lines = self.get_lines(market, fighter)
        if not lines:
            return None
        return max(lines, key=lambda l: l.decimal_odds)

    def consensus_implied(self, market: str, fighter: str) -> Optional[float]:
        """Average implied probability across all books (vig-included)."""
        lines = self.get_lines(market, fighter)
        if not lines:
            return None
        return sum(l.implied_prob for l in lines) / len(lines)


@dataclass
class Mispricing:
    fight: str
    type: str            # "cross_book", "consistency", "model_vs_market"
    description: str
    edge: float          # signed edge (positive = value)
    confidence: str      # "high", "medium", "low"
    details: Dict = field(default_factory=dict)


# ============================================================
# ODDS MATH
# ============================================================

def american_to_implied(ml: int) -> float:
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)

def american_to_decimal(ml: int) -> float:
    if ml < 0:
        return 1 + (100 / abs(ml))
    return 1 + (ml / 100)

def decimal_to_american(dec: float) -> int:
    if dec >= 2.0:
        return round((dec - 1) * 100)
    return round(-100 / (dec - 1))

def remove_vig(prob_a: float, prob_b: float) -> Tuple[float, float]:
    """Remove overround to get true probabilities."""
    total = prob_a + prob_b
    return prob_a / total, prob_b / total

def implied_to_american(prob: float) -> int:
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    return round(100 * (1 - prob) / prob)


# ============================================================
# API CLIENT
# ============================================================

class OddsClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError(
                "No API key. Get one free at https://the-odds-api.com\n"
                "Then: export ODDS_API_KEY='your_key'"
            )
        self.remaining_requests = None

    def _get(self, endpoint: str, params: dict = None) -> dict:
        params = params or {}
        params["apiKey"] = self.api_key
        resp = requests.get(f"{BASE_URL}/{endpoint}", params=params)
        # Track quota
        self.remaining_requests = resp.headers.get("x-requests-remaining")
        resp.raise_for_status()
        return resp.json()

    def get_upcoming_events(self) -> List[dict]:
        """Get upcoming MMA events."""
        return self._get(f"sports/{SPORT}/events")

    def get_odds(self, markets: str = "h2h", regions: str = "us",
                 event_id: str = None) -> List[dict]:
        """
        Fetch odds for upcoming MMA fights.
        markets: comma-separated — h2h, spreads, totals, outrights
        """
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
        }
        if event_id:
            return self._get(f"sports/{SPORT}/events/{event_id}/odds", params)
        return self._get(f"sports/{SPORT}/odds", params)

    def get_event_odds_all_markets(self, event_id: str) -> dict:
        """Fetch all available markets for a specific event."""
        return self.get_odds(
            markets="h2h,totals",
            event_id=event_id,
        )


# ============================================================
# ODDS SCANNER — THE CORE
# ============================================================

class OddsScanner:
    def __init__(self, api_key: str = None):
        self.client = OddsClient(api_key)
        self.fights: List[FightOdds] = []
        self.mispricings: List[Mispricing] = []

    def fetch_all_ufc_odds(self, event_date: str = None) -> List[FightOdds]:
        """
        Pull odds for upcoming UFC fights.
        event_date: filter to a specific date, e.g. "2026-04-04" for Vegas 115.
                    Matches fights on that date and the next day (cards span midnight UTC).
        """
        raw = self.client.get_odds(markets="h2h,totals")
        self.fights = []

        for event in raw:
            # Filter by date if specified
            if event_date:
                ct = event.get("commence_time", "")
                if ct:
                    fight_date = ct[:10]  # "2026-04-04"
                    # Cards span 2 days in UTC (e.g. Apr 4 prelims -> Apr 5 main card)
                    from datetime import timedelta
                    base = datetime.strptime(event_date, "%Y-%m-%d")
                    next_day = (base + timedelta(days=1)).strftime("%Y-%m-%d")
                    if fight_date != event_date and fight_date != next_day:
                        continue

            fight = FightOdds(
                fighter_a=event["home_team"],
                fighter_b=event["away_team"],
                commence_time=event.get("commence_time", ""),
            )

            for bookmaker in event.get("bookmakers", []):
                book = bookmaker["key"]
                if book not in PREFERRED_BOOKS:
                    continue

                for market in bookmaker.get("markets", []):
                    market_key = market["key"]

                    if market_key == "h2h":
                        for outcome in market["outcomes"]:
                            ml = outcome["price"]
                            fight.lines.append(OddsLine(
                                book=book,
                                fighter=outcome["name"],
                                market="ml",
                                american=ml,
                                implied_prob=american_to_implied(ml),
                                decimal_odds=american_to_decimal(ml),
                                timestamp=bookmaker.get("last_update", ""),
                            ))

                    elif market_key == "totals":
                        for outcome in market["outcomes"]:
                            side = "over" if outcome["name"] == "Over" else "under"
                            ml = outcome["price"]
                            fight.lines.append(OddsLine(
                                book=book,
                                fighter=f"{outcome.get('point', '')}",
                                market=side,
                                american=ml,
                                implied_prob=american_to_implied(ml),
                                decimal_odds=american_to_decimal(ml),
                                timestamp=bookmaker.get("last_update", ""),
                            ))

            if fight.lines:
                self.fights.append(fight)

        return self.fights

    # ----------------------------------------------------------
    # SCAN 1: Cross-book divergence
    # ----------------------------------------------------------

    def scan_cross_book(self) -> List[Mispricing]:
        """Find same prop priced very differently across books."""
        results = []

        for fight in self.fights:
            for fighter in [fight.fighter_a, fight.fighter_b]:
                ml_lines = fight.get_lines("ml", fighter)
                if len(ml_lines) < 2:
                    continue

                probs = [l.implied_prob for l in ml_lines]
                spread = max(probs) - min(probs)

                if spread > CROSS_BOOK_SPREAD:
                    best = max(ml_lines, key=lambda l: l.decimal_odds)
                    worst = min(ml_lines, key=lambda l: l.decimal_odds)

                    results.append(Mispricing(
                        fight=f"{fight.fighter_a} vs {fight.fighter_b}",
                        type="cross_book",
                        description=(
                            f"{fighter} ML ranges from {worst.american:+d} ({worst.book}) "
                            f"to {best.american:+d} ({best.book}) — "
                            f"{spread:.1%} implied prob spread"
                        ),
                        edge=spread,
                        confidence="high" if spread > 0.12 else "medium",
                        details={
                            "fighter": fighter,
                            "best_book": best.book,
                            "best_odds": best.american,
                            "worst_book": worst.book,
                            "worst_odds": worst.american,
                            "spread": spread,
                            "all_lines": [(l.book, l.american) for l in ml_lines],
                        },
                    ))

        self.mispricings.extend(results)
        return results

    # ----------------------------------------------------------
    # SCAN 2: Internal consistency (ML vs distance props)
    # ----------------------------------------------------------

    def scan_consistency(self) -> List[Mispricing]:
        """
        Check if moneyline implied probs are consistent with
        over/under round totals within the same book.

        Logic: Heavy favorite + high over (fight goes long) can conflict.
        If a fighter is -400 (80% win prob) but the over 2.5 is -200
        (67% goes to decision), that implies only ~13% chance of early finish
        by the favorite — which may not square with their KO rate.
        """
        results = []

        for fight in self.fights:
            # Get consensus ML
            a_prob = fight.consensus_implied("ml", fight.fighter_a)
            b_prob = fight.consensus_implied("ml", fight.fighter_b)

            if a_prob is None or b_prob is None:
                continue

            # Remove vig from ML
            a_true, b_true = remove_vig(a_prob, b_prob)

            # Get over/under lines
            over_lines = fight.get_lines("over")
            under_lines = fight.get_lines("under")

            if not over_lines or not under_lines:
                continue

            avg_over_prob = sum(l.implied_prob for l in over_lines) / len(over_lines)
            avg_under_prob = sum(l.implied_prob for l in under_lines) / len(under_lines)
            dist_prob, finish_prob = remove_vig(avg_over_prob, avg_under_prob)

            # The favorite's finish rate should make sense with the distance prop
            fav, fav_prob = (fight.fighter_a, a_true) if a_true > b_true else (fight.fighter_b, b_true)
            dog, dog_prob = (fight.fighter_b, b_true) if a_true > b_true else (fight.fighter_a, a_true)

            # Implied finish-by-favorite = fav_prob - (fav's share of decisions)
            # If fight goes distance at dist_prob, and favorite is fav_prob to win,
            # then fav's decision wins ≈ fav_prob * dist_prob (rough proxy)
            # So fav's finish wins ≈ fav_prob - fav_prob * dist_prob = fav_prob * finish_prob
            fav_finish_implied = fav_prob * finish_prob
            fav_dec_implied = fav_prob * dist_prob

            # Flag if the favorite is huge but finish rate is tiny
            if fav_prob > 0.65 and fav_finish_implied < 0.15 and finish_prob > 0.30:
                results.append(Mispricing(
                    fight=f"{fight.fighter_a} vs {fight.fighter_b}",
                    type="consistency",
                    description=(
                        f"{fav} is {implied_to_american(fav_prob):+d} ({fav_prob:.0%}) "
                        f"but distance prop implies only {fav_finish_implied:.0%} finish rate "
                        f"for {fav}. If {fav} finishes more than that, the under has value."
                    ),
                    edge=fav_prob * finish_prob - fav_finish_implied,
                    confidence="medium",
                    details={
                        "favorite": fav,
                        "fav_true_prob": fav_prob,
                        "distance_prob": dist_prob,
                        "finish_prob": finish_prob,
                        "fav_finish_implied": fav_finish_implied,
                    },
                ))

            # Flag if distance prop conflicts with both fighters being finishers
            if finish_prob > 0.50 and dist_prob > 0.55:
                results.append(Mispricing(
                    fight=f"{fight.fighter_a} vs {fight.fighter_b}",
                    type="consistency",
                    description=(
                        f"Books imply {finish_prob:.0%} finish rate AND {dist_prob:.0%} "
                        f"distance rate — these overlap. One side has value."
                    ),
                    edge=abs(finish_prob + dist_prob - 1.0),
                    confidence="low",
                    details={
                        "finish_prob": finish_prob,
                        "distance_prob": dist_prob,
                    },
                ))

        self.mispricings.extend(results)
        return results

    # ----------------------------------------------------------
    # SCAN 3: Model vs market
    # ----------------------------------------------------------

    def scan_model_vs_market(self, model_probs: Dict[str, Dict[str, float]]) -> List[Mispricing]:
        """
        Compare your simulation output against live market odds.

        model_probs format:
        {
            "Fighter Name": {
                "ml": 0.67,        # win probability
                "by_ko": 0.25,     # win by KO probability
                "by_sub": 0.10,
                "by_dec": 0.32,
                "distance": 0.45,  # goes to distance
            },
            ...
        }
        """
        results = []

        for fight in self.fights:
            for fighter in [fight.fighter_a, fight.fighter_b]:
                # Fuzzy match fighter name to model keys
                model_key = self._match_fighter(fighter, model_probs)
                if not model_key:
                    continue

                model = model_probs[model_key]

                # Compare moneyline
                consensus = fight.consensus_implied("ml", fighter)
                if consensus and "ml" in model:
                    # Remove vig (approximate — use opponent's line)
                    opp = fight.fighter_b if fighter == fight.fighter_a else fight.fighter_a
                    opp_consensus = fight.consensus_implied("ml", opp)
                    if opp_consensus:
                        true_market, _ = remove_vig(consensus, opp_consensus)
                    else:
                        true_market = consensus * 0.95  # rough vig removal

                    edge = model["ml"] - true_market

                    if abs(edge) > EDGE_THRESHOLD:
                        best = fight.best_odds("ml", fighter)
                        results.append(Mispricing(
                            fight=f"{fight.fighter_a} vs {fight.fighter_b}",
                            type="model_vs_market",
                            description=(
                                f"{fighter} ML — Model: {model['ml']:.1%} vs "
                                f"Market: {true_market:.1%} "
                                f"({'VALUE' if edge > 0 else 'FADE'}: {edge:+.1%})"
                                + (f" | Best: {best.american:+d} @ {best.book}" if best else "")
                            ),
                            edge=edge,
                            confidence="high" if abs(edge) > 0.10 else "medium",
                            details={
                                "fighter": fighter,
                                "model_prob": model["ml"],
                                "market_prob": true_market,
                                "best_book": best.book if best else None,
                                "best_odds": best.american if best else None,
                            },
                        ))

        self.mispricings.extend(results)
        return results

    def _match_fighter(self, api_name: str, model_probs: dict) -> Optional[str]:
        """Fuzzy match API fighter name to model keys (last name match)."""
        api_lower = api_name.lower()
        for key in model_probs:
            if key.lower() in api_lower or api_lower in key.lower():
                return key
            # Try last name
            api_last = api_lower.split()[-1] if " " in api_lower else api_lower
            key_last = key.lower().split()[-1] if " " in key.lower() else key.lower()
            if api_last == key_last:
                return key
        return None

    # ----------------------------------------------------------
    # RUN ALL SCANS
    # ----------------------------------------------------------

    def scan_all(self, model_probs: Dict[str, Dict[str, float]] = None,
                 event_date: str = None) -> List[Mispricing]:
        """Run all scans and return sorted mispricings."""
        print(f"\n{'='*70}")
        print(f"  UFC ODDS SCANNER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*70}\n")

        print("Fetching live odds...")
        self.fetch_all_ufc_odds(event_date=event_date)
        print(f"Found {len(self.fights)} fights with odds\n")

        if self.client.remaining_requests:
            print(f"API quota remaining: {self.client.remaining_requests} requests\n")

        self.mispricings = []

        # Scan 1
        print("--- SCAN 1: Cross-Book Divergences ---\n")
        cross = self.scan_cross_book()
        if cross:
            for m in sorted(cross, key=lambda x: -x.edge):
                print(f"  [{m.confidence.upper()}] {m.description}")
        else:
            print("  No significant cross-book divergences found.")

        # Scan 2
        print("\n--- SCAN 2: Internal Consistency ---\n")
        consistency = self.scan_consistency()
        if consistency:
            for m in sorted(consistency, key=lambda x: -abs(x.edge)):
                print(f"  [{m.confidence.upper()}] {m.description}")
        else:
            print("  All props look internally consistent.")

        # Scan 3
        if model_probs:
            print("\n--- SCAN 3: Model vs Market ---\n")
            model = self.scan_model_vs_market(model_probs)
            if model:
                for m in sorted(model, key=lambda x: -abs(x.edge)):
                    tag = "VALUE" if m.edge > 0 else " FADE"
                    print(f"  [{m.confidence.upper()}] [{tag}] {m.description}")
            else:
                print("  No significant model-market disagreements.")

        # Summary
        print(f"\n{'='*70}")
        print(f"  SUMMARY: {len(self.mispricings)} potential mispricings found")
        print(f"{'='*70}\n")

        high = [m for m in self.mispricings if m.confidence == "high"]
        if high:
            print("  TOP OPPORTUNITIES:")
            for m in sorted(high, key=lambda x: -abs(x.edge))[:5]:
                print(f"    * {m.fight} — {m.description}")
            print()

        return self.mispricings


# ============================================================
# BRIDGE: Convert v2 engine output to scanner input
# ============================================================

def engine_to_model_probs(analysis_results: Dict[str, dict]) -> Dict[str, Dict[str, float]]:
    """
    Convert your v2 engine's analyze() output into the format
    scan_model_vs_market() expects.

    analysis_results: dict from your engine keyed by fighter name,
    where each value has keys like "ml", "ko", "sub", "dec", etc.
    """
    model_probs = {}
    for fighter, data in analysis_results.items():
        model_probs[fighter] = {
            "ml": data.get("ml", data.get("win_prob", 0)),
            "by_ko": data.get("ko", 0),
            "by_sub": data.get("sub", 0),
            "by_dec": data.get("dec", 0),
            "distance": data.get("distance", 0),
        }
    return model_probs


# ============================================================
# STANDALONE QUICK SCAN (no model needed)
# ============================================================

def quick_scan(event_date: str = None):
    """Just scan for cross-book and consistency issues — no model required."""
    scanner = OddsScanner()

    label = f" ({event_date})" if event_date else ""
    print(f"\n{'='*70}")
    print(f"  UFC ODDS QUICK SCAN{label} — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    print("Fetching live odds...")
    scanner.fetch_all_ufc_odds(event_date=event_date)
    print(f"Found {len(scanner.fights)} fights\n")

    if scanner.client.remaining_requests:
        print(f"API quota: {scanner.client.remaining_requests} requests remaining\n")

    # Show all fights and their odds
    for fight in scanner.fights:
        print(f"\n  {fight.fighter_a} vs {fight.fighter_b}")
        print(f"  {'─'*50}")

        for fighter in [fight.fighter_a, fight.fighter_b]:
            lines = fight.get_lines("ml", fighter)
            if lines:
                odds_str = " | ".join(f"{l.book}: {l.american:+d}" for l in lines)
                best = max(lines, key=lambda l: l.decimal_odds)
                print(f"    {fighter}: {odds_str}")
                print(f"      Best: {best.american:+d} @ {best.book}")

    # Run scans
    print(f"\n{'='*70}")
    scanner.scan_cross_book()
    scanner.scan_consistency()

    high = [m for m in scanner.mispricings if m.confidence == "high"]
    med = [m for m in scanner.mispricings if m.confidence == "medium"]

    if high or med:
        print("\n  FINDINGS:\n")
        for m in sorted(scanner.mispricings, key=lambda x: -abs(x.edge)):
            print(f"  [{m.confidence.upper():6s}] [{m.type:12s}] {m.description}")
    else:
        print("\n  No significant mispricings found in current lines.")

    print()
    return scanner


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys

    # Pass a date as argument: python3 odds_scanner.py 2026-04-04
    # Or run with no args to get the next upcoming card
    event_date = sys.argv[1] if len(sys.argv) > 1 else None

    if not API_KEY:
        print("\n  To use the live scanner, get a free API key:")
        print("    1. Sign up at https://the-odds-api.com")
        print("    2. export ODDS_API_KEY='your_key'")
        print("    3. python odds_scanner.py")
        print("\n  Running in DEMO mode with sample data...\n")

        # Demo with your hardcoded book_odds to show the concept
        print("="*70)
        print("  DEMO: Internal Consistency Check (hardcoded odds)")
        print("="*70)

        # Import your engine's book_odds if available
        try:
            from ufc_vegas_115_v2 import book_odds, round_props, american_to_implied as ati

            print("\n  Checking method props sum to ~100% per fight...\n")

            fights = [
                ("Moicano", "Duncan"), ("Jandiroba", "Ricci"),
                ("Yakhyaev", "Ribeiro"), ("Estevam", "Ewing"),
                ("McMillen", "Zecchini"), ("Ruchala", "Delano"),
                ("Vannata", "Flowers"), ("Bekoev", "Gore"),
                ("Petersen", "Pat"), ("Costa", "Nicoll"),
                ("Gatto", "Barbosa"), ("Cowan", "Pereira"),
            ]

            for a, b in fights:
                if a not in book_odds or b not in book_odds:
                    continue

                # Sum implied probs for all method outcomes
                total_implied = 0
                breakdown = {}
                for fighter in [a, b]:
                    odds = book_odds[fighter]
                    for method in ["by_ko", "by_sub", "by_dec"]:
                        imp = ati(odds[method])
                        total_implied += imp
                        breakdown[f"{fighter} {method}"] = imp

                overround = total_implied - 1.0

                # Check: do the method props roughly make sense?
                print(f"  {a} vs {b}")
                print(f"    Method props total implied: {total_implied:.1%} "
                      f"(overround: {overround:+.1%})")

                # Flag extremes
                if overround > 0.40:
                    print(f"    *** HIGH OVERROUND — books are pricing in a lot of vig "
                          f"on method props. Look for the most inflated line.")
                elif overround < -0.05:
                    print(f"    *** UNDERROUND — rare! Method props sum to LESS than 100%. "
                          f"Free edge exists somewhere in these lines.")

                # Check if ML is consistent with sum of methods
                a_methods = sum(ati(book_odds[a][m]) for m in ["by_ko", "by_sub", "by_dec"])
                b_methods = sum(ati(book_odds[b][m]) for m in ["by_ko", "by_sub", "by_dec"])
                a_ml_imp = ati(book_odds[a]["ml"])
                b_ml_imp = ati(book_odds[b]["ml"])

                a_gap = a_methods - a_ml_imp
                b_gap = b_methods - b_ml_imp

                if abs(a_gap) > CONSISTENCY_GAP:
                    direction = "OVER" if a_gap > 0 else "UNDER"
                    print(f"    >>> {a}: Method props imply {a_methods:.1%} win rate "
                          f"but ML implies {a_ml_imp:.1%} — "
                          f"methods {direction}-PRICED by {abs(a_gap):.1%}")

                if abs(b_gap) > CONSISTENCY_GAP:
                    direction = "OVER" if b_gap > 0 else "UNDER"
                    print(f"    >>> {b}: Method props imply {b_methods:.1%} win rate "
                          f"but ML implies {b_ml_imp:.1%} — "
                          f"methods {direction}-PRICED by {abs(b_gap):.1%}")

                print()

        except ImportError:
            print("  Could not import ufc_vegas_115_v2.py for demo.")
            print("  Run with ODDS_API_KEY set for live scanning.")

    else:
        quick_scan(event_date=event_date)
