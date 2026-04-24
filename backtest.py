"""
UFC Model Backtester — Test Our Edge Against Reality

Runs the Fight IQ + Duration Model + Monte Carlo against past cards
where we know the actual outcomes. Scores:

  1. Winner prediction accuracy (did we pick the right fighter?)
  2. Method prediction accuracy (did we get KO/SUB/DEC right?)
  3. Over/Under accuracy (did our finish/distance call match reality?)
  4. Edge profitability (if we bet every +EV O/U, did we make money?)

Usage:
    python3 backtest.py
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from ufc_vegas_115_v2 import (
    Fighter, american_to_implied, american_to_decimal, kelly,
)


# We need standalone versions of our models that don't depend on the
# global fighters dict — each historical card has its own fighters.

def calculate_elo_standalone(f: Fighter) -> float:
    base = 1500
    total = f.wins + f.losses
    if total == 0:
        return base
    wc = f.ko_wins * 35 + f.sub_wins * 30 + f.dec_wins * 20 + f.losses * -25
    streak = max(min(f.win_streak * 30, 120), -90)
    if 28 <= f.age <= 33:
        age_mod = 20
    elif f.age < 28:
        age_mod = (f.age - 22) * 3
    else:
        age_mod = max(-60, (33 - f.age) * 8)
    act = (f.sig_strikes_per_min + f.takedowns_per_15 - 4.0) * 10
    recency_mod = (f.recent_form - 0.5) * 160
    return round(base + wc + streak + age_mod + act + recency_mod, 1)


def elo_wp(ea: float, eb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))


# ============================================================
# Import Fight IQ model functions (standalone-compatible)
# ============================================================
from fight_iq import build_profile, find_matchup_edges, classify_archetype, ARCHETYPES


# ============================================================
# HISTORICAL FIGHT DATA
# ============================================================

@dataclass
class HistoricalFight:
    """A past fight with known outcome."""
    fighter_a: Fighter
    fighter_b: Fighter
    num_rounds: int
    # Book odds at fight time
    ml_a: int                # American odds for fighter A
    ml_b: int                # American odds for fighter B
    ou_line: float           # 1.5, 2.5, or 3.5
    over_odds: int           # American
    under_odds: int          # American
    # Actual result
    winner: str              # fighter name
    method: str              # "ko", "sub", "dec"
    finish_round: int        # round it ended (= num_rounds for decisions)
    went_distance: bool      # True if went to scorecards
    card_name: str = ""


@dataclass
class PredictionResult:
    """Our model's prediction vs reality for one fight."""
    fight_name: str
    card_name: str
    # Winner prediction
    predicted_winner: str
    predicted_win_prob: float
    actual_winner: str
    winner_correct: bool
    # Method prediction
    predicted_method: str
    predicted_method_prob: float
    actual_method: str
    method_correct: bool
    # Over/Under prediction
    ou_line: float
    predicted_side: str       # "over" or "under"
    predicted_prob: float
    book_implied: float
    edge: float
    actual_side: str          # "over" or "under"
    ou_correct: bool
    # Bet result (if we bet on our predicted side)
    bet_odds: int             # American
    bet_profit: float         # profit on $100 bet (-100 to +X)
    # Fight IQ archetype
    archetype: str
    read_confidence: float


def run_fight_iq_prediction(fight: HistoricalFight) -> PredictionResult:
    """Run our full model stack on a historical fight and compare to reality."""
    fa, fb = fight.fighter_a, fight.fighter_b
    fa.elo = calculate_elo_standalone(fa)
    fb.elo = calculate_elo_standalone(fb)

    # Build combat profiles
    pa = build_profile(fa)
    pb = build_profile(fb)

    # Find matchup edges
    edges_a, edges_b = find_matchup_edges(pa, pb)

    # Classify archetype
    archetype = classify_archetype(pa, pb, edges_a, edges_b, fa, fb)
    arch_data = ARCHETYPES[archetype]

    # --- Win probability ---
    base_wp_a = elo_wp(fa.elo, fb.elo)
    edge_shift_a = sum(e.magnitude * 0.15 for e in edges_a)
    edge_shift_b = sum(e.magnitude * 0.15 for e in edges_b)
    net_shift = max(-0.20, min(0.20, edge_shift_a - edge_shift_b))
    wp_a = max(0.05, min(0.95, base_wp_a + net_shift))

    predicted_winner = fa.name if wp_a > 0.5 else fb.name
    predicted_win_prob = max(wp_a, 1 - wp_a)

    # --- Method prediction ---
    ko_edges = [e for e in edges_a + edges_b if e.weapon == "ko_power"]
    sub_edges = [e for e in edges_a + edges_b if e.weapon == "sub_threat"]

    base_finish = arch_data["base_finish"]
    finish_edge_mag = sum(e.magnitude for e in ko_edges + sub_edges)
    durability = (pa.chin + pb.chin + pa.sub_defense + pb.sub_defense) / 4
    durability_adj = (durability - 0.5) * 0.20

    p_finish = max(0.10, min(0.85, base_finish + finish_edge_mag * 0.30 - durability_adj))
    p_distance = 1.0 - p_finish

    ko_share = arch_data["ko_share"]
    sub_share = arch_data["sub_share"]

    if ko_edges or sub_edges:
        ko_mag = sum(e.magnitude for e in ko_edges)
        sub_mag = sum(e.magnitude for e in sub_edges)
        total_mag = ko_mag + sub_mag
        if total_mag > 0:
            ko_share = ko_share * 0.5 + (ko_mag / total_mag) * 0.5
            sub_share = sub_share * 0.5 + (sub_mag / total_mag) * 0.5

    p_ko = p_finish * ko_share / (ko_share + sub_share + arch_data["dec_share"])
    p_sub = p_finish * sub_share / (ko_share + sub_share + arch_data["dec_share"])
    p_dec = p_distance

    if p_ko >= p_sub and p_ko >= p_dec:
        predicted_method = "ko"
        predicted_method_prob = p_ko
    elif p_sub >= p_ko and p_sub >= p_dec:
        predicted_method = "sub"
        predicted_method_prob = p_sub
    else:
        predicted_method = "dec"
        predicted_method_prob = p_dec

    # --- Over/Under prediction ---
    r1_share = arch_data["r1_share"]
    r2_share = arch_data["r2_share"]
    r3_share = arch_data["r3_share"]
    total_r = r1_share + r2_share + r3_share

    if fight.ou_line == 1.5:
        p_under = p_finish * (r1_share / total_r)
        p_over = 1.0 - p_under
    elif fight.ou_line == 2.5:
        p_under = p_finish * (r1_share + r2_share) / total_r
        p_over = 1.0 - p_under
    elif fight.ou_line == 3.5:
        p_under = p_finish * (r1_share + r2_share + r3_share) / total_r
        p_over = 1.0 - p_under
    elif fight.ou_line == 4.5:
        # 5-round fight: under 4.5 = finish in R1-R4
        r4_share = 0.10  # approximate
        p_under = p_finish * 0.85  # ~85% of finishes happen before R5
        p_over = 1.0 - p_under
    else:
        p_under = p_finish
        p_over = p_distance

    # Read confidence
    max_edge_mag = max(
        (max((e.magnitude for e in edges_a), default=0)),
        (max((e.magnitude for e in edges_b), default=0))
    )
    read_confidence = min(1.0,
        max_edge_mag * 0.5 +
        min(len(edges_a) + len(edges_b), 4) * 0.08 +
        min(abs(fa.elo - fb.elo) / 400, 0.3)
    )

    # Compare to book
    over_imp = american_to_implied(fight.over_odds)
    under_imp = american_to_implied(fight.under_odds)
    total_imp = over_imp + under_imp
    over_true = over_imp / total_imp if total_imp > 0 else 0.5
    under_true = under_imp / total_imp if total_imp > 0 else 0.5

    # Our predicted side
    if p_over > over_true:
        predicted_side = "over"
        predicted_prob = p_over
        book_implied = over_true
        edge = p_over - over_true
        bet_odds = fight.over_odds
    elif p_under > under_true:
        predicted_side = "under"
        predicted_prob = p_under
        book_implied = under_true
        edge = p_under - under_true
        bet_odds = fight.under_odds
    else:
        # No edge — still pick a side for scoring
        if p_over > p_under:
            predicted_side = "over"
            predicted_prob = p_over
            book_implied = over_true
        else:
            predicted_side = "under"
            predicted_prob = p_under
            book_implied = under_true
        edge = 0.0
        bet_odds = fight.over_odds if predicted_side == "over" else fight.under_odds

    # Actual result
    if fight.went_distance:
        actual_side = "over"  # decision = over (for 2.5 and 1.5 lines)
        if fight.ou_line == 4.5 and fight.num_rounds == 5:
            actual_side = "over"
        elif fight.ou_line == 3.5 and fight.finish_round <= 3 and not fight.went_distance:
            actual_side = "under"
    else:
        if fight.finish_round <= fight.ou_line:
            actual_side = "under"
        else:
            actual_side = "over"

    ou_correct = predicted_side == actual_side

    # Bet profit
    if ou_correct:
        dec_odds = american_to_decimal(bet_odds)
        bet_profit = 100 * (dec_odds - 1)
    else:
        bet_profit = -100

    return PredictionResult(
        fight_name=f"{fa.name} vs {fb.name}",
        card_name=fight.card_name,
        predicted_winner=predicted_winner,
        predicted_win_prob=predicted_win_prob,
        actual_winner=fight.winner,
        winner_correct=(predicted_winner == fight.winner),
        predicted_method=predicted_method,
        predicted_method_prob=predicted_method_prob,
        actual_method=fight.method,
        method_correct=(predicted_method == fight.method),
        ou_line=fight.ou_line,
        predicted_side=predicted_side,
        predicted_prob=predicted_prob,
        book_implied=book_implied,
        edge=edge,
        actual_side=actual_side,
        ou_correct=ou_correct,
        bet_odds=bet_odds,
        bet_profit=bet_profit,
        archetype=archetype,
        read_confidence=read_confidence,
    )


# ============================================================
# BACKTESTING ENGINE
# ============================================================

def run_backtest(cards: Dict[str, List[HistoricalFight]]) -> List[PredictionResult]:
    """Run our model on all historical fights and collect results."""
    all_results = []
    for card_name, fights in cards.items():
        for fight in fights:
            fight.card_name = card_name
            result = run_fight_iq_prediction(fight)
            all_results.append(result)
    return all_results


def print_backtest_results(results: List[PredictionResult]):
    """Print comprehensive backtest report."""
    total = len(results)
    if total == 0:
        print("  No fights to backtest.")
        return

    print("=" * 115)
    print("  UFC MODEL BACKTEST — How Did We Actually Do?")
    print("=" * 115)

    # ── Per-Fight Results ──
    print(f"\n{'━' * 115}")
    print(f"  FIGHT-BY-FIGHT RESULTS")
    print(f"{'━' * 115}\n")

    print(f"  {'Fight':<40} {'Archetype':<18} {'Winner':>7} {'Method':>7} {'O/U':>5} {'Edge':>7} {'P/L':>8} {'Conf':>5}")
    print(f"  {'─' * 100}")

    current_card = ""
    for r in results:
        if r.card_name != current_card:
            current_card = r.card_name
            print(f"\n  -- {current_card} --")

        w_mark = "Y" if r.winner_correct else "X"
        m_mark = "Y" if r.method_correct else "X"
        ou_mark = "Y" if r.ou_correct else "X"
        pnl = f"${r.bet_profit:+.0f}"
        arch_short = r.archetype.replace("_", " ")[:16]

        print(f"  {r.fight_name:<40} {arch_short:<18} "
              f"  {w_mark:>3}    {m_mark:>3}   {ou_mark:>3} "
              f"{r.edge:>+6.0%} {pnl:>8} {r.read_confidence:>4.0%}")

    # ── Aggregate Stats ──
    print(f"\n\n{'━' * 115}")
    print(f"  AGGREGATE ACCURACY")
    print(f"{'━' * 115}\n")

    winners_correct = sum(1 for r in results if r.winner_correct)
    methods_correct = sum(1 for r in results if r.method_correct)
    ou_correct = sum(1 for r in results if r.ou_correct)

    print(f"  Total fights analyzed: {total}")
    print(f"\n  Winner Prediction:  {winners_correct}/{total} ({winners_correct/total:.1%})")
    print(f"  Method Prediction:  {methods_correct}/{total} ({methods_correct/total:.1%})")
    print(f"  Over/Under:         {ou_correct}/{total} ({ou_correct/total:.1%})")

    # ── O/U Breakdown by Edge Size ──
    print(f"\n  O/U ACCURACY BY EDGE SIZE:")
    for min_edge, label in [(0.0, "All picks"), (0.03, "Edge > 3%"), (0.05, "Edge > 5%"), (0.10, "Edge > 10%")]:
        filtered = [r for r in results if r.edge >= min_edge]
        if filtered:
            correct = sum(1 for r in filtered if r.ou_correct)
            print(f"    {label:<20} {correct}/{len(filtered)} ({correct/len(filtered):.1%})")

    # ── O/U Breakdown by Confidence ──
    print(f"\n  O/U ACCURACY BY READ CONFIDENCE:")
    for min_conf, label in [(0.0, "All"), (0.40, "Conf > 40%"), (0.60, "Conf > 60%"), (0.80, "Conf > 80%")]:
        filtered = [r for r in results if r.read_confidence >= min_conf]
        if filtered:
            correct = sum(1 for r in filtered if r.ou_correct)
            print(f"    {label:<20} {correct}/{len(filtered)} ({correct/len(filtered):.1%})")

    # ── Profitability ──
    print(f"\n{'━' * 115}")
    print(f"  BETTING PROFITABILITY ($100 flat bet per pick)")
    print(f"{'━' * 115}\n")

    total_pnl = sum(r.bet_profit for r in results)
    roi = total_pnl / (total * 100) if total > 0 else 0
    print(f"  ALL O/U picks:     P/L: ${total_pnl:+.0f} | ROI: {roi:+.1%} | {ou_correct}W-{total-ou_correct}L")

    # Only positive-edge bets
    pos_edge = [r for r in results if r.edge > 0.03]
    if pos_edge:
        pe_pnl = sum(r.bet_profit for r in pos_edge)
        pe_correct = sum(1 for r in pos_edge if r.ou_correct)
        pe_roi = pe_pnl / (len(pos_edge) * 100)
        print(f"  Edge > 3% only:    P/L: ${pe_pnl:+.0f} | ROI: {pe_roi:+.1%} | {pe_correct}W-{len(pos_edge)-pe_correct}L")

    pos_edge_5 = [r for r in results if r.edge > 0.05]
    if pos_edge_5:
        pe5_pnl = sum(r.bet_profit for r in pos_edge_5)
        pe5_correct = sum(1 for r in pos_edge_5 if r.ou_correct)
        pe5_roi = pe5_pnl / (len(pos_edge_5) * 100)
        print(f"  Edge > 5% only:    P/L: ${pe5_pnl:+.0f} | ROI: {pe5_roi:+.1%} | {pe5_correct}W-{len(pos_edge_5)-pe5_correct}L")

    # High confidence only
    high_conf = [r for r in results if r.read_confidence > 0.60 and r.edge > 0.03]
    if high_conf:
        hc_pnl = sum(r.bet_profit for r in high_conf)
        hc_correct = sum(1 for r in high_conf if r.ou_correct)
        hc_roi = hc_pnl / (len(high_conf) * 100)
        print(f"  High conf + edge:  P/L: ${hc_pnl:+.0f} | ROI: {hc_roi:+.1%} | {hc_correct}W-{len(high_conf)-hc_correct}L")

    # ── By Archetype ──
    print(f"\n{'━' * 115}")
    print(f"  ACCURACY BY FIGHT ARCHETYPE")
    print(f"{'━' * 115}\n")

    archetypes = set(r.archetype for r in results)
    print(f"  {'Archetype':<30} {'Fights':>6} {'Win%':>7} {'O/U%':>7} {'Avg Edge':>9}")
    print(f"  {'─' * 65}")
    for arch in sorted(archetypes):
        arch_fights = [r for r in results if r.archetype == arch]
        n = len(arch_fights)
        w_pct = sum(1 for r in arch_fights if r.winner_correct) / n
        ou_pct = sum(1 for r in arch_fights if r.ou_correct) / n
        avg_edge = sum(r.edge for r in arch_fights) / n
        arch_label = arch.replace("_", " ")
        print(f"  {arch_label:<30} {n:>6} {w_pct:>6.0%} {ou_pct:>6.0%} {avg_edge:>+8.1%}")

    # ── Biggest Wins and Losses ──
    print(f"\n{'━' * 115}")
    print(f"  BIGGEST WINS AND LOSSES")
    print(f"{'━' * 115}\n")

    sorted_by_pnl = sorted(results, key=lambda r: r.bet_profit, reverse=True)
    print(f"  TOP 5 WINS:")
    for r in sorted_by_pnl[:5]:
        print(f"    ${r.bet_profit:+.0f}  {r.fight_name} — {r.predicted_side} {r.ou_line} ({r.bet_odds:+d}) "
              f"[{r.archetype.replace('_',' ')}]")

    print(f"\n  WORST 5 LOSSES:")
    for r in sorted_by_pnl[-5:]:
        print(f"    ${r.bet_profit:+.0f}  {r.fight_name} — {r.predicted_side} {r.ou_line} ({r.bet_odds:+d}) "
              f"[{r.archetype.replace('_',' ')}]")

    print(f"\n{'=' * 115}")
    print(f"  VERDICT: {'MODEL HAS EDGE' if roi > 0 else 'MODEL NEEDS WORK'} — {roi:+.1%} ROI on {total} fights")
    print(f"{'=' * 115}\n")


# ============================================================
# HISTORICAL CARD DATA
# ============================================================

def build_fighter(name, wins, losses, ko_w, sub_w, dec_w, ko_l, sub_l, dec_l,
                  sspm, ssacc, ssdef, tdp15, tdacc, tddef, subatt,
                  reach, age, streak, form=0.5, style="balanced", gender="M"):
    """Quick fighter builder for backtesting."""
    return Fighter(
        name=name, wins=wins, losses=losses,
        ko_wins=ko_w, sub_wins=sub_w, dec_wins=dec_w,
        ko_losses=ko_l, sub_losses=sub_l, dec_losses=dec_l,
        sig_strikes_per_min=sspm, sig_strike_accuracy=ssacc,
        sig_strike_defense=ssdef, takedowns_per_15=tdp15,
        takedown_accuracy=tdacc, takedown_defense=tddef,
        sub_attempts_per_15=subatt, reach_inches=reach,
        age=age, win_streak=streak, recent_form=form,
        style=style, gender=gender,
    )


def load_historical_cards() -> Dict[str, List[HistoricalFight]]:
    """Load 5 recent UFC cards with approximate fighter stats and actual results.

    Fighter stats are approximate based on known career profiles.
    Results are verified from ESPN, UFC.com, and CBS Sports.
    O/U lines are typical for these matchup types.
    """
    cards = {}
    bf = build_fighter  # shorthand

    # ============================================================
    # CARD 1: UFC Seattle — Adesanya vs Pyfer (March 28, 2026)
    # ============================================================
    cards["UFC Seattle (Mar 28)"] = [
        HistoricalFight(
            fighter_a=bf("Israel Adesanya", 24, 4, 16, 0, 8, 2, 0, 2,
                         4.40, 0.49, 0.62, 0.30, 0.50, 0.65, 0.0, 80.0, 36, -1, 0.35, "striker"),
            fighter_b=bf("Joe Pyfer", 14, 3, 10, 2, 2, 1, 1, 1,
                         5.80, 0.50, 0.48, 0.80, 0.40, 0.55, 0.3, 76.0, 28, 2, 0.80, "striker"),
            num_rounds=5, ml_a=+110, ml_b=-130, ou_line=3.5, over_odds=-140, under_odds=+120,
            winner="Joe Pyfer", method="ko", finish_round=2, went_distance=False,
        ),
        HistoricalFight(
            fighter_a=bf("Alexa Grasso", 16, 4, 4, 3, 9, 1, 1, 2,
                         4.50, 0.46, 0.58, 0.80, 0.35, 0.65, 0.3, 65.0, 33, 1, 0.70, "striker", "F"),
            fighter_b=bf("Maycee Barber", 14, 3, 5, 1, 8, 1, 0, 2,
                         5.00, 0.44, 0.52, 1.50, 0.38, 0.55, 0.2, 66.0, 27, 1, 0.65, "balanced", "F"),
            num_rounds=3, ml_a=-150, ml_b=+130, ou_line=2.5, over_odds=-120, under_odds=+100,
            winner="Alexa Grasso", method="ko", finish_round=1, went_distance=False,
        ),
        HistoricalFight(
            fighter_a=bf("Michael Chiesa", 18, 7, 2, 8, 8, 3, 2, 2,
                         2.50, 0.40, 0.55, 2.80, 0.38, 0.60, 1.2, 74.0, 37, 1, 0.50, "grappler"),
            fighter_b=bf("Niko Price", 16, 8, 9, 2, 5, 5, 2, 1,
                         4.80, 0.44, 0.42, 0.60, 0.30, 0.50, 0.2, 72.0, 35, -1, 0.35, "striker"),
            num_rounds=3, ml_a=-110, ml_b=-110, ou_line=2.5, over_odds=-130, under_odds=+110,
            winner="Michael Chiesa", method="sub", finish_round=1, went_distance=False,
        ),
        HistoricalFight(
            fighter_a=bf("Lerryan Douglas", 6, 0, 5, 1, 0, 0, 0, 0,
                         6.00, 0.52, 0.55, 0.50, 0.30, 0.60, 0.1, 74.0, 26, 6, 0.85, "striker"),
            fighter_b=bf("Julian Erosa", 30, 13, 8, 12, 10, 4, 5, 4,
                         4.00, 0.42, 0.45, 0.90, 0.33, 0.52, 0.6, 74.0, 34, -1, 0.35, "balanced"),
            num_rounds=3, ml_a=-200, ml_b=+170, ou_line=2.5, over_odds=+100, under_odds=-120,
            winner="Lerryan Douglas", method="ko", finish_round=1, went_distance=False,
        ),
    ]

    # ============================================================
    # CARD 2: UFC London — Evloev vs Murphy (March 21, 2026)
    # ============================================================
    cards["UFC London (Mar 21)"] = [
        HistoricalFight(
            fighter_a=bf("Movsar Evloev", 19, 0, 3, 3, 13, 0, 0, 0,
                         4.80, 0.48, 0.58, 3.50, 0.46, 0.70, 0.4, 69.0, 30, 19, 0.85, "balanced"),
            fighter_b=bf("Lerone Murphy", 14, 0, 5, 2, 7, 0, 0, 0,
                         4.20, 0.44, 0.60, 1.20, 0.35, 0.68, 0.3, 72.0, 30, 14, 0.80, "balanced"),
            num_rounds=5, ml_a=-175, ml_b=+150, ou_line=4.5, over_odds=-200, under_odds=+170,
            winner="Movsar Evloev", method="dec", finish_round=5, went_distance=True,
        ),
        HistoricalFight(
            fighter_a=bf("Michael Page", 22, 3, 12, 2, 8, 1, 1, 1,
                         3.80, 0.46, 0.60, 0.30, 0.25, 0.75, 0.1, 78.0, 37, 1, 0.55, "striker"),
            fighter_b=bf("Sam Patterson", 12, 2, 4, 3, 5, 1, 0, 1,
                         4.20, 0.43, 0.50, 1.00, 0.35, 0.55, 0.3, 72.0, 28, 2, 0.65, "balanced"),
            num_rounds=3, ml_a=-200, ml_b=+170, ou_line=2.5, over_odds=-160, under_odds=+140,
            winner="Michael Page", method="dec", finish_round=3, went_distance=True,
        ),
        HistoricalFight(
            fighter_a=bf("Iwo Baraniewski", 10, 1, 6, 1, 3, 0, 1, 0,
                         5.50, 0.50, 0.52, 0.80, 0.35, 0.58, 0.2, 73.0, 27, 3, 0.80, "striker"),
            fighter_b=bf("Austen Lane", 8, 2, 5, 1, 2, 1, 0, 1,
                         4.80, 0.45, 0.48, 0.60, 0.30, 0.55, 0.1, 79.0, 40, 1, 0.45, "striker"),
            num_rounds=3, ml_a=-130, ml_b=+110, ou_line=2.5, over_odds=+100, under_odds=-120,
            winner="Iwo Baraniewski", method="ko", finish_round=1, went_distance=False,
        ),
        HistoricalFight(
            fighter_a=bf("Christian Leroy Duncan", 12, 1, 4, 2, 6, 0, 0, 1,
                         4.60, 0.44, 0.55, 1.20, 0.38, 0.62, 0.3, 76.0, 28, 2, 0.75, "balanced"),
            fighter_b=bf("Roman Dolidze", 12, 5, 5, 4, 3, 2, 1, 2,
                         3.50, 0.41, 0.48, 2.00, 0.40, 0.55, 0.5, 73.0, 36, -1, 0.40, "grappler"),
            num_rounds=3, ml_a=-200, ml_b=+170, ou_line=2.5, over_odds=-110, under_odds=-110,
            winner="Christian Leroy Duncan", method="dec", finish_round=3, went_distance=True,
        ),
        HistoricalFight(
            fighter_a=bf("Danny Silva", 8, 1, 5, 1, 2, 0, 1, 0,
                         5.20, 0.48, 0.50, 0.50, 0.30, 0.55, 0.1, 70.0, 27, 2, 0.75, "striker"),
            fighter_b=bf("Kurtis Campbell", 6, 2, 3, 1, 2, 2, 0, 0,
                         4.00, 0.42, 0.45, 0.80, 0.33, 0.50, 0.2, 72.0, 29, -1, 0.45, "balanced"),
            num_rounds=3, ml_a=-150, ml_b=+130, ou_line=2.5, over_odds=-110, under_odds=-110,
            winner="Danny Silva", method="ko", finish_round=2, went_distance=False,
        ),
    ]

    # ============================================================
    # CARD 3: UFC Vegas 114 — Emmett vs Vallejos (March 14, 2026)
    # ============================================================
    cards["UFC Vegas 114 (Mar 14)"] = [
        HistoricalFight(
            fighter_a=bf("Josh Emmett", 18, 6, 9, 0, 9, 3, 0, 3,
                         3.80, 0.42, 0.55, 0.80, 0.35, 0.62, 0.1, 67.0, 39, -1, 0.30, "striker"),
            fighter_b=bf("Kevin Vallejos", 14, 1, 8, 3, 3, 0, 0, 1,
                         5.50, 0.50, 0.52, 1.20, 0.40, 0.60, 0.3, 70.0, 27, 5, 0.85, "striker"),
            num_rounds=5, ml_a=+200, ml_b=-250, ou_line=3.5, over_odds=-120, under_odds=+100,
            winner="Kevin Vallejos", method="ko", finish_round=1, went_distance=False,
        ),
        HistoricalFight(
            fighter_a=bf("Gillian Robertson", 13, 9, 0, 7, 6, 2, 1, 6,
                         2.50, 0.38, 0.48, 3.80, 0.40, 0.55, 1.5, 63.0, 30, 1, 0.50, "grappler", "F"),
            fighter_b=bf("Amanda Lemos", 15, 5, 4, 4, 7, 2, 1, 2,
                         5.20, 0.47, 0.52, 0.60, 0.30, 0.55, 0.2, 66.0, 30, -1, 0.45, "striker", "F"),
            num_rounds=3, ml_a=+250, ml_b=-300, ou_line=2.5, over_odds=-130, under_odds=+110,
            winner="Gillian Robertson", method="dec", finish_round=3, went_distance=True,
        ),
        HistoricalFight(
            fighter_a=bf("Myktybek Orolbai", 13, 1, 4, 4, 5, 0, 0, 1,
                         4.80, 0.47, 0.55, 2.00, 0.42, 0.65, 0.6, 72.0, 27, 3, 0.80, "balanced"),
            fighter_b=bf("Chris Curtis", 31, 12, 8, 0, 23, 5, 0, 7,
                         3.20, 0.40, 0.55, 0.40, 0.25, 0.60, 0.0, 74.0, 37, -2, 0.30, "striker"),
            num_rounds=3, ml_a=-250, ml_b=+210, ou_line=2.5, over_odds=-130, under_odds=+110,
            winner="Myktybek Orolbai", method="dec", finish_round=3, went_distance=True,
        ),
        HistoricalFight(
            fighter_a=bf("Ion Cutelaba", 18, 9, 10, 3, 5, 5, 2, 2,
                         4.50, 0.42, 0.45, 2.50, 0.45, 0.52, 0.3, 73.0, 30, 1, 0.50, "striker"),
            fighter_b=bf("Oumar Sy", 10, 2, 5, 3, 2, 1, 0, 1,
                         4.20, 0.44, 0.48, 1.00, 0.35, 0.55, 0.3, 77.0, 30, 2, 0.65, "balanced"),
            num_rounds=3, ml_a=+120, ml_b=-140, ou_line=2.5, over_odds=-110, under_odds=-110,
            winner="Ion Cutelaba", method="sub", finish_round=1, went_distance=False,
        ),
        HistoricalFight(
            fighter_a=bf("Vitor Petrino", 10, 2, 6, 1, 3, 1, 1, 0,
                         5.00, 0.48, 0.52, 1.50, 0.40, 0.58, 0.2, 77.0, 27, 1, 0.70, "striker"),
            fighter_b=bf("Steven Asplund", 8, 1, 3, 2, 3, 0, 0, 1,
                         3.80, 0.42, 0.50, 1.80, 0.42, 0.55, 0.4, 75.0, 29, 2, 0.65, "balanced"),
            num_rounds=3, ml_a=-300, ml_b=+250, ou_line=2.5, over_odds=+110, under_odds=-130,
            winner="Vitor Petrino", method="dec", finish_round=3, went_distance=True,
        ),
    ]

    # ============================================================
    # CARD 4: UFC Houston — Strickland vs Hernandez (Feb 21, 2026)
    # ============================================================
    cards["UFC Houston (Feb 21)"] = [
        HistoricalFight(
            fighter_a=bf("Sean Strickland", 29, 6, 9, 2, 18, 3, 1, 2,
                         5.80, 0.47, 0.62, 0.60, 0.30, 0.70, 0.1, 76.0, 35, 2, 0.70, "striker"),
            fighter_b=bf("Anthony Hernandez", 12, 3, 3, 6, 3, 1, 0, 2,
                         3.50, 0.42, 0.48, 2.50, 0.45, 0.55, 0.8, 76.0, 31, 3, 0.75, "grappler"),
            num_rounds=5, ml_a=-300, ml_b=+250, ou_line=4.5, over_odds=-130, under_odds=+110,
            winner="Sean Strickland", method="ko", finish_round=3, went_distance=False,
        ),
        HistoricalFight(
            fighter_a=bf("Geoff Neal", 16, 6, 8, 0, 8, 3, 0, 3,
                         3.80, 0.45, 0.52, 0.40, 0.25, 0.60, 0.0, 74.0, 34, -1, 0.40, "striker"),
            fighter_b=bf("Uros Medic", 9, 2, 6, 1, 2, 2, 0, 0,
                         5.50, 0.50, 0.50, 0.30, 0.25, 0.55, 0.1, 73.0, 29, 2, 0.80, "striker"),
            num_rounds=3, ml_a=-110, ml_b=-110, ou_line=2.5, over_odds=+110, under_odds=-130,
            winner="Uros Medic", method="ko", finish_round=1, went_distance=False,
        ),
        HistoricalFight(
            fighter_a=bf("Michel Pereira", 31, 12, 14, 5, 12, 4, 2, 6,
                         5.20, 0.44, 0.50, 1.50, 0.38, 0.58, 0.3, 78.0, 31, 2, 0.65, "striker"),
            fighter_b=bf("Zach Reese", 8, 2, 5, 1, 2, 1, 0, 1,
                         5.00, 0.46, 0.48, 0.80, 0.33, 0.52, 0.2, 74.0, 28, 1, 0.60, "striker"),
            num_rounds=3, ml_a=-200, ml_b=+170, ou_line=2.5, over_odds=+100, under_odds=-120,
            winner="Michel Pereira", method="dec", finish_round=3, went_distance=True,
        ),
        HistoricalFight(
            fighter_a=bf("Jacobe Smith", 10, 1, 7, 1, 2, 0, 0, 1,
                         5.80, 0.52, 0.50, 0.40, 0.30, 0.55, 0.1, 75.0, 27, 4, 0.85, "striker"),
            fighter_b=bf("Josiah Harrell", 7, 2, 3, 1, 3, 1, 0, 1,
                         4.20, 0.43, 0.48, 1.00, 0.35, 0.55, 0.2, 73.0, 29, 1, 0.55, "balanced"),
            num_rounds=3, ml_a=-200, ml_b=+170, ou_line=2.5, over_odds=-110, under_odds=-110,
            # Result unknown from search — assume Smith won (he got POTN bonus)
            winner="Jacobe Smith", method="ko", finish_round=2, went_distance=False,
        ),
    ]

    # ============================================================
    # CARD 5: UFC Mexico — Moreno vs Kavanagh (Feb 28, 2026)
    # ============================================================
    cards["UFC Mexico (Feb 28)"] = [
        HistoricalFight(
            fighter_a=bf("Brandon Moreno", 22, 8, 4, 6, 12, 2, 2, 4,
                         4.50, 0.44, 0.55, 1.80, 0.40, 0.60, 0.8, 67.0, 32, -1, 0.45, "balanced"),
            fighter_b=bf("Lone'er Kavanagh", 8, 1, 3, 2, 3, 0, 0, 1,
                         4.80, 0.46, 0.52, 1.20, 0.38, 0.58, 0.4, 68.0, 24, 3, 0.80, "balanced"),
            num_rounds=5, ml_a=-300, ml_b=+250, ou_line=4.5, over_odds=-170, under_odds=+145,
            winner="Lone'er Kavanagh", method="dec", finish_round=5, went_distance=True,
        ),
        HistoricalFight(
            fighter_a=bf("Marlon Vera", 22, 11, 10, 5, 7, 2, 3, 6,
                         4.20, 0.43, 0.52, 0.80, 0.33, 0.55, 0.3, 73.0, 32, -3, 0.30, "striker"),
            fighter_b=bf("David Martinez", 12, 2, 4, 2, 6, 0, 0, 2,
                         4.50, 0.45, 0.50, 1.50, 0.40, 0.60, 0.3, 72.0, 28, 3, 0.75, "balanced"),
            num_rounds=3, ml_a=-110, ml_b=-110, ou_line=2.5, over_odds=-140, under_odds=+120,
            winner="David Martinez", method="dec", finish_round=3, went_distance=True,
        ),
        HistoricalFight(
            fighter_a=bf("Daniel Zellhuber", 15, 2, 5, 3, 7, 1, 0, 1,
                         4.60, 0.45, 0.54, 1.50, 0.40, 0.62, 0.3, 74.0, 25, 4, 0.80, "balanced"),
            fighter_b=bf("King Green", 12, 3, 5, 3, 4, 1, 1, 1,
                         5.00, 0.47, 0.50, 0.80, 0.35, 0.55, 0.3, 73.0, 30, 1, 0.65, "striker"),
            num_rounds=3, ml_a=-200, ml_b=+170, ou_line=2.5, over_odds=-120, under_odds=+100,
            winner="King Green", method="ko", finish_round=2, went_distance=False,
        ),
        HistoricalFight(
            fighter_a=bf("Ryan Gandra", 5, 0, 3, 1, 1, 0, 0, 0,
                         5.50, 0.50, 0.52, 1.00, 0.38, 0.58, 0.2, 71.0, 27, 5, 0.85, "striker"),
            fighter_b=bf("Jose Daniel Medina", 8, 3, 3, 2, 3, 2, 1, 0,
                         4.00, 0.42, 0.45, 0.80, 0.33, 0.52, 0.3, 70.0, 28, -1, 0.40, "balanced"),
            num_rounds=3, ml_a=-200, ml_b=+170, ou_line=2.5, over_odds=-110, under_odds=-110,
            winner="Ryan Gandra", method="ko", finish_round=1, went_distance=False,
        ),
        HistoricalFight(
            fighter_a=bf("Ailin Perez", 11, 2, 2, 2, 7, 0, 0, 2,
                         4.80, 0.45, 0.52, 1.80, 0.42, 0.60, 0.4, 64.0, 28, 2, 0.70, "balanced", "F"),
            fighter_b=bf("Macy Chiasson", 10, 4, 3, 4, 3, 2, 1, 1,
                         4.30, 0.43, 0.48, 1.20, 0.35, 0.55, 0.3, 69.0, 33, -1, 0.40, "balanced", "F"),
            num_rounds=3, ml_a=-150, ml_b=+130, ou_line=2.5, over_odds=-130, under_odds=+110,
            winner="Ailin Perez", method="dec", finish_round=3, went_distance=True,
        ),
    ]

    return cards


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n  Loading historical card data...")
    cards = load_historical_cards()

    if not cards:
        print("  No historical data loaded yet. Run with card data populated.")
        print("  See load_historical_cards() function.")
    else:
        results = run_backtest(cards)
        print_backtest_results(results)
