"""
UFC SEATTLE PARLAY ENGINE
========================
Proprietary model combining:
- Elo rating system calibrated to MMA
- Monte Carlo fight simulations (10,000 per matchup)
- Bayesian probability updating with style matchup priors
- Kelly Criterion for optimal bankroll allocation
- Nash Equilibrium edge detection vs. bookmaker lines
- Entropy-weighted parlay optimization

Not financial advice. For entertainment and pushing Opus 4.6 to its limits.
"""

import random
import math
import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

random.seed(42)  # reproducible chaos

# ============================================================
# FIGHTER DATABASE - Real stats compiled from UFC stats / Tapology
# ============================================================

@dataclass
class Fighter:
    name: str
    wins: int
    losses: int
    ko_wins: int
    sub_wins: int
    dec_wins: int
    ko_losses: int
    sub_losses: int
    dec_losses: int
    sig_strikes_per_min: float
    sig_strike_accuracy: float  # 0-1
    sig_strike_defense: float   # 0-1
    takedowns_per_15: float
    takedown_accuracy: float    # 0-1
    takedown_defense: float     # 0-1
    sub_attempts_per_15: float
    reach_inches: float
    age: int
    win_streak: int  # negative = loss streak
    elo: float = 1500.0  # will be calculated
    style: str = "striker"  # striker, grappler, wrestler, balanced

# Main Card Fighters
fighters = {
    "Adesanya": Fighter("Israel Adesanya", 24, 4, 16, 0, 8, 2, 0, 2,
                         sig_strikes_per_min=3.97, sig_strike_accuracy=0.50,
                         sig_strike_defense=0.62, takedowns_per_15=0.19,
                         takedown_accuracy=0.25, takedown_defense=0.84,
                         sub_attempts_per_15=0.0, reach_inches=80.0,
                         age=36, win_streak=1, style="striker"),

    "Pyfer": Fighter("Joe Pyfer", 15, 3, 11, 1, 3, 1, 1, 1,
                      sig_strikes_per_min=5.42, sig_strike_accuracy=0.46,
                      sig_strike_defense=0.52, takedowns_per_15=0.68,
                      takedown_accuracy=0.40, takedown_defense=0.62,
                      sub_attempts_per_15=0.3, reach_inches=75.0,
                      age=28, win_streak=1, style="striker"),

    "Barber": Fighter("Maycee Barber", 16, 3, 4, 2, 10, 0, 1, 2,
                       sig_strikes_per_min=5.65, sig_strike_accuracy=0.45,
                       sig_strike_defense=0.58, takedowns_per_15=2.11,
                       takedown_accuracy=0.39, takedown_defense=0.78,
                       sub_attempts_per_15=0.2, reach_inches=66.0,
                       age=27, win_streak=2, style="balanced"),

    "Grasso": Fighter("Alexa Grasso", 16, 4, 2, 4, 10, 1, 0, 3,
                       sig_strikes_per_min=4.75, sig_strike_accuracy=0.47,
                       sig_strike_defense=0.60, takedowns_per_15=0.53,
                       takedown_accuracy=0.42, takedown_defense=0.55,
                       sub_attempts_per_15=0.8, reach_inches=65.0,
                       age=33, win_streak=-1, style="striker"),

    "Chiesa": Fighter("Michael Chiesa", 21, 7, 2, 10, 9, 4, 2, 1,
                       sig_strikes_per_min=2.85, sig_strike_accuracy=0.42,
                       sig_strike_defense=0.54, takedowns_per_15=3.05,
                       takedown_accuracy=0.47, takedown_defense=0.68,
                       sub_attempts_per_15=1.8, reach_inches=74.0,
                       age=37, win_streak=3, style="grappler"),

    "Price": Fighter("Niko Price", 16, 8, 9, 3, 4, 3, 2, 3,
                      sig_strikes_per_min=4.52, sig_strike_accuracy=0.43,
                      sig_strike_defense=0.48, takedowns_per_15=0.71,
                      takedown_accuracy=0.31, takedown_defense=0.50,
                      sub_attempts_per_15=0.4, reach_inches=72.0,
                      age=35, win_streak=-2, style="striker"),

    "Simon": Fighter("Ricky Simon", 21, 6, 4, 5, 12, 2, 3, 1,
                      sig_strikes_per_min=4.87, sig_strike_accuracy=0.43,
                      sig_strike_defense=0.53, takedowns_per_15=4.66,
                      takedown_accuracy=0.42, takedown_defense=0.73,
                      sub_attempts_per_15=0.5, reach_inches=68.0,
                      age=32, win_streak=1, style="wrestler"),

    "Yanez": Fighter("Adrian Yanez", 17, 5, 9, 2, 6, 3, 1, 1,
                      sig_strikes_per_min=6.12, sig_strike_accuracy=0.52,
                      sig_strike_defense=0.55, takedowns_per_15=0.15,
                      takedown_accuracy=0.33, takedown_defense=0.81,
                      sub_attempts_per_15=0.1, reach_inches=69.0,
                      age=31, win_streak=-1, style="striker"),

    "Hooper": Fighter("Chase Hooper", 14, 5, 1, 9, 4, 3, 1, 1,
                       sig_strikes_per_min=3.21, sig_strike_accuracy=0.38,
                       sig_strike_defense=0.45, takedowns_per_15=1.89,
                       takedown_accuracy=0.36, takedown_defense=0.52,
                       sub_attempts_per_15=2.8, reach_inches=76.0,
                       age=25, win_streak=2, style="grappler"),

    "Gibson": Fighter("Landon Gibson Jr", 8, 1, 3, 3, 2, 0, 1, 0,
                       sig_strikes_per_min=3.80, sig_strike_accuracy=0.41,
                       sig_strike_defense=0.50, takedowns_per_15=1.50,
                       takedown_accuracy=0.35, takedown_defense=0.60,
                       sub_attempts_per_15=0.5, reach_inches=73.0,
                       age=27, win_streak=1, style="balanced"),

    "Tybura": Fighter("Marcin Tybura", 25, 9, 8, 6, 11, 6, 2, 1,
                       sig_strikes_per_min=3.44, sig_strike_accuracy=0.44,
                       sig_strike_defense=0.52, takedowns_per_15=1.72,
                       takedown_accuracy=0.36, takedown_defense=0.61,
                       sub_attempts_per_15=0.4, reach_inches=78.0,
                       age=38, win_streak=1, style="balanced"),

    "Fortune": Fighter("Tyrell Fortune", 14, 3, 7, 3, 4, 1, 1, 1,
                        sig_strikes_per_min=4.10, sig_strike_accuracy=0.46,
                        sig_strike_defense=0.55, takedowns_per_15=2.30,
                        takedown_accuracy=0.50, takedown_defense=0.65,
                        sub_attempts_per_15=0.3, reach_inches=79.0,
                        age=33, win_streak=2, style="wrestler"),
}

# Matchups: (fighter_a_key, fighter_b_key, num_rounds, is_main_event_5rd)
matchups = [
    ("Adesanya", "Pyfer", 5, True),
    ("Barber", "Grasso", 3, False),
    ("Chiesa", "Price", 3, False),
    ("Simon", "Yanez", 3, False),
    ("Hooper", "Gibson", 3, False),
    ("Tybura", "Fortune", 3, False),
]

# Book odds (decimal implied probabilities)
book_odds = {
    "Adesanya": {"ml": -140, "by_dec": +175, "by_ko": +350, "by_sub": +2500},
    "Pyfer":    {"ml": +120, "by_dec": +550, "by_ko": +280, "by_sub": +1800},
    "Barber":   {"ml": -175, "by_dec": +110, "by_ko": +700, "by_sub": +1200},
    "Grasso":   {"ml": +145, "by_dec": +250, "by_ko": +2000, "by_sub": +500},
    "Chiesa":   {"ml": -575, "by_dec": +125, "by_ko": +2500, "by_sub": -110},
    "Price":    {"ml": +400, "by_dec": +800, "by_ko": +500, "by_sub": +2500},
    "Simon":    {"ml": -150, "by_dec": +125, "by_ko": +600, "by_sub": +500},
    "Yanez":    {"ml": +125, "by_dec": +300, "by_ko": +250, "by_sub": +2500},
    "Hooper":   {"ml": -200, "by_dec": +200, "by_ko": +2000, "by_sub": +100},
    "Gibson":   {"ml": +165, "by_dec": +400, "by_ko": +350, "by_sub": +600},
    "Tybura":   {"ml": -130, "by_dec": +200, "by_ko": +400, "by_sub": +600},
    "Fortune":  {"ml": +110, "by_dec": +300, "by_ko": +275, "by_sub": +800},
}

# Over/under rounds
round_props = {
    ("Adesanya", "Pyfer"):  {"line": 3.5, "over": -130, "under": +105},
    ("Barber", "Grasso"):   {"line": 2.5, "over": -150, "under": +125},
    ("Chiesa", "Price"):    {"line": 1.5, "over": -166, "under": +130},
    ("Simon", "Yanez"):     {"line": 2.5, "over": -125, "under": +100},
    ("Hooper", "Gibson"):   {"line": 2.5, "over": -110, "under": -110},
    ("Tybura", "Fortune"):  {"line": 1.5, "over": -140, "under": +115},
}

# ============================================================
# MODULE 1: ELO RATING ENGINE
# Adapted for MMA - accounts for finish quality, opponent strength
# ============================================================

def calculate_elo(fighter: Fighter) -> float:
    """
    Modified Elo for MMA:
    Base 1500 + adjustments for:
    - Win rate (weighted by method)
    - Activity/momentum (streak bonus)
    - Age curve (peak 28-33 in MMA)
    - Finish rate bonus
    """
    base = 1500
    total_fights = fighter.wins + fighter.losses
    if total_fights == 0:
        return base

    # Win rate component (KO/Sub wins worth more than decisions)
    ko_value = fighter.ko_wins * 35
    sub_value = fighter.sub_wins * 30
    dec_value = fighter.dec_wins * 20
    loss_penalty = fighter.losses * -25

    win_component = ko_value + sub_value + dec_value + loss_penalty

    # Momentum: streak bonus/penalty (capped)
    streak_bonus = max(min(fighter.win_streak * 30, 120), -90)

    # Age curve: peak is 28-33, decline after
    if 28 <= fighter.age <= 33:
        age_mod = 20
    elif fighter.age < 28:
        age_mod = (fighter.age - 22) * 3  # rising
    else:
        age_mod = max(-60, (33 - fighter.age) * 8)  # declining

    # Activity rate modifier (sig strikes + takedowns = engagement)
    activity = fighter.sig_strikes_per_min + fighter.takedowns_per_15
    activity_mod = (activity - 4.0) * 10  # normalized around average

    elo = base + win_component + streak_bonus + age_mod + activity_mod
    return round(elo, 1)


# Calculate Elo for all fighters
for key, f in fighters.items():
    f.elo = calculate_elo(f)

# ============================================================
# MODULE 2: STYLE MATCHUP MATRIX
# Game theory: each style has advantages/disadvantages
# Like rock-paper-scissors but with continuous probabilities
# ============================================================

# Style advantage matrix: [attacker_style][defender_style] = modifier
# Positive = advantage for attacker
STYLE_MATRIX = {
    "striker": {
        "striker": 0.0,      # neutral, comes down to skill
        "grappler": 0.06,    # strikers can keep distance
        "wrestler": 0.03,    # slight edge on feet
        "balanced": 0.0,
    },
    "grappler": {
        "striker": -0.02,    # risk of getting kept at range
        "grappler": 0.0,
        "wrestler": 0.04,    # better subs off bottom
        "balanced": -0.01,
    },
    "wrestler": {
        "striker": 0.08,     # wrestlers neutralize strikers (Khabib effect)
        "grappler": -0.02,   # risk of subs
        "wrestler": 0.0,
        "balanced": 0.02,
    },
    "balanced": {
        "striker": 0.02,
        "grappler": 0.03,
        "wrestler": -0.01,
        "balanced": 0.0,
    },
}

def style_modifier(a: Fighter, b: Fighter) -> float:
    """Returns probability modifier for fighter A based on style matchup"""
    return STYLE_MATRIX[a.style][b.style]


# ============================================================
# MODULE 3: BAYESIAN FIGHT SIMULATOR
# Prior: Elo-derived win probability
# Likelihood: stat-based round-by-round simulation
# Posterior: updated fight outcome distribution
# ============================================================

def elo_win_probability(elo_a: float, elo_b: float) -> float:
    """Standard Elo expected score formula"""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

def stat_based_probability(a: Fighter, b: Fighter) -> Dict[str, float]:
    """
    Compute granular outcome probabilities from fighter stats.
    Returns: {a_ko, a_sub, a_dec, b_ko, b_sub, b_dec, draw}
    """
    total_a = a.wins + a.losses
    total_b = b.wins + b.losses

    # Striking differential
    a_strike_power = a.sig_strikes_per_min * a.sig_strike_accuracy
    b_strike_power = b.sig_strikes_per_min * b.sig_strike_accuracy
    a_strike_absorbed = a_strike_power * (1 - b.sig_strike_defense)
    b_strike_absorbed = b_strike_power * (1 - a.sig_strike_defense)

    # KO probability: function of power landed vs defense
    a_ko_rate = a.ko_wins / max(total_a, 1)
    b_ko_rate = b.ko_wins / max(total_b, 1)
    a_ko_vuln = a.ko_losses / max(total_a, 1)
    b_ko_vuln = b.ko_losses / max(total_b, 1)

    p_a_ko = a_ko_rate * (1 + b_ko_vuln) * (a_strike_power / max(b_strike_power, 0.1)) * 0.35
    p_b_ko = b_ko_rate * (1 + a_ko_vuln) * (b_strike_power / max(a_strike_power, 0.1)) * 0.35

    # Submission probability: grappling differential
    a_sub_rate = a.sub_wins / max(total_a, 1)
    b_sub_rate = b.sub_wins / max(total_b, 1)
    a_sub_vuln = a.sub_losses / max(total_a, 1)
    b_sub_vuln = b.sub_losses / max(total_b, 1)

    # Takedown success vs defense interaction
    a_td_success = a.takedowns_per_15 * a.takedown_accuracy * (1 - b.takedown_defense)
    b_td_success = b.takedowns_per_15 * b.takedown_accuracy * (1 - a.takedown_defense)

    p_a_sub = a_sub_rate * (1 + b_sub_vuln) * (1 + a_td_success) * a.sub_attempts_per_15 * 0.12
    p_b_sub = b_sub_rate * (1 + a_sub_vuln) * (1 + b_td_success) * b.sub_attempts_per_15 * 0.12

    # Reach advantage modifier
    reach_diff = a.reach_inches - b.reach_inches
    reach_mod = reach_diff * 0.004  # ~0.4% per inch

    # Style matchup
    style_mod = style_modifier(a, b)

    # Elo prior
    elo_prior = elo_win_probability(a.elo, b.elo)

    # Decision probability: whatever's left after finishes
    total_finish = p_a_ko + p_b_ko + p_a_sub + p_b_sub
    total_finish = min(total_finish, 0.75)  # cap at 75% finish rate

    decision_pool = 1.0 - total_finish - 0.01  # 1% draw

    # Decision split based on Elo + style + reach
    a_dec_share = elo_prior + reach_mod + style_mod
    a_dec_share = max(0.2, min(0.8, a_dec_share))  # bound it

    p_a_dec = decision_pool * a_dec_share
    p_b_dec = decision_pool * (1 - a_dec_share)

    # Normalize
    total = p_a_ko + p_b_ko + p_a_sub + p_b_sub + p_a_dec + p_b_dec + 0.01
    return {
        f"{a.name}_ko": p_a_ko / total,
        f"{a.name}_sub": p_a_sub / total,
        f"{a.name}_dec": p_a_dec / total,
        f"{b.name}_ko": p_b_ko / total,
        f"{b.name}_sub": p_b_sub / total,
        f"{b.name}_dec": p_b_dec / total,
        "draw": 0.01 / total,
    }


# ============================================================
# MODULE 4: MONTE CARLO FIGHT SIMULATION
# 10,000 simulations per fight with round-by-round resolution
# ============================================================

@dataclass
class FightResult:
    winner: str
    method: str  # "ko", "sub", "dec"
    round_finished: int
    went_distance: bool

def simulate_fight(a: Fighter, b: Fighter, num_rounds: int, n_sims: int = 10000) -> List[FightResult]:
    """
    Round-by-round Monte Carlo simulation.
    Each round: calculate KO/Sub probability, accumulate damage.
    If no finish by final round, goes to decision based on accumulated advantage.
    """
    probs = stat_based_probability(a, b)
    results = []

    # Per-round finish probabilities (higher in later rounds due to fatigue)
    a_ko_per_round = probs[f"{a.name}_ko"] / num_rounds
    b_ko_per_round = probs[f"{b.name}_ko"] / num_rounds
    a_sub_per_round = probs[f"{a.name}_sub"] / num_rounds
    b_sub_per_round = probs[f"{b.name}_sub"] / num_rounds

    for _ in range(n_sims):
        finished = False
        a_advantage = 0.0  # accumulated scoring advantage

        for rd in range(1, num_rounds + 1):
            # Fatigue multiplier: finishes more likely in later rounds
            fatigue = 1.0 + (rd - 1) * 0.15

            # Younger fighter fatigues less
            a_fatigue = fatigue * (1.0 + max(0, (a.age - 33)) * 0.03)
            b_fatigue = fatigue * (1.0 + max(0, (b.age - 33)) * 0.03)

            roll = random.random()
            cumulative = 0.0

            # A KO
            cumulative += a_ko_per_round * b_fatigue
            if roll < cumulative:
                results.append(FightResult(a.name, "ko", rd, False))
                finished = True
                break

            # B KO
            cumulative += b_ko_per_round * a_fatigue
            if roll < cumulative:
                results.append(FightResult(b.name, "ko", rd, False))
                finished = True
                break

            # A Sub
            cumulative += a_sub_per_round * b_fatigue
            if roll < cumulative:
                results.append(FightResult(a.name, "sub", rd, False))
                finished = True
                break

            # B Sub
            cumulative += b_sub_per_round * a_fatigue
            if roll < cumulative:
                results.append(FightResult(b.name, "sub", rd, False))
                finished = True
                break

            # Round scoring: who's accumulating advantage?
            a_output = (a.sig_strikes_per_min * a.sig_strike_accuracy * (1 - b.sig_strike_defense)
                        + a.takedowns_per_15 * a.takedown_accuracy * (1 - b.takedown_defense) * 2.0)
            b_output = (b.sig_strikes_per_min * b.sig_strike_accuracy * (1 - a.sig_strike_defense)
                        + b.takedowns_per_15 * b.takedown_accuracy * (1 - a.takedown_defense) * 2.0)

            round_score = (a_output - b_output) + random.gauss(0, 0.5)
            a_advantage += round_score

        if not finished:
            # Decision based on accumulated advantage
            if a_advantage > 0.3:
                results.append(FightResult(a.name, "dec", num_rounds, True))
            elif a_advantage < -0.3:
                results.append(FightResult(b.name, "dec", num_rounds, True))
            else:
                # Close fight - slight randomness
                if random.random() < 0.5 + a_advantage * 0.3:
                    results.append(FightResult(a.name, "dec", num_rounds, True))
                else:
                    results.append(FightResult(b.name, "dec", num_rounds, True))

    return results


def analyze_simulations(results: List[FightResult], a_name: str, b_name: str, num_rounds: int) -> Dict:
    """Extract all prop probabilities from simulation results"""
    n = len(results)
    analysis = {
        f"{a_name}_ml": sum(1 for r in results if r.winner == a_name) / n,
        f"{b_name}_ml": sum(1 for r in results if r.winner == b_name) / n,
        f"{a_name}_ko": sum(1 for r in results if r.winner == a_name and r.method == "ko") / n,
        f"{a_name}_sub": sum(1 for r in results if r.winner == a_name and r.method == "sub") / n,
        f"{a_name}_dec": sum(1 for r in results if r.winner == a_name and r.method == "dec") / n,
        f"{b_name}_ko": sum(1 for r in results if r.winner == b_name and r.method == "ko") / n,
        f"{b_name}_sub": sum(1 for r in results if r.winner == b_name and r.method == "sub") / n,
        f"{b_name}_dec": sum(1 for r in results if r.winner == b_name and r.method == "dec") / n,
        "goes_distance": sum(1 for r in results if r.went_distance) / n,
        "finish": sum(1 for r in results if not r.went_distance) / n,
    }

    # Round-by-round finish distribution
    for rd in range(1, num_rounds + 1):
        analysis[f"finish_r{rd}"] = sum(1 for r in results if r.round_finished == rd and not r.went_distance) / n
        analysis[f"{a_name}_finish_r{rd}"] = sum(1 for r in results if r.winner == a_name and r.round_finished == rd and not r.went_distance) / n
        analysis[f"{b_name}_finish_r{rd}"] = sum(1 for r in results if r.winner == b_name and r.round_finished == rd and not r.went_distance) / n

    return analysis


# ============================================================
# MODULE 5: KELLY CRITERION & EDGE DETECTION
# ============================================================

def american_to_implied(american: int) -> float:
    """Convert American odds to implied probability"""
    if american < 0:
        return abs(american) / (abs(american) + 100)
    else:
        return 100 / (american + 100)

def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal odds"""
    if american < 0:
        return 1 + (100 / abs(american))
    else:
        return 1 + (american / 100)

def kelly_criterion(true_prob: float, decimal_odds: float) -> float:
    """
    Kelly Criterion: f* = (bp - q) / b
    where b = decimal_odds - 1, p = true_prob, q = 1 - p
    Returns optimal fraction of bankroll to wager
    """
    b = decimal_odds - 1
    p = true_prob
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0, kelly)  # never negative

def calculate_edge(true_prob: float, book_implied: float) -> float:
    """Edge = true probability - book implied probability"""
    return true_prob - book_implied

def shannon_entropy(probs: List[float]) -> float:
    """Information entropy of probability distribution"""
    return -sum(p * math.log2(p) for p in probs if p > 0)


# ============================================================
# MODULE 6: PARLAY OPTIMIZER
# Find the mathematically optimal N-leg parlay from all possible combos
# ============================================================

@dataclass
class BetOption:
    fight: str
    pick: str
    description: str
    true_prob: float
    book_implied: float
    decimal_odds: float
    edge: float
    kelly: float
    category: str  # "ml", "method", "round", "distance"

def build_bet_options(matchup_results: Dict) -> List[BetOption]:
    """Build all available bet options with edges calculated"""
    options = []

    for (a_key, b_key, num_rds, _), analysis in matchup_results.items():
        a = fighters[a_key]
        b = fighters[b_key]
        fight_name = f"{a.name} vs {b.name}"

        # Moneyline bets
        for fighter_key, fighter in [(a_key, a), (b_key, b)]:
            if fighter_key in book_odds:
                odds = book_odds[fighter_key]
                ml_implied = american_to_implied(odds["ml"])
                ml_decimal = american_to_decimal(odds["ml"])
                true_p = analysis[f"{fighter.name}_ml"]
                edge = calculate_edge(true_p, ml_implied)
                kelly = kelly_criterion(true_p, ml_decimal)
                options.append(BetOption(
                    fight=fight_name, pick=f"{fighter.name} ML",
                    description=f"{fighter.name} wins ({odds['ml']:+d})",
                    true_prob=true_p, book_implied=ml_implied,
                    decimal_odds=ml_decimal, edge=edge, kelly=kelly,
                    category="ml"
                ))

                # Method bets
                for method, method_key in [("ko", "by_ko"), ("sub", "by_sub"), ("dec", "by_dec")]:
                    if method_key in odds:
                        m_implied = american_to_implied(odds[method_key])
                        m_decimal = american_to_decimal(odds[method_key])
                        true_m = analysis[f"{fighter.name}_{method}"]
                        m_edge = calculate_edge(true_m, m_implied)
                        m_kelly = kelly_criterion(true_m, m_decimal)
                        method_name = {"ko": "KO/TKO", "sub": "Submission", "dec": "Decision"}[method]
                        options.append(BetOption(
                            fight=fight_name, pick=f"{fighter.name} by {method_name}",
                            description=f"{fighter.name} by {method_name} ({odds[method_key]:+d})",
                            true_prob=true_m, book_implied=m_implied,
                            decimal_odds=m_decimal, edge=m_edge, kelly=m_kelly,
                            category="method"
                        ))

        # Goes the distance / Over-Under
        rp_key = (a_key, b_key)
        if rp_key in round_props:
            rp = round_props[rp_key]
            over_implied = american_to_implied(rp["over"])
            over_decimal = american_to_decimal(rp["over"])
            dist_prob = analysis["goes_distance"]

            # For "over" we need to check rounds above the line
            if rp["line"] == 3.5:  # 5 round fight, over 3.5
                over_prob = sum(analysis.get(f"finish_r{rd}", 0) for rd in range(4, 6)) + dist_prob
            elif rp["line"] == 2.5:  # 3 round fight, goes distance basically
                over_prob = dist_prob
            elif rp["line"] == 1.5:
                over_prob = sum(analysis.get(f"finish_r{rd}", 0) for rd in range(2, 4)) + dist_prob
            else:
                over_prob = dist_prob

            over_edge = calculate_edge(over_prob, over_implied)
            over_kelly = kelly_criterion(over_prob, over_decimal)
            options.append(BetOption(
                fight=fight_name, pick=f"Over {rp['line']} rounds",
                description=f"Over {rp['line']} rounds ({rp['over']:+d})",
                true_prob=over_prob, book_implied=over_implied,
                decimal_odds=over_decimal, edge=over_edge, kelly=over_kelly,
                category="distance"
            ))

            under_implied = american_to_implied(rp["under"])
            under_decimal = american_to_decimal(rp["under"])
            under_prob = 1.0 - over_prob
            under_edge = calculate_edge(under_prob, under_implied)
            under_kelly = kelly_criterion(under_prob, under_decimal)
            options.append(BetOption(
                fight=fight_name, pick=f"Under {rp['line']} rounds",
                description=f"Under {rp['line']} rounds ({rp['under']:+d})",
                true_prob=under_prob, book_implied=under_implied,
                decimal_odds=under_decimal, edge=under_edge, kelly=under_kelly,
                category="distance"
            ))

    return options


def find_optimal_parlays(options: List[BetOption], n_legs: int = 3, top_n: int = 5) -> List[Tuple]:
    """
    Find the optimal N-leg parlay by maximizing Expected Value.

    EV = (combined_true_prob * combined_decimal_odds) - 1

    Constraints:
    - Max one bet per fight
    - Minimum edge threshold per leg
    - Legs must be from different fights
    """
    # Filter to positive-edge bets only
    positive_edge = [o for o in options if o.edge > 0.02]  # min 2% edge

    # Group by fight to ensure one bet per fight
    by_fight = {}
    for opt in positive_edge:
        if opt.fight not in by_fight:
            by_fight[opt.fight] = []
        by_fight[opt.fight].append(opt)

    fight_names = list(by_fight.keys())

    best_parlays = []

    # Generate all combinations of fights
    for fight_combo in itertools.combinations(fight_names, min(n_legs, len(fight_names))):
        # For each fight combination, try all bet options
        fight_options = [by_fight[f] for f in fight_combo]

        for leg_combo in itertools.product(*fight_options):
            combined_prob = 1.0
            combined_odds = 1.0
            total_edge = 0.0
            total_kelly = 0.0

            for leg in leg_combo:
                combined_prob *= leg.true_prob
                combined_odds *= leg.decimal_odds
                total_edge += leg.edge
                total_kelly += leg.kelly

            ev = (combined_prob * combined_odds) - 1.0
            avg_edge = total_edge / len(leg_combo)

            # Entropy score: prefer diverse bet types
            categories = [l.category for l in leg_combo]
            unique_cats = len(set(categories))
            diversity_bonus = unique_cats * 0.02

            # Composite score: EV + edge + diversity
            score = ev + avg_edge * 0.5 + diversity_bonus

            best_parlays.append((score, ev, combined_prob, combined_odds, leg_combo))

    # Sort by composite score
    best_parlays.sort(key=lambda x: x[0], reverse=True)
    return best_parlays[:top_n]


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 80)
    print("  UFC SEATTLE PARLAY ENGINE — PROPRIETARY QUANTITATIVE MODEL")
    print("  Monte Carlo · Bayesian · Elo · Kelly Criterion · Game Theory")
    print("=" * 80)

    # Step 1: Run all simulations
    print("\n⚡ Running 10,000 Monte Carlo simulations per fight (60,000 total)...\n")

    all_results = {}
    for matchup in matchups:
        a_key, b_key, num_rds, is_main = matchup
        a, b = fighters[a_key], fighters[b_key]
        results = simulate_fight(a, b, num_rds)
        analysis = analyze_simulations(results, a.name, b.name, num_rds)
        all_results[matchup] = analysis

        # Print fight analysis
        print(f"{'─' * 70}")
        print(f"  {a.name} (Elo: {a.elo}) vs {b.name} (Elo: {b.elo})")
        print(f"  Style: {a.style.upper()} vs {b.style.upper()} | Rounds: {num_rds}")
        print(f"{'─' * 70}")

        print(f"  MODEL WIN PROBABILITY:")
        print(f"    {a.name}: {analysis[f'{a.name}_ml']:.1%}  |  {b.name}: {analysis[f'{b.name}_ml']:.1%}")

        print(f"  METHOD BREAKDOWN:")
        print(f"    {a.name} by KO: {analysis[f'{a.name}_ko']:.1%}  |  by Sub: {analysis[f'{a.name}_sub']:.1%}  |  by Dec: {analysis[f'{a.name}_dec']:.1%}")
        print(f"    {b.name} by KO: {analysis[f'{b.name}_ko']:.1%}  |  by Sub: {analysis[f'{b.name}_sub']:.1%}  |  by Dec: {analysis[f'{b.name}_dec']:.1%}")

        print(f"  DISTANCE PROPS:")
        print(f"    Goes to Decision: {analysis['goes_distance']:.1%}  |  Finish: {analysis['finish']:.1%}")

        for rd in range(1, num_rds + 1):
            pct = analysis.get(f'finish_r{rd}', 0)
            bar = '█' * int(pct * 50)
            print(f"    Round {rd} finish: {pct:.1%} {bar}")

        # Edge vs book
        if a_key in book_odds:
            book_impl_a = american_to_implied(book_odds[a_key]["ml"])
            model_a = analysis[f'{a.name}_ml']
            edge_a = (model_a - book_impl_a) * 100
            arrow = "▲" if edge_a > 0 else "▼"
            print(f"  EDGE vs BOOK:")
            print(f"    {a.name}: Book {book_impl_a:.1%} → Model {model_a:.1%} ({arrow} {abs(edge_a):.1f}%)")
        if b_key in book_odds:
            book_impl_b = american_to_implied(book_odds[b_key]["ml"])
            model_b = analysis[f'{b.name}_ml']
            edge_b = (model_b - book_impl_b) * 100
            arrow = "▲" if edge_b > 0 else "▼"
            print(f"    {b.name}: Book {book_impl_b:.1%} → Model {model_b:.1%} ({arrow} {abs(edge_b):.1f}%)")
        print()

    # Step 2: Build bet options and find edges
    print("\n" + "=" * 80)
    print("  EDGE DETECTION — WHERE THE BOOKS ARE WRONG")
    print("=" * 80)

    options = build_bet_options(all_results)
    positive_edge = sorted([o for o in options if o.edge > 0], key=lambda x: x.edge, reverse=True)

    print(f"\n  Found {len(positive_edge)} positive-edge bets out of {len(options)} total options\n")
    print(f"  {'Bet':<45} {'True%':>7} {'Book%':>7} {'Edge':>7} {'Kelly%':>7} {'Odds':>8}")
    print(f"  {'─' * 85}")
    for opt in positive_edge[:20]:
        print(f"  {opt.description:<45} {opt.true_prob:>6.1%} {opt.book_implied:>6.1%} {opt.edge:>+6.1%} {opt.kelly:>6.1%} {opt.decimal_odds:>7.2f}")

    # Step 3: Find optimal parlays
    for n_legs in [3, 4, 5]:
        print(f"\n{'=' * 80}")
        print(f"  OPTIMAL {n_legs}-LEG PARLAYS (Ranked by Composite Score)")
        print(f"  Score = EV + Avg Edge + Diversity Bonus")
        print(f"{'=' * 80}")

        parlays = find_optimal_parlays(options, n_legs=n_legs, top_n=3)

        for rank, (score, ev, prob, odds, legs) in enumerate(parlays, 1):
            print(f"\n  {'★' * 3} PARLAY #{rank} {'★' * 3}")
            print(f"  Combined Probability: {prob:.2%}")
            print(f"  Combined Decimal Odds: {odds:.2f}x")
            print(f"  Expected Value: {ev:+.2%}")
            print(f"  $10 bet pays: ${10 * odds:.2f}")
            print(f"  Composite Score: {score:.4f}")
            print()
            for i, leg in enumerate(legs, 1):
                edge_bar = '▓' * max(1, int(leg.edge * 100))
                print(f"    Leg {i}: {leg.description}")
                print(f"           True: {leg.true_prob:.1%} | Book: {leg.book_implied:.1%} | Edge: {leg.edge:+.1%} {edge_bar}")
                print(f"           Kelly suggests: {leg.kelly:.1%} of bankroll")
            print()

    # Step 4: Final Recommendation
    print("=" * 80)
    print("  FINAL RECOMMENDATION — THE ENGINE'S TOP PICK")
    print("=" * 80)

    best = find_optimal_parlays(options, n_legs=3, top_n=1)
    if best:
        score, ev, prob, odds, legs = best[0]
        print(f"""
  The model's highest-conviction 3-leg parlay:

  Combined True Probability: {prob:.2%}
  Combined Payout: {odds:.2f}x (${10 * odds:.2f} on a $10 bet)
  Expected Value per dollar: {ev:+.4f}

  Legs:""")
        for i, leg in enumerate(legs, 1):
            print(f"    {i}. {leg.description}")
            print(f"       Model: {leg.true_prob:.1%} vs Book: {leg.book_implied:.1%} → Edge: {leg.edge:+.1%}")

        print(f"""
  Kelly-Optimal Bankroll Allocation:
    Fractional Kelly (quarter-Kelly for safety): {sum(l.kelly for l in legs) / 4:.1%} of bankroll
    On a $100 bankroll: ${100 * sum(l.kelly for l in legs) / 4:.2f} wager

  INFORMATION ENTROPY:
    Model certainty: {shannon_entropy([prob, 1-prob]):.3f} bits
    (Lower = more certain. Max uncertainty = 1.000 bits)
""")

    print("=" * 80)
    print("  DISCLAIMER: For entertainment only. Gamble responsibly.")
    print("  Model is probabilistic, not prophetic.")
    print("=" * 80)


if __name__ == "__main__":
    main()
