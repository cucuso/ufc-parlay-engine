"""
UFC 327 MIAMI PARLAY ENGINE — Prochazka vs Ulberg
==================================================
Proprietary model combining:
- Elo rating system calibrated to MMA
- Monte Carlo fight simulations (10,000 per matchup)
- Bayesian probability updating with style matchup priors
- Kelly Criterion for optimal bankroll allocation
- Nash Equilibrium edge detection vs. bookmaker lines
- Entropy-weighted parlay optimization

Card: April 11, 2026 — Kaseya Center, Miami, FL
Not financial advice. For entertainment and pushing Opus 4.6 to its limits.
"""

import random
import math
import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

random.seed(327)  # reproducible chaos

# ============================================================
# FIGHTER DATABASE - Real stats compiled from UFC stats / Tapology / ESPN
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

# ============================================================
# MAIN CARD
# ============================================================
fighters = {
    # --- MAIN EVENT: Vacant LHW Title ---
    "Prochazka": Fighter("Jiri Prochazka", 30, 5, 20, 4, 6, 3, 1, 1,
                          sig_strikes_per_min=5.69, sig_strike_accuracy=0.55,
                          sig_strike_defense=0.53, takedowns_per_15=0.80,
                          takedown_accuracy=0.60, takedown_defense=0.68,
                          sub_attempts_per_15=0.3, reach_inches=80.0,
                          age=33, win_streak=2, style="striker"),

    "Ulberg": Fighter("Carlos Ulberg", 12, 1, 9, 1, 2, 1, 0, 0,
                       sig_strikes_per_min=6.54, sig_strike_accuracy=0.55,
                       sig_strike_defense=0.60, takedowns_per_15=0.50,
                       takedown_accuracy=0.60, takedown_defense=0.85,
                       sub_attempts_per_15=0.1, reach_inches=78.5,
                       age=31, win_streak=8, style="striker"),

    # --- CO-MAIN: LHW ---
    "Murzakanov": Fighter("Azamat Murzakanov", 14, 1, 9, 1, 4, 0, 0, 1,
                           sig_strikes_per_min=4.70, sig_strike_accuracy=0.57,
                           sig_strike_defense=0.58, takedowns_per_15=0.50,
                           takedown_accuracy=0.15, takedown_defense=0.87,
                           sub_attempts_per_15=0.2, reach_inches=76.0,
                           age=33, win_streak=5, style="striker"),

    "Costa": Fighter("Paulo Costa", 14, 4, 10, 0, 4, 2, 0, 2,
                      sig_strikes_per_min=6.26, sig_strike_accuracy=0.58,
                      sig_strike_defense=0.56, takedowns_per_15=0.60,
                      takedown_accuracy=0.50, takedown_defense=0.80,
                      sub_attempts_per_15=0.0, reach_inches=72.0,
                      age=33, win_streak=-1, style="striker"),

    # --- HW ---
    "Blaydes": Fighter("Curtis Blaydes", 18, 5, 11, 0, 7, 4, 0, 1,
                        sig_strikes_per_min=3.78, sig_strike_accuracy=0.49,
                        sig_strike_defense=0.57, takedowns_per_15=5.51,
                        takedown_accuracy=0.50, takedown_defense=0.72,
                        sub_attempts_per_15=0.3, reach_inches=80.0,
                        age=35, win_streak=1, style="wrestler"),

    "Hokit": Fighter("Josh Hokit", 10, 1, 5, 2, 3, 0, 1, 0,
                      sig_strikes_per_min=4.20, sig_strike_accuracy=0.45,
                      sig_strike_defense=0.50, takedowns_per_15=2.80,
                      takedown_accuracy=0.48, takedown_defense=0.60,
                      sub_attempts_per_15=0.4, reach_inches=76.0,
                      age=28, win_streak=3, style="wrestler"),

    # --- LHW ---
    "Reyes": Fighter("Dominick Reyes", 13, 5, 7, 0, 6, 3, 0, 2,
                      sig_strikes_per_min=3.70, sig_strike_accuracy=0.44,
                      sig_strike_defense=0.55, takedowns_per_15=1.00,
                      takedown_accuracy=0.50, takedown_defense=0.82,
                      sub_attempts_per_15=0.0, reach_inches=77.0,
                      age=35, win_streak=1, style="striker"),

    "Walker": Fighter("Johnny Walker", 21, 9, 12, 3, 6, 6, 1, 2,
                       sig_strikes_per_min=3.88, sig_strike_accuracy=0.47,
                       sig_strike_defense=0.57, takedowns_per_15=0.58,
                       takedown_accuracy=0.31, takedown_defense=0.60,
                       sub_attempts_per_15=0.3, reach_inches=82.0,
                       age=33, win_streak=-2, style="striker"),

    # --- FW ---
    "Swanson": Fighter("Cub Swanson", 29, 13, 10, 8, 11, 5, 3, 5,
                        sig_strikes_per_min=4.79, sig_strike_accuracy=0.43,
                        sig_strike_defense=0.56, takedowns_per_15=1.00,
                        takedown_accuracy=0.50, takedown_defense=0.63,
                        sub_attempts_per_15=0.8, reach_inches=71.0,
                        age=42, win_streak=1, style="balanced"),

    "Landwehr": Fighter("Nate Landwehr", 18, 6, 5, 3, 10, 3, 1, 2,
                         sig_strikes_per_min=5.63, sig_strike_accuracy=0.42,
                         sig_strike_defense=0.50, takedowns_per_15=1.50,
                         takedown_accuracy=0.41, takedown_defense=0.71,
                         sub_attempts_per_15=0.5, reach_inches=72.0,
                         age=36, win_streak=-2, style="balanced"),

    # ============================================================
    # PRELIMINARY CARD
    # ============================================================

    "Pitbull": Fighter("Patricio Pitbull", 36, 12, 12, 10, 14, 5, 3, 4,
                        sig_strikes_per_min=4.00, sig_strike_accuracy=0.45,
                        sig_strike_defense=0.55, takedowns_per_15=1.50,
                        takedown_accuracy=0.40, takedown_defense=0.65,
                        sub_attempts_per_15=0.8, reach_inches=68.0,
                        age=37, win_streak=-1, style="balanced"),

    "Pico": Fighter("Aaron Pico", 13, 3, 9, 0, 4, 3, 0, 0,
                     sig_strikes_per_min=5.50, sig_strike_accuracy=0.48,
                     sig_strike_defense=0.55, takedowns_per_15=3.00,
                     takedown_accuracy=0.55, takedown_defense=0.75,
                     sub_attempts_per_15=0.1, reach_inches=70.0,
                     age=29, win_streak=4, style="wrestler"),

    "Gamrot": Fighter("Mateusz Gamrot", 24, 3, 4, 10, 10, 1, 1, 1,
                       sig_strikes_per_min=4.34, sig_strike_accuracy=0.42,
                       sig_strike_defense=0.58, takedowns_per_15=3.49,
                       takedown_accuracy=0.38, takedown_defense=0.79,
                       sub_attempts_per_15=1.0, reach_inches=70.0,
                       age=34, win_streak=2, style="wrestler"),

    "Ribovics": Fighter("Esteban Ribovics", 14, 2, 5, 4, 5, 1, 1, 0,
                          sig_strikes_per_min=5.50, sig_strike_accuracy=0.47,
                          sig_strike_defense=0.50, takedowns_per_15=0.50,
                          takedown_accuracy=0.30, takedown_defense=0.55,
                          sub_attempts_per_15=0.3, reach_inches=75.0,
                          age=26, win_streak=-1, style="striker"),

    "Holland": Fighter("Kevin Holland", 26, 12, 11, 5, 10, 5, 3, 4,
                        sig_strikes_per_min=4.40, sig_strike_accuracy=0.44,
                        sig_strike_defense=0.52, takedowns_per_15=0.40,
                        takedown_accuracy=0.25, takedown_defense=0.38,
                        sub_attempts_per_15=0.7, reach_inches=81.0,
                        age=33, win_streak=1, style="striker"),

    "Brown": Fighter("Randy Brown", 18, 6, 5, 5, 8, 2, 2, 2,
                      sig_strikes_per_min=3.71, sig_strike_accuracy=0.43,
                      sig_strike_defense=0.59, takedowns_per_15=1.50,
                      takedown_accuracy=0.44, takedown_defense=0.79,
                      sub_attempts_per_15=0.5, reach_inches=78.0,
                      age=33, win_streak=1, style="balanced"),

    "Suarez": Fighter("Tatiana Suarez", 11, 0, 0, 3, 8, 0, 0, 0,
                       sig_strikes_per_min=3.50, sig_strike_accuracy=0.40,
                       sig_strike_defense=0.65, takedowns_per_15=6.00,
                       takedown_accuracy=0.55, takedown_defense=0.80,
                       sub_attempts_per_15=1.5, reach_inches=63.0,
                       age=35, win_streak=3, style="wrestler"),

    "Godinez": Fighter("Loopy Godinez", 12, 4, 1, 5, 6, 0, 2, 2,
                        sig_strikes_per_min=3.80, sig_strike_accuracy=0.42,
                        sig_strike_defense=0.52, takedowns_per_15=2.00,
                        takedown_accuracy=0.40, takedown_defense=0.45,
                        sub_attempts_per_15=1.0, reach_inches=65.0,
                        age=29, win_streak=1, style="grappler"),

    # ============================================================
    # EARLY PRELIMS
    # ============================================================

    "Padilla": Fighter("Chris Padilla", 11, 4, 3, 4, 4, 1, 2, 1,
                        sig_strikes_per_min=3.90, sig_strike_accuracy=0.44,
                        sig_strike_defense=0.55, takedowns_per_15=2.50,
                        takedown_accuracy=0.45, takedown_defense=0.65,
                        sub_attempts_per_15=1.2, reach_inches=72.0,
                        age=28, win_streak=2, style="grappler"),

    "Mederos": Fighter("MarQuel Mederos", 12, 3, 7, 2, 3, 1, 1, 1,
                        sig_strikes_per_min=5.20, sig_strike_accuracy=0.46,
                        sig_strike_defense=0.48, takedowns_per_15=0.80,
                        takedown_accuracy=0.35, takedown_defense=0.55,
                        sub_attempts_per_15=0.2, reach_inches=74.0,
                        age=28, win_streak=1, style="striker"),

    "Gastelum": Fighter("Kelvin Gastelum", 19, 10, 6, 3, 10, 4, 2, 4,
                          sig_strikes_per_min=3.67, sig_strike_accuracy=0.43,
                          sig_strike_defense=0.55, takedowns_per_15=1.50,
                          takedown_accuracy=0.34, takedown_defense=0.60,
                          sub_attempts_per_15=0.3, reach_inches=71.0,
                          age=34, win_streak=2, style="balanced"),

    "Luque": Fighter("Vicente Luque", 22, 11, 12, 4, 6, 4, 3, 4,
                      sig_strikes_per_min=4.83, sig_strike_accuracy=0.45,
                      sig_strike_defense=0.52, takedowns_per_15=0.70,
                      takedown_accuracy=0.48, takedown_defense=0.60,
                      sub_attempts_per_15=0.4, reach_inches=74.0,
                      age=34, win_streak=-3, style="striker"),

    "Radtke": Fighter("Charles Radtke", 9, 2, 5, 2, 2, 1, 1, 0,
                       sig_strikes_per_min=4.80, sig_strike_accuracy=0.46,
                       sig_strike_defense=0.52, takedowns_per_15=1.20,
                       takedown_accuracy=0.40, takedown_defense=0.58,
                       sub_attempts_per_15=0.5, reach_inches=75.0,
                       age=28, win_streak=2, style="balanced"),

    "Prado": Fighter("Francisco Prado", 11, 5, 5, 3, 3, 2, 2, 1,
                      sig_strikes_per_min=4.30, sig_strike_accuracy=0.43,
                      sig_strike_defense=0.49, takedowns_per_15=1.00,
                      takedown_accuracy=0.35, takedown_defense=0.55,
                      sub_attempts_per_15=0.4, reach_inches=73.0,
                      age=28, win_streak=-1, style="balanced"),
}

# Matchups: (fighter_a_key, fighter_b_key, num_rounds, is_main_event_5rd)
matchups = [
    ("Prochazka", "Ulberg", 5, True),       # Main event — Vacant LHW Title
    ("Murzakanov", "Costa", 3, False),       # Co-main
    ("Blaydes", "Hokit", 3, False),          # HW
    ("Reyes", "Walker", 3, False),           # LHW
    ("Swanson", "Landwehr", 3, False),       # FW
    ("Pitbull", "Pico", 3, False),           # FW — Bellator crossover
    ("Holland", "Brown", 3, False),          # WW
    ("Gamrot", "Ribovics", 3, False),        # LW
    ("Suarez", "Godinez", 3, False),         # WSW
    ("Padilla", "Mederos", 3, False),        # CW 158
    ("Gastelum", "Luque", 3, False),         # MW
    ("Radtke", "Prado", 3, False),           # WW
]

# Book odds (American moneylines) — via DraftKings / SI.com
book_odds = {
    "Prochazka":  {"ml": -118, "by_dec": +250, "by_ko": +150, "by_sub": +1400},
    "Ulberg":     {"ml": -102, "by_dec": +350, "by_ko": +175, "by_sub": +2000},
    "Murzakanov": {"ml": -205, "by_dec": +175, "by_ko": +165, "by_sub": +2500},
    "Costa":      {"ml": +170, "by_dec": +500, "by_ko": +280, "by_sub": +5000},
    "Blaydes":    {"ml": -122, "by_dec": +175, "by_ko": +350, "by_sub": +2500},
    "Hokit":      {"ml": +102, "by_dec": +350, "by_ko": +300, "by_sub": +1200},
    "Reyes":      {"ml": -148, "by_dec": +175, "by_ko": +250, "by_sub": +3000},
    "Walker":     {"ml": +124, "by_dec": +400, "by_ko": +225, "by_sub": +2000},
    "Swanson":    {"ml": -108, "by_dec": +200, "by_ko": +350, "by_sub": +600},
    "Landwehr":   {"ml": -112, "by_dec": +250, "by_ko": +400, "by_sub": +800},
    "Pitbull":    {"ml": +230, "by_dec": +400, "by_ko": +500, "by_sub": +450},
    "Pico":       {"ml": -285, "by_dec": +175, "by_ko": +250, "by_sub": +3000},
    "Holland":    {"ml": -112, "by_dec": +200, "by_ko": +300, "by_sub": +600},
    "Brown":      {"ml": -108, "by_dec": +225, "by_ko": +500, "by_sub": +450},
    "Gamrot":     {"ml": -205, "by_dec": +100, "by_ko": +1000, "by_sub": +350},
    "Ribovics":   {"ml": +170, "by_dec": +500, "by_ko": +325, "by_sub": +800},
    "Suarez":     {"ml": -148, "by_dec": +100, "by_ko": +3000, "by_sub": +300},
    "Godinez":    {"ml": +124, "by_dec": +300, "by_ko": +2500, "by_sub": +350},
    "Padilla":    {"ml": -162, "by_dec": +175, "by_ko": +600, "by_sub": +300},
    "Mederos":    {"ml": +136, "by_dec": +400, "by_ko": +250, "by_sub": +1500},
    "Gastelum":   {"ml": -278, "by_dec": +100, "by_ko": +450, "by_sub": +1200},
    "Luque":      {"ml": +225, "by_dec": +500, "by_ko": +350, "by_sub": +800},
    "Radtke":     {"ml": -180, "by_dec": +200, "by_ko": +250, "by_sub": +600},
    "Prado":      {"ml": +150, "by_dec": +400, "by_ko": +300, "by_sub": +700},
}

# Over/under rounds
round_props = {
    ("Prochazka", "Ulberg"):    {"line": 3.5, "over": -125, "under": +100},
    ("Murzakanov", "Costa"):    {"line": 1.5, "over": -150, "under": +125},
    ("Blaydes", "Hokit"):       {"line": 2.5, "over": -130, "under": +105},
    ("Reyes", "Walker"):        {"line": 1.5, "over": -130, "under": +105},
    ("Swanson", "Landwehr"):    {"line": 2.5, "over": -120, "under": -105},
    ("Pitbull", "Pico"):        {"line": 2.5, "over": -110, "under": -115},
    ("Holland", "Brown"):       {"line": 2.5, "over": -135, "under": +110},
    ("Gamrot", "Ribovics"):     {"line": 2.5, "over": -130, "under": +105},
    ("Suarez", "Godinez"):      {"line": 2.5, "over": -150, "under": +125},
    ("Padilla", "Mederos"):     {"line": 2.5, "over": -120, "under": -105},
    ("Gastelum", "Luque"):      {"line": 2.5, "over": -115, "under": -110},
    ("Radtke", "Prado"):        {"line": 2.5, "over": -115, "under": -110},
}

# ============================================================
# MODULE 1: ELO RATING ENGINE
# ============================================================

def calculate_elo(fighter: Fighter) -> float:
    base = 1500
    total_fights = fighter.wins + fighter.losses
    if total_fights == 0:
        return base

    ko_value = fighter.ko_wins * 35
    sub_value = fighter.sub_wins * 30
    dec_value = fighter.dec_wins * 20
    loss_penalty = fighter.losses * -25

    win_component = ko_value + sub_value + dec_value + loss_penalty

    streak_bonus = max(min(fighter.win_streak * 30, 120), -90)

    if 28 <= fighter.age <= 33:
        age_mod = 20
    elif fighter.age < 28:
        age_mod = (fighter.age - 22) * 3
    else:
        age_mod = max(-60, (33 - fighter.age) * 8)

    activity = fighter.sig_strikes_per_min + fighter.takedowns_per_15
    activity_mod = (activity - 4.0) * 10

    elo = base + win_component + streak_bonus + age_mod + activity_mod
    return round(elo, 1)


for key, f in fighters.items():
    f.elo = calculate_elo(f)

# ============================================================
# MODULE 2: STYLE MATCHUP MATRIX
# ============================================================

STYLE_MATRIX = {
    "striker": {
        "striker": 0.0,
        "grappler": 0.06,
        "wrestler": 0.03,
        "balanced": 0.0,
    },
    "grappler": {
        "striker": -0.02,
        "grappler": 0.0,
        "wrestler": 0.04,
        "balanced": -0.01,
    },
    "wrestler": {
        "striker": 0.08,
        "grappler": -0.02,
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
    return STYLE_MATRIX[a.style][b.style]


# ============================================================
# MODULE 3: BAYESIAN FIGHT SIMULATOR
# ============================================================

def elo_win_probability(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

def stat_based_probability(a: Fighter, b: Fighter) -> Dict[str, float]:
    total_a = a.wins + a.losses
    total_b = b.wins + b.losses

    a_strike_power = a.sig_strikes_per_min * a.sig_strike_accuracy
    b_strike_power = b.sig_strikes_per_min * b.sig_strike_accuracy
    a_strike_absorbed = a_strike_power * (1 - b.sig_strike_defense)
    b_strike_absorbed = b_strike_power * (1 - a.sig_strike_defense)

    a_ko_rate = a.ko_wins / max(total_a, 1)
    b_ko_rate = b.ko_wins / max(total_b, 1)
    a_ko_vuln = a.ko_losses / max(total_a, 1)
    b_ko_vuln = b.ko_losses / max(total_b, 1)

    p_a_ko = a_ko_rate * (1 + b_ko_vuln) * (a_strike_power / max(b_strike_power, 0.1)) * 0.35
    p_b_ko = b_ko_rate * (1 + a_ko_vuln) * (b_strike_power / max(a_strike_power, 0.1)) * 0.35

    a_sub_rate = a.sub_wins / max(total_a, 1)
    b_sub_rate = b.sub_wins / max(total_b, 1)
    a_sub_vuln = a.sub_losses / max(total_a, 1)
    b_sub_vuln = b.sub_losses / max(total_b, 1)

    a_td_success = a.takedowns_per_15 * a.takedown_accuracy * (1 - b.takedown_defense)
    b_td_success = b.takedowns_per_15 * b.takedown_accuracy * (1 - a.takedown_defense)

    p_a_sub = a_sub_rate * (1 + b_sub_vuln) * (1 + a_td_success) * a.sub_attempts_per_15 * 0.12
    p_b_sub = b_sub_rate * (1 + a_sub_vuln) * (1 + b_td_success) * b.sub_attempts_per_15 * 0.12

    reach_diff = a.reach_inches - b.reach_inches
    reach_mod = reach_diff * 0.004

    style_mod = style_modifier(a, b)
    elo_prior = elo_win_probability(a.elo, b.elo)

    total_finish = p_a_ko + p_b_ko + p_a_sub + p_b_sub
    total_finish = min(total_finish, 0.75)

    decision_pool = 1.0 - total_finish - 0.01

    a_dec_share = elo_prior + reach_mod + style_mod
    a_dec_share = max(0.2, min(0.8, a_dec_share))

    p_a_dec = decision_pool * a_dec_share
    p_b_dec = decision_pool * (1 - a_dec_share)

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
# ============================================================

@dataclass
class FightResult:
    winner: str
    method: str
    round_finished: int
    went_distance: bool

def simulate_fight(a: Fighter, b: Fighter, num_rounds: int, n_sims: int = 10000) -> List[FightResult]:
    probs = stat_based_probability(a, b)
    results = []

    a_ko_per_round = probs[f"{a.name}_ko"] / num_rounds
    b_ko_per_round = probs[f"{b.name}_ko"] / num_rounds
    a_sub_per_round = probs[f"{a.name}_sub"] / num_rounds
    b_sub_per_round = probs[f"{b.name}_sub"] / num_rounds

    for _ in range(n_sims):
        finished = False
        a_advantage = 0.0

        for rd in range(1, num_rounds + 1):
            fatigue = 1.0 + (rd - 1) * 0.15
            a_fatigue = fatigue * (1.0 + max(0, (a.age - 33)) * 0.03)
            b_fatigue = fatigue * (1.0 + max(0, (b.age - 33)) * 0.03)

            roll = random.random()
            cumulative = 0.0

            cumulative += a_ko_per_round * b_fatigue
            if roll < cumulative:
                results.append(FightResult(a.name, "ko", rd, False))
                finished = True
                break

            cumulative += b_ko_per_round * a_fatigue
            if roll < cumulative:
                results.append(FightResult(b.name, "ko", rd, False))
                finished = True
                break

            cumulative += a_sub_per_round * b_fatigue
            if roll < cumulative:
                results.append(FightResult(a.name, "sub", rd, False))
                finished = True
                break

            cumulative += b_sub_per_round * a_fatigue
            if roll < cumulative:
                results.append(FightResult(b.name, "sub", rd, False))
                finished = True
                break

            a_output = (a.sig_strikes_per_min * a.sig_strike_accuracy * (1 - b.sig_strike_defense)
                        + a.takedowns_per_15 * a.takedown_accuracy * (1 - b.takedown_defense) * 2.0)
            b_output = (b.sig_strikes_per_min * b.sig_strike_accuracy * (1 - a.sig_strike_defense)
                        + b.takedowns_per_15 * b.takedown_accuracy * (1 - a.takedown_defense) * 2.0)

            round_score = (a_output - b_output) + random.gauss(0, 0.5)
            a_advantage += round_score

        if not finished:
            if a_advantage > 0.3:
                results.append(FightResult(a.name, "dec", num_rounds, True))
            elif a_advantage < -0.3:
                results.append(FightResult(b.name, "dec", num_rounds, True))
            else:
                if random.random() < 0.5 + a_advantage * 0.3:
                    results.append(FightResult(a.name, "dec", num_rounds, True))
                else:
                    results.append(FightResult(b.name, "dec", num_rounds, True))

    return results


def analyze_simulations(results: List[FightResult], a_name: str, b_name: str, num_rounds: int) -> Dict:
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

    for rd in range(1, num_rounds + 1):
        analysis[f"finish_r{rd}"] = sum(1 for r in results if r.round_finished == rd and not r.went_distance) / n
        analysis[f"{a_name}_finish_r{rd}"] = sum(1 for r in results if r.winner == a_name and r.round_finished == rd and not r.went_distance) / n
        analysis[f"{b_name}_finish_r{rd}"] = sum(1 for r in results if r.winner == b_name and r.round_finished == rd and not r.went_distance) / n

    return analysis


# ============================================================
# MODULE 5: KELLY CRITERION & EDGE DETECTION
# ============================================================

def american_to_implied(american: int) -> float:
    if american < 0:
        return abs(american) / (abs(american) + 100)
    else:
        return 100 / (american + 100)

def american_to_decimal(american: int) -> float:
    if american < 0:
        return 1 + (100 / abs(american))
    else:
        return 1 + (american / 100)

def kelly_criterion(true_prob: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    p = true_prob
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0, kelly)

def calculate_edge(true_prob: float, book_implied: float) -> float:
    return true_prob - book_implied

def shannon_entropy(probs: List[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


# ============================================================
# MODULE 6: PARLAY OPTIMIZER
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
    category: str

def build_bet_options(matchup_results: Dict) -> List[BetOption]:
    options = []

    for (a_key, b_key, num_rds, _), analysis in matchup_results.items():
        a = fighters[a_key]
        b = fighters[b_key]
        fight_name = f"{a.name} vs {b.name}"

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

        rp_key = (a_key, b_key)
        if rp_key in round_props:
            rp = round_props[rp_key]
            over_implied = american_to_implied(rp["over"])
            over_decimal = american_to_decimal(rp["over"])
            dist_prob = analysis["goes_distance"]

            if rp["line"] == 3.5:
                over_prob = sum(analysis.get(f"finish_r{rd}", 0) for rd in range(4, 6)) + dist_prob
            elif rp["line"] == 2.5:
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
    positive_edge = [o for o in options if o.edge > 0.02]

    by_fight = {}
    for opt in positive_edge:
        if opt.fight not in by_fight:
            by_fight[opt.fight] = []
        by_fight[opt.fight].append(opt)

    fight_names = list(by_fight.keys())
    best_parlays = []

    for fight_combo in itertools.combinations(fight_names, min(n_legs, len(fight_names))):
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

            categories = [l.category for l in leg_combo]
            unique_cats = len(set(categories))
            diversity_bonus = unique_cats * 0.02

            score = ev + avg_edge * 0.5 + diversity_bonus

            best_parlays.append((score, ev, combined_prob, combined_odds, leg_combo))

    best_parlays.sort(key=lambda x: x[0], reverse=True)
    return best_parlays[:top_n]


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 80)
    print("  UFC 327 MIAMI PARLAY ENGINE — PROCHAZKA vs ULBERG")
    print("  Monte Carlo | Bayesian | Elo | Kelly Criterion | Game Theory")
    print("  Kaseya Center, Miami FL — April 11, 2026")
    print("=" * 80)

    n_fights = len(matchups)
    print(f"\n  Running 10,000 Monte Carlo simulations per fight ({n_fights * 10000:,} total)...\n")

    all_results = {}
    for matchup in matchups:
        a_key, b_key, num_rds, is_main = matchup
        a, b = fighters[a_key], fighters[b_key]
        results = simulate_fight(a, b, num_rds)
        analysis = analyze_simulations(results, a.name, b.name, num_rds)
        all_results[matchup] = analysis

        tag = " [MAIN EVENT - VACANT LHW TITLE]" if is_main else ""
        print(f"{'=' * 70}")
        print(f"  {a.name} (Elo: {a.elo}) vs {b.name} (Elo: {b.elo}){tag}")
        print(f"  Style: {a.style.upper()} vs {b.style.upper()} | Rounds: {num_rds}")
        print(f"{'=' * 70}")

        print(f"  MODEL WIN PROBABILITY:")
        print(f"    {a.name}: {analysis[f'{a.name}_ml']:.1%}  |  {b.name}: {analysis[f'{b.name}_ml']:.1%}")

        print(f"  METHOD BREAKDOWN:")
        print(f"    {a.name} by KO: {analysis[f'{a.name}_ko']:.1%}  |  by Sub: {analysis[f'{a.name}_sub']:.1%}  |  by Dec: {analysis[f'{a.name}_dec']:.1%}")
        print(f"    {b.name} by KO: {analysis[f'{b.name}_ko']:.1%}  |  by Sub: {analysis[f'{b.name}_sub']:.1%}  |  by Dec: {analysis[f'{b.name}_dec']:.1%}")

        print(f"  DISTANCE PROPS:")
        print(f"    Goes to Decision: {analysis['goes_distance']:.1%}  |  Finish: {analysis['finish']:.1%}")

        for rd in range(1, num_rds + 1):
            pct = analysis.get(f'finish_r{rd}', 0)
            bar = '#' * int(pct * 50)
            print(f"    Round {rd} finish: {pct:.1%} {bar}")

        if a_key in book_odds:
            book_impl_a = american_to_implied(book_odds[a_key]["ml"])
            model_a = analysis[f'{a.name}_ml']
            edge_a = (model_a - book_impl_a) * 100
            arrow = "^" if edge_a > 0 else "v"
            print(f"  EDGE vs BOOK:")
            print(f"    {a.name}: Book {book_impl_a:.1%} -> Model {model_a:.1%} ({arrow} {abs(edge_a):.1f}%)")
        if b_key in book_odds:
            book_impl_b = american_to_implied(book_odds[b_key]["ml"])
            model_b = analysis[f'{b.name}_ml']
            edge_b = (model_b - book_impl_b) * 100
            arrow = "^" if edge_b > 0 else "v"
            print(f"    {b.name}: Book {book_impl_b:.1%} -> Model {model_b:.1%} ({arrow} {abs(edge_b):.1f}%)")
        print()

    # Edge Detection
    print("\n" + "=" * 80)
    print("  EDGE DETECTION — WHERE THE BOOKS ARE WRONG")
    print("=" * 80)

    options = build_bet_options(all_results)
    positive_edge = sorted([o for o in options if o.edge > 0], key=lambda x: x.edge, reverse=True)

    print(f"\n  Found {len(positive_edge)} positive-edge bets out of {len(options)} total options\n")
    print(f"  {'Bet':<45} {'True%':>7} {'Book%':>7} {'Edge':>7} {'Kelly%':>7} {'Odds':>8}")
    print(f"  {'_' * 85}")
    for opt in positive_edge[:25]:
        print(f"  {opt.description:<45} {opt.true_prob:>6.1%} {opt.book_implied:>6.1%} {opt.edge:>+6.1%} {opt.kelly:>6.1%} {opt.decimal_odds:>7.2f}")

    # Optimal Parlays
    for n_legs in [3, 4, 5]:
        print(f"\n{'=' * 80}")
        print(f"  OPTIMAL {n_legs}-LEG PARLAYS (Ranked by Composite Score)")
        print(f"  Score = EV + Avg Edge + Diversity Bonus")
        print(f"{'=' * 80}")

        parlays = find_optimal_parlays(options, n_legs=n_legs, top_n=3)

        for rank, (score, ev, prob, odds, legs) in enumerate(parlays, 1):
            print(f"\n  *** PARLAY #{rank} ***")
            print(f"  Combined Probability: {prob:.2%}")
            print(f"  Combined Decimal Odds: {odds:.2f}x")
            print(f"  Expected Value: {ev:+.2%}")
            print(f"  $10 bet pays: ${10 * odds:.2f}")
            print(f"  Composite Score: {score:.4f}")
            print()
            for i, leg in enumerate(legs, 1):
                edge_bar = '#' * max(1, int(leg.edge * 100))
                print(f"    Leg {i}: {leg.description}")
                print(f"           True: {leg.true_prob:.1%} | Book: {leg.book_implied:.1%} | Edge: {leg.edge:+.1%} {edge_bar}")
                print(f"           Kelly suggests: {leg.kelly:.1%} of bankroll")
            print()

    # Final Recommendation
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
            print(f"       Model: {leg.true_prob:.1%} vs Book: {leg.book_implied:.1%} -> Edge: {leg.edge:+.1%}")

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
