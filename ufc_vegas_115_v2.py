"""
UFC VEGAS 115: MOICANO vs DUNCAN — April 4, 2026
v2 ENGINE — Quantitative Analyst Upgrade

New in v2:
- Recency-weighted stats (last 3 fights dampening)
- X-Factor sensitivity analysis per fight
- Market disagreement alerts (confidence calibration)
- Scorecard simulator for decision fights (10-9, 10-8 round modeling)
- Situational notes per fighter
- Full parlay optimizer with method/distance props
"""

import random
import math
import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from copy import deepcopy

random.seed(42)

# ============================================================
# FIGHTER MODEL — v2 with recency + situational fields
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
    sig_strike_accuracy: float
    sig_strike_defense: float
    takedowns_per_15: float
    takedown_accuracy: float
    takedown_defense: float
    sub_attempts_per_15: float
    reach_inches: float
    age: int
    win_streak: int
    # --- v2 fields ---
    recent_form: float = 0.5       # 0.0 (terrible recent) to 1.0 (peak recent form)
    situational: str = ""           # camp changes, weight cuts, short notice, etc.
    style: str = "striker"
    gender: str = "M"
    elo: float = 1500.0
    # --- v3: competition level ---
    # 0.3 = mostly regional (RUF, LFA, Contender Series)
    # 0.5 = early UFC, unranked opponents
    # 0.7 = mid-level UFC, established opponents
    # 0.85 = ranked opponents regularly
    # 1.0 = title fights, elite competition
    competition_level: float = 0.5


# ============================================================
# FIGHTER DATABASE — with recent_form and situational notes
# ============================================================

fighters = {
    "Moicano": Fighter("Renato Moicano", 20, 7, 5, 8, 7, 2, 4, 1,
                        sig_strikes_per_min=2.14, sig_strike_accuracy=0.37,
                        sig_strike_defense=0.63, takedowns_per_15=2.98,
                        takedown_accuracy=0.44, takedown_defense=0.73,
                        sub_attempts_per_15=1.2, reach_inches=71.0,
                        age=36, win_streak=1, style="grappler",
                        recent_form=0.55,
                        situational="Coming off interim title loss to Islam; grappling-first gameplan expected",
                        competition_level=0.90),  # fought for interim title, top-5 wins
    "Duncan": Fighter("Chris Duncan", 15, 2, 7, 4, 4, 1, 1, 0,
                       sig_strikes_per_min=4.85, sig_strike_accuracy=0.48,
                       sig_strike_defense=0.56, takedowns_per_15=1.20,
                       takedown_accuracy=0.38, takedown_defense=0.65,
                       sub_attempts_per_15=0.8, reach_inches=71.5,
                       age=32, win_streak=4, style="balanced",
                       recent_form=0.85,
                       situational="4-fight win streak, rising contender with heavy hands; first main event",
                        competition_level=0.65),
    "Jandiroba": Fighter("Virna Jandiroba", 22, 4, 2, 10, 10, 1, 0, 3,
                          sig_strikes_per_min=3.45, sig_strike_accuracy=0.42,
                          sig_strike_defense=0.55, takedowns_per_15=3.80,
                          takedown_accuracy=0.40, takedown_defense=0.70,
                          sub_attempts_per_15=1.5, reach_inches=63.0,
                          age=37, win_streak=1, style="grappler", gender="F",
                          recent_form=0.45,
                          situational="Age 37, last 3: 1W 2L, lost title fight; may be declining",
                        competition_level=0.85),
    "Ricci": Fighter("Tabatha Ricci", 13, 3, 3, 3, 7, 0, 2, 1,
                      sig_strikes_per_min=4.60, sig_strike_accuracy=0.44,
                      sig_strike_defense=0.58, takedowns_per_15=1.10,
                      takedown_accuracy=0.35, takedown_defense=0.62,
                      sub_attempts_per_15=0.4, reach_inches=64.0,
                      age=28, win_streak=2, style="striker", gender="F",
                      recent_form=0.70,
                      situational="On 2-fight win streak, improving standup; youth advantage",
                        competition_level=0.6),
    "Yakhyaev": Fighter("Abdul Rakhman Yakhyaev", 10, 1, 6, 2, 2, 0, 1, 0,
                         sig_strikes_per_min=5.10, sig_strike_accuracy=0.50,
                         sig_strike_defense=0.52, takedowns_per_15=1.80,
                         takedown_accuracy=0.45, takedown_defense=0.70,
                         sub_attempts_per_15=0.3, reach_inches=76.0,
                         age=28, win_streak=3, style="striker",
                         recent_form=0.80,
                         situational="Explosive power striker; small UFC sample size (2 fights)",
                        competition_level=0.35),
    "Ribeiro": Fighter("Brendson Ribeiro", 16, 4, 8, 5, 3, 2, 1, 1,
                        sig_strikes_per_min=4.20, sig_strike_accuracy=0.45,
                        sig_strike_defense=0.48, takedowns_per_15=0.90,
                        takedown_accuracy=0.33, takedown_defense=0.55,
                        sub_attempts_per_15=0.5, reach_inches=75.0,
                        age=29, win_streak=-1, style="balanced",
                        recent_form=0.40,
                        situational="Coming off a loss; durable but hittable",
                        competition_level=0.5),
    "Estevam": Fighter("Rafael Estevam", 14, 2, 5, 4, 5, 1, 1, 0,
                        sig_strikes_per_min=4.70, sig_strike_accuracy=0.46,
                        sig_strike_defense=0.54, takedowns_per_15=2.50,
                        takedown_accuracy=0.42, takedown_defense=0.68,
                        sub_attempts_per_15=0.6, reach_inches=67.0,
                        age=26, win_streak=2, style="balanced",
                        recent_form=0.75,
                        situational="Young prospect, 2-fight streak; solid all-around game",
                        competition_level=0.4),
    "Ewing": Fighter("Ethyn Ewing", 11, 3, 4, 2, 5, 1, 1, 1,
                      sig_strikes_per_min=4.30, sig_strike_accuracy=0.43,
                      sig_strike_defense=0.50, takedowns_per_15=1.60,
                      takedown_accuracy=0.37, takedown_defense=0.60,
                      sub_attempts_per_15=0.3, reach_inches=68.0,
                      age=27, win_streak=1, style="balanced",
                      recent_form=0.50,
                      situational="Solid but unspectacular; step up in competition",
                        competition_level=0.45),
    "McMillen": Fighter("Tommy McMillen", 10, 1, 5, 3, 2, 0, 1, 0,
                         sig_strikes_per_min=5.20, sig_strike_accuracy=0.49,
                         sig_strike_defense=0.58, takedowns_per_15=1.50,
                         takedown_accuracy=0.40, takedown_defense=0.72,
                         sub_attempts_per_15=0.4, reach_inches=72.0,
                         age=27, win_streak=5, style="striker",
                         recent_form=0.85,
                         situational="5-fight win streak; dangerous finisher on the rise",
                        competition_level=0.35),
    "Zecchini": Fighter("Manolo Zecchini", 8, 3, 3, 2, 3, 1, 1, 1,
                         sig_strikes_per_min=3.80, sig_strike_accuracy=0.40,
                         sig_strike_defense=0.45, takedowns_per_15=1.00,
                         takedown_accuracy=0.30, takedown_defense=0.50,
                         sub_attempts_per_15=0.3, reach_inches=70.0,
                         age=30, win_streak=-1, style="balanced",
                         recent_form=0.35,
                         situational="Coming off a loss; outmatched on paper",
                        competition_level=0.4),
    "Ruchala": Fighter("Robert Ruchala", 14, 3, 6, 3, 5, 1, 1, 1,
                        sig_strikes_per_min=4.40, sig_strike_accuracy=0.44,
                        sig_strike_defense=0.52, takedowns_per_15=1.20,
                        takedown_accuracy=0.35, takedown_defense=0.58,
                        sub_attempts_per_15=0.3, reach_inches=70.0,
                        age=28, win_streak=1, style="striker",
                        recent_form=0.55,
                        situational="Decent striker but facing a more complete fighter",
                        competition_level=0.4),
    "Delano": Fighter("Jose Delano", 15, 2, 6, 5, 4, 1, 0, 1,
                       sig_strikes_per_min=4.80, sig_strike_accuracy=0.47,
                       sig_strike_defense=0.56, takedowns_per_15=2.00,
                       takedown_accuracy=0.42, takedown_defense=0.66,
                       sub_attempts_per_15=0.7, reach_inches=71.0,
                       age=27, win_streak=3, style="balanced",
                       recent_form=0.80,
                       situational="3-fight streak; well-rounded with finishing ability",
                        competition_level=0.45),
    "Vannata": Fighter("Lando Vannata", 12, 7, 5, 3, 4, 3, 2, 2,
                        sig_strikes_per_min=4.50, sig_strike_accuracy=0.44,
                        sig_strike_defense=0.50, takedowns_per_15=0.80,
                        takedown_accuracy=0.33, takedown_defense=0.60,
                        sub_attempts_per_15=0.5, reach_inches=72.0,
                        age=33, win_streak=1, style="striker",
                        recent_form=0.50,
                        situational="Veteran, inconsistent; capable of highlight KO or flat performance",
                        competition_level=0.7),
    "Flowers": Fighter("Darrius Flowers", 8, 1, 4, 2, 2, 0, 1, 0,
                        sig_strikes_per_min=4.00, sig_strike_accuracy=0.42,
                        sig_strike_defense=0.52, takedowns_per_15=1.30,
                        takedown_accuracy=0.36, takedown_defense=0.58,
                        sub_attempts_per_15=0.3, reach_inches=74.0,
                        age=27, win_streak=3, style="balanced",
                        recent_form=0.70,
                        situational="Small sample UFC debut; athletic but untested at this level",
                        competition_level=0.3),
    "Bekoev": Fighter("Azamat Bekoev", 16, 0, 8, 5, 3, 0, 0, 0,
                       sig_strikes_per_min=5.50, sig_strike_accuracy=0.52,
                       sig_strike_defense=0.60, takedowns_per_15=2.40,
                       takedown_accuracy=0.48, takedown_defense=0.75,
                       sub_attempts_per_15=0.8, reach_inches=74.0,
                       age=28, win_streak=16, style="balanced",
                       recent_form=0.90,
                       situational="Undefeated, 16-0; elite prospect but regional competition caveat",
                        competition_level=0.35),
    "Gore": Fighter("Tresean Gore", 6, 4, 4, 0, 2, 3, 0, 1,
                     sig_strikes_per_min=3.90, sig_strike_accuracy=0.41,
                     sig_strike_defense=0.42, takedowns_per_15=0.50,
                     takedown_accuracy=0.25, takedown_defense=0.45,
                     sub_attempts_per_15=0.1, reach_inches=75.0,
                     age=28, win_streak=-1, style="striker",
                     recent_form=0.30,
                     situational="3 KO losses; very hittable, chin may be compromised",
                        competition_level=0.55),
    "Petersen": Fighter("Thomas Petersen", 10, 4, 5, 2, 3, 2, 1, 1,
                         sig_strikes_per_min=4.10, sig_strike_accuracy=0.43,
                         sig_strike_defense=0.50, takedowns_per_15=1.40,
                         takedown_accuracy=0.38, takedown_defense=0.55,
                         sub_attempts_per_15=0.3, reach_inches=77.0,
                         age=30, win_streak=1, style="balanced",
                         recent_form=0.55,
                         situational="Reach advantage; solid but not elite",
                        competition_level=0.5),
    "Pat": Fighter("Guilherme Pat", 10, 2, 5, 3, 2, 1, 0, 1,
                    sig_strikes_per_min=4.30, sig_strike_accuracy=0.45,
                    sig_strike_defense=0.52, takedowns_per_15=1.20,
                    takedown_accuracy=0.35, takedown_defense=0.60,
                    sub_attempts_per_15=0.4, reach_inches=76.0,
                    age=28, win_streak=2, style="balanced",
                    recent_form=0.65,
                    situational="2-fight streak; finishing ability but limited UFC data",
                        competition_level=0.4),
    "Costa": Fighter("Alessandro Costa", 16, 4, 7, 5, 4, 2, 1, 1,
                      sig_strikes_per_min=5.00, sig_strike_accuracy=0.47,
                      sig_strike_defense=0.55, takedowns_per_15=2.10,
                      takedown_accuracy=0.43, takedown_defense=0.65,
                      sub_attempts_per_15=0.6, reach_inches=66.0,
                      age=29, win_streak=2, style="balanced",
                      recent_form=0.75,
                      situational="Well-rounded; 2-fight streak with solid output",
                        competition_level=0.6),
    "Nicoll": Fighter("Stewart Nicoll", 8, 2, 3, 2, 3, 1, 0, 1,
                       sig_strikes_per_min=3.60, sig_strike_accuracy=0.40,
                       sig_strike_defense=0.48, takedowns_per_15=0.80,
                       takedown_accuracy=0.30, takedown_defense=0.55,
                       sub_attempts_per_15=0.2, reach_inches=67.0,
                       age=27, win_streak=1, style="striker",
                       recent_form=0.50,
                       situational="Limited UFC experience; outgunned statistically",
                        competition_level=0.35),
    "Gatto": Fighter("Melissa Gatto", 14, 4, 4, 5, 5, 1, 1, 2,
                      sig_strikes_per_min=4.20, sig_strike_accuracy=0.44,
                      sig_strike_defense=0.52, takedowns_per_15=0.90,
                      takedown_accuracy=0.35, takedown_defense=0.60,
                      sub_attempts_per_15=0.6, reach_inches=66.0,
                      age=31, win_streak=1, style="balanced", gender="F",
                      recent_form=0.60,
                      situational="Inconsistent; capable of subs but can be outpointed",
                        competition_level=0.6),
    "Barbosa": Fighter("Dione Barbosa", 10, 3, 3, 2, 5, 1, 1, 1,
                        sig_strikes_per_min=3.80, sig_strike_accuracy=0.41,
                        sig_strike_defense=0.50, takedowns_per_15=1.20,
                        takedown_accuracy=0.33, takedown_defense=0.55,
                        sub_attempts_per_15=0.3, reach_inches=65.0,
                        age=29, win_streak=1, style="balanced", gender="F",
                        recent_form=0.55,
                        situational="Decision-heavy fighter; limited finishing power",
                        competition_level=0.5),
    "Cowan": Fighter("Hailey Cowan", 10, 3, 3, 3, 4, 1, 1, 1,
                      sig_strikes_per_min=4.00, sig_strike_accuracy=0.43,
                      sig_strike_defense=0.50, takedowns_per_15=1.50,
                      takedown_accuracy=0.38, takedown_defense=0.58,
                      sub_attempts_per_15=0.4, reach_inches=66.0,
                      age=27, win_streak=1, style="balanced", gender="F",
                      recent_form=0.55,
                      situational="Balanced skillset; competitive but not dominant",
                        competition_level=0.45),
    "Pereira": Fighter("Alice Pereira", 11, 2, 4, 3, 4, 0, 1, 1,
                        sig_strikes_per_min=4.30, sig_strike_accuracy=0.45,
                        sig_strike_defense=0.53, takedowns_per_15=1.10,
                        takedown_accuracy=0.36, takedown_defense=0.60,
                        sub_attempts_per_15=0.5, reach_inches=65.0,
                        age=28, win_streak=2, style="balanced", gender="F",
                        recent_form=0.65,
                        situational="2-fight streak; slight edge in recent momentum",
                        competition_level=0.5),
}

matchups = [
    ("Moicano", "Duncan", 5, True),
    ("Jandiroba", "Ricci", 3, False),
    ("Yakhyaev", "Ribeiro", 3, False),
    ("Estevam", "Ewing", 3, False),
    ("McMillen", "Zecchini", 3, False),
    ("Ruchala", "Delano", 3, False),
    ("Vannata", "Flowers", 3, False),
    ("Bekoev", "Gore", 3, False),
    ("Petersen", "Pat", 3, False),
    ("Costa", "Nicoll", 3, False),
    ("Gatto", "Barbosa", 3, False),
    ("Cowan", "Pereira", 3, False),
]

book_odds = {
    "Moicano":   {"ml": +170, "by_ko": +1200, "by_sub": +500, "by_dec": +350},
    "Duncan":    {"ml": -205, "by_ko": -110,  "by_sub": +800, "by_dec": +350},
    "Jandiroba": {"ml": +110, "by_ko": +2500, "by_sub": +300, "by_dec": +200},
    "Ricci":     {"ml": -130, "by_ko": +500,  "by_sub": +1200,"by_dec": -110},
    "Yakhyaev":  {"ml": -250, "by_ko": -110,  "by_sub": +1500,"by_dec": +200},
    "Ribeiro":   {"ml": +210, "by_ko": +400,  "by_sub": +600, "by_dec": +700},
    "Estevam":   {"ml": -200, "by_ko": +350,  "by_sub": +600, "by_dec": +110},
    "Ewing":     {"ml": +170, "by_ko": +400,  "by_sub": +800, "by_dec": +350},
    "McMillen":  {"ml": -500, "by_ko": -110,  "by_sub": +500, "by_dec": +300},
    "Zecchini":  {"ml": +385, "by_ko": +700,  "by_sub": +1000,"by_dec": +800},
    "Ruchala":   {"ml": +235, "by_ko": +450,  "by_sub": +1200,"by_dec": +500},
    "Delano":    {"ml": -275, "by_ko": +200,  "by_sub": +400, "by_dec": +110},
    "Vannata":   {"ml": -170, "by_ko": +150,  "by_sub": +600, "by_dec": +200},
    "Flowers":   {"ml": +145, "by_ko": +350,  "by_sub": +800, "by_dec": +300},
    "Bekoev":    {"ml": -850, "by_ko": +100,  "by_sub": +400, "by_dec": +200},
    "Gore":      {"ml": +625, "by_ko": +700,  "by_sub": +3000,"by_dec": +1500},
    "Petersen":  {"ml": -130, "by_ko": +300,  "by_sub": +800, "by_dec": +175},
    "Pat":       {"ml": +110, "by_ko": +300,  "by_sub": +500, "by_dec": +250},
    "Costa":     {"ml": -450, "by_ko": +100,  "by_sub": +400, "by_dec": +200},
    "Nicoll":    {"ml": +350, "by_ko": +600,  "by_sub": +1000,"by_dec": +800},
    "Gatto":     {"ml": -125, "by_ko": +500,  "by_sub": +300, "by_dec": +110},
    "Barbosa":   {"ml": +105, "by_ko": +600,  "by_sub": +800, "by_dec": +200},
    "Cowan":     {"ml": +120, "by_ko": +500,  "by_sub": +400, "by_dec": +225},
    "Pereira":   {"ml": -140, "by_ko": +400,  "by_sub": +350, "by_dec": +150},
}

round_props = {
    ("Moicano", "Duncan"):   {"line": 3.5, "over": -130, "under": +105},
    ("Jandiroba", "Ricci"):  {"line": 2.5, "over": -110, "under": -110},
    ("Yakhyaev", "Ribeiro"): {"line": 2.5, "over": +110, "under": -130},
    ("Estevam", "Ewing"):    {"line": 2.5, "over": -150, "under": +125},
    ("McMillen", "Zecchini"):{"line": 2.5, "over": +100, "under": -120},
    ("Ruchala", "Delano"):   {"line": 2.5, "over": -130, "under": +105},
    ("Vannata", "Flowers"):  {"line": 2.5, "over": -140, "under": +115},
    ("Bekoev", "Gore"):      {"line": 1.5, "over": -110, "under": -110},
    ("Petersen", "Pat"):     {"line": 2.5, "over": -130, "under": +105},
    ("Costa", "Nicoll"):     {"line": 2.5, "over": -110, "under": -110},
    ("Gatto", "Barbosa"):    {"line": 2.5, "over": -170, "under": +140},
    ("Cowan", "Pereira"):    {"line": 2.5, "over": -150, "under": +125},
}


# ============================================================
# MODULE 1: RECENCY-WEIGHTED ELO RATING
# ============================================================

STYLE_MATRIX = {
    "striker":  {"striker": 0.0, "grappler": 0.06, "wrestler": 0.03, "balanced": 0.0},
    "grappler": {"striker": -0.02, "grappler": 0.0, "wrestler": 0.04, "balanced": -0.01},
    "wrestler": {"striker": 0.08, "grappler": -0.02, "wrestler": 0.0, "balanced": 0.02},
    "balanced": {"striker": 0.02, "grappler": 0.03, "wrestler": -0.01, "balanced": 0.0},
}

def calculate_elo(f: Fighter) -> float:
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

    # v2: Recency adjustment — recent_form shifts Elo up to +/- 80 points
    # 0.5 = neutral, 1.0 = +80, 0.0 = -80
    recency_mod = (f.recent_form - 0.5) * 160

    return round(base + wc + streak + age_mod + act + recency_mod, 1)


def elo_wp(ea: float, eb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))


# ============================================================
# MODULE 2: STAT-BASED PROBABILITY (recency-weighted)
# ============================================================

def fighter_finish_rate(f: Fighter) -> float:
    """What % of this fighter's WINS come by finish (KO + SUB)?
    This is the personal 'closer' rate — a fighter who finishes 80% of
    their wins should have amplified finish probs vs the engine's default."""
    if f.wins == 0:
        return 0.5
    return (f.ko_wins + f.sub_wins) / f.wins


def mismatch_multiplier(elo_a: float, elo_b: float) -> Tuple[float, float]:
    """When there's a big Elo gap, the dominant fighter finishes more often.
    Historical UFC data: -500 favorites finish ~65-70% of the time.
    Returns (dominant_boost, underdog_boost)."""
    gap = abs(elo_a - elo_b)
    if gap < 150:
        return (1.0, 1.0)  # competitive fight, no adjustment
    # Scale: 150 gap = 1.0x, 250 = 1.3x, 400+ = 1.7x for dominant fighter
    dominant_boost = 1.0 + max(0, (gap - 150)) / 400
    dominant_boost = min(dominant_boost, 2.0)  # cap at 2x
    # Underdog's finish rate slightly suppressed in mismatches
    underdog_dampen = max(0.6, 1.0 - (gap - 150) / 800)
    if elo_a > elo_b:
        return (dominant_boost, underdog_dampen)
    else:
        return (underdog_dampen, dominant_boost)


def stat_prob(a: Fighter, b: Fighter) -> Dict[str, float]:
    ta = max(a.wins + a.losses, 1)
    tb = max(b.wins + b.losses, 1)

    asp = a.sig_strikes_per_min * a.sig_strike_accuracy
    bsp = b.sig_strikes_per_min * b.sig_strike_accuracy

    # v2: Recency dampens/amplifies finish rates
    a_rf = 0.7 + 0.6 * a.recent_form   # range: 0.7 to 1.3
    b_rf = 0.7 + 0.6 * b.recent_form

    # v3: Personal finish rate — fighters who close the show get boosted
    # A fighter with 80% finish rate on wins gets ~1.3x; 20% gets ~0.7x
    a_closer = 0.7 + 0.6 * fighter_finish_rate(a)
    b_closer = 0.7 + 0.6 * fighter_finish_rate(b)

    # v3: Mismatch multiplier — big Elo gaps = more finishes for the favorite
    a_mm, b_mm = mismatch_multiplier(a.elo, b.elo)

    p_a_ko = (a.ko_wins / ta) * (1 + b.ko_losses / tb) * (asp / max(bsp, 0.1)) * 0.35 * a_rf * a_closer * a_mm
    p_b_ko = (b.ko_wins / tb) * (1 + a.ko_losses / ta) * (bsp / max(asp, 0.1)) * 0.35 * b_rf * b_closer * b_mm

    atd = a.takedowns_per_15 * a.takedown_accuracy * (1 - b.takedown_defense)
    btd = b.takedowns_per_15 * b.takedown_accuracy * (1 - a.takedown_defense)

    p_a_sub = (a.sub_wins / ta) * (1 + b.sub_losses / tb) * (1 + atd) * a.sub_attempts_per_15 * 0.12 * a_rf * a_closer * a_mm
    p_b_sub = (b.sub_wins / tb) * (1 + a.sub_losses / ta) * (1 + btd) * b.sub_attempts_per_15 * 0.12 * b_rf * b_closer * b_mm

    rm = (a.reach_inches - b.reach_inches) * 0.004
    sm = STYLE_MATRIX[a.style][b.style]
    ep = elo_wp(a.elo, b.elo)

    # v3: Raise finish cap to 88% for extreme mismatches
    elo_gap = abs(a.elo - b.elo)
    finish_cap = min(0.88, 0.75 + max(0, elo_gap - 200) / 1000)

    tf = min(p_a_ko + p_b_ko + p_a_sub + p_b_sub, finish_cap)
    dp = 1.0 - tf - 0.01
    ads = max(0.2, min(0.8, ep + rm + sm))
    p_a_dec = dp * ads
    p_b_dec = dp * (1 - ads)

    t = p_a_ko + p_b_ko + p_a_sub + p_b_sub + p_a_dec + p_b_dec + 0.01
    return {
        f"{a.name}_ko": p_a_ko / t, f"{a.name}_sub": p_a_sub / t, f"{a.name}_dec": p_a_dec / t,
        f"{b.name}_ko": p_b_ko / t, f"{b.name}_sub": p_b_sub / t, f"{b.name}_dec": p_b_dec / t,
        "draw": 0.01 / t,
    }


# ============================================================
# MODULE 3: MONTE CARLO FIGHT SIMULATION
# ============================================================

def simulate(a: Fighter, b: Fighter, nr: int, ns: int = 10000) -> list:
    random.seed(42)
    pr = stat_prob(a, b)
    res = []

    # v3: Front-loaded round weights instead of even distribution
    # Finishes are more likely early, especially in mismatches
    # R1 gets the most weight, declining per round
    elo_gap = abs(a.elo - b.elo)
    if nr == 3:
        if elo_gap > 300:
            # Heavy mismatch: very front-loaded
            round_weights = [0.45, 0.33, 0.22]
        elif elo_gap > 150:
            # Moderate mismatch
            round_weights = [0.40, 0.33, 0.27]
        else:
            # Competitive: slight front-load (R1 still most common for finishes)
            round_weights = [0.37, 0.33, 0.30]
    elif nr == 5:
        if elo_gap > 300:
            round_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        elif elo_gap > 150:
            round_weights = [0.27, 0.23, 0.20, 0.17, 0.13]
        else:
            round_weights = [0.24, 0.22, 0.20, 0.18, 0.16]
    else:
        round_weights = [1.0 / nr] * nr  # fallback: even

    for _ in range(ns):
        fin = False
        adv = 0.0
        for rd in range(1, nr + 1):
            rw = round_weights[rd - 1]
            fat = 1.0 + (rd - 1) * 0.15
            af = fat * (1 + max(0, (a.age - 33)) * 0.03)
            bf = fat * (1 + max(0, (b.age - 33)) * 0.03)

            # v3: Per-round finish probs use front-loaded weights
            akr = pr[f"{a.name}_ko"] * rw
            bkr = pr[f"{b.name}_ko"] * rw
            asr = pr[f"{a.name}_sub"] * rw
            bsr = pr[f"{b.name}_sub"] * rw

            roll = random.random()
            c = 0.0
            c += akr * bf
            if roll < c:
                res.append((a.name, "ko", rd, False))
                fin = True
                break
            c += bkr * af
            if roll < c:
                res.append((b.name, "ko", rd, False))
                fin = True
                break
            c += asr * bf
            if roll < c:
                res.append((a.name, "sub", rd, False))
                fin = True
                break
            c += bsr * af
            if roll < c:
                res.append((b.name, "sub", rd, False))
                fin = True
                break

            ao = (a.sig_strikes_per_min * a.sig_strike_accuracy * (1 - b.sig_strike_defense)
                  + a.takedowns_per_15 * a.takedown_accuracy * (1 - b.takedown_defense) * 2)
            bo = (b.sig_strikes_per_min * b.sig_strike_accuracy * (1 - a.sig_strike_defense)
                  + b.takedowns_per_15 * b.takedown_accuracy * (1 - a.takedown_defense) * 2)
            adv += (ao - bo) + random.gauss(0, 0.5)

        if not fin:
            if adv > 0.3:
                res.append((a.name, "dec", nr, True))
            elif adv < -0.3:
                res.append((b.name, "dec", nr, True))
            else:
                if random.random() < 0.5 + adv * 0.3:
                    res.append((a.name, "dec", nr, True))
                else:
                    res.append((b.name, "dec", nr, True))
    return res


def analyze(res: list, an: str, bn: str, nr: int) -> Dict:
    n = len(res)
    d = {
        f"{an}_ml": sum(1 for w, m, r, dd in res if w == an) / n,
        f"{bn}_ml": sum(1 for w, m, r, dd in res if w == bn) / n,
        f"{an}_ko": sum(1 for w, m, r, dd in res if w == an and m == "ko") / n,
        f"{an}_sub": sum(1 for w, m, r, dd in res if w == an and m == "sub") / n,
        f"{an}_dec": sum(1 for w, m, r, dd in res if w == an and m == "dec") / n,
        f"{bn}_ko": sum(1 for w, m, r, dd in res if w == bn and m == "ko") / n,
        f"{bn}_sub": sum(1 for w, m, r, dd in res if w == bn and m == "sub") / n,
        f"{bn}_dec": sum(1 for w, m, r, dd in res if w == bn and m == "dec") / n,
        "distance": sum(1 for w, m, r, dd in res if dd) / n,
        "finish": sum(1 for w, m, r, dd in res if not dd) / n,
    }
    for rd in range(1, nr + 1):
        d[f"r{rd}_finish"] = sum(1 for w, m, r, dd in res if r == rd and not dd) / n
    return d


# ============================================================
# MODULE 4: SCORECARD SIMULATOR
# Models 10-9 and 10-8 rounds for decision fights
# ============================================================

def simulate_scorecard(a: Fighter, b: Fighter, nr: int, ns: int = 5000) -> Dict:
    """Simulate round-by-round scoring for fights that go to decision."""
    random.seed(42)
    scorecards = []

    ao_base = (a.sig_strikes_per_min * a.sig_strike_accuracy * (1 - b.sig_strike_defense)
               + a.takedowns_per_15 * a.takedown_accuracy * (1 - b.takedown_defense) * 2.0)
    bo_base = (b.sig_strikes_per_min * b.sig_strike_accuracy * (1 - a.sig_strike_defense)
               + b.takedowns_per_15 * b.takedown_accuracy * (1 - a.takedown_defense) * 2.0)

    # Recency affects round control
    ao_adj = ao_base * (0.85 + 0.30 * a.recent_form)
    bo_adj = bo_base * (0.85 + 0.30 * b.recent_form)

    for _ in range(ns):
        a_score = 0
        b_score = 0
        rounds_a = 0
        rounds_b = 0
        a_10_8s = 0
        b_10_8s = 0

        for rd in range(1, nr + 1):
            # Each round: output differential + noise
            margin = (ao_adj - bo_adj) + random.gauss(0, 0.6)

            if margin > 1.2:
                # 10-8 round for A (dominant)
                a_score += 10
                b_score += 8
                rounds_a += 1
                a_10_8s += 1
            elif margin > 0.15:
                # 10-9 round for A
                a_score += 10
                b_score += 9
                rounds_a += 1
            elif margin < -1.2:
                # 10-8 round for B
                a_score += 8
                b_score += 10
                rounds_b += 1
                b_10_8s += 1
            elif margin < -0.15:
                # 10-9 round for B
                a_score += 9
                b_score += 10
                rounds_b += 1
            else:
                # Swing round — could go either way
                if random.random() < 0.5:
                    a_score += 10
                    b_score += 9
                    rounds_a += 1
                else:
                    a_score += 9
                    b_score += 10
                    rounds_b += 1

        scorecards.append({
            "a_score": a_score, "b_score": b_score,
            "rounds_a": rounds_a, "rounds_b": rounds_b,
            "a_10_8s": a_10_8s, "b_10_8s": b_10_8s,
        })

    # Aggregate
    n = len(scorecards)
    a_wins_dec = sum(1 for s in scorecards if s["a_score"] > s["b_score"]) / n
    b_wins_dec = sum(1 for s in scorecards if s["b_score"] > s["a_score"]) / n
    draws = sum(1 for s in scorecards if s["a_score"] == s["b_score"]) / n

    avg_a = sum(s["a_score"] for s in scorecards) / n
    avg_b = sum(s["b_score"] for s in scorecards) / n

    # Most common scorecard
    from collections import Counter
    card_counts = Counter((s["a_score"], s["b_score"]) for s in scorecards)
    top_cards = card_counts.most_common(5)

    avg_10_8_a = sum(s["a_10_8s"] for s in scorecards) / n
    avg_10_8_b = sum(s["b_10_8s"] for s in scorecards) / n

    return {
        "a_dec_pct": a_wins_dec,
        "b_dec_pct": b_wins_dec,
        "draw_pct": draws,
        "avg_a_score": avg_a,
        "avg_b_score": avg_b,
        "top_scorecards": top_cards,
        "avg_10_8_a": avg_10_8_a,
        "avg_10_8_b": avg_10_8_b,
    }


# ============================================================
# MODULE 5: X-FACTOR SENSITIVITY ANALYSIS
# Perturb each variable +/- 10% and measure impact on win prob
# ============================================================

def sensitivity_analysis(a_key: str, b_key: str, nr: int, base_ana: Dict) -> Dict:
    """
    Scenario-based sensitivity: instead of perturbing individual stats (which
    lets triple-counted variables like sig_strikes_per_min always dominate),
    simulate concrete fight scenarios that test real strategic questions:

    1. "What if the fight stays standing?" — boost both fighters' strike defense,
       nerf takedown success. Measures how much the outcome depends on grappling.
    2. "What if the grappler gets the fight down?" — boost TD rates, nerf TDD.
       Measures how much the ground game swings things.
    3. "What if the underdog has a career-best night?" — boost underdog's form
       and key offensive stats. Measures upset variance.
    4. "What if fatigue is a factor?" — penalize the older/less-conditioned fighter.
    5. "What if the finisher lands early?" — boost dominant fighter's KO/sub rates.

    Each scenario gets a plain-English label so the output is INSIGHT, not a
    stat name with a multiplier.
    """
    a_orig = fighters[a_key]
    b_orig = fighters[b_key]
    a_base_wp = base_ana[f"{a_orig.name}_ml"]

    # Identify roles
    a_is_favorite = a_base_wp > 0.5
    fav, dog = (a_orig, b_orig) if a_is_favorite else (b_orig, a_orig)
    fav_key, dog_key = (a_key, b_key) if a_is_favorite else (b_key, a_key)
    fav_is_a = a_is_favorite

    scenarios = []

    def run_scenario(mod_a: Fighter, mod_b: Fighter) -> float:
        """Run a quick sim with modified fighters, return A's win prob."""
        mod_a.elo = calculate_elo(mod_a)
        mod_b.elo = calculate_elo(mod_b)
        res = simulate(mod_a, mod_b, nr, ns=2000)
        ana = analyze(res, mod_a.name, mod_b.name, nr)
        return ana[f"{mod_a.name}_ml"]

    # --- Scenario 1: Fight stays standing (TDD boosted, TD rate nerfed) ---
    ma, mb = deepcopy(a_orig), deepcopy(b_orig)
    ma.takedown_defense = min(0.95, ma.takedown_defense * 1.3)
    mb.takedown_defense = min(0.95, mb.takedown_defense * 1.3)
    ma.takedowns_per_15 *= 0.5
    mb.takedowns_per_15 *= 0.5
    ma.sub_attempts_per_15 *= 0.5
    mb.sub_attempts_per_15 *= 0.5
    wp_standing = run_scenario(ma, mb)
    delta_standing = wp_standing - a_base_wp
    # Who benefits when it stays standing? The better striker.
    standing_benefits = a_orig.name if delta_standing > 0 else b_orig.name
    scenarios.append({
        "scenario": f"Fight stays on the feet",
        "detail": f"Benefits {standing_benefits} ({abs(delta_standing):+.1%} shift)",
        "delta": abs(delta_standing),
        "wp_scenario": wp_standing,
        "direction": "favors A" if delta_standing > 0 else "favors B",
    })

    # --- Scenario 2: Fight goes to the ground (TD rates boosted, TDD nerfed) ---
    ma, mb = deepcopy(a_orig), deepcopy(b_orig)
    ma.takedowns_per_15 *= 1.5
    mb.takedowns_per_15 *= 1.5
    ma.takedown_defense *= 0.7
    mb.takedown_defense *= 0.7
    ma.sub_attempts_per_15 *= 1.5
    mb.sub_attempts_per_15 *= 1.5
    wp_ground = run_scenario(ma, mb)
    delta_ground = wp_ground - a_base_wp
    ground_benefits = a_orig.name if delta_ground > 0 else b_orig.name
    scenarios.append({
        "scenario": f"Fight goes to the ground",
        "detail": f"Benefits {ground_benefits} ({abs(delta_ground):+.1%} shift)",
        "delta": abs(delta_ground),
        "wp_scenario": wp_ground,
        "direction": "favors A" if delta_ground > 0 else "favors B",
    })

    # --- Scenario 3: Underdog has career-best night ---
    ma, mb = deepcopy(a_orig), deepcopy(b_orig)
    if fav_is_a:
        mb.recent_form = min(1.0, mb.recent_form + 0.3)
        mb.sig_strike_accuracy = min(0.65, mb.sig_strike_accuracy * 1.15)
        mb.takedown_accuracy = min(0.65, mb.takedown_accuracy * 1.15)
        mb.sig_strike_defense = min(0.75, mb.sig_strike_defense * 1.15)
    else:
        ma.recent_form = min(1.0, ma.recent_form + 0.3)
        ma.sig_strike_accuracy = min(0.65, ma.sig_strike_accuracy * 1.15)
        ma.takedown_accuracy = min(0.65, ma.takedown_accuracy * 1.15)
        ma.sig_strike_defense = min(0.75, ma.sig_strike_defense * 1.15)
    wp_upset = run_scenario(ma, mb)
    delta_upset = abs(wp_upset - a_base_wp)
    scenarios.append({
        "scenario": f"{dog.name} has career-best performance",
        "detail": f"Fav win prob drops to {wp_upset:.1%}" if fav_is_a else f"Fav win prob drops to {1-wp_upset:.1%}",
        "delta": delta_upset,
        "wp_scenario": wp_upset,
        "direction": "upset risk",
    })

    # --- Scenario 4: Fatigue/age factor (older fighter fades) ---
    older = a_orig if a_orig.age > b_orig.age else b_orig
    younger = b_orig if a_orig.age > b_orig.age else a_orig
    if abs(a_orig.age - b_orig.age) >= 2:
        ma, mb = deepcopy(a_orig), deepcopy(b_orig)
        if a_orig.age > b_orig.age:
            ma.sig_strikes_per_min *= 0.85
            ma.takedowns_per_15 *= 0.85
            ma.recent_form = max(0.0, ma.recent_form - 0.15)
        else:
            mb.sig_strikes_per_min *= 0.85
            mb.takedowns_per_15 *= 0.85
            mb.recent_form = max(0.0, mb.recent_form - 0.15)
        wp_fatigue = run_scenario(ma, mb)
        delta_fatigue = abs(wp_fatigue - a_base_wp)
        scenarios.append({
            "scenario": f"{older.name} fades late (age {older.age})",
            "detail": f"{younger.name} benefits ({delta_fatigue:.1%} swing)",
            "delta": delta_fatigue,
            "wp_scenario": wp_fatigue,
            "direction": f"favors {younger.name}",
        })

    # --- Scenario 5: Early finish (dominant fighter's power amplified) ---
    ma, mb = deepcopy(a_orig), deepcopy(b_orig)
    if fav_is_a:
        ma.ko_wins = int(ma.ko_wins * 1.5) + 1
        ma.sub_wins = int(ma.sub_wins * 1.3)
    else:
        mb.ko_wins = int(mb.ko_wins * 1.5) + 1
        mb.sub_wins = int(mb.sub_wins * 1.3)
    wp_finish = run_scenario(ma, mb)
    delta_finish = abs(wp_finish - a_base_wp)
    scenarios.append({
        "scenario": f"{fav.name}'s power shows up early",
        "detail": f"Finish prob increases, win prob -> {wp_finish:.1%}" if fav_is_a else f"Finish prob increases, fav win prob -> {1-wp_finish:.1%}",
        "delta": delta_finish,
        "wp_scenario": wp_finish,
        "direction": f"favors {fav.name}",
    })

    # Sort by impact
    scenarios.sort(key=lambda x: x["delta"], reverse=True)
    return {
        "x_factor": scenarios[0] if scenarios else None,
        "top_scenarios": scenarios,
    }


# ============================================================
# MODULE 6: ODDS & EDGE UTILITIES
# ============================================================

def american_to_implied(ml: int) -> float:
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)

def american_to_decimal(ml: int) -> float:
    if ml < 0:
        return 1 + (100 / abs(ml))
    return 1 + (ml / 100)

def kelly(tp: float, do: float) -> float:
    b = do - 1
    if b == 0:
        return 0
    return max(0, (b * tp - (1 - tp)) / b)

def shannon_entropy(probs: List[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


# ============================================================
# MODULE 7: BET OPTIONS & PARLAY OPTIMIZER
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
    kelly_frac: float
    category: str   # ml, method, distance

def build_bet_options(all_analysis: Dict, matchup_list: list,
                      adj_probs: Dict = None) -> List[BetOption]:
    options = []
    for a_key, b_key, nr, _ in matchup_list:
        a, b = fighters[a_key], fighters[b_key]
        ana = all_analysis[(a_key, b_key)]
        fight = f"{a.name} vs {b.name}"

        # Get variance adjustment ratio if available
        adj_ratio = 1.0
        if adj_probs and (a_key, b_key) in adj_probs:
            ap = adj_probs[(a_key, b_key)]
            if ap["base_wp"] > 0:
                # How much did adjustment shrink the favorite's probability?
                # Apply the same ratio to all method probs for that fighter
                adj_ratio_fav = ap["adj_wp"] / ap["base_wp"]
                adj_ratio_dog = (1.0 - ap["adj_wp"]) / max(1.0 - ap["base_wp"], 0.01)

        for fkey, fn in [(a_key, a.name), (b_key, b.name)]:
            if fkey not in book_odds:
                continue
            odds = book_odds[fkey]

            # Apply variance adjustment: favorite's probs get discounted,
            # underdog's probs get boosted proportionally
            if adj_probs and (a_key, b_key) in adj_probs:
                ap = adj_probs[(a_key, b_key)]
                is_fav = (fn == ap["fav_name"])
                ratio = adj_ratio_fav if is_fav else adj_ratio_dog
            else:
                ratio = 1.0

            # Moneyline — use variance-adjusted probability
            ml_imp = american_to_implied(odds["ml"])
            ml_dec = american_to_decimal(odds["ml"])
            tp = ana[f"{fn}_ml"] * ratio
            e = tp - ml_imp
            k = kelly(tp, ml_dec)
            options.append(BetOption(fight, f"{fn} ML", f"{fn} ML ({odds['ml']:+d})",
                                     tp, ml_imp, ml_dec, e, k, "ml"))

            # Method bets — also variance-adjusted
            for method, odds_key in [("ko", "by_ko"), ("sub", "by_sub"), ("dec", "by_dec")]:
                if odds_key not in odds:
                    continue
                m_imp = american_to_implied(odds[odds_key])
                m_dec = american_to_decimal(odds[odds_key])
                m_tp = ana.get(f"{fn}_{method}", 0) * ratio
                m_e = m_tp - m_imp
                m_k = kelly(m_tp, m_dec)
                label = {"ko": "KO/TKO", "sub": "SUB", "dec": "DEC"}[method]
                options.append(BetOption(fight, f"{fn} by {label}",
                                         f"{fn} by {label} ({odds[odds_key]:+d})",
                                         m_tp, m_imp, m_dec, m_e, m_k, "method"))

        # Distance props
        rp_key = (a_key, b_key)
        if rp_key in round_props:
            rp = round_props[rp_key]
            dist_prob = ana["distance"]

            if rp["line"] == 3.5:
                over_prob = sum(ana.get(f"r{rd}_finish", 0) for rd in range(4, 6)) + dist_prob
            elif rp["line"] == 2.5:
                over_prob = dist_prob
            elif rp["line"] == 1.5:
                over_prob = sum(ana.get(f"r{rd}_finish", 0) for rd in range(2, nr + 1)) + dist_prob
            else:
                over_prob = dist_prob

            over_imp = american_to_implied(rp["over"])
            over_dec = american_to_decimal(rp["over"])
            over_e = over_prob - over_imp
            over_k = kelly(over_prob, over_dec)
            options.append(BetOption(fight, f"Over {rp['line']}", f"Over {rp['line']} rds ({rp['over']:+d})",
                                     over_prob, over_imp, over_dec, over_e, over_k, "distance"))

            under_prob = 1.0 - over_prob
            under_imp = american_to_implied(rp["under"])
            under_dec = american_to_decimal(rp["under"])
            under_e = under_prob - under_imp
            under_k = kelly(under_prob, under_dec)
            options.append(BetOption(fight, f"Under {rp['line']}", f"Under {rp['line']} rds ({rp['under']:+d})",
                                     under_prob, under_imp, under_dec, under_e, under_k, "distance"))

    return options


def find_optimal_parlays(options: List[BetOption], n_legs: int = 3, top_n: int = 5) -> list:
    positive_edge = [o for o in options if o.edge > 0.02]

    by_fight = {}
    for opt in positive_edge:
        by_fight.setdefault(opt.fight, []).append(opt)

    fight_names = list(by_fight.keys())
    parlays = []

    for fight_combo in itertools.combinations(fight_names, min(n_legs, len(fight_names))):
        fight_options = [by_fight[f] for f in fight_combo]
        for leg_combo in itertools.product(*fight_options):
            cp = 1.0
            co = 1.0
            te = 0.0
            for leg in leg_combo:
                cp *= leg.true_prob
                co *= leg.decimal_odds
                te += leg.edge

            ev = cp * co - 1.0
            cats = set(l.category for l in leg_combo)
            diversity = len(cats) * 0.02
            score = ev + (te / len(leg_combo)) * 0.5 + diversity

            parlays.append((score, ev, cp, co, te, leg_combo))

    parlays.sort(key=lambda x: x[0], reverse=True)
    return parlays[:top_n]


# ============================================================
# MAIN — v2 OUTPUT
# ============================================================

def main():
    # Calculate Elo for all fighters
    for k, f in fighters.items():
        f.elo = calculate_elo(f)

    print("=" * 115)
    print("  UFC VEGAS 115: MOICANO vs DUNCAN — APRIL 4, 2026")
    print("  v2 ENGINE — Recency-Weighted | X-Factor Analysis | Scorecard Sim | Confidence Calibration")
    print("=" * 115)

    all_analysis = {}
    all_xfactors = {}
    adjusted_probs = {}  # variance-adjusted win probabilities

    # Phase 1: Run base simulations
    for a_key, b_key, nr, is_main in matchups:
        a, b = fighters[a_key], fighters[b_key]
        res = simulate(a, b, nr)
        ana = analyze(res, a.name, b.name, nr)
        all_analysis[(a_key, b_key)] = ana

    # Phase 2: Run scenario analysis and compute variance-adjusted confidence
    # The scenarios produce alternate win probabilities under plausible conditions.
    # We blend them into the base prediction:
    #   adjusted_wp = base_wp * stability + worst_case_wp * fragility
    # This penalizes predictions that collapse under realistic scenarios.
    for a_key, b_key, nr, is_main in matchups:
        a, b = fighters[a_key], fighters[b_key]
        ana = all_analysis[(a_key, b_key)]
        sens = sensitivity_analysis(a_key, b_key, nr, ana)
        all_xfactors[(a_key, b_key)] = sens

        a_ml = ana[f"{a.name}_ml"]
        b_ml = ana[f"{b.name}_ml"]
        base_wp = max(a_ml, b_ml)
        fav_name = a.name if a_ml > b_ml else b.name
        fav_is_a = a_ml > b_ml

        # Compute plausibility-weighted adjusted probability.
        # Each scenario has a different likelihood of actually happening.
        # Instead of uniform blend, weight each scenario's pull on the
        # prediction by how plausible it is for THIS specific fight.

        scenario_pulls = []  # list of (plausibility, scenario_fav_wp)
        for sc in sens["top_scenarios"]:
            sc_wp = sc["wp_scenario"]
            fav_wp = sc_wp if fav_is_a else (1.0 - sc_wp)
            scenario_name = sc["scenario"]

            # Assign plausibility based on scenario type + fighter context
            if "stays on the feet" in scenario_name:
                # How likely is it to stay standing? Based on lower TD rate
                # in the fight and higher combined TDD
                avg_tdd = (a.takedown_defense + b.takedown_defense) / 2
                avg_tdr = (a.takedowns_per_15 + b.takedowns_per_15) / 2
                # High TDD + low TD attempts = very likely to stay standing
                plausibility = avg_tdd * (1.0 - min(avg_tdr / 5.0, 0.8))
                # Clamp 0.1-0.6
                plausibility = max(0.1, min(0.6, plausibility))

            elif "goes to the ground" in scenario_name:
                # Inverse of standing — how likely is grappling?
                max_tdr = max(a.takedowns_per_15, b.takedowns_per_15)
                min_tdd = min(a.takedown_defense, b.takedown_defense)
                plausibility = (max_tdr / 5.0) * (1.0 - min_tdd)
                plausibility = max(0.05, min(0.5, plausibility))

            elif "career-best" in scenario_name:
                # Underdog having a great night. More plausible if:
                # - the dog's recent form is decent (not totally shot)
                # - the dog has shown flashes (some wins by finish)
                dog_fighter = b if fav_is_a else a
                dog_form = dog_fighter.recent_form
                dog_finish_rate = fighter_finish_rate(dog_fighter)
                # A dog with 0.7 form and finishing ability = ~0.25 plausibility
                # A dog with 0.3 form and no finishes = ~0.08
                plausibility = 0.05 + 0.25 * dog_form + 0.10 * dog_finish_rate
                plausibility = max(0.05, min(0.35, plausibility))

            elif "fades late" in scenario_name:
                # Age-based fatigue. More plausible with bigger age gap
                # and older fighter being 34+
                older_age = max(a.age, b.age)
                age_gap = abs(a.age - b.age)
                if older_age >= 35:
                    plausibility = 0.15 + age_gap * 0.04
                elif older_age >= 33:
                    plausibility = 0.08 + age_gap * 0.03
                else:
                    plausibility = 0.03 + age_gap * 0.01
                plausibility = max(0.03, min(0.35, plausibility))

            elif "power shows up early" in scenario_name:
                # Favorite finishing early. More plausible if fav has high finish rate
                fav_fighter = a if fav_is_a else b
                plausibility = 0.10 + 0.25 * fighter_finish_rate(fav_fighter)
                plausibility = max(0.05, min(0.40, plausibility))

            else:
                plausibility = 0.10

            scenario_pulls.append((plausibility, fav_wp, scenario_name))

        if scenario_pulls:
            # Weighted average of all scenarios, where each pulls the base
            # prediction proportionally to its plausibility
            total_plaus = sum(p for p, _, _ in scenario_pulls)
            # The base prediction also gets a weight — it's the "everything
            # goes as the model expects" scenario
            base_weight = 1.0  # base model gets weight of 1.0
            weighted_sum = base_wp * base_weight
            total_weight = base_weight

            for plaus, sc_wp, _ in scenario_pulls:
                weighted_sum += sc_wp * plaus
                total_weight += plaus

            adj_wp = weighted_sum / total_weight

            worst_case = min(wp for _, wp, _ in scenario_pulls)
            variance = max(wp for _, wp, _ in scenario_pulls) - worst_case
            fragility = base_wp - worst_case
        else:
            adj_wp = base_wp
            fragility = 0.0
            variance = 0.0
            worst_case = base_wp
            scenario_pulls = []

        adjusted_probs[(a_key, b_key)] = {
            "base_wp": base_wp,
            "adj_wp": adj_wp,
            "fragility": fragility,
            "variance": variance,
            "worst_case": worst_case,
            "fav_name": fav_name,
            "fav_is_a": fav_is_a,
            "scenario_pulls": scenario_pulls,
        }

    # ============================================================
    # SECTION 1: FULL CARD PREDICTIONS — Variance-Adjusted
    # ============================================================
    print(f"\n{'━' * 115}")
    print(f"  FULL CARD PREDICTIONS — Variance-Adjusted Confidence")
    print(f"{'━' * 115}\n")

    print(f"  {'#':<3} {'Fight':<42} {'Winner':<22} {'Method':<10} {'Base':>6} {'Adj':>6} {'Book':>6} {'Edge':>7} {'Tier':>8}")
    print(f"  {'─' * 112}")

    for i, (a_key, b_key, nr, is_main) in enumerate(matchups, 1):
        a, b = fighters[a_key], fighters[b_key]
        ana = all_analysis[(a_key, b_key)]
        ap = adjusted_probs[(a_key, b_key)]

        winner = ap["fav_name"]
        base_wp = ap["base_wp"]
        adj_wp = ap["adj_wp"]
        w_key = a_key if ap["fav_is_a"] else b_key
        w_fighter = a if ap["fav_is_a"] else b

        methods = {"KO/TKO": ana.get(f"{winner}_ko", 0), "SUB": ana.get(f"{winner}_sub", 0), "DEC": ana.get(f"{winner}_dec", 0)}
        best_method = max(methods, key=methods.get)

        book_imp = american_to_implied(book_odds[w_key]["ml"])
        # Edge based on ADJUSTED probability, not raw
        edge = adj_wp - book_imp

        # Tiers based on adjusted probability
        if adj_wp > 0.85:
            tier = "LOCK"
        elif adj_wp > 0.70:
            tier = "HIGH"
        elif adj_wp > 0.55:
            tier = "MEDIUM"
        else:
            tier = "LEAN"

        tag = " *" if is_main else ""
        fight = f"{a.name} vs {b.name}{tag}"
        print(f"  {i:<3} {fight:<42} {winner:<22} {best_method:<10} {base_wp:>5.1%} {adj_wp:>5.1%} {book_imp:>5.1%} {edge:>+6.1%} {tier:>8}")

    # ============================================================
    # SECTION 2: DETAILED FIGHT-BY-FIGHT + X-FACTOR
    # ============================================================
    print(f"\n\n{'=' * 115}")
    print(f"  DETAILED FIGHT-BY-FIGHT — with X-Factor & Situational Context")
    print(f"{'=' * 115}")

    for i, (a_key, b_key, nr, is_main) in enumerate(matchups, 1):
        a, b = fighters[a_key], fighters[b_key]
        ana = all_analysis[(a_key, b_key)]
        ap = adjusted_probs[(a_key, b_key)]

        winner = ap["fav_name"]
        loser = b.name if ap["fav_is_a"] else a.name
        win_prob = ap["adj_wp"]
        base_wp = ap["base_wp"]
        w_key = a_key if ap["fav_is_a"] else b_key

        book_imp = american_to_implied(book_odds[w_key]["ml"])
        edge = win_prob - book_imp

        tag = " * MAIN EVENT" if is_main else ""
        print(f"\n  ┌─ Fight {i}{tag}")
        print(f"  │  {a.name} (Elo: {a.elo} | {a.style} | Form: {a.recent_form:.1f})")
        print(f"  │  vs {b.name} (Elo: {b.elo} | {b.style} | Form: {b.recent_form:.1f}) | {nr}R")
        print(f"  │")

        # Situational notes
        print(f"  │  SITUATIONAL CONTEXT:")
        print(f"  │    {a.name}: {a.situational}")
        print(f"  │    {b.name}: {b.situational}")
        print(f"  │")

        print(f"  │  PREDICTION:  {winner} def. {loser}")
        print(f"  │  BASE PROB:   {base_wp:.1%}  |  ADJ PROB: {win_prob:.1%}  |  Book: {book_imp:.1%}  |  Edge: {edge:+.1%}")
        print(f"  │  Fragility: {ap['fragility']:.1%}  |  Worst-case: {ap['worst_case']:.1%}  |  Variance: {ap['variance']:.1%}")

        # Confidence calibration
        if abs(edge) > 0.20:
            print(f"  │")
            print(f"  │  *** MARKET DISAGREEMENT ALERT ***")
            if edge > 0.20:
                print(f"  │  Model has {edge:+.1%} edge over books. Possible explanations:")
                print(f"  │    1. Genuine value — books underpricing {winner}")
                print(f"  │    2. Model overconfidence — books may know something (injury, camp issues)")
                print(f"  │    3. Sharp money may not have moved the line yet")
            else:
                print(f"  │  Model disagrees with market by {abs(edge):.1%} — books heavily favor this side")

        print(f"  │")

        # Outcome probabilities
        outcomes = []
        for fkey in [a_key, b_key]:
            fn = fighters[fkey].name
            for method, label in [("ko", "KO/TKO"), ("sub", "SUB"), ("dec", "DEC")]:
                p = ana.get(f"{fn}_{method}", 0)
                if p > 0.005:
                    outcomes.append((f"{fn} by {label}", p))
        outcomes.sort(key=lambda x: x[1], reverse=True)

        print(f"  │  OUTCOME PROBABILITIES:")
        for outcome, prob in outcomes:
            bar = '█' * int(prob * 50)
            marker = " <-- MOST LIKELY" if outcome == outcomes[0][0] else ""
            print(f"  │    {outcome:<32} {prob:>5.1%} {bar}{marker}")

        # Distance
        print(f"  │")
        print(f"  │  Goes to Decision: {ana['distance']:.1%}  |  Finish: {ana['finish']:.1%}")

        # Scorecard (if likely to go distance)
        if ana["distance"] > 0.40:
            sc = simulate_scorecard(a, b, nr)
            print(f"  │")
            print(f"  │  PROJECTED SCORECARDS (if goes to decision):")
            for (sa, sb), count in sc["top_scorecards"]:
                pct = count / 5000
                print(f"  │    {sa}-{sb}  ({pct:.1%} of sims)")
            print(f"  │    Avg 10-8 rounds: {a.name} {sc['avg_10_8_a']:.2f} | {b.name} {sc['avg_10_8_b']:.2f}")

        # X-Factor
        print(f"  │")
        print(f"  │  X-FACTOR ANALYSIS (scenario-based sensitivity):")
        sens = all_xfactors[(a_key, b_key)]

        pulls = ap.get("scenario_pulls", [])
        # Sort by plausibility descending to show most likely scenarios first
        pulls_sorted = sorted(pulls, key=lambda x: x[0], reverse=True)

        if pulls_sorted:
            print(f"  │  Scenario                                Plausibility  Fav WP if true")
            for plaus, sc_wp, sc_name in pulls_sorted:
                shift = sc_wp - base_wp
                print(f"  │    {sc_name:<40} {plaus:>8.0%}       {sc_wp:>5.1%} ({shift:+.1%})")

        print(f"  └{'─' * 80}")

    # ============================================================
    # SECTION 3: EDGE DETECTION — ALL BET TYPES
    # ============================================================
    print(f"\n\n{'=' * 115}")
    print(f"  EDGE DETECTION — MONEYLINE + METHOD + DISTANCE PROPS")
    print(f"{'=' * 115}\n")

    options = build_bet_options(all_analysis, matchups, adjusted_probs)
    positive = sorted([o for o in options if o.edge > 0], key=lambda x: x.edge, reverse=True)

    print(f"  Found {len(positive)} positive-edge bets out of {len(options)} total\n")
    print(f"  {'Bet':<48} {'True%':>7} {'Book%':>7} {'Edge':>7} {'Kelly%':>7} {'Type':>8}")
    print(f"  {'─' * 90}")
    for opt in positive[:25]:
        print(f"  {opt.description:<48} {opt.true_prob:>6.1%} {opt.book_implied:>6.1%} {opt.edge:>+6.1%} {opt.kelly_frac:>6.1%} {opt.category:>8}")

    # ============================================================
    # SECTION 4: OPTIMAL PARLAYS (3, 4, 5 legs)
    # ============================================================
    for n_legs in [3, 4, 5]:
        print(f"\n{'=' * 115}")
        print(f"  OPTIMAL {n_legs}-LEG PARLAYS (Score = EV + Avg Edge + Diversity)")
        print(f"{'=' * 115}")

        parlays = find_optimal_parlays(options, n_legs=n_legs, top_n=3)

        for rank, (score, ev, prob, odds, te, legs) in enumerate(parlays, 1):
            print(f"\n  {'*' * 3} PARLAY #{rank} {'*' * 3}")
            print(f"  EV: {ev:+.1%} | Hit Rate: {prob:.1%} | Payout: {odds:.1f}x | $10 -> ${10 * odds:.0f}")
            print(f"  Composite Score: {score:.4f} | Total Edge: {te:.1%}")
            for leg in legs:
                print(f"    {leg.description:<48} Model: {leg.true_prob:.1%} vs Book: {leg.book_implied:.1%} -> Edge: {leg.edge:+.1%}  [{leg.category}]")
            print()

    # ============================================================
    # SECTION 5: FINAL RECOMMENDATION + BANKROLL
    # ============================================================
    print(f"{'=' * 115}")
    print(f"  FINAL RECOMMENDATION — v2 ENGINE TOP PICKS")
    print(f"{'=' * 115}")

    best_3 = find_optimal_parlays(options, n_legs=3, top_n=1)
    if best_3:
        score, ev, prob, odds, te, legs = best_3[0]
        print(f"\n  TOP 3-LEG PARLAY:")
        print(f"  Combined True Probability: {prob:.2%}")
        print(f"  Payout: {odds:.2f}x ($10 -> ${10 * odds:.2f})")
        print(f"  Expected Value: {ev:+.4f} per dollar")
        print(f"  Legs:")
        for j, leg in enumerate(legs, 1):
            print(f"    {j}. {leg.description}")
            print(f"       Model: {leg.true_prob:.1%} vs Book: {leg.book_implied:.1%} -> Edge: {leg.edge:+.1%}")

        total_kelly = sum(l.kelly_frac for l in legs)
        qk = total_kelly / 4
        print(f"\n  Kelly-Optimal Bankroll (quarter-Kelly): {qk:.1%} of bankroll")
        print(f"  On $100 bankroll: ${100 * qk:.2f} wager")
        print(f"  Model Certainty: {shannon_entropy([prob, 1 - prob]):.3f} bits (lower = more certain)")

    # Best value single bets
    print(f"\n  TOP 5 SINGLE-BET VALUES:")
    for j, opt in enumerate(positive[:5], 1):
        print(f"    {j}. {opt.description:<48} Edge: {opt.edge:+.1%} | Kelly: {opt.kelly_frac:.1%}")

    # Quick tier summary
    print(f"\n{'=' * 115}")
    print(f"  QUICK SUMMARY — CONFIDENCE TIERS")
    print(f"{'=' * 115}\n")

    for tier_name in ["LOCK", "HIGH", "MEDIUM", "LEAN"]:
        tier_fights = []
        for a_key, b_key, nr, _ in matchups:
            ap = adjusted_probs[(a_key, b_key)]
            adj_wp = ap["adj_wp"]
            base_wp = ap["base_wp"]
            wn = ap["fav_name"]
            wk = a_key if ap["fav_is_a"] else b_key
            if adj_wp > 0.85:
                t = "LOCK"
            elif adj_wp > 0.70:
                t = "HIGH"
            elif adj_wp > 0.55:
                t = "MEDIUM"
            else:
                t = "LEAN"
            if t == tier_name:
                xf = all_xfactors.get((a_key, b_key), {})
                xf_str = ""
                if xf and xf.get("x_factor"):
                    xf_str = f" | X-Factor: {xf['x_factor']['scenario']}"
                tier_fights.append((wn, base_wp, adj_wp, fighters[wk].recent_form, xf_str))

        if tier_fights:
            print(f"  {tier_name}:")
            for wn, bwp, awp, rf, xf_str in tier_fights:
                print(f"    -> {wn} (Base: {bwp:.1%} -> Adj: {awp:.1%} | Form: {rf:.1f}){xf_str}")
            print()

    print(f"{'=' * 115}")
    print(f"  DISCLAIMER: Entertainment only. Gamble responsibly.")
    print(f"  v2 improvements: recency weighting, sensitivity analysis, scorecard modeling,")
    print(f"  market disagreement alerts, method/distance prop edges, full parlay optimizer.")
    print(f"{'=' * 115}")


if __name__ == "__main__":
    main()
