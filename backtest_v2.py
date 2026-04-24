"""
UFC Predictor Backtester — Score our predictions against reality.

Feeds historical fights into predictor.py and scores:
  1. Winner accuracy
  2. Method accuracy (KO/SUB/DEC)
  3. Round accuracy (exact round or within 1)
  4. Accuracy by archetype and confidence level

Usage:
    python3 backtest_v2.py
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from predictor import predict_fight, build_fighter, Prediction, Fighter


@dataclass
class FightResult:
    fighter_a: Fighter
    fighter_b: Fighter
    num_rounds: int
    winner: str
    method: str         # "KO/TKO", "SUB", "DEC"
    round: int
    card: str = ""


@dataclass
class Score:
    fight: str
    card: str
    pred_winner: str
    pred_method: str
    pred_round: int
    actual_winner: str
    actual_method: str
    actual_round: int
    winner_correct: bool
    method_correct: bool
    round_exact: bool
    round_within_1: bool
    archetype: str
    confidence: float
    win_prob: float


def score_fight(result: FightResult) -> Score:
    p = predict_fight(result.fighter_a, result.fighter_b, result.num_rounds)

    return Score(
        fight=f"{result.fighter_a.name} vs {result.fighter_b.name}",
        card=result.card,
        pred_winner=p.winner, pred_method=p.method, pred_round=p.round,
        actual_winner=result.winner, actual_method=result.method, actual_round=result.round,
        winner_correct=(p.winner == result.winner),
        method_correct=(p.method == result.method),
        round_exact=(p.round == result.round),
        round_within_1=(abs(p.round - result.round) <= 1),
        archetype=p.archetype, confidence=p.confidence, win_prob=p.win_prob,
    )


# ============================================================
# HISTORICAL DATA — 5 Cards, verified results
# ============================================================

def load_cards() -> List[FightResult]:
    bf = build_fighter
    fights = []

    # ── UFC Seattle: Adesanya vs Pyfer (Mar 28, 2026) ──
    card = "UFC Seattle (Mar 28)"
    fights.extend([
        FightResult(
            bf("Israel Adesanya", 24, 4, 16, 0, 8, 2, 0, 2, 4.40, 0.49, 0.62, 0.30, 0.50, 0.65, 0.0, 80.0, 36, -1, 0.35, "striker", comp=1.0),
            bf("Joe Pyfer", 14, 3, 10, 2, 2, 1, 1, 1, 5.80, 0.50, 0.48, 0.80, 0.40, 0.55, 0.3, 76.0, 28, 2, 0.80, "striker", comp=0.55),
            5, "Joe Pyfer", "KO/TKO", 2, card),
        FightResult(
            bf("Alexa Grasso", 16, 4, 4, 3, 9, 1, 1, 2, 4.50, 0.46, 0.58, 0.80, 0.35, 0.65, 0.3, 65.0, 33, 1, 0.70, "striker", "F", comp=0.9),
            bf("Maycee Barber", 14, 3, 5, 1, 8, 1, 0, 2, 5.00, 0.44, 0.52, 1.50, 0.38, 0.55, 0.2, 66.0, 27, 1, 0.65, "balanced", "F", comp=0.65),
            3, "Alexa Grasso", "KO/TKO", 1, card),
        FightResult(
            bf("Michael Chiesa", 18, 7, 2, 8, 8, 3, 2, 2, 2.50, 0.40, 0.55, 2.80, 0.38, 0.60, 1.2, 74.0, 37, 1, 0.50, "grappler", comp=0.75),
            bf("Niko Price", 16, 8, 9, 2, 5, 5, 2, 1, 4.80, 0.44, 0.42, 0.60, 0.30, 0.50, 0.2, 72.0, 35, -1, 0.35, "striker", comp=0.7),
            3, "Michael Chiesa", "SUB", 1, card),
        FightResult(
            bf("Lerryan Douglas", 6, 0, 5, 1, 0, 0, 0, 0, 6.00, 0.52, 0.55, 0.50, 0.30, 0.60, 0.1, 74.0, 26, 6, 0.85, "striker", comp=0.3),
            bf("Julian Erosa", 30, 13, 8, 12, 10, 4, 5, 4, 4.00, 0.42, 0.45, 0.90, 0.33, 0.52, 0.6, 74.0, 34, -1, 0.35, "balanced", comp=0.65),
            3, "Lerryan Douglas", "KO/TKO", 1, card),
    ])

    # ── UFC London: Evloev vs Murphy (Mar 21, 2026) ──
    card = "UFC London (Mar 21)"
    fights.extend([
        FightResult(
            bf("Movsar Evloev", 19, 0, 3, 3, 13, 0, 0, 0, 4.80, 0.48, 0.58, 3.50, 0.46, 0.70, 0.4, 69.0, 30, 19, 0.85, "balanced", comp=0.8),
            bf("Lerone Murphy", 14, 0, 5, 2, 7, 0, 0, 0, 4.20, 0.44, 0.60, 1.20, 0.35, 0.68, 0.3, 72.0, 30, 14, 0.80, "balanced", comp=0.75),
            5, "Movsar Evloev", "DEC", 5, card),
        FightResult(
            bf("Michael Page", 22, 3, 12, 2, 8, 1, 1, 1, 3.80, 0.46, 0.60, 0.30, 0.25, 0.75, 0.1, 78.0, 37, 1, 0.55, "striker", comp=0.6),
            bf("Sam Patterson", 12, 2, 4, 3, 5, 1, 0, 1, 4.20, 0.43, 0.50, 1.00, 0.35, 0.55, 0.3, 72.0, 28, 2, 0.65, "balanced", comp=0.45),
            3, "Michael Page", "DEC", 3, card),
        FightResult(
            bf("Iwo Baraniewski", 10, 1, 6, 1, 3, 0, 1, 0, 5.50, 0.50, 0.52, 0.80, 0.35, 0.58, 0.2, 73.0, 27, 3, 0.80, "striker", comp=0.4),
            bf("Austen Lane", 8, 2, 5, 1, 2, 1, 0, 1, 4.80, 0.45, 0.48, 0.60, 0.30, 0.55, 0.1, 79.0, 40, 1, 0.45, "striker", comp=0.35),
            3, "Iwo Baraniewski", "KO/TKO", 1, card),
        FightResult(
            bf("Christian Leroy Duncan", 12, 1, 4, 2, 6, 0, 0, 1, 4.60, 0.44, 0.55, 1.20, 0.38, 0.62, 0.3, 76.0, 28, 2, 0.75, "balanced", comp=0.55),
            bf("Roman Dolidze", 12, 5, 5, 4, 3, 2, 1, 2, 3.50, 0.41, 0.48, 2.00, 0.40, 0.55, 0.5, 73.0, 36, -1, 0.40, "grappler", comp=0.65),
            3, "Christian Leroy Duncan", "DEC", 3, card),
        FightResult(
            bf("Danny Silva", 8, 1, 5, 1, 2, 0, 1, 0, 5.20, 0.48, 0.50, 0.50, 0.30, 0.55, 0.1, 70.0, 27, 2, 0.75, "striker", comp=0.4),
            bf("Kurtis Campbell", 6, 2, 3, 1, 2, 2, 0, 0, 4.00, 0.42, 0.45, 0.80, 0.33, 0.50, 0.2, 72.0, 29, -1, 0.45, "balanced", comp=0.35),
            3, "Danny Silva", "KO/TKO", 2, card),
    ])

    # ── UFC Vegas 114: Emmett vs Vallejos (Mar 14, 2026) ──
    card = "UFC Vegas 114 (Mar 14)"
    fights.extend([
        FightResult(
            bf("Josh Emmett", 18, 6, 9, 0, 9, 3, 0, 3, 3.80, 0.42, 0.55, 0.80, 0.35, 0.62, 0.1, 67.0, 39, -1, 0.30, "striker", comp=0.85),
            bf("Kevin Vallejos", 14, 1, 8, 3, 3, 0, 0, 1, 5.50, 0.50, 0.52, 1.20, 0.40, 0.60, 0.3, 70.0, 27, 5, 0.85, "striker", comp=0.45),
            5, "Kevin Vallejos", "KO/TKO", 1, card),
        FightResult(
            bf("Gillian Robertson", 13, 9, 0, 7, 6, 2, 1, 6, 2.50, 0.38, 0.48, 3.80, 0.40, 0.55, 1.5, 63.0, 30, 1, 0.50, "grappler", "F", comp=0.65),
            bf("Amanda Lemos", 15, 5, 4, 4, 7, 2, 1, 2, 5.20, 0.47, 0.52, 0.60, 0.30, 0.55, 0.2, 66.0, 30, -1, 0.45, "striker", "F", comp=0.75),
            3, "Gillian Robertson", "DEC", 3, card),
        FightResult(
            bf("Myktybek Orolbai", 13, 1, 4, 4, 5, 0, 0, 1, 4.80, 0.47, 0.55, 2.00, 0.42, 0.65, 0.6, 72.0, 27, 3, 0.80, "balanced", comp=0.55),
            bf("Chris Curtis", 31, 12, 8, 0, 23, 5, 0, 7, 3.20, 0.40, 0.55, 0.40, 0.25, 0.60, 0.0, 74.0, 37, -2, 0.30, "striker", comp=0.7),
            3, "Myktybek Orolbai", "DEC", 3, card),
        FightResult(
            bf("Ion Cutelaba", 18, 9, 10, 3, 5, 5, 2, 2, 4.50, 0.42, 0.45, 2.50, 0.45, 0.52, 0.3, 73.0, 30, 1, 0.50, "striker", comp=0.7),
            bf("Oumar Sy", 10, 2, 5, 3, 2, 1, 0, 1, 4.20, 0.44, 0.48, 1.00, 0.35, 0.55, 0.3, 77.0, 30, 2, 0.65, "balanced", comp=0.4),
            3, "Ion Cutelaba", "SUB", 1, card),
        FightResult(
            bf("Vitor Petrino", 10, 2, 6, 1, 3, 1, 1, 0, 5.00, 0.48, 0.52, 1.50, 0.40, 0.58, 0.2, 77.0, 27, 1, 0.70, "striker", comp=0.5),
            bf("Steven Asplund", 8, 1, 3, 2, 3, 0, 0, 1, 3.80, 0.42, 0.50, 1.80, 0.42, 0.55, 0.4, 75.0, 29, 2, 0.65, "balanced", comp=0.35),
            3, "Vitor Petrino", "DEC", 3, card),
    ])

    # ── UFC Houston: Strickland vs Hernandez (Feb 21, 2026) ──
    card = "UFC Houston (Feb 21)"
    fights.extend([
        FightResult(
            bf("Sean Strickland", 29, 6, 9, 2, 18, 3, 1, 2, 5.80, 0.47, 0.62, 0.60, 0.30, 0.70, 0.1, 76.0, 35, 2, 0.70, "striker", comp=0.95),
            bf("Anthony Hernandez", 12, 3, 3, 6, 3, 1, 0, 2, 3.50, 0.42, 0.48, 2.50, 0.45, 0.55, 0.8, 76.0, 31, 3, 0.75, "grappler", comp=0.6),
            5, "Sean Strickland", "KO/TKO", 3, card),
        FightResult(
            bf("Geoff Neal", 16, 6, 8, 0, 8, 3, 0, 3, 3.80, 0.45, 0.52, 0.40, 0.25, 0.60, 0.0, 74.0, 34, -1, 0.40, "striker", comp=0.75),
            bf("Uros Medic", 9, 2, 6, 1, 2, 2, 0, 0, 5.50, 0.50, 0.50, 0.30, 0.25, 0.55, 0.1, 73.0, 29, 2, 0.80, "striker", comp=0.45),
            3, "Uros Medic", "KO/TKO", 1, card),
        FightResult(
            bf("Michel Pereira", 31, 12, 14, 5, 12, 4, 2, 6, 5.20, 0.44, 0.50, 1.50, 0.38, 0.58, 0.3, 78.0, 31, 2, 0.65, "striker", comp=0.65),
            bf("Zach Reese", 8, 2, 5, 1, 2, 1, 0, 1, 5.00, 0.46, 0.48, 0.80, 0.33, 0.52, 0.2, 74.0, 28, 1, 0.60, "striker", comp=0.35),
            3, "Michel Pereira", "DEC", 3, card),
        FightResult(
            bf("Jacobe Smith", 10, 1, 7, 1, 2, 0, 0, 1, 5.80, 0.52, 0.50, 0.40, 0.30, 0.55, 0.1, 75.0, 27, 4, 0.85, "striker", comp=0.4),
            bf("Josiah Harrell", 7, 2, 3, 1, 3, 1, 0, 1, 4.20, 0.43, 0.48, 1.00, 0.35, 0.55, 0.2, 73.0, 29, 1, 0.55, "balanced", comp=0.35),
            3, "Jacobe Smith", "KO/TKO", 2, card),
    ])

    # ── UFC Mexico: Moreno vs Kavanagh (Feb 28, 2026) ──
    card = "UFC Mexico (Feb 28)"
    fights.extend([
        FightResult(
            bf("Brandon Moreno", 22, 8, 4, 6, 12, 2, 2, 4, 4.50, 0.44, 0.55, 1.80, 0.40, 0.60, 0.8, 67.0, 32, -1, 0.45, "balanced", comp=0.95),
            bf("Lone'er Kavanagh", 8, 1, 3, 2, 3, 0, 0, 1, 4.80, 0.46, 0.52, 1.20, 0.38, 0.58, 0.4, 68.0, 24, 3, 0.80, "balanced", comp=0.4),
            5, "Lone'er Kavanagh", "DEC", 5, card),
        FightResult(
            bf("Marlon Vera", 22, 11, 10, 5, 7, 2, 3, 6, 4.20, 0.43, 0.52, 0.80, 0.33, 0.55, 0.3, 73.0, 32, -3, 0.30, "striker", comp=0.85),
            bf("David Martinez", 12, 2, 4, 2, 6, 0, 0, 2, 4.50, 0.45, 0.50, 1.50, 0.40, 0.60, 0.3, 72.0, 28, 3, 0.75, "balanced", comp=0.45),
            3, "David Martinez", "DEC", 3, card),
        FightResult(
            bf("Daniel Zellhuber", 15, 2, 5, 3, 7, 1, 0, 1, 4.60, 0.45, 0.54, 1.50, 0.40, 0.62, 0.3, 74.0, 25, 4, 0.80, "balanced", comp=0.55),
            bf("King Green", 12, 3, 5, 3, 4, 1, 1, 1, 5.00, 0.47, 0.50, 0.80, 0.35, 0.55, 0.3, 73.0, 30, 1, 0.65, "striker", comp=0.5),
            3, "King Green", "KO/TKO", 2, card),
        FightResult(
            bf("Ryan Gandra", 5, 0, 3, 1, 1, 0, 0, 0, 5.50, 0.50, 0.52, 1.00, 0.38, 0.58, 0.2, 71.0, 27, 5, 0.85, "striker", comp=0.3),
            bf("Jose Daniel Medina", 8, 3, 3, 2, 3, 2, 1, 0, 4.00, 0.42, 0.45, 0.80, 0.33, 0.52, 0.3, 70.0, 28, -1, 0.40, "balanced", comp=0.35),
            3, "Ryan Gandra", "KO/TKO", 1, card),
        FightResult(
            bf("Ailin Perez", 11, 2, 2, 2, 7, 0, 0, 2, 4.80, 0.45, 0.52, 1.80, 0.42, 0.60, 0.4, 64.0, 28, 2, 0.70, "balanced", "F", comp=0.55),
            bf("Macy Chiasson", 10, 4, 3, 4, 3, 2, 1, 1, 4.30, 0.43, 0.48, 1.20, 0.35, 0.55, 0.3, 69.0, 33, -1, 0.40, "balanced", "F", comp=0.6),
            3, "Ailin Perez", "DEC", 3, card),
    ])

    # ── UFC 324: Gaethje vs Pimblett (Jan 24, 2026) ──
    card = "UFC 324 (Jan 24)"
    fights.extend([
        FightResult(
            bf("Justin Gaethje", 26, 5, 20, 0, 6, 4, 0, 1, 7.50, 0.50, 0.55, 0.80, 0.50, 0.70, 0.0, 70.0, 36, 1, 0.60, "striker", comp=1.0),
            bf("Paddy Pimblett", 22, 4, 8, 8, 6, 1, 1, 2, 4.50, 0.44, 0.48, 1.80, 0.40, 0.55, 0.5, 74.0, 30, 4, 0.75, "balanced", comp=0.65),
            5, "Justin Gaethje", "DEC", 5, card),
        FightResult(
            bf("Nikita Krylov", 31, 10, 16, 10, 5, 5, 3, 2, 4.20, 0.44, 0.50, 1.00, 0.35, 0.55, 0.5, 76.5, 33, 1, 0.55, "balanced", comp=0.75),
            bf("Modestas Bukauskas", 16, 7, 10, 2, 4, 4, 1, 2, 4.80, 0.46, 0.45, 0.60, 0.30, 0.50, 0.2, 76.0, 30, 1, 0.55, "striker", comp=0.5),
            3, "Nikita Krylov", "KO/TKO", 3, card),
        FightResult(
            bf("Alex Perez", 25, 8, 6, 3, 16, 4, 1, 3, 4.00, 0.43, 0.52, 2.20, 0.42, 0.60, 0.3, 65.0, 33, -1, 0.40, "balanced", comp=0.7),
            bf("Charles Johnson", 18, 6, 5, 5, 8, 2, 2, 2, 4.50, 0.44, 0.48, 1.50, 0.38, 0.55, 0.4, 67.0, 31, -1, 0.45, "balanced", comp=0.5),
            3, "Alex Perez", "KO/TKO", 1, card),
        FightResult(
            bf("Umar Nurmagomedov", 18, 0, 4, 7, 7, 0, 0, 0, 4.60, 0.46, 0.60, 3.50, 0.48, 0.72, 0.6, 69.0, 29, 18, 0.85, "balanced", comp=0.75),
            bf("Deiveson Figueiredo", 24, 4, 10, 5, 9, 1, 2, 1, 4.80, 0.45, 0.55, 1.50, 0.40, 0.58, 0.3, 66.0, 38, 2, 0.50, "balanced", comp=0.95),
            3, "Umar Nurmagomedov", "DEC", 3, card),
    ])

    # ── UFC Vegas 112: Royval vs Kape (Dec 13, 2025) ──
    card = "UFC Vegas 112 (Dec 13)"
    fights.extend([
        FightResult(
            bf("Brandon Royval", 17, 7, 3, 6, 8, 2, 2, 3, 5.50, 0.47, 0.48, 1.80, 0.40, 0.55, 0.8, 68.0, 33, 2, 0.60, "balanced", comp=0.8),
            bf("Manel Kape", 20, 7, 10, 2, 8, 3, 0, 4, 5.80, 0.50, 0.52, 0.50, 0.30, 0.55, 0.1, 66.0, 31, 2, 0.70, "striker", comp=0.65),
            5, "Manel Kape", "KO/TKO", 1, card),
        FightResult(
            bf("Giga Chikadze", 15, 4, 7, 0, 8, 3, 0, 1, 4.80, 0.47, 0.55, 0.30, 0.25, 0.70, 0.0, 74.0, 37, -2, 0.35, "striker", comp=0.75),
            bf("Kevin Vallejos", 13, 1, 7, 3, 3, 0, 0, 1, 5.50, 0.50, 0.52, 1.20, 0.40, 0.60, 0.3, 70.0, 27, 4, 0.85, "striker", comp=0.45),
            3, "Kevin Vallejos", "KO/TKO", 2, card),
        FightResult(
            bf("Melquizael Costa", 12, 2, 8, 1, 3, 1, 0, 1, 5.50, 0.50, 0.48, 0.30, 0.25, 0.55, 0.1, 72.0, 28, 3, 0.80, "striker", comp=0.45),
            bf("Morgan Charriere", 20, 8, 7, 4, 9, 3, 2, 3, 4.50, 0.44, 0.45, 0.80, 0.33, 0.52, 0.3, 68.0, 30, 1, 0.50, "balanced", comp=0.5),
            3, "Melquizael Costa", "KO/TKO", 1, card),
        FightResult(
            bf("Yaroslav Amosov", 28, 0, 2, 10, 16, 0, 0, 0, 3.00, 0.40, 0.60, 4.50, 0.50, 0.75, 1.0, 73.0, 31, 28, 0.85, "grappler", comp=0.7),
            bf("Neil Magny", 28, 14, 5, 3, 20, 3, 5, 6, 3.80, 0.42, 0.48, 1.80, 0.38, 0.55, 0.2, 80.0, 38, -1, 0.30, "balanced", comp=0.8),
            3, "Yaroslav Amosov", "SUB", 1, card),
    ])

    # ── UFC Qatar: Tsarukyan vs Hooker (Nov 22, 2025) ──
    card = "UFC Qatar (Nov 22)"
    fights.extend([
        FightResult(
            bf("Arman Tsarukyan", 22, 3, 6, 5, 11, 0, 1, 2, 5.50, 0.48, 0.58, 3.20, 0.45, 0.68, 0.4, 72.0, 28, 3, 0.85, "balanced", comp=0.9),
            bf("Dan Hooker", 24, 13, 11, 3, 10, 5, 3, 5, 4.50, 0.44, 0.50, 0.80, 0.35, 0.55, 0.2, 75.0, 35, 2, 0.55, "striker", comp=0.85),
            5, "Arman Tsarukyan", "SUB", 2, card),
        FightResult(
            bf("Ian Machado Garry", 16, 0, 4, 2, 10, 0, 0, 0, 5.20, 0.48, 0.58, 1.50, 0.40, 0.65, 0.2, 77.0, 27, 16, 0.80, "striker", comp=0.65),
            bf("Belal Muhammad", 24, 4, 3, 1, 20, 0, 0, 4, 3.50, 0.40, 0.55, 3.50, 0.45, 0.58, 0.1, 72.0, 37, -1, 0.45, "balanced", comp=0.95),
            3, "Ian Machado Garry", "DEC", 3, card),
        FightResult(
            bf("Volkan Oezdemir", 20, 8, 11, 0, 9, 5, 0, 3, 4.80, 0.46, 0.50, 0.30, 0.25, 0.60, 0.0, 74.0, 33, 1, 0.50, "striker", comp=0.75),
            bf("Alonzo Menifield", 15, 5, 10, 0, 5, 4, 0, 1, 5.50, 0.50, 0.45, 0.60, 0.30, 0.50, 0.1, 75.0, 33, 1, 0.55, "striker", comp=0.6),
            3, "Volkan Oezdemir", "KO/TKO", 1, card),
        FightResult(
            bf("Kyoji Horiguchi", 35, 5, 12, 10, 13, 2, 2, 1, 4.50, 0.44, 0.55, 2.00, 0.42, 0.62, 0.6, 66.0, 35, 1, 0.55, "balanced", comp=0.8),
            bf("Tagir Ulanbekov", 16, 3, 2, 5, 9, 0, 1, 2, 3.20, 0.40, 0.52, 3.80, 0.45, 0.60, 0.5, 66.0, 32, 2, 0.65, "grappler", comp=0.6),
            3, "Kyoji Horiguchi", "SUB", 3, card),
    ])

    # ── UFC Vegas 110: Bonfim vs Brown (Nov 8, 2025) ──
    card = "UFC Vegas 110 (Nov 8)"
    fights.extend([
        FightResult(
            bf("Gabriel Bonfim", 20, 2, 8, 4, 8, 0, 1, 1, 5.00, 0.47, 0.55, 2.50, 0.45, 0.62, 0.3, 74.0, 27, 4, 0.80, "balanced", comp=0.55),
            bf("Randy Brown", 18, 6, 7, 2, 9, 2, 1, 3, 4.00, 0.43, 0.50, 0.80, 0.33, 0.55, 0.2, 78.0, 34, -1, 0.40, "striker", comp=0.7),
            5, "Gabriel Bonfim", "KO/TKO", 1, card),
        FightResult(
            bf("Uros Medic", 8, 2, 5, 1, 2, 2, 0, 0, 5.50, 0.50, 0.50, 0.30, 0.25, 0.55, 0.1, 73.0, 29, 1, 0.75, "striker", comp=0.45),
            bf("Muslim Salikhov", 20, 4, 10, 1, 9, 2, 0, 2, 3.50, 0.42, 0.52, 0.40, 0.25, 0.60, 0.0, 72.0, 41, -1, 0.30, "striker", comp=0.65),
            3, "Uros Medic", "KO/TKO", 1, card),
        FightResult(
            bf("Christian Leroy Duncan", 11, 1, 3, 2, 6, 0, 0, 1, 4.60, 0.44, 0.55, 1.20, 0.38, 0.62, 0.3, 76.0, 28, 1, 0.70, "balanced", comp=0.55),
            bf("Marco Tulio", 8, 1, 4, 2, 2, 0, 0, 1, 4.80, 0.46, 0.48, 0.80, 0.33, 0.52, 0.2, 74.0, 27, 3, 0.75, "striker", comp=0.35),
            3, "Christian Leroy Duncan", "KO/TKO", 2, card),
        FightResult(
            bf("Raoni Barcelos", 19, 7, 5, 4, 10, 2, 2, 3, 4.50, 0.44, 0.52, 1.50, 0.38, 0.58, 0.3, 68.0, 37, 1, 0.45, "balanced", comp=0.65),
            bf("Ricky Simon", 21, 6, 2, 6, 13, 1, 2, 3, 4.80, 0.45, 0.50, 3.50, 0.45, 0.55, 0.4, 67.0, 32, -2, 0.40, "grappler", comp=0.75),
            3, "Raoni Barcelos", "DEC", 3, card),
    ])

    # ── UFC Vegas 109: Garcia vs Onama (Nov 1, 2025) ──
    card = "UFC Vegas 109 (Nov 1)"
    fights.extend([
        FightResult(
            bf("Steve Garcia", 17, 5, 8, 5, 4, 2, 1, 2, 5.80, 0.50, 0.50, 1.00, 0.35, 0.55, 0.4, 72.0, 32, 7, 0.85, "striker", comp=0.6),
            bf("David Onama", 12, 4, 5, 2, 5, 2, 1, 1, 5.00, 0.46, 0.48, 1.20, 0.38, 0.55, 0.3, 74.0, 28, 2, 0.65, "balanced", comp=0.55),
            5, "Steve Garcia", "KO/TKO", 1, card),
        FightResult(
            bf("Waldo Cortes-Acosta", 10, 2, 6, 0, 4, 1, 0, 1, 5.00, 0.48, 0.50, 0.50, 0.30, 0.55, 0.1, 76.0, 33, 3, 0.70, "striker", comp=0.45),
            bf("Ante Delija", 17, 8, 10, 3, 4, 5, 1, 2, 4.20, 0.43, 0.45, 0.80, 0.33, 0.50, 0.2, 79.0, 35, -2, 0.35, "striker", comp=0.5),
            3, "Waldo Cortes-Acosta", "KO/TKO", 1, card),
        FightResult(
            bf("Allan Nascimento", 22, 6, 3, 12, 7, 1, 2, 3, 3.50, 0.40, 0.50, 2.50, 0.42, 0.58, 1.2, 65.0, 33, 2, 0.60, "grappler", comp=0.55),
            bf("Cody Durden", 17, 6, 4, 3, 10, 2, 2, 2, 4.20, 0.43, 0.48, 2.80, 0.42, 0.55, 0.3, 66.0, 32, -1, 0.45, "balanced", comp=0.55),
            3, "Allan Nascimento", "SUB", 2, card),
        FightResult(
            bf("Norma Dumont", 11, 3, 2, 1, 8, 0, 1, 2, 4.50, 0.44, 0.52, 1.50, 0.38, 0.60, 0.2, 68.0, 35, 2, 0.55, "balanced", "F", comp=0.6),
            bf("Ketlen Vieira", 14, 4, 2, 1, 11, 0, 1, 3, 3.80, 0.42, 0.50, 1.80, 0.40, 0.55, 0.3, 67.0, 33, -1, 0.45, "balanced", "F", comp=0.75),
            3, "Norma Dumont", "DEC", 3, card),
    ])

    return fights


# ============================================================
# MAIN — RUN BACKTEST
# ============================================================

def main():
    fights = load_cards()
    scores = [score_fight(f) for f in fights]
    total = len(scores)

    print("=" * 110)
    print("  UFC PREDICTOR BACKTEST — 10 Cards, {total} Fights")
    print("  Scoring: Winner | Method | Round")
    print("=" * 110)

    # ── Per-fight results ──
    print(f"\n  {'Fight':<40} {'Pred':<22} {'Actual':<22} {'W':>2} {'M':>2} {'R':>2} {'Conf':>5}")
    print(f"  {'─' * 98}")

    current_card = ""
    for s in scores:
        if s.card != current_card:
            current_card = s.card
            print(f"\n  -- {current_card} --")

        pred_str = f"{s.pred_winner.split()[-1]} {s.pred_method} R{s.pred_round}"
        actual_str = f"{s.actual_winner.split()[-1]} {s.actual_method} R{s.actual_round}"
        w = "Y" if s.winner_correct else "X"
        m = "Y" if s.method_correct else "X"
        r = "Y" if s.round_exact else ("~" if s.round_within_1 else "X")

        print(f"  {s.fight:<40} {pred_str:<22} {actual_str:<22} {w:>2} {m:>2} {r:>2} {s.confidence:>4.0%}")

    # ── Aggregate ──
    print(f"\n\n{'━' * 110}")
    print(f"  SCORECARD")
    print(f"{'━' * 110}\n")

    w_correct = sum(1 for s in scores if s.winner_correct)
    m_correct = sum(1 for s in scores if s.method_correct)
    r_exact = sum(1 for s in scores if s.round_exact)
    r_close = sum(1 for s in scores if s.round_within_1)

    print(f"  Winner:          {w_correct}/{total} ({w_correct/total:.1%})")
    print(f"  Method:          {m_correct}/{total} ({m_correct/total:.1%})")
    print(f"  Round (exact):   {r_exact}/{total} ({r_exact/total:.1%})")
    print(f"  Round (within 1):{r_close}/{total} ({r_close/total:.1%})")
    print(f"  All 3 correct:   {sum(1 for s in scores if s.winner_correct and s.method_correct and s.round_exact)}/{total}")

    # ── By confidence ──
    print(f"\n  WINNER ACCURACY BY CONFIDENCE:")
    for min_c, label in [(0.0, "All"), (0.40, ">40%"), (0.60, ">60%"), (0.70, ">70%")]:
        filt = [s for s in scores if s.confidence >= min_c]
        if filt:
            c = sum(1 for s in filt if s.winner_correct)
            m = sum(1 for s in filt if s.method_correct)
            print(f"    {label:<8} Winner: {c}/{len(filt)} ({c/len(filt):.0%})  Method: {m}/{len(filt)} ({m/len(filt):.0%})")

    # ── By archetype ──
    print(f"\n  ACCURACY BY ARCHETYPE:")
    print(f"  {'Archetype':<25} {'N':>3} {'Winner':>8} {'Method':>8} {'Round~1':>8}")
    print(f"  {'─' * 58}")
    for arch in sorted(set(s.archetype for s in scores)):
        filt = [s for s in scores if s.archetype == arch]
        n = len(filt)
        w = sum(1 for s in filt if s.winner_correct) / n
        m = sum(1 for s in filt if s.method_correct) / n
        r = sum(1 for s in filt if s.round_within_1) / n
        print(f"  {arch:<25} {n:>3} {w:>7.0%} {m:>7.0%} {r:>7.0%}")

    # ── Misses ──
    misses = [s for s in scores if not s.winner_correct]
    if misses:
        print(f"\n  WRONG WINNER PICKS ({len(misses)}):")
        for s in misses:
            print(f"    {s.fight}: Picked {s.pred_winner.split()[-1]}, actual {s.actual_winner.split()[-1]} "
                  f"({s.archetype}, {s.confidence:.0%} conf)")

    print(f"\n{'=' * 110}")


if __name__ == "__main__":
    main()
