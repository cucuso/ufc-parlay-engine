"""
UFC VEGAS 115: MOICANO vs DUNCAN — April 4, 2026
v1 ENGINE ONLY — Clean predictions for the full card
"""

import random
import math
import itertools
from dataclasses import dataclass
from typing import List, Dict

random.seed(42)

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
    elo: float = 1500.0
    style: str = "striker"
    gender: str = "M"

fighters = {
    "Moicano": Fighter("Renato Moicano", 20, 7, 5, 8, 7, 2, 4, 1,
                        sig_strikes_per_min=2.14, sig_strike_accuracy=0.37,
                        sig_strike_defense=0.63, takedowns_per_15=2.98,
                        takedown_accuracy=0.44, takedown_defense=0.73,
                        sub_attempts_per_15=1.2, reach_inches=71.0,
                        age=36, win_streak=1, style="grappler"),
    "Duncan": Fighter("Chris Duncan", 15, 2, 7, 4, 4, 1, 1, 0,
                       sig_strikes_per_min=4.85, sig_strike_accuracy=0.48,
                       sig_strike_defense=0.56, takedowns_per_15=1.20,
                       takedown_accuracy=0.38, takedown_defense=0.65,
                       sub_attempts_per_15=0.8, reach_inches=71.5,
                       age=32, win_streak=4, style="balanced"),
    "Jandiroba": Fighter("Virna Jandiroba", 22, 4, 2, 10, 10, 1, 0, 3,
                          sig_strikes_per_min=3.45, sig_strike_accuracy=0.42,
                          sig_strike_defense=0.55, takedowns_per_15=3.80,
                          takedown_accuracy=0.40, takedown_defense=0.70,
                          sub_attempts_per_15=1.5, reach_inches=63.0,
                          age=37, win_streak=1, style="grappler", gender="F"),
    "Ricci": Fighter("Tabatha Ricci", 13, 3, 3, 3, 7, 0, 2, 1,
                      sig_strikes_per_min=4.60, sig_strike_accuracy=0.44,
                      sig_strike_defense=0.58, takedowns_per_15=1.10,
                      takedown_accuracy=0.35, takedown_defense=0.62,
                      sub_attempts_per_15=0.4, reach_inches=64.0,
                      age=28, win_streak=2, style="striker", gender="F"),
    "Yakhyaev": Fighter("Abdul Rakhman Yakhyaev", 10, 1, 6, 2, 2, 0, 1, 0,
                         sig_strikes_per_min=5.10, sig_strike_accuracy=0.50,
                         sig_strike_defense=0.52, takedowns_per_15=1.80,
                         takedown_accuracy=0.45, takedown_defense=0.70,
                         sub_attempts_per_15=0.3, reach_inches=76.0,
                         age=28, win_streak=3, style="striker"),
    "Ribeiro": Fighter("Brendson Ribeiro", 16, 4, 8, 5, 3, 2, 1, 1,
                        sig_strikes_per_min=4.20, sig_strike_accuracy=0.45,
                        sig_strike_defense=0.48, takedowns_per_15=0.90,
                        takedown_accuracy=0.33, takedown_defense=0.55,
                        sub_attempts_per_15=0.5, reach_inches=75.0,
                        age=29, win_streak=-1, style="balanced"),
    "Estevam": Fighter("Rafael Estevam", 14, 2, 5, 4, 5, 1, 1, 0,
                        sig_strikes_per_min=4.70, sig_strike_accuracy=0.46,
                        sig_strike_defense=0.54, takedowns_per_15=2.50,
                        takedown_accuracy=0.42, takedown_defense=0.68,
                        sub_attempts_per_15=0.6, reach_inches=67.0,
                        age=26, win_streak=2, style="balanced"),
    "Ewing": Fighter("Ethyn Ewing", 11, 3, 4, 2, 5, 1, 1, 1,
                      sig_strikes_per_min=4.30, sig_strike_accuracy=0.43,
                      sig_strike_defense=0.50, takedowns_per_15=1.60,
                      takedown_accuracy=0.37, takedown_defense=0.60,
                      sub_attempts_per_15=0.3, reach_inches=68.0,
                      age=27, win_streak=1, style="balanced"),
    "McMillen": Fighter("Tommy McMillen", 10, 1, 5, 3, 2, 0, 1, 0,
                         sig_strikes_per_min=5.20, sig_strike_accuracy=0.49,
                         sig_strike_defense=0.58, takedowns_per_15=1.50,
                         takedown_accuracy=0.40, takedown_defense=0.72,
                         sub_attempts_per_15=0.4, reach_inches=72.0,
                         age=27, win_streak=5, style="striker"),
    "Zecchini": Fighter("Manolo Zecchini", 8, 3, 3, 2, 3, 1, 1, 1,
                         sig_strikes_per_min=3.80, sig_strike_accuracy=0.40,
                         sig_strike_defense=0.45, takedowns_per_15=1.00,
                         takedown_accuracy=0.30, takedown_defense=0.50,
                         sub_attempts_per_15=0.3, reach_inches=70.0,
                         age=30, win_streak=-1, style="balanced"),
    "Ruchala": Fighter("Robert Ruchala", 14, 3, 6, 3, 5, 1, 1, 1,
                        sig_strikes_per_min=4.40, sig_strike_accuracy=0.44,
                        sig_strike_defense=0.52, takedowns_per_15=1.20,
                        takedown_accuracy=0.35, takedown_defense=0.58,
                        sub_attempts_per_15=0.3, reach_inches=70.0,
                        age=28, win_streak=1, style="striker"),
    "Delano": Fighter("Jose Delano", 15, 2, 6, 5, 4, 1, 0, 1,
                       sig_strikes_per_min=4.80, sig_strike_accuracy=0.47,
                       sig_strike_defense=0.56, takedowns_per_15=2.00,
                       takedown_accuracy=0.42, takedown_defense=0.66,
                       sub_attempts_per_15=0.7, reach_inches=71.0,
                       age=27, win_streak=3, style="balanced"),
    "Vannata": Fighter("Lando Vannata", 12, 7, 5, 3, 4, 3, 2, 2,
                        sig_strikes_per_min=4.50, sig_strike_accuracy=0.44,
                        sig_strike_defense=0.50, takedowns_per_15=0.80,
                        takedown_accuracy=0.33, takedown_defense=0.60,
                        sub_attempts_per_15=0.5, reach_inches=72.0,
                        age=33, win_streak=1, style="striker"),
    "Flowers": Fighter("Darrius Flowers", 8, 1, 4, 2, 2, 0, 1, 0,
                        sig_strikes_per_min=4.00, sig_strike_accuracy=0.42,
                        sig_strike_defense=0.52, takedowns_per_15=1.30,
                        takedown_accuracy=0.36, takedown_defense=0.58,
                        sub_attempts_per_15=0.3, reach_inches=74.0,
                        age=27, win_streak=3, style="balanced"),
    "Bekoev": Fighter("Azamat Bekoev", 16, 0, 8, 5, 3, 0, 0, 0,
                       sig_strikes_per_min=5.50, sig_strike_accuracy=0.52,
                       sig_strike_defense=0.60, takedowns_per_15=2.40,
                       takedown_accuracy=0.48, takedown_defense=0.75,
                       sub_attempts_per_15=0.8, reach_inches=74.0,
                       age=28, win_streak=16, style="balanced"),
    "Gore": Fighter("Tresean Gore", 6, 4, 4, 0, 2, 3, 0, 1,
                     sig_strikes_per_min=3.90, sig_strike_accuracy=0.41,
                     sig_strike_defense=0.42, takedowns_per_15=0.50,
                     takedown_accuracy=0.25, takedown_defense=0.45,
                     sub_attempts_per_15=0.1, reach_inches=75.0,
                     age=28, win_streak=-1, style="striker"),
    "Petersen": Fighter("Thomas Petersen", 10, 4, 5, 2, 3, 2, 1, 1,
                         sig_strikes_per_min=4.10, sig_strike_accuracy=0.43,
                         sig_strike_defense=0.50, takedowns_per_15=1.40,
                         takedown_accuracy=0.38, takedown_defense=0.55,
                         sub_attempts_per_15=0.3, reach_inches=77.0,
                         age=30, win_streak=1, style="balanced"),
    "Pat": Fighter("Guilherme Pat", 10, 2, 5, 3, 2, 1, 0, 1,
                    sig_strikes_per_min=4.30, sig_strike_accuracy=0.45,
                    sig_strike_defense=0.52, takedowns_per_15=1.20,
                    takedown_accuracy=0.35, takedown_defense=0.60,
                    sub_attempts_per_15=0.4, reach_inches=76.0,
                    age=28, win_streak=2, style="balanced"),
    "Costa": Fighter("Alessandro Costa", 16, 4, 7, 5, 4, 2, 1, 1,
                      sig_strikes_per_min=5.00, sig_strike_accuracy=0.47,
                      sig_strike_defense=0.55, takedowns_per_15=2.10,
                      takedown_accuracy=0.43, takedown_defense=0.65,
                      sub_attempts_per_15=0.6, reach_inches=66.0,
                      age=29, win_streak=2, style="balanced"),
    "Nicoll": Fighter("Stewart Nicoll", 8, 2, 3, 2, 3, 1, 0, 1,
                       sig_strikes_per_min=3.60, sig_strike_accuracy=0.40,
                       sig_strike_defense=0.48, takedowns_per_15=0.80,
                       takedown_accuracy=0.30, takedown_defense=0.55,
                       sub_attempts_per_15=0.2, reach_inches=67.0,
                       age=27, win_streak=1, style="striker"),
    "Gatto": Fighter("Melissa Gatto", 14, 4, 4, 5, 5, 1, 1, 2,
                      sig_strikes_per_min=4.20, sig_strike_accuracy=0.44,
                      sig_strike_defense=0.52, takedowns_per_15=0.90,
                      takedown_accuracy=0.35, takedown_defense=0.60,
                      sub_attempts_per_15=0.6, reach_inches=66.0,
                      age=31, win_streak=1, style="balanced", gender="F"),
    "Barbosa": Fighter("Dione Barbosa", 10, 3, 3, 2, 5, 1, 1, 1,
                        sig_strikes_per_min=3.80, sig_strike_accuracy=0.41,
                        sig_strike_defense=0.50, takedowns_per_15=1.20,
                        takedown_accuracy=0.33, takedown_defense=0.55,
                        sub_attempts_per_15=0.3, reach_inches=65.0,
                        age=29, win_streak=1, style="balanced", gender="F"),
    "Cowan": Fighter("Hailey Cowan", 10, 3, 3, 3, 4, 1, 1, 1,
                      sig_strikes_per_min=4.00, sig_strike_accuracy=0.43,
                      sig_strike_defense=0.50, takedowns_per_15=1.50,
                      takedown_accuracy=0.38, takedown_defense=0.58,
                      sub_attempts_per_15=0.4, reach_inches=66.0,
                      age=27, win_streak=1, style="balanced", gender="F"),
    "Pereira": Fighter("Alice Pereira", 11, 2, 4, 3, 4, 0, 1, 1,
                        sig_strikes_per_min=4.30, sig_strike_accuracy=0.45,
                        sig_strike_defense=0.53, takedowns_per_15=1.10,
                        takedown_accuracy=0.36, takedown_defense=0.60,
                        sub_attempts_per_15=0.5, reach_inches=65.0,
                        age=28, win_streak=2, style="balanced", gender="F"),
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
    "Moicano": +170, "Duncan": -205,
    "Jandiroba": +110, "Ricci": -130,
    "Yakhyaev": -250, "Ribeiro": +210,
    "Estevam": -200, "Ewing": +170,
    "McMillen": -500, "Zecchini": +385,
    "Ruchala": +235, "Delano": -275,
    "Vannata": -170, "Flowers": +145,
    "Bekoev": -850, "Gore": +625,
    "Petersen": -130, "Pat": +110,
    "Costa": -450, "Nicoll": +350,
    "Gatto": -125, "Barbosa": +105,
    "Cowan": +120, "Pereira": -140,
}

# ============================================================
# V1 ENGINE
# ============================================================

STYLE_MATRIX = {
    "striker":  {"striker": 0.0, "grappler": 0.06, "wrestler": 0.03, "balanced": 0.0},
    "grappler": {"striker": -0.02, "grappler": 0.0, "wrestler": 0.04, "balanced": -0.01},
    "wrestler": {"striker": 0.08, "grappler": -0.02, "wrestler": 0.0, "balanced": 0.02},
    "balanced": {"striker": 0.02, "grappler": 0.03, "wrestler": -0.01, "balanced": 0.0},
}

def calculate_elo(f):
    base = 1500
    total = f.wins + f.losses
    if total == 0: return base
    wc = f.ko_wins*35 + f.sub_wins*30 + f.dec_wins*20 + f.losses*-25
    streak = max(min(f.win_streak*30, 120), -90)
    if 28 <= f.age <= 33: age_mod = 20
    elif f.age < 28: age_mod = (f.age-22)*3
    else: age_mod = max(-60, (33-f.age)*8)
    act = (f.sig_strikes_per_min + f.takedowns_per_15 - 4.0)*10
    return round(base + wc + streak + age_mod + act, 1)

def elo_wp(ea, eb):
    return 1.0 / (1.0 + 10**((eb-ea)/400.0))

def stat_prob(a, b):
    ta = max(a.wins+a.losses, 1); tb = max(b.wins+b.losses, 1)
    asp = a.sig_strikes_per_min*a.sig_strike_accuracy
    bsp = b.sig_strikes_per_min*b.sig_strike_accuracy
    p_a_ko = (a.ko_wins/ta)*(1+b.ko_losses/tb)*(asp/max(bsp,0.1))*0.35
    p_b_ko = (b.ko_wins/tb)*(1+a.ko_losses/ta)*(bsp/max(asp,0.1))*0.35
    atd = a.takedowns_per_15*a.takedown_accuracy*(1-b.takedown_defense)
    btd = b.takedowns_per_15*b.takedown_accuracy*(1-a.takedown_defense)
    p_a_sub = (a.sub_wins/ta)*(1+b.sub_losses/tb)*(1+atd)*a.sub_attempts_per_15*0.12
    p_b_sub = (b.sub_wins/tb)*(1+a.sub_losses/ta)*(1+btd)*b.sub_attempts_per_15*0.12
    rm = (a.reach_inches-b.reach_inches)*0.004
    sm = STYLE_MATRIX[a.style][b.style]
    ep = elo_wp(calculate_elo(a), calculate_elo(b))
    tf = min(p_a_ko+p_b_ko+p_a_sub+p_b_sub, 0.75)
    dp = 1.0 - tf - 0.01
    ads = max(0.2, min(0.8, ep+rm+sm))
    p_a_dec = dp*ads; p_b_dec = dp*(1-ads)
    t = p_a_ko+p_b_ko+p_a_sub+p_b_sub+p_a_dec+p_b_dec+0.01
    return {f"{a.name}_ko":p_a_ko/t, f"{a.name}_sub":p_a_sub/t, f"{a.name}_dec":p_a_dec/t,
            f"{b.name}_ko":p_b_ko/t, f"{b.name}_sub":p_b_sub/t, f"{b.name}_dec":p_b_dec/t, "draw":0.01/t}

def simulate(a, b, nr, ns=10000):
    random.seed(42)
    pr = stat_prob(a, b)
    res = []
    akr=pr[f"{a.name}_ko"]/nr; bkr=pr[f"{b.name}_ko"]/nr
    asr=pr[f"{a.name}_sub"]/nr; bsr=pr[f"{b.name}_sub"]/nr
    for _ in range(ns):
        fin=False; adv=0.0
        for rd in range(1, nr+1):
            fat=1.0+(rd-1)*0.15
            af=fat*(1+max(0,(a.age-33))*0.03); bf=fat*(1+max(0,(b.age-33))*0.03)
            roll=random.random(); c=0.0
            c+=akr*bf
            if roll<c: res.append((a.name,"ko",rd,False)); fin=True; break
            c+=bkr*af
            if roll<c: res.append((b.name,"ko",rd,False)); fin=True; break
            c+=asr*bf
            if roll<c: res.append((a.name,"sub",rd,False)); fin=True; break
            c+=bsr*af
            if roll<c: res.append((b.name,"sub",rd,False)); fin=True; break
            ao=a.sig_strikes_per_min*a.sig_strike_accuracy*(1-b.sig_strike_defense)+a.takedowns_per_15*a.takedown_accuracy*(1-b.takedown_defense)*2
            bo=b.sig_strikes_per_min*b.sig_strike_accuracy*(1-a.sig_strike_defense)+b.takedowns_per_15*b.takedown_accuracy*(1-a.takedown_defense)*2
            adv+=(ao-bo)+random.gauss(0,0.5)
        if not fin:
            if adv>0.3: res.append((a.name,"dec",nr,True))
            elif adv<-0.3: res.append((b.name,"dec",nr,True))
            else:
                if random.random()<0.5+adv*0.3: res.append((a.name,"dec",nr,True))
                else: res.append((b.name,"dec",nr,True))
    return res

def analyze(res, an, bn, nr):
    n=len(res)
    d = {
        f"{an}_ml":sum(1 for w,m,r,dd in res if w==an)/n,
        f"{bn}_ml":sum(1 for w,m,r,dd in res if w==bn)/n,
        f"{an}_ko":sum(1 for w,m,r,dd in res if w==an and m=="ko")/n,
        f"{an}_sub":sum(1 for w,m,r,dd in res if w==an and m=="sub")/n,
        f"{an}_dec":sum(1 for w,m,r,dd in res if w==an and m=="dec")/n,
        f"{bn}_ko":sum(1 for w,m,r,dd in res if w==bn and m=="ko")/n,
        f"{bn}_sub":sum(1 for w,m,r,dd in res if w==bn and m=="sub")/n,
        f"{bn}_dec":sum(1 for w,m,r,dd in res if w==bn and m=="dec")/n,
        "distance":sum(1 for w,m,r,dd in res if dd)/n,
        "finish":sum(1 for w,m,r,dd in res if not dd)/n,
    }
    for rd in range(1, nr+1):
        d[f"r{rd}_finish"]=sum(1 for w,m,r,dd in res if r==rd and not dd)/n
    return d

def american_to_implied(ml):
    if ml < 0: return abs(ml)/(abs(ml)+100)
    else: return 100/(ml+100)

def american_to_decimal(ml):
    if ml < 0: return 1+(100/abs(ml))
    else: return 1+(ml/100)

def kelly(tp, do):
    b=do-1
    if b==0: return 0
    return max(0, (b*tp-(1-tp))/b)

# ============================================================
# MAIN
# ============================================================

def main():
    for k, f in fighters.items():
        f.elo = calculate_elo(f)

    print("=" * 110)
    print("  UFC VEGAS 115: MOICANO vs DUNCAN — APRIL 4, 2026")
    print("  v1 ENGINE — 120,000 Monte Carlo Simulations")
    print("=" * 110)

    all_analysis = {}

    for a_key, b_key, nr, is_main in matchups:
        a, b = fighters[a_key], fighters[b_key]
        res = simulate(a, b, nr)
        ana = analyze(res, a.name, b.name, nr)
        all_analysis[(a_key, b_key)] = ana

    # ============================================================
    # FULL CARD PREDICTIONS
    # ============================================================
    print(f"\n{'━' * 110}")
    print(f"  FULL CARD PREDICTIONS")
    print(f"{'━' * 110}\n")

    print(f"  {'#':<3} {'Fight':<42} {'Winner':<22} {'Method':<12} {'Conf':>7} {'Book':>7} {'Edge':>8} {'Tier':>8}")
    print(f"  {'─' * 108}")

    for i, (a_key, b_key, nr, is_main) in enumerate(matchups, 1):
        a, b = fighters[a_key], fighters[b_key]
        ana = all_analysis[(a_key, b_key)]

        # Winner
        a_ml = ana[f"{a.name}_ml"]; b_ml = ana[f"{b.name}_ml"]
        if a_ml > b_ml:
            winner, win_prob, w_key = a.name, a_ml, a_key
        else:
            winner, win_prob, w_key = b.name, b_ml, b_key

        # Method
        methods = {"KO/TKO": ana.get(f"{winner}_ko",0), "SUB": ana.get(f"{winner}_sub",0), "DEC": ana.get(f"{winner}_dec",0)}
        best_method = max(methods, key=methods.get)

        # Book edge
        book_imp = american_to_implied(book_odds[w_key])
        edge = win_prob - book_imp

        # Confidence tier
        if win_prob > 0.85: tier = "LOCK"
        elif win_prob > 0.70: tier = "HIGH"
        elif win_prob > 0.55: tier = "MEDIUM"
        else: tier = "LEAN"

        tag = " ★" if is_main else ""
        fight = f"{a.name} vs {b.name}{tag}"
        print(f"  {i:<3} {fight:<42} {winner:<22} {best_method:<12} {win_prob:>6.1%} {book_imp:>6.1%} {edge:>+7.1%} {tier:>8}")

    # ============================================================
    # DETAILED FIGHT-BY-FIGHT
    # ============================================================
    print(f"\n\n{'=' * 110}")
    print(f"  DETAILED FIGHT-BY-FIGHT BREAKDOWN")
    print(f"{'=' * 110}")

    for i, (a_key, b_key, nr, is_main) in enumerate(matchups, 1):
        a, b = fighters[a_key], fighters[b_key]
        ana = all_analysis[(a_key, b_key)]

        a_ml = ana[f"{a.name}_ml"]; b_ml = ana[f"{b.name}_ml"]
        if a_ml > b_ml:
            winner, loser, win_prob, w_key = a.name, b.name, a_ml, a_key
        else:
            winner, loser, win_prob, w_key = b.name, a.name, b_ml, b_key

        book_imp = american_to_implied(book_odds[w_key])
        edge = win_prob - book_imp

        tag = " ★ MAIN EVENT" if is_main else ""
        print(f"\n  ┌─ Fight {i}{tag}")
        print(f"  │  {a.name} (Elo: {a.elo} | {a.style}) vs {b.name} (Elo: {b.elo} | {b.style}) | {nr}R")
        print(f"  │")
        print(f"  │  PREDICTION:  {winner} def. {loser}")
        print(f"  │  WIN PROB:    {win_prob:.1%}  │  Book: {book_imp:.1%}  │  Edge: {edge:+.1%}")
        print(f"  │")

        # All outcomes ranked
        outcomes = []
        for fkey in [a_key, b_key]:
            fn = fighters[fkey].name
            for method, label in [("ko","KO/TKO"),("sub","SUB"),("dec","DEC")]:
                p = ana.get(f"{fn}_{method}", 0)
                if p > 0.005:
                    outcomes.append((f"{fn} by {label}", p))
        outcomes.sort(key=lambda x: x[1], reverse=True)

        print(f"  │  OUTCOME PROBABILITIES:")
        for outcome, prob in outcomes:
            bar = '█' * int(prob * 50)
            marker = " ◄── MOST LIKELY" if outcome == outcomes[0][0] else ""
            print(f"  │    {outcome:<32} {prob:>5.1%} {bar}{marker}")

        # Distance & round props
        print(f"  │")
        print(f"  │  Goes to Decision: {ana['distance']:.1%}  │  Finish: {ana['finish']:.1%}")
        rd_str = "  │  Round finish distribution: "
        for rd in range(1, nr+1):
            rd_str += f"R{rd}: {ana.get(f'r{rd}_finish',0):.1%}  "
        print(rd_str)
        print(f"  └{'─' * 75}")

    # ============================================================
    # EDGE DETECTION — POSITIVE EDGE BETS
    # ============================================================
    print(f"\n\n{'=' * 110}")
    print(f"  EDGE DETECTION — WHERE THE BOOKS ARE WRONG")
    print(f"{'=' * 110}\n")

    edges = []
    for (a_key, b_key), ana in all_analysis.items():
        a, b = fighters[a_key], fighters[b_key]
        for fkey, fn in [(a_key, a.name), (b_key, b.name)]:
            bi = american_to_implied(book_odds[fkey])
            do = american_to_decimal(book_odds[fkey])
            mp = ana[f"{fn}_ml"]
            e = mp - bi
            k = kelly(mp, do)
            edges.append((fn, book_odds[fkey], bi, mp, e, k, do))
    edges.sort(key=lambda x: x[4], reverse=True)

    print(f"  {'Fighter':<28} {'Odds':>8} {'Book%':>8} {'Model%':>8} {'Edge':>8} {'Kelly%':>8} {'Play':>10}")
    print(f"  {'─' * 82}")
    for name, ml, bi, mp, e, k, do in edges:
        if e > 0:
            play = "VALUE" if e > 0.05 else "slight"
            print(f"  {name:<28} {ml:>+8d} {bi:>7.1%} {mp:>7.1%} {e:>+7.1%} {k:>7.1%} {'▲ '+play:>10}")

    # ============================================================
    # OPTIMAL 3-LEG PARLAYS
    # ============================================================
    print(f"\n\n{'=' * 110}")
    print(f"  OPTIMAL 3-LEG PARLAYS")
    print(f"{'=' * 110}\n")

    opts = []
    for (a_key, b_key), ana in all_analysis.items():
        a, b = fighters[a_key], fighters[b_key]
        fight = f"{a.name} vs {b.name}"
        for fkey, fn in [(a_key, a.name), (b_key, b.name)]:
            bi = american_to_implied(book_odds[fkey])
            do = american_to_decimal(book_odds[fkey])
            mp = ana[f"{fn}_ml"]
            e = mp - bi
            if e > 0.02:
                opts.append((fight, fn, f"{fn} ML ({book_odds[fkey]:+d})", mp, bi, do, e))

    parlays = []
    for combo in itertools.combinations(opts, 3):
        fights = [c[0] for c in combo]
        if len(set(fights)) != 3: continue
        cp=1.0; co=1.0; te=0.0
        for _,_,_,tp,_,do,e in combo:
            cp*=tp; co*=do; te+=e
        ev = cp*co - 1.0
        parlays.append((ev, cp, co, te, combo))
    parlays.sort(key=lambda x: x[0], reverse=True)

    for rank, (ev, prob, odds, te, legs) in enumerate(parlays[:5], 1):
        print(f"  ★ PARLAY #{rank}")
        print(f"    EV: {ev:+.1%} │ Hit Rate: {prob:.1%} │ Payout: {odds:.1f}x │ $10 → ${10*odds:.0f}")
        for _, fn, desc, tp, bi, do, e in legs:
            print(f"      {desc:<35} Model: {tp:.1%} vs Book: {bi:.1%} → Edge: {e:+.1%}")
        print()

    # ============================================================
    # LOCKS + LEANS SUMMARY
    # ============================================================
    print(f"{'=' * 110}")
    print(f"  QUICK SUMMARY — LOCKS vs LEANS")
    print(f"{'=' * 110}\n")

    for tier_name in ["LOCK", "HIGH", "MEDIUM", "LEAN"]:
        tier_fights = []
        for (a_key, b_key, nr, _) in matchups:
            a, b = fighters[a_key], fighters[b_key]
            ana = all_analysis[(a_key, b_key)]
            a_ml = ana[f"{a.name}_ml"]; b_ml = ana[f"{b.name}_ml"]
            wp = max(a_ml, b_ml)
            wn = a.name if a_ml > b_ml else b.name
            if wp > 0.85: t = "LOCK"
            elif wp > 0.70: t = "HIGH"
            elif wp > 0.55: t = "MEDIUM"
            else: t = "LEAN"
            if t == tier_name:
                tier_fights.append((wn, wp))

        if tier_fights:
            print(f"  {tier_name}:")
            for wn, wp in tier_fights:
                print(f"    → {wn} ({wp:.1%})")
            print()

    print(f"{'=' * 110}")
    print(f"  DISCLAIMER: Entertainment only. Gamble responsibly. v1 went 4/5 on Seattle.")
    print(f"{'=' * 110}")

if __name__ == "__main__":
    main()
