"""
UFC Fight Predictor — Two Fighters In, Prediction Out

Takes two fighters, builds combat profiles, analyzes the matchup,
and outputs three predictions:
  1. Winner
  2. Method (KO/TKO, SUB, DEC)
  3. Round

No odds. No books. Just: who wins, how, and when.

Usage:
    from predictor import predict_fight, build_fighter

    fighter_a = build_fighter("Israel Adesanya", ...)
    fighter_b = build_fighter("Joe Pyfer", ...)
    result = predict_fight(fighter_a, fighter_b, num_rounds=5)
    print(result)

    # Or run the full Vegas 115 card:
    python3 predictor.py
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from ufc_vegas_115_v2 import Fighter


# ============================================================
# ELO
# ============================================================

def calculate_elo(f: Fighter) -> float:
    base = 1500
    total = f.wins + f.losses
    if total == 0:
        return base
    wc = f.ko_wins * 35 + f.sub_wins * 30 + f.dec_wins * 20 + f.losses * -25
    streak = max(min(f.win_streak * 30, 120), -90)
    # Age curve — MMA is a young person's game.
    # Prime: 26-31. Sharp dropoff after 33.
    # The old curve was too gentle (-8 per year past 33, capped at -60).
    # Real data: fighters over 34 lose at way higher rates.
    if 26 <= f.age <= 31:
        age_mod = 30  # prime years
    elif f.age < 26:
        age_mod = (f.age - 21) * 5  # developing
    elif f.age <= 33:
        age_mod = 30 - (f.age - 31) * 15  # starting to fade
    else:
        # 34+: steep decline. -25 per year past 33, no floor.
        age_mod = 0 - (f.age - 33) * 25
    act = (f.sig_strikes_per_min + f.takedowns_per_15 - 4.0) * 10
    recency_mod = (f.recent_form - 0.5) * 160

    raw_elo = base + wc + streak + age_mod + act + recency_mod

    # Competition level scales how much we trust the record.
    # A 10-1 record in regional shows (level=0.3) gets compressed toward baseline.
    # A 10-1 record against UFC-ranked fighters (level=0.85) keeps most of its value.
    #
    # BUT: a declining veteran (high comp, low form) shouldn't keep the full boost.
    # Vera at comp=0.85 but form=0.30 means he USED to fight elite guys but is
    # clearly fading. His past doesn't help him now.
    #
    # effective_comp = comp * (0.5 + 0.5 * recent_form)
    #   - Peak form (1.0): keeps full competition level
    #   - Neutral (0.5): keeps 75% of competition level
    #   - Declining (0.2): keeps only 60% of competition level
    comp = getattr(f, 'competition_level', 0.5)
    form = f.recent_form
    effective_comp = comp * (0.5 + 0.5 * form)
    adjusted_elo = base + (raw_elo - base) * effective_comp

    return round(adjusted_elo, 1)


def elo_wp(ea: float, eb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))


# ============================================================
# COMBAT PROFILE — 8 Dimensions (0.0–1.0)
# ============================================================

@dataclass
class CombatProfile:
    name: str
    ko_power: float
    sub_threat: float
    td_pressure: float
    volume: float
    chin: float
    sub_defense: float
    td_defense: float
    cardio: float
    finishing_instinct: float
    vulnerability: str      # "ko", "sub", "dec", "none"
    primary_weapon: str     # "ko", "sub", "dec"
    style_tag: str


def build_profile(f: Fighter) -> CombatProfile:
    total = max(f.wins + f.losses, 1)
    wins = max(f.wins, 1)

    ko_win_rate = f.ko_wins / total
    striking_output = f.sig_strikes_per_min * f.sig_strike_accuracy
    ko_power = min(1.0, (ko_win_rate * 2.0) + (striking_output / 6.0) * 0.3)

    sub_win_rate = f.sub_wins / total
    sub_threat = min(1.0, (sub_win_rate * 2.5) +
                     (f.sub_attempts_per_15 / 2.0) * 0.3 +
                     (f.takedowns_per_15 * f.takedown_accuracy / 3.0) * 0.2)

    td_success_rate = f.takedowns_per_15 * f.takedown_accuracy
    td_pressure = min(1.0, td_success_rate / 3.0)

    volume = min(1.0, f.sig_strikes_per_min / 6.5)

    ko_loss_rate = f.ko_losses / total
    chin = max(0.0, min(1.0,
        (1.0 - ko_loss_rate * 3.0) * 0.6 + f.sig_strike_defense * 0.4))

    sub_loss_rate = f.sub_losses / total
    sub_defense = max(0.0, min(1.0,
        (1.0 - sub_loss_rate * 3.0) * 0.5 + f.takedown_defense * 0.3 + 0.2))

    td_defense = min(1.0, f.takedown_defense + 0.05)

    age_factor = max(0.0, 1.0 - max(0, f.age - 30) * 0.08)
    cardio = min(1.0, age_factor * 0.3 + f.recent_form * 0.3 + volume * 0.2 + 0.2)

    finishing_instinct = (f.ko_wins + f.sub_wins) / wins if f.wins > 0 else 0.5

    if f.ko_losses >= 2 and ko_loss_rate > 0.15:
        vulnerability = "ko"
    elif f.sub_losses >= 2 and sub_loss_rate > 0.10:
        vulnerability = "sub"
    elif f.dec_losses >= 2:
        vulnerability = "dec"
    else:
        vulnerability = "none"

    if f.ko_wins >= f.sub_wins and f.ko_wins >= f.dec_wins:
        primary_weapon = "ko"
    elif f.sub_wins >= f.ko_wins and f.sub_wins >= f.dec_wins:
        primary_weapon = "sub"
    else:
        primary_weapon = "dec"

    if ko_power > 0.6 and td_pressure < 0.3:
        style_tag = "power striker"
    elif ko_power > 0.5 and volume > 0.6:
        style_tag = "volume striker"
    elif sub_threat > 0.5 and td_pressure > 0.4:
        style_tag = "submission grappler"
    elif td_pressure > 0.5 and sub_threat < 0.3:
        style_tag = "wrestle-controller"
    elif td_pressure > 0.4 and sub_threat > 0.3:
        style_tag = "well-rounded grappler"
    elif volume > 0.5 and td_pressure > 0.3:
        style_tag = "pressure fighter"
    elif chin > 0.7 and volume > 0.4:
        style_tag = "durable brawler"
    else:
        style_tag = "balanced"

    return CombatProfile(
        name=f.name,
        ko_power=round(ko_power, 3), sub_threat=round(sub_threat, 3),
        td_pressure=round(td_pressure, 3), volume=round(volume, 3),
        chin=round(chin, 3), sub_defense=round(sub_defense, 3),
        td_defense=round(td_defense, 3), cardio=round(cardio, 3),
        finishing_instinct=round(finishing_instinct, 3),
        vulnerability=vulnerability, primary_weapon=primary_weapon,
        style_tag=style_tag,
    )


# ============================================================
# MATCHUP EDGE DETECTION
# ============================================================

@dataclass
class MatchupEdge:
    attacker: str
    weapon: str
    target_vuln: str
    magnitude: float
    description: str


def find_edges(pa: CombatProfile, pb: CombatProfile) -> Tuple[List[MatchupEdge], List[MatchupEdge]]:
    edges_a = []
    edges_b = []

    # KO power vs chin
    for attacker, defender, ea, eb in [(pa, pb, edges_a, edges_b), (pb, pa, edges_b, edges_a)]:
        if attacker.ko_power > 0.3:
            chin_vuln = max(0, 0.75 - defender.chin)
            if chin_vuln > 0:
                mag = attacker.ko_power * chin_vuln * 1.5
                label = "questionable" if defender.chin < 0.55 else "exploitable"
                ea.append(MatchupEdge(
                    attacker.name, "ko_power", "weak_chin", mag,
                    f"{attacker.name}'s KO power ({attacker.ko_power:.0%}) vs {defender.name}'s {label} chin ({defender.chin:.0%})"
                ))

    # Sub threat vs sub defense
    for attacker, defender, ea, eb in [(pa, pb, edges_a, edges_b), (pb, pa, edges_b, edges_a)]:
        if attacker.sub_threat > 0.25:
            sub_gap = attacker.sub_threat - defender.sub_defense
            if sub_gap > -0.10:
                mag = max(0, attacker.sub_threat * (1.1 - defender.sub_defense))
                label = "weak" if defender.sub_defense < 0.6 else "testable"
                ea.append(MatchupEdge(
                    attacker.name, "sub_threat", "sub_vulnerable", mag,
                    f"{attacker.name}'s sub game ({attacker.sub_threat:.0%}) vs {defender.name}'s {label} sub defense ({defender.sub_defense:.0%})"
                ))

    # TD pressure vs TD defense
    for attacker, defender, ea, eb in [(pa, pb, edges_a, edges_b), (pb, pa, edges_b, edges_a)]:
        if attacker.td_pressure > 0.2:
            td_gap = attacker.td_pressure - defender.td_defense * 0.8
            if td_gap > 0:
                mag = td_gap * 1.2
                ea.append(MatchupEdge(
                    attacker.name, "td_pressure", "weak_tdd", mag,
                    f"{attacker.name}'s TD threat ({attacker.td_pressure:.0%}) vs {defender.name}'s TDD ({defender.td_defense:.0%})"
                ))

    # Volume vs low cardio
    for attacker, defender, ea, eb in [(pa, pb, edges_a, edges_b), (pb, pa, edges_b, edges_a)]:
        if attacker.volume > 0.4 and defender.cardio < 0.55:
            mag = attacker.volume * (1.0 - defender.cardio) * 0.8
            ea.append(MatchupEdge(
                attacker.name, "volume", "low_cardio", mag,
                f"{attacker.name}'s output ({attacker.volume:.0%}) can drown {defender.name}'s cardio ({defender.cardio:.0%})"
            ))

    # Wrestler neutralizing striker
    for attacker, defender, ea, eb in [(pa, pb, edges_a, edges_b), (pb, pa, edges_b, edges_a)]:
        if attacker.td_pressure > 0.3 and defender.td_defense < 0.70 and defender.volume > attacker.volume:
            mag = attacker.td_pressure * (1.0 - defender.td_defense)
            ea.append(MatchupEdge(
                attacker.name, "wrestling", "striker_neutralized", mag,
                f"{attacker.name} can wrestle-neutralize {defender.name}'s striking"
            ))

    # Volume edge
    vol_gap = pa.volume - pb.volume
    if abs(vol_gap) > 0.15:
        attacker = pa if vol_gap > 0 else pb
        defender = pb if vol_gap > 0 else pa
        mag = abs(vol_gap) * 0.6
        edge_list = edges_a if vol_gap > 0 else edges_b
        edge_list.append(MatchupEdge(
            attacker.name, "volume", "outstriked", mag,
            f"{attacker.name}'s volume ({attacker.volume:.0%}) overwhelms {defender.name}'s ({defender.volume:.0%})"
        ))

    return edges_a, edges_b


# ============================================================
# ARCHETYPE CLASSIFIER
# ============================================================

ARCHETYPES = {
    "ko_mismatch": {
        "desc": "KO Mismatch",
        "base_finish": 0.55, "ko_share": 0.75, "sub_share": 0.10, "dec_share": 0.15,
        "r1_share": 0.35, "r2_share": 0.35, "r3_share": 0.30,
    },
    "sub_mismatch": {
        "desc": "Submission Mismatch",
        "base_finish": 0.45, "ko_share": 0.15, "sub_share": 0.65, "dec_share": 0.20,
        "r1_share": 0.20, "r2_share": 0.35, "r3_share": 0.45,
    },
    "wrestle_vs_striker": {
        "desc": "Wrestler vs Striker",
        "base_finish": 0.30, "ko_share": 0.30, "sub_share": 0.20, "dec_share": 0.50,
        "r1_share": 0.20, "r2_share": 0.30, "r3_share": 0.50,
    },
    "two_strikers": {
        "desc": "Striker vs Striker",
        "base_finish": 0.40, "ko_share": 0.80, "sub_share": 0.05, "dec_share": 0.15,
        "r1_share": 0.30, "r2_share": 0.35, "r3_share": 0.35,
    },
    "two_grapplers": {
        "desc": "Grappler vs Grappler",
        "base_finish": 0.30, "ko_share": 0.10, "sub_share": 0.50, "dec_share": 0.40,
        "r1_share": 0.20, "r2_share": 0.30, "r3_share": 0.50,
    },
    "two_wrestlers": {
        "desc": "Wrestler vs Wrestler",
        "base_finish": 0.20, "ko_share": 0.25, "sub_share": 0.10, "dec_share": 0.65,
        "r1_share": 0.15, "r2_share": 0.30, "r3_share": 0.55,
    },
    "heavy_favorite": {
        "desc": "Heavy Mismatch",
        "base_finish": 0.55, "ko_share": 0.45, "sub_share": 0.25, "dec_share": 0.30,
        "r1_share": 0.35, "r2_share": 0.35, "r3_share": 0.30,
    },
    "competitive_balanced": {
        "desc": "Competitive",
        "base_finish": 0.35, "ko_share": 0.35, "sub_share": 0.20, "dec_share": 0.45,
        "r1_share": 0.25, "r2_share": 0.35, "r3_share": 0.40,
    },
    "aging_fighter": {
        "desc": "Age Factor",
        "base_finish": 0.45, "ko_share": 0.50, "sub_share": 0.15, "dec_share": 0.35,
        "r1_share": 0.25, "r2_share": 0.35, "r3_share": 0.40,
    },
}


def classify_archetype(pa: CombatProfile, pb: CombatProfile,
                        edges_a: List[MatchupEdge], edges_b: List[MatchupEdge],
                        fa: Fighter, fb: Fighter) -> str:
    all_edges = edges_a + edges_b

    ko_edges = [e for e in all_edges if e.weapon == "ko_power" and e.target_vuln == "weak_chin"]
    if ko_edges and max(e.magnitude for e in ko_edges) > 0.10:
        return "ko_mismatch"

    elo_gap = abs(fa.elo - fb.elo)
    if elo_gap > 300:
        return "heavy_favorite"

    sub_edges = [e for e in all_edges if e.weapon == "sub_threat" and e.target_vuln == "sub_vulnerable"]
    if sub_edges and max(e.magnitude for e in sub_edges) > 0.08:
        return "sub_mismatch"

    wrestler_edges = [e for e in all_edges if e.weapon in ("td_pressure", "wrestling")]
    if wrestler_edges and max(e.magnitude for e in wrestler_edges) > 0.05:
        wrestler = pa if pa.td_pressure > pb.td_pressure else pb
        striker = pb if pa.td_pressure > pb.td_pressure else pa
        if striker.td_pressure < wrestler.td_pressure * 0.6:
            return "wrestle_vs_striker"

    if abs(fa.age - fb.age) >= 4 and max(fa.age, fb.age) >= 34:
        return "aging_fighter"

    if pa.td_pressure > 0.35 and pb.td_pressure > 0.35:
        if pa.sub_threat > 0.25 or pb.sub_threat > 0.25:
            return "two_grapplers"
        return "two_wrestlers"

    if pa.td_pressure < 0.30 and pb.td_pressure < 0.30:
        if pa.volume > 0.35 and pb.volume > 0.35:
            return "two_strikers"

    a_grapply = pa.sub_threat > 0.35 or pa.td_pressure > 0.30
    b_grapply = pb.sub_threat > 0.35 or pb.td_pressure > 0.30
    if a_grapply and not b_grapply:
        return "wrestle_vs_striker"
    if b_grapply and not a_grapply:
        return "wrestle_vs_striker"
    if a_grapply and b_grapply:
        return "two_grapplers"

    return "competitive_balanced"


# ============================================================
# PREDICTION — The whole point
# ============================================================

@dataclass
class Prediction:
    fighter_a: str
    fighter_b: str
    winner: str
    win_prob: float
    method: str             # "KO/TKO", "SUB", "DEC"
    method_prob: float
    round: int              # predicted round of finish (or last round for DEC)
    goes_distance: bool
    finish_prob: float
    archetype: str
    confidence: float       # 0-1 how clear the matchup read is
    key_edge: str           # the single most important factor
    narrative: str


def predict_fight(fa: Fighter, fb: Fighter, num_rounds: int = 3) -> Prediction:
    """Two fighters in, prediction out."""
    fa.elo = calculate_elo(fa)
    fb.elo = calculate_elo(fb)

    pa = build_profile(fa)
    pb = build_profile(fb)
    edges_a, edges_b = find_edges(pa, pb)
    archetype = classify_archetype(pa, pb, edges_a, edges_b, fa, fb)
    arch = ARCHETYPES[archetype]

    # --- Winner ---
    base_wp_a = elo_wp(fa.elo, fb.elo)
    edge_shift_a = sum(e.magnitude * 0.15 for e in edges_a)
    edge_shift_b = sum(e.magnitude * 0.15 for e in edges_b)
    net_shift = max(-0.20, min(0.20, edge_shift_a - edge_shift_b))
    wp_a = max(0.05, min(0.95, base_wp_a + net_shift))
    wp_b = 1.0 - wp_a

    winner = fa.name if wp_a > wp_b else fb.name
    win_prob = max(wp_a, wp_b)

    # --- Finish Rate ---
    # FIX 1: Defense stats now PROPERLY suppress finish probability.
    # The loser's chin/sub_defense directly gates whether a finish happens.
    # A fighter with 90% chin doesn't get KO'd just because the archetype says so.

    # Who's the likely loser? Their defense is what matters for finish rate.
    if wp_a > wp_b:
        loser_chin = pb.chin
        loser_sub_def = pb.sub_defense
        loser_f = fb
        winner_f = fa
    else:
        loser_chin = pa.chin
        loser_sub_def = pa.sub_defense
        loser_f = fb if wp_a > wp_b else fa
        winner_f = fa if wp_a > wp_b else fb

    # Loser's defensive resistance — how hard are they to finish?
    # chin=0.80+ means very hard to KO, sub_defense=0.80+ means very hard to sub
    ko_resistance = loser_chin        # 0-1, higher = harder to KO
    sub_resistance = loser_sub_def    # 0-1, higher = harder to sub

    # Scale finish edges by how much the defense ALLOWS them through
    ko_edges_mag = sum(e.magnitude for e in edges_a + edges_b if e.weapon == "ko_power")
    sub_edges_mag = sum(e.magnitude for e in edges_a + edges_b if e.weapon == "sub_threat")

    # Effective finish threat = attack magnitude * (1 - defense)
    # High defense BLOCKS the finish even if the attacker has power
    effective_ko_threat = ko_edges_mag * (1.0 - ko_resistance * 0.7)
    effective_sub_threat = sub_edges_mag * (1.0 - sub_resistance * 0.6)
    total_effective_threat = effective_ko_threat + effective_sub_threat

    # FIX 2: Decision fighter recognition.
    # If both fighters historically go to decision, the fight probably will too.
    a_dec_rate = fa.dec_wins / max(fa.wins, 1)  # what % of A's wins are decisions
    b_dec_rate = fb.dec_wins / max(fb.wins, 1)
    a_total_dec = (fa.dec_wins + fa.dec_losses) / max(fa.wins + fa.losses, 1)  # total fights going distance
    b_total_dec = (fb.dec_wins + fb.dec_losses) / max(fb.wins + fb.losses, 1)

    # Combined "decision gravity" — how much both fighters pull toward distance
    # Two decision fighters (Curtis at 74% dec) = massive pull toward DEC
    decision_gravity = (a_total_dec + b_total_dec) / 2  # 0-1

    # Base finish from archetype, then adjust
    base_finish = arch["base_finish"]

    # Boost from effective finish threats (attack that gets past defense)
    threat_boost = total_effective_threat * 0.25

    # Suppression from defense stats — this is the key fix
    # Average of both fighters' defensive quality, weighted toward the loser
    defense_suppression = (ko_resistance * 0.3 + sub_resistance * 0.3 +
                           loser_chin * 0.2 + loser_sub_def * 0.2)
    # Defense above 0.6 starts suppressing finishes significantly
    if defense_suppression > 0.6:
        defense_penalty = (defense_suppression - 0.6) * 0.60  # up to -24% at max defense
    else:
        defense_penalty = 0.0

    # Decision gravity pull — high-DEC fighters drag finish rate down
    dec_pull = decision_gravity * 0.30  # 0-30% pull toward decision

    p_finish = base_finish + threat_boost - defense_penalty - dec_pull
    p_finish = max(0.10, min(0.75, p_finish))
    p_distance = 1.0 - p_finish

    # --- Method Distribution ---
    ko_share = arch["ko_share"]
    sub_share = arch["sub_share"]
    dec_share = arch["dec_share"]

    # Weight method shares by effective threats (defense-adjusted)
    if effective_ko_threat + effective_sub_threat > 0:
        ko_share = ko_share * 0.4 + (effective_ko_threat / (effective_ko_threat + effective_sub_threat + 0.01)) * 0.6
        sub_share = sub_share * 0.4 + (effective_sub_threat / (effective_ko_threat + effective_sub_threat + 0.01)) * 0.6

    total_share = ko_share + sub_share + dec_share
    ko_share /= total_share
    sub_share /= total_share
    dec_share /= total_share

    p_ko = p_finish * ko_share
    p_sub = p_finish * sub_share
    p_dec = p_distance

    if p_ko >= p_sub and p_ko >= p_dec:
        method = "KO/TKO"
        method_prob = p_ko
    elif p_sub >= p_ko and p_sub >= p_dec:
        method = "SUB"
        method_prob = p_sub
    else:
        method = "DEC"
        method_prob = p_dec

    # --- Round Prediction ---
    # FIX 3: Fighter-specific timing, not just archetype defaults.
    r1 = arch["r1_share"]
    r2 = arch["r2_share"]
    r3 = arch["r3_share"]

    # Counter-strikers and grapplers finish later (need to read/establish position)
    winner_pa = pa if wp_a > wp_b else pb
    if winner_pa.style_tag in ("submission grappler", "well-rounded grappler"):
        # Grapplers need time to get it down and work
        r1 -= 0.10
        r2 += 0.05
        r3 += 0.05
    elif winner_pa.volume < 0.4:
        # Low-volume fighters = counter-strikers, finish later
        r1 -= 0.05
        r2 += 0.02
        r3 += 0.03

    # High loser durability pushes finish later
    if loser_chin > 0.70:
        r1 -= 0.08
        r2 += 0.03
        r3 += 0.05

    # Normalize
    total_r = r1 + r2 + r3
    r1 /= total_r
    r2 /= total_r
    r3 /= total_r

    if method == "DEC":
        predicted_round = num_rounds
        goes_distance = True
    else:
        round_probs = [(1, r1), (2, r2), (3, r3)]
        if num_rounds == 5:
            # Spread remaining probability across R4/R5
            round_probs = [(1, r1 * 0.7), (2, r2 * 0.8), (3, r3 * 0.9),
                           (4, r1 * 0.15 + r2 * 0.1 + r3 * 0.05),
                           (5, r1 * 0.15 + r2 * 0.1 + r3 * 0.05)]
        predicted_round = max(round_probs, key=lambda x: x[1])[0]
        goes_distance = False

    # --- Confidence ---
    max_edge = max(
        max((e.magnitude for e in edges_a), default=0),
        max((e.magnitude for e in edges_b), default=0)
    )
    edge_count = len(edges_a) + len(edges_b)
    elo_gap = abs(fa.elo - fb.elo)
    confidence = min(1.0, max_edge * 0.5 + min(edge_count, 4) * 0.08 + min(elo_gap / 400, 0.3))

    # --- Key Edge ---
    all_edges = sorted(edges_a + edges_b, key=lambda e: -e.magnitude)
    key_edge = all_edges[0].description if all_edges else "No clear stylistic edge"

    # --- Narrative ---
    fav_name = winner
    dog_name = fb.name if winner == fa.name else fa.name

    if method == "DEC":
        how = f"outpoints {dog_name} over {num_rounds} rounds"
    elif method == "KO/TKO":
        how = f"stops {dog_name} in round {predicted_round}"
    else:
        how = f"submits {dog_name} in round {predicted_round}"

    narrative = f"{arch['desc']}: {fav_name} ({win_prob:.0%}) {how}."

    return Prediction(
        fighter_a=fa.name, fighter_b=fb.name,
        winner=winner, win_prob=win_prob,
        method=method, method_prob=method_prob,
        round=predicted_round, goes_distance=goes_distance,
        finish_prob=p_finish,
        archetype=arch["desc"], confidence=confidence,
        key_edge=key_edge, narrative=narrative,
    )


# ============================================================
# QUICK FIGHTER BUILDER
# ============================================================

def build_fighter(name, wins, losses, ko_w, sub_w, dec_w, ko_l, sub_l, dec_l,
                  sspm, ssacc, ssdef, tdp15, tdacc, tddef, subatt,
                  reach, age, streak, form=0.5, style="balanced", gender="M",
                  comp=0.5):
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
        competition_level=comp,
    )


# ============================================================
# MAIN — Run Vegas 115 Card
# ============================================================

if __name__ == "__main__":
    from ufc_vegas_115_v2 import fighters, matchups

    for k, f in fighters.items():
        f.elo = calculate_elo(f)

    print("=" * 100)
    print("  UFC VEGAS 115 — FIGHT PREDICTOR")
    print("  Winner | Method | Round")
    print("=" * 100)

    print(f"\n  {'#':<3} {'Fight':<40} {'Pick':<22} {'Method':<8} {'Rd':>3} {'Prob':>6} {'Conf':>6} {'Archetype':<20}")
    print(f"  {'─' * 97}")

    for i, (a_key, b_key, nr, is_main) in enumerate(matchups, 1):
        fa, fb = fighters[a_key], fighters[b_key]
        p = predict_fight(fa, fb, nr)

        tag = " *" if is_main else ""
        fight = f"{fa.name} vs {fb.name}{tag}"
        print(f"  {i:<3} {fight:<40} {p.winner:<22} {p.method:<8} R{p.round:>1} {p.win_prob:>5.0%} {p.confidence:>5.0%} {p.archetype:<20}")

    print(f"\n{'─' * 100}")
    print(f"  DETAILED READS:\n")

    for i, (a_key, b_key, nr, is_main) in enumerate(matchups, 1):
        fa, fb = fighters[a_key], fighters[b_key]
        p = predict_fight(fa, fb, nr)

        tag = " [MAIN]" if is_main else ""
        print(f"  {i}. {fa.name} vs {fb.name}{tag}")
        print(f"     {p.narrative}")
        print(f"     Key: {p.key_edge}")
        print(f"     Finish: {p.finish_prob:.0%} | Distance: {1-p.finish_prob:.0%}")
        print()

    print("=" * 100)
