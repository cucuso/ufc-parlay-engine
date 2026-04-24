"""
Fight IQ — Matchup Intelligence Engine

Mimics an expert analyst who has watched every fighter's tape.
Instead of running stats through a generic formula, this module:

  1. Profiles each fighter along 8 combat dimensions
  2. Identifies the specific INTERACTION between those profiles
  3. Classifies the fight into a matchup archetype
  4. Generates calibrated predictions for that archetype

The key insight: a wrestler vs a chinny striker is NOT the same fight
as a wrestler vs a BJJ specialist. The specific vulnerabilities one
fighter has that the other can EXPLOIT are what determine outcomes.

This gives us a genuinely independent signal from the Monte Carlo sim
because it reasons about matchup dynamics, not just stat averages.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from ufc_vegas_115_v2 import (
    fighters, matchups, Fighter, calculate_elo, elo_wp,
)


# ============================================================
# FIGHTER COMBAT PROFILE — 8 Dimensions
#
# Each dimension is normalized to 0.0–1.0 where:
#   0.0 = bottom of UFC roster
#   0.5 = average UFC fighter
#   1.0 = elite / historically great at this
# ============================================================

@dataclass
class CombatProfile:
    name: str
    # Offensive
    ko_power: float        # how likely to KO opponents
    sub_threat: float      # how likely to submit opponents
    td_pressure: float     # ability to get fight to the ground
    volume: float          # striking output / activity level
    # Defensive
    chin: float            # ability to absorb strikes without getting KO'd
    sub_defense: float     # ability to avoid submissions
    td_defense: float      # ability to stay on feet
    cardio: float          # ability to maintain output over rounds
    # Meta
    finishing_instinct: float  # % of wins that come by finish (not decision)
    vulnerability: str     # primary way this fighter loses ("ko", "sub", "dec", "none")
    primary_weapon: str    # primary way this fighter wins
    style_tag: str         # human-readable style description


def build_profile(f: Fighter) -> CombatProfile:
    """Convert raw fighter stats into a normalized combat profile."""
    total = max(f.wins + f.losses, 1)
    wins = max(f.wins, 1)

    # --- KO Power ---
    # KO win rate on total fights, boosted by striking accuracy and volume
    ko_win_rate = f.ko_wins / total
    striking_output = f.sig_strikes_per_min * f.sig_strike_accuracy
    # Scale: 0 KOs + low output = 0.0, 40% KO rate + high output = 1.0
    ko_power = min(1.0, (ko_win_rate * 2.0) + (striking_output / 6.0) * 0.3)

    # --- Submission Threat ---
    sub_win_rate = f.sub_wins / total
    # Scale: sub rate + sub attempts + TD ability (need to get it down first)
    sub_threat = min(1.0, (sub_win_rate * 2.5) +
                     (f.sub_attempts_per_15 / 2.0) * 0.3 +
                     (f.takedowns_per_15 * f.takedown_accuracy / 3.0) * 0.2)

    # --- Takedown Pressure ---
    td_success_rate = f.takedowns_per_15 * f.takedown_accuracy
    # 3+ successful TDs per 15 min = elite (Khabib territory)
    td_pressure = min(1.0, td_success_rate / 3.0)

    # --- Volume ---
    # Sig strikes per min: UFC average ~4.0, elite ~6.0+
    volume = min(1.0, f.sig_strikes_per_min / 6.5)

    # --- Chin ---
    # How often does this fighter get KO'd? Lower = worse chin
    ko_loss_rate = f.ko_losses / total
    # Combine with strike defense
    # 0 KO losses + 60% defense = great chin
    # 3 KO losses in 10 fights + 40% defense = bad chin
    chin = max(0.0, min(1.0,
        (1.0 - ko_loss_rate * 3.0) * 0.6 +
        f.sig_strike_defense * 0.4
    ))

    # --- Sub Defense ---
    sub_loss_rate = f.sub_losses / total
    sub_defense = max(0.0, min(1.0,
        (1.0 - sub_loss_rate * 3.0) * 0.5 +
        f.takedown_defense * 0.3 +
        0.2  # base — everyone trains sub defense
    ))

    # --- TD Defense ---
    td_defense = min(1.0, f.takedown_defense + 0.05)  # slight boost from raw stat

    # --- Cardio ---
    # Younger + higher recent form + higher volume = better cardio
    age_factor = max(0.0, 1.0 - max(0, f.age - 30) * 0.08)
    cardio = min(1.0, (age_factor * 0.3 +
                       f.recent_form * 0.3 +
                       volume * 0.2 +
                       0.2))  # base

    # --- Finishing Instinct ---
    if f.wins > 0:
        finishing_instinct = (f.ko_wins + f.sub_wins) / f.wins
    else:
        finishing_instinct = 0.5

    # --- Primary Vulnerability ---
    if f.ko_losses >= 2 and ko_loss_rate > 0.15:
        vulnerability = "ko"
    elif f.sub_losses >= 2 and sub_loss_rate > 0.10:
        vulnerability = "sub"
    elif f.dec_losses >= 2:
        vulnerability = "dec"
    else:
        vulnerability = "none"

    # --- Primary Weapon ---
    if f.ko_wins >= f.sub_wins and f.ko_wins >= f.dec_wins:
        primary_weapon = "ko"
    elif f.sub_wins >= f.ko_wins and f.sub_wins >= f.dec_wins:
        primary_weapon = "sub"
    else:
        primary_weapon = "dec"

    # --- Style Tag ---
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
        ko_power=round(ko_power, 3),
        sub_threat=round(sub_threat, 3),
        td_pressure=round(td_pressure, 3),
        volume=round(volume, 3),
        chin=round(chin, 3),
        sub_defense=round(sub_defense, 3),
        td_defense=round(td_defense, 3),
        cardio=round(cardio, 3),
        finishing_instinct=round(finishing_instinct, 3),
        vulnerability=vulnerability,
        primary_weapon=primary_weapon,
        style_tag=style_tag,
    )


# ============================================================
# MATCHUP INTERACTION ANALYSIS
#
# This is the core insight: it's not about each fighter's stats
# in isolation. It's about how Fighter A's weapons interact with
# Fighter B's vulnerabilities (and vice versa).
# ============================================================

@dataclass
class MatchupEdge:
    """A specific advantage one fighter has over the other."""
    attacker: str
    weapon: str          # "ko_power", "sub_threat", "td_pressure", etc.
    target_vuln: str     # what vulnerability it exploits
    magnitude: float     # 0.0–1.0, how big the mismatch is
    description: str


@dataclass
class FightProfile:
    """Complete matchup analysis for a fight."""
    fighter_a: str
    fighter_b: str
    profile_a: CombatProfile
    profile_b: CombatProfile
    # Matchup edges
    edges_a: List[MatchupEdge]
    edges_b: List[MatchupEdge]
    # Archetype classification
    archetype: str
    archetype_desc: str
    # Predicted outcomes
    p_a_ko: float
    p_b_ko: float
    p_a_sub: float
    p_b_sub: float
    p_a_dec: float
    p_b_dec: float
    p_draw: float
    p_distance: float
    p_finish: float
    p_finish_r1: float
    p_finish_r2: float
    p_finish_r3: float
    # Confidence in our read
    read_confidence: float   # 0.0–1.0: how clear-cut is this matchup?
    narrative: str           # plain-English fight prediction


def find_matchup_edges(pa: CombatProfile, pb: CombatProfile) -> Tuple[List[MatchupEdge], List[MatchupEdge]]:
    """Find specific advantages each fighter has by matching weapons to vulnerabilities."""
    edges_a = []
    edges_b = []

    # --- KO POWER vs CHIN ---
    # Even moderate power is dangerous against a bad chin.
    # Any power > 0.3 matters; chin < 0.7 is exploitable.
    if pa.ko_power > 0.3:
        chin_vuln = max(0, 0.75 - pb.chin)  # how far below "solid" chin is
        if chin_vuln > 0:
            mag = pa.ko_power * chin_vuln * 1.5
            label = "questionable" if pb.chin < 0.55 else "exploitable"
            edges_a.append(MatchupEdge(
                pa.name, "ko_power", "weak_chin", mag,
                f"{pa.name}'s KO power ({pa.ko_power:.0%}) vs {pb.name}'s {label} chin ({pb.chin:.0%})"
            ))
    if pb.ko_power > 0.3:
        chin_vuln = max(0, 0.75 - pa.chin)
        if chin_vuln > 0:
            mag = pb.ko_power * chin_vuln * 1.5
            label = "questionable" if pa.chin < 0.55 else "exploitable"
            edges_b.append(MatchupEdge(
                pb.name, "ko_power", "weak_chin", mag,
                f"{pb.name}'s KO power ({pb.ko_power:.0%}) vs {pa.name}'s {label} chin ({pa.chin:.0%})"
            ))

    # --- SUBMISSION THREAT vs SUB DEFENSE ---
    # Sub threats matter even against decent defense if the gap is significant
    if pa.sub_threat > 0.25:
        sub_gap = pa.sub_threat - pb.sub_defense
        if sub_gap > -0.10:  # attacker within 10% of defender, or better
            mag = max(0, pa.sub_threat * (1.1 - pb.sub_defense))
            label = "weak" if pb.sub_defense < 0.6 else "testable"
            edges_a.append(MatchupEdge(
                pa.name, "sub_threat", "sub_vulnerable", mag,
                f"{pa.name}'s sub game ({pa.sub_threat:.0%}) vs {pb.name}'s {label} sub defense ({pb.sub_defense:.0%})"
            ))
    if pb.sub_threat > 0.25:
        sub_gap = pb.sub_threat - pa.sub_defense
        if sub_gap > -0.10:
            mag = max(0, pb.sub_threat * (1.1 - pa.sub_defense))
            label = "weak" if pa.sub_defense < 0.6 else "testable"
            edges_b.append(MatchupEdge(
                pb.name, "sub_threat", "sub_vulnerable", mag,
                f"{pb.name}'s sub game ({pb.sub_threat:.0%}) vs {pa.name}'s {label} sub defense ({pa.sub_defense:.0%})"
            ))

    # --- TD PRESSURE vs TD DEFENSE ---
    # Grappler with significant TD advantage = dictates where fight happens
    if pa.td_pressure > 0.2:
        td_gap = pa.td_pressure - pb.td_defense * 0.8  # discount TDD slightly
        if td_gap > 0:
            mag = td_gap * 1.2
            edges_a.append(MatchupEdge(
                pa.name, "td_pressure", "weak_tdd", mag,
                f"{pa.name}'s TD threat ({pa.td_pressure:.0%}) vs {pb.name}'s TDD ({pb.td_defense:.0%})"
            ))
    if pb.td_pressure > 0.2:
        td_gap = pb.td_pressure - pa.td_defense * 0.8
        if td_gap > 0:
            mag = td_gap * 1.2
            edges_b.append(MatchupEdge(
                pb.name, "td_pressure", "weak_tdd", mag,
                f"{pb.name}'s TD threat ({pb.td_pressure:.0%}) vs {pa.name}'s TDD ({pa.td_defense:.0%})"
            ))

    # --- VOLUME vs LOW CARDIO ---
    if pa.volume > 0.4 and pb.cardio < 0.55:
        mag = pa.volume * (1.0 - pb.cardio) * 0.8
        edges_a.append(MatchupEdge(
            pa.name, "volume", "low_cardio", mag,
            f"{pa.name}'s output ({pa.volume:.0%}) vs {pb.name}'s cardio ({pb.cardio:.0%}) — can drown in volume"
        ))
    if pb.volume > 0.4 and pa.cardio < 0.55:
        mag = pb.volume * (1.0 - pa.cardio) * 0.8
        edges_b.append(MatchupEdge(
            pb.name, "volume", "low_cardio", mag,
            f"{pb.name}'s output ({pb.volume:.0%}) vs {pa.name}'s cardio ({pa.cardio:.0%}) — can drown in volume"
        ))

    # --- WRESTLE-CONTROL vs STRIKER ---
    # Wrestler/grappler who can neutralize a striker's game
    if pa.td_pressure > 0.3 and pb.td_defense < 0.70 and pb.volume > pa.volume:
        mag = pa.td_pressure * (1.0 - pb.td_defense)
        edges_a.append(MatchupEdge(
            pa.name, "wrestling", "striker_neutralized", mag,
            f"{pa.name} can wrestle-neutralize {pb.name}'s striking (TDP {pa.td_pressure:.0%} vs TDD {pb.td_defense:.0%})"
        ))
    if pb.td_pressure > 0.3 and pa.td_defense < 0.70 and pa.volume > pb.volume:
        mag = pb.td_pressure * (1.0 - pa.td_defense)
        edges_b.append(MatchupEdge(
            pb.name, "wrestling", "striker_neutralized", mag,
            f"{pb.name} can wrestle-neutralize {pa.name}'s striking (TDP {pb.td_pressure:.0%} vs TDD {pa.td_defense:.0%})"
        ))

    # --- STRIKING VOLUME EDGE ---
    # One fighter massively outstrikes the other on the feet
    vol_gap = pa.volume - pb.volume
    if abs(vol_gap) > 0.15:
        attacker = pa if vol_gap > 0 else pb
        defender = pb if vol_gap > 0 else pa
        mag = abs(vol_gap) * 0.6
        edge_list = edges_a if vol_gap > 0 else edges_b
        edge_list.append(MatchupEdge(
            attacker.name, "volume", "outstriked",  mag,
            f"{attacker.name}'s volume ({attacker.volume:.0%}) overwhelms {defender.name}'s ({defender.volume:.0%})"
        ))

    return edges_a, edges_b


# ============================================================
# MATCHUP ARCHETYPE CLASSIFIER
#
# Based on the profiles and edges, classify the fight into one
# of these archetypes, each with calibrated outcome distributions
# derived from historical UFC patterns.
# ============================================================

ARCHETYPES = {
    "ko_mismatch": {
        # One fighter has serious KO power, opponent has a bad chin
        "desc": "KO Mismatch — power vs chin",
        "base_finish": 0.70, "ko_share": 0.75, "sub_share": 0.10, "dec_share": 0.15,
        "r1_share": 0.40, "r2_share": 0.30, "r3_share": 0.20,
    },
    "sub_mismatch": {
        # Grappler vs sub-vulnerable opponent
        "desc": "Submission Mismatch — grappler vs sub-vulnerable",
        "base_finish": 0.60, "ko_share": 0.15, "sub_share": 0.65, "dec_share": 0.20,
        "r1_share": 0.25, "r2_share": 0.35, "r3_share": 0.30,
    },
    "wrestle_vs_striker": {
        # Wrestler who can control vs pure striker. Grinding fight.
        "desc": "Wrestler vs Striker — grind it out",
        "base_finish": 0.35, "ko_share": 0.30, "sub_share": 0.20, "dec_share": 0.50,
        "r1_share": 0.20, "r2_share": 0.30, "r3_share": 0.35,
    },
    "two_strikers": {
        # Both fighters prefer to stand and bang
        "desc": "Striker vs Striker — fireworks likely",
        "base_finish": 0.55, "ko_share": 0.70, "sub_share": 0.05, "dec_share": 0.25,
        "r1_share": 0.35, "r2_share": 0.30, "r3_share": 0.25,
    },
    "two_grapplers": {
        # Both want the ground, positions cancel out
        "desc": "Grappler vs Grappler — positional chess",
        "base_finish": 0.40, "ko_share": 0.10, "sub_share": 0.50, "dec_share": 0.40,
        "r1_share": 0.20, "r2_share": 0.30, "r3_share": 0.35,
    },
    "two_wrestlers": {
        # Both wrestle, neither subs. Cage-stalling potential.
        "desc": "Wrestler vs Wrestler — TD stalemate, goes long",
        "base_finish": 0.25, "ko_share": 0.25, "sub_share": 0.10, "dec_share": 0.65,
        "r1_share": 0.15, "r2_share": 0.25, "r3_share": 0.35,
    },
    "heavy_favorite": {
        # Huge skill gap, favorite should dominate
        "desc": "Heavy Mismatch — class gap",
        "base_finish": 0.65, "ko_share": 0.45, "sub_share": 0.25, "dec_share": 0.30,
        "r1_share": 0.40, "r2_share": 0.30, "r3_share": 0.20,
    },
    "competitive_balanced": {
        # Close fight, both well-rounded, no glaring edges
        "desc": "Competitive — no clear stylistic edge",
        "base_finish": 0.40, "ko_share": 0.35, "sub_share": 0.20, "dec_share": 0.45,
        "r1_share": 0.25, "r2_share": 0.30, "r3_share": 0.30,
    },
    "aging_fighter": {
        # One fighter is significantly older, potential for durability issues
        "desc": "Age Factor — older fighter at risk of being stopped",
        "base_finish": 0.55, "ko_share": 0.50, "sub_share": 0.15, "dec_share": 0.35,
        "r1_share": 0.25, "r2_share": 0.30, "r3_share": 0.30,
    },
}


def classify_archetype(pa: CombatProfile, pb: CombatProfile,
                        edges_a: List[MatchupEdge], edges_b: List[MatchupEdge],
                        fa: Fighter, fb: Fighter) -> str:
    """Classify the fight into a matchup archetype.

    Priority order matters — check the most decisive dynamics first.
    A KO mismatch trumps everything. A big Elo gap is the next
    strongest signal. Then specific style interactions."""
    all_edges = edges_a + edges_b

    # 1. KO mismatch: one fighter has serious power vs a bad chin
    ko_edges = [e for e in all_edges if e.weapon == "ko_power" and e.target_vuln == "weak_chin"]
    if ko_edges and max(e.magnitude for e in ko_edges) > 0.10:
        return "ko_mismatch"

    # 2. Big Elo gap = heavy favorite (class difference)
    elo_gap = abs(fa.elo - fb.elo)
    if elo_gap > 300:
        return "heavy_favorite"

    # 3. Sub mismatch: grappler vs sub-vulnerable opponent
    sub_edges = [e for e in all_edges if e.weapon == "sub_threat" and e.target_vuln == "sub_vulnerable"]
    if sub_edges and max(e.magnitude for e in sub_edges) > 0.08:
        return "sub_mismatch"

    # 4. Wrestler vs striker: one fighter can dictate the fight location
    wrestler_edges = [e for e in all_edges if e.weapon in ("td_pressure", "wrestling")]
    if wrestler_edges and max(e.magnitude for e in wrestler_edges) > 0.05:
        # Which fighter is the wrestler?
        a_is_wrestler = pa.td_pressure > pb.td_pressure
        wrestler = pa if a_is_wrestler else pb
        striker = pb if a_is_wrestler else pa
        # Confirm the other guy is actually a striker (not also a wrestler)
        if striker.td_pressure < wrestler.td_pressure * 0.6:
            return "wrestle_vs_striker"

    # 5. Age factor: meaningful age gap with an older fighter
    if abs(fa.age - fb.age) >= 4 and max(fa.age, fb.age) >= 34:
        return "aging_fighter"

    # 6. Both wrestlers / both grapplers
    both_wrestlers = pa.td_pressure > 0.35 and pb.td_pressure > 0.35
    if both_wrestlers:
        if pa.sub_threat > 0.25 or pb.sub_threat > 0.25:
            return "two_grapplers"
        return "two_wrestlers"

    # 7. Both strikers: low TD, high volume on both sides
    if pa.td_pressure < 0.30 and pb.td_pressure < 0.30:
        if pa.volume > 0.35 and pb.volume > 0.35:
            return "two_strikers"

    # 8. One grappler vs one striker (but no dominant TD edge)
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
# PREDICTION ENGINE
#
# Combines: archetype base rates + specific matchup edges +
# Elo-derived win probability to produce final predictions.
# ============================================================

def analyze_matchup(a_key: str, b_key: str, nr: int) -> FightProfile:
    """Full matchup analysis for a single fight."""
    fa, fb = fighters[a_key], fighters[b_key]
    pa = build_profile(fa)
    pb = build_profile(fb)

    edges_a, edges_b = find_matchup_edges(pa, pb)
    archetype = classify_archetype(pa, pb, edges_a, edges_b, fa, fb)
    arch_data = ARCHETYPES[archetype]

    # Base Elo win probability
    base_wp_a = elo_wp(fa.elo, fb.elo)
    base_wp_b = 1.0 - base_wp_a

    # --- Adjust win probability based on matchup edges ---
    # Each edge shifts the fight toward the attacker.
    # But we cap the total shift to prevent runaway predictions.
    edge_shift_a = sum(e.magnitude * 0.15 for e in edges_a)
    edge_shift_b = sum(e.magnitude * 0.15 for e in edges_b)
    net_shift = edge_shift_a - edge_shift_b
    net_shift = max(-0.20, min(0.20, net_shift))  # cap at +/- 20%

    wp_a = max(0.05, min(0.95, base_wp_a + net_shift))
    wp_b = 1.0 - wp_a

    # --- Build outcome distribution ---
    base_finish = arch_data["base_finish"]

    # Adjust finish rate based on how many finish-related edges exist
    finish_edges = [e for e in edges_a + edges_b
                    if e.weapon in ("ko_power", "sub_threat")]
    if finish_edges:
        avg_finish_mag = sum(e.magnitude for e in finish_edges) / len(finish_edges)
        # Strong finish edges push finish rate up
        finish_adj = avg_finish_mag * 0.30
    else:
        finish_adj = 0.0

    # Both fighters being durable pushes toward distance
    durability_factor = (pa.chin + pb.chin + pa.sub_defense + pb.sub_defense) / 4
    durability_adj = (durability_factor - 0.5) * 0.20  # durable fighters = more distance

    # Adjust for 5-round fights (more time = more chance to finish, but also pace slower)
    round_adj = 0.0
    if nr == 5:
        round_adj = 0.05  # slightly more finishes in championship rounds

    p_finish = max(0.10, min(0.85, base_finish + finish_adj - durability_adj + round_adj))
    p_distance = 1.0 - p_finish
    p_draw = 0.005

    # --- Distribute finishes by method ---
    ko_share = arch_data["ko_share"]
    sub_share = arch_data["sub_share"]
    dec_share = arch_data["dec_share"]

    # Adjust shares based on specific edges
    ko_edge_mag = sum(e.magnitude for e in edges_a + edges_b if e.weapon == "ko_power")
    sub_edge_mag = sum(e.magnitude for e in edges_a + edges_b if e.weapon == "sub_threat")

    if ko_edge_mag > 0 or sub_edge_mag > 0:
        total_finish_edge = ko_edge_mag + sub_edge_mag
        if total_finish_edge > 0:
            ko_share = ko_share * 0.5 + (ko_edge_mag / total_finish_edge) * 0.5
            sub_share = sub_share * 0.5 + (sub_edge_mag / total_finish_edge) * 0.5

    # Renormalize shares
    total_share = ko_share + sub_share + dec_share
    ko_share /= total_share
    sub_share /= total_share
    dec_share /= total_share

    # Distribute among fighters by win probability
    p_a_ko = p_finish * ko_share * wp_a
    p_b_ko = p_finish * ko_share * wp_b
    p_a_sub = p_finish * sub_share * wp_a
    p_b_sub = p_finish * sub_share * wp_b
    p_a_dec = p_distance * wp_a * (1 - p_draw)
    p_b_dec = p_distance * wp_b * (1 - p_draw)

    # But: if fighter A has a specific KO edge over B (weak chin),
    # A should get MORE than their proportional share of KOs
    for edge in edges_a:
        if edge.weapon == "ko_power" and edge.target_vuln == "weak_chin":
            # Transfer some of B's KO wins to A
            transfer = min(p_b_ko * 0.5, edge.magnitude * 0.10)
            p_a_ko += transfer
            p_b_ko -= transfer
        if edge.weapon == "sub_threat" and edge.target_vuln == "sub_vulnerable":
            transfer = min(p_b_sub * 0.5, edge.magnitude * 0.08)
            p_a_sub += transfer
            p_b_sub -= transfer

    for edge in edges_b:
        if edge.weapon == "ko_power" and edge.target_vuln == "weak_chin":
            transfer = min(p_a_ko * 0.5, edge.magnitude * 0.10)
            p_b_ko += transfer
            p_a_ko -= transfer
        if edge.weapon == "sub_threat" and edge.target_vuln == "sub_vulnerable":
            transfer = min(p_a_sub * 0.5, edge.magnitude * 0.08)
            p_b_sub += transfer
            p_a_sub -= transfer

    # Normalize everything to sum to 1
    total = p_a_ko + p_b_ko + p_a_sub + p_b_sub + p_a_dec + p_b_dec + p_draw
    p_a_ko /= total
    p_b_ko /= total
    p_a_sub /= total
    p_b_sub /= total
    p_a_dec /= total
    p_b_dec /= total
    p_draw /= total

    p_finish = p_a_ko + p_b_ko + p_a_sub + p_b_sub
    p_distance = p_a_dec + p_b_dec + p_draw

    # --- Round distribution for finishes ---
    r1_share = arch_data["r1_share"]
    r2_share = arch_data["r2_share"]
    r3_share = arch_data["r3_share"]

    # Front-load more if there's a big KO mismatch edge
    if ko_edge_mag > 0.25:
        r1_share += 0.10
        r3_share -= 0.10
    # Back-load more if sub-heavy (subs take time to set up)
    if sub_edge_mag > ko_edge_mag:
        r1_share -= 0.05
        r2_share += 0.03
        r3_share += 0.02

    # Normalize
    total_r = r1_share + r2_share + r3_share
    p_finish_r1 = p_finish * (r1_share / total_r)
    p_finish_r2 = p_finish * (r2_share / total_r)
    p_finish_r3 = p_finish * (r3_share / total_r)

    # --- Read Confidence ---
    # How clear-cut is this matchup? High when there are big, obvious edges.
    # Low when fighters are well-matched with no clear interaction.
    max_edge_a = max((e.magnitude for e in edges_a), default=0)
    max_edge_b = max((e.magnitude for e in edges_b), default=0)
    max_edge = max(max_edge_a, max_edge_b)
    edge_count = len(edges_a) + len(edges_b)
    elo_gap = abs(fa.elo - fb.elo)

    read_confidence = min(1.0,
        max_edge * 0.5 +
        min(edge_count, 4) * 0.08 +
        min(elo_gap / 400, 0.3)
    )

    # --- Build Narrative ---
    narrative = _build_narrative(fa, fb, pa, pb, edges_a, edges_b,
                                 archetype, arch_data, wp_a,
                                 p_finish, p_distance, p_a_ko, p_b_ko,
                                 p_a_sub, p_b_sub, p_a_dec, p_b_dec)

    return FightProfile(
        fighter_a=fa.name, fighter_b=fb.name,
        profile_a=pa, profile_b=pb,
        edges_a=edges_a, edges_b=edges_b,
        archetype=archetype, archetype_desc=arch_data["desc"],
        p_a_ko=p_a_ko, p_b_ko=p_b_ko,
        p_a_sub=p_a_sub, p_b_sub=p_b_sub,
        p_a_dec=p_a_dec, p_b_dec=p_b_dec,
        p_draw=p_draw,
        p_distance=p_distance, p_finish=p_finish,
        p_finish_r1=p_finish_r1, p_finish_r2=p_finish_r2, p_finish_r3=p_finish_r3,
        read_confidence=read_confidence,
        narrative=narrative,
    )


def _build_narrative(fa, fb, pa, pb, edges_a, edges_b,
                      archetype, arch_data, wp_a,
                      p_finish, p_distance, p_a_ko, p_b_ko,
                      p_a_sub, p_b_sub, p_a_dec, p_b_dec) -> str:
    """Generate a plain-English fight read like an analyst would give."""
    fav = fa if wp_a > 0.5 else fb
    dog = fb if wp_a > 0.5 else fa
    fav_wp = max(wp_a, 1 - wp_a)
    fav_pa = pa if wp_a > 0.5 else pb
    dog_pa = pb if wp_a > 0.5 else pa

    lines = []
    lines.append(f"ARCHETYPE: {arch_data['desc']}")
    lines.append(f"{fav.name} ({fav_pa.style_tag}) should be the {fav_wp:.0%} favorite.")

    # Key edges
    fav_edges = edges_a if wp_a > 0.5 else edges_b
    dog_edges = edges_b if wp_a > 0.5 else edges_a

    if fav_edges:
        best_edge = max(fav_edges, key=lambda e: e.magnitude)
        lines.append(f"KEY EDGE: {best_edge.description}")

    if dog_edges:
        best_dog_edge = max(dog_edges, key=lambda e: e.magnitude)
        lines.append(f"UPSET PATH: {best_dog_edge.description}")
    else:
        lines.append(f"UPSET PATH: No clear stylistic path for {dog.name}. Must outwork on volume or land a flash KO.")

    # Distance call
    if p_distance > 0.60:
        lines.append(f"DISTANCE: Likely goes to decision ({p_distance:.0%}). Style matchup and durability suggest a full fight.")
    elif p_finish > 0.60:
        lines.append(f"FINISH: High finish probability ({p_finish:.0%}). Look for the {archetype.replace('_', ' ')} dynamic to produce a stoppage.")
    else:
        lines.append(f"COULD GO EITHER WAY: {p_finish:.0%} finish / {p_distance:.0%} distance. No strong lean on duration.")

    return " | ".join(lines)


# ============================================================
# MAIN — Full Card Analysis
# ============================================================

def analyze_full_card() -> Dict[Tuple[str, str], FightProfile]:
    """Analyze every fight on the card."""
    for k, f in fighters.items():
        f.elo = calculate_elo(f)

    results = {}
    for a_key, b_key, nr, is_main in matchups:
        fp = analyze_matchup(a_key, b_key, nr)
        results[(a_key, b_key)] = fp
    return results


def print_card_analysis():
    """Print full card analysis with combat profiles and matchup reads."""
    results = analyze_full_card()

    print("=" * 110)
    print("  FIGHT IQ — Matchup Intelligence Report")
    print("  Style Profiling | Vulnerability Mapping | Archetype Classification")
    print("=" * 110)

    for i, (a_key, b_key, nr, is_main) in enumerate(matchups, 1):
        fp = results[(a_key, b_key)]
        pa, pb = fp.profile_a, fp.profile_b
        tag = " [MAIN EVENT]" if is_main else ""

        print(f"\n{'━' * 110}")
        print(f"  Fight {i}{tag}: {fp.fighter_a} vs {fp.fighter_b}")
        print(f"{'━' * 110}")

        # Combat Profiles side by side
        print(f"\n  {'COMBAT PROFILE':<20} {fp.fighter_a:<25} {fp.fighter_b:<25}")
        print(f"  {'─' * 72}")
        dims = [
            ("KO Power",     pa.ko_power,    pb.ko_power),
            ("Sub Threat",   pa.sub_threat,   pb.sub_threat),
            ("TD Pressure",  pa.td_pressure,  pb.td_pressure),
            ("Volume",       pa.volume,       pb.volume),
            ("Chin",         pa.chin,         pb.chin),
            ("Sub Defense",  pa.sub_defense,  pb.sub_defense),
            ("TD Defense",   pa.td_defense,   pb.td_defense),
            ("Cardio",       pa.cardio,       pb.cardio),
        ]
        for label, val_a, val_b in dims:
            bar_a = "█" * int(val_a * 20)
            bar_b = "█" * int(val_b * 20)
            leader = "<" if val_a > val_b + 0.05 else (">" if val_b > val_a + 0.05 else "=")
            print(f"  {label:<20} {bar_a:<20} {val_a:.0%}  {leader}  {val_b:.0%} {bar_b:<20}")

        print(f"\n  Style:       {pa.style_tag:<25} {pb.style_tag:<25}")
        print(f"  Weapon:      {pa.primary_weapon:<25} {pb.primary_weapon:<25}")
        print(f"  Vulnerable:  {pa.vulnerability:<25} {pb.vulnerability:<25}")
        print(f"  Finisher:    {pa.finishing_instinct:.0%}{'':<22} {pb.finishing_instinct:.0%}")

        # Matchup Edges
        print(f"\n  MATCHUP EDGES:")
        if fp.edges_a:
            for e in sorted(fp.edges_a, key=lambda x: -x.magnitude):
                print(f"    >> {e.description} (magnitude: {e.magnitude:.2f})")
        if fp.edges_b:
            for e in sorted(fp.edges_b, key=lambda x: -x.magnitude):
                print(f"    >> {e.description} (magnitude: {e.magnitude:.2f})")
        if not fp.edges_a and not fp.edges_b:
            print(f"    No major stylistic mismatches — competitive fight")

        # Archetype & Predictions
        print(f"\n  ARCHETYPE: {fp.archetype_desc}")
        print(f"  Read Confidence: {fp.read_confidence:.0%}")

        # Outcome table
        print(f"\n  {'OUTCOME':<35} {'Prob':>7}")
        print(f"  {'─' * 44}")
        outcomes = [
            (f"{fp.fighter_a} by KO/TKO", fp.p_a_ko),
            (f"{fp.fighter_a} by SUB", fp.p_a_sub),
            (f"{fp.fighter_a} by DEC", fp.p_a_dec),
            (f"{fp.fighter_b} by KO/TKO", fp.p_b_ko),
            (f"{fp.fighter_b} by SUB", fp.p_b_sub),
            (f"{fp.fighter_b} by DEC", fp.p_b_dec),
        ]
        outcomes.sort(key=lambda x: -x[1])
        for label, prob in outcomes:
            bar = "█" * int(prob * 40)
            marker = " <-- MOST LIKELY" if prob == outcomes[0][1] else ""
            print(f"  {label:<35} {prob:>6.1%} {bar}{marker}")

        print(f"\n  Finish: {fp.p_finish:.0%} | Distance: {fp.p_distance:.0%}")
        print(f"  If finish: R1 {fp.p_finish_r1:.0%} | R2 {fp.p_finish_r2:.0%} | R3 {fp.p_finish_r3:.0%}")

        # Narrative
        print(f"\n  ANALYST READ:")
        for part in fp.narrative.split(" | "):
            print(f"    {part}")

    # Summary table
    print(f"\n\n{'=' * 110}")
    print(f"  FULL CARD SUMMARY")
    print(f"{'=' * 110}\n")

    print(f"  {'#':<3} {'Fight':<40} {'Archetype':<30} {'Finish%':>8} {'Dist%':>8} {'Conf':>6}")
    print(f"  {'─' * 98}")

    for i, (a_key, b_key, nr, is_main) in enumerate(matchups, 1):
        fp = results[(a_key, b_key)]
        tag = " *" if is_main else ""
        fight = f"{fp.fighter_a} vs {fp.fighter_b}{tag}"
        print(f"  {i:<3} {fight:<40} {fp.archetype_desc:<30} {fp.p_finish:>7.0%} {fp.p_distance:>7.0%} {fp.read_confidence:>5.0%}")

    print(f"\n{'=' * 110}")

    return results


if __name__ == "__main__":
    print_card_analysis()
