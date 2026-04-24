"""
Global Elo System — Every opponent gets a real rating.

Instead of starting all opponents at 1500, this system:
1. Fetches fight histories for the target fighters
2. Fetches fight histories for ALL their opponents (1 level deep)
3. Sorts all fights chronologically
4. Processes them into one global Elo pool

Now when Moicano loses to Makhachev, the system knows Makhachev
is 2200+ rated — not a generic 1500.

Usage:
    from global_elo import build_global_profiles
    profiles = build_global_profiles("Renato Moicano", "Chris Duncan")
"""

import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from profile_builder import find_espn_id, fetch_espn_profile, parse_espn_data, get_ufcstats_career, _curl_json

DATA_DIR = Path(__file__).parent / "data"
GLOBAL_ELO_CACHE = DATA_DIR / "global_elo_cache.json"


def _load_cache():
    if GLOBAL_ELO_CACHE.exists():
        return json.loads(GLOBAL_ELO_CACHE.read_text())
    return {}


def _save_cache(cache):
    GLOBAL_ELO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_ELO_CACHE.write_text(json.dumps(cache, indent=2))


def fetch_fighter_history(espn_id: str, cache: dict) -> Optional[dict]:
    """Fetch and parse a fighter's history. Uses cache to avoid repeat calls."""
    if espn_id in cache:
        return cache[espn_id]

    raw = fetch_espn_profile(espn_id)
    if not raw or 'athlete' not in raw:
        return None

    parsed = parse_espn_data(raw)
    cache[espn_id] = parsed
    time.sleep(0.3)  # be nice to ESPN
    return parsed


def collect_all_fights(fighter_a_name: str, fighter_b_name: str,
                        cutoff_date: str = "2026-04-04") -> Tuple[List[dict], dict]:
    """
    Collect ALL fights for both fighters and their opponents.
    Returns a list of all fights (sorted chronologically) and
    a mapping of fighter names to ESPN IDs.
    """
    cache = _load_cache()
    name_to_id = {}
    all_parsed = {}  # espn_id -> parsed data

    # Level 0: Get both main fighters
    fighters_to_fetch = []
    for name in [fighter_a_name, fighter_b_name]:
        eid = find_espn_id(name)
        if eid:
            name_to_id[name] = eid
            fighters_to_fetch.append(eid)

    # Fetch main fighters
    for eid in fighters_to_fetch:
        parsed = fetch_fighter_history(eid, cache)
        if parsed:
            all_parsed[eid] = parsed

    # Level 1: Get LAST 5 opponents only (most relevant, saves API calls)
    opponent_ids = set()
    for eid, parsed in all_parsed.items():
        recent_fights = parsed.get('fights', [])[-5:]  # last 5 (most recent)
        for fight in recent_fights:
            opp_id = fight.get('opponent_id', '')
            if opp_id and opp_id not in all_parsed:
                opponent_ids.add(opp_id)

    print(f"    Fetching {len(opponent_ids)} opponent histories...")
    fetched = 0
    for opp_id in opponent_ids:
        parsed = fetch_fighter_history(opp_id, cache)
        if parsed:
            all_parsed[opp_id] = parsed
            fetched += 1
        if fetched % 20 == 0 and fetched > 0:
            print(f"      ...{fetched}/{len(opponent_ids)}")

    # Save cache
    _save_cache(cache)
    print(f"    Fetched {fetched} opponent profiles (cached for reuse)")

    # Now build a global fight list from ALL fighters
    # Each fight entry: (date, winner_name, loser_name, method, event)
    all_fights = []
    seen_fights = set()  # avoid duplicates

    for eid, parsed in all_parsed.items():
        fighter_name = parsed.get('name', '')
        for fight in parsed.get('fights', []):
            date = fight.get('date', '')
            if date > cutoff_date:
                continue

            opp_name = fight.get('opponent', '')
            result = fight.get('result', '')
            method = fight.get('method', 'DEC')
            event = fight.get('event', '')

            # Create a unique fight key to avoid processing same fight twice
            fight_key = tuple(sorted([fighter_name, opp_name])) + (date,)
            if fight_key in seen_fights:
                continue
            seen_fights.add(fight_key)

            if result == 'W':
                all_fights.append({
                    'date': date,
                    'winner': fighter_name,
                    'loser': opp_name,
                    'method': method,
                    'event': event,
                })
            elif result == 'L':
                all_fights.append({
                    'date': date,
                    'winner': opp_name,
                    'loser': fighter_name,
                    'method': method,
                    'event': event,
                })

    # Sort chronologically
    all_fights.sort(key=lambda f: f['date'])
    print(f"    Total fights in global pool: {len(all_fights)}")

    return all_fights, name_to_id


def is_ufc_event(event_name: str) -> bool:
    return any(k in event_name for k in ['UFC', 'Dana White', 'DWCS', 'TUF'])


def compute_global_elo(all_fights: List[dict], target_names: List[str]) -> dict:
    """
    Process all fights chronologically and compute Elo for everyone.
    Returns final Elo for each target fighter plus their opponent history.
    """
    K = 80
    BASE = 1500
    ratings = defaultdict(lambda: BASE)
    fight_counts = defaultdict(int)

    # Track per-fighter opponent Elos for strength of schedule
    fighter_opp_elos = defaultdict(list)
    fighter_win_opp_elos = defaultdict(list)
    fighter_loss_opp_elos = defaultdict(list)

    for fight in all_fights:
        winner = fight['winner']
        loser = fight['loser']
        method = fight['method']
        is_ufc = is_ufc_event(fight.get('event', ''))

        ra = ratings[winner]
        rb = ratings[loser]

        # Record opponent Elo BEFORE update
        fighter_opp_elos[winner].append(rb)
        fighter_opp_elos[loser].append(ra)
        fighter_win_opp_elos[winner].append(rb)
        fighter_loss_opp_elos[loser].append(ra)

        # K-factor adjustments
        k_mult = 1.4 if method == 'KO/TKO' else (1.3 if method == 'SUB' else 1.0)
        level_mult = 1.5 if is_ufc else 0.7  # UFC fights count more

        def get_k(fighter):
            c = fight_counts[fighter]
            base = K * k_mult * level_mult
            if c < 3: return base * 2.0
            if c < 6: return base * 1.5
            return base

        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        ratings[winner] += get_k(winner) * (1 - ea)
        ratings[loser] -= get_k(loser) * ea

        fight_counts[winner] += 1
        fight_counts[loser] += 1

    # Build results for target fighters
    results = {}
    for name in target_names:
        elo = ratings[name]
        opp_elos = fighter_opp_elos[name]
        win_elos = fighter_win_opp_elos[name]
        loss_elos = fighter_loss_opp_elos[name]

        results[name] = {
            'elo': elo,
            'competition_level': max(0.1, min(1.0, (elo - 1100) / 800)),
            'avg_opp_elo': float(np.mean(opp_elos)) if opp_elos else BASE,
            'recent_opp_elo': float(np.mean(opp_elos[-3:])) if opp_elos else BASE,
            'best_win_elo': max(win_elos) if win_elos else BASE,
            'worst_loss_elo': min(loss_elos) if loss_elos else BASE,
            'avg_loss_opp_elo': float(np.mean(loss_elos)) if loss_elos else BASE,
            'losses_to_elite': sum(1 for e in loss_elos if e > 1650),
            'strength_of_schedule': (float(np.mean(opp_elos)) - 1500) / 200 if opp_elos else 0,
            'quality_of_losses': (float(np.mean(loss_elos)) - 1500) / 200 if loss_elos else 0,
        }

    return results


def build_global_profiles(name_a: str, name_b: str,
                           cutoff_date: str = "2026-04-04") -> Tuple[Optional[dict], Optional[dict]]:
    """
    Build complete profiles for two fighters using global Elo.
    This is the main entry point.
    """
    print(f"  Building global Elo for {name_a} vs {name_b}...")

    # Collect all fights
    all_fights, name_to_id = collect_all_fights(name_a, name_b, cutoff_date)

    if not all_fights:
        print("    No fights found")
        return None, None

    # Compute global Elo
    elo_results = compute_global_elo(all_fights, [name_a, name_b])

    # Now build full profiles using ESPN data + global Elo
    profiles = {}
    for name in [name_a, name_b]:
        eid = name_to_id.get(name) or find_espn_id(name)
        if not eid:
            continue

        raw = fetch_espn_profile(eid)
        if not raw:
            continue

        parsed = parse_espn_data(raw)
        fights = [f for f in parsed['fights'] if f['date'] < cutoff_date]
        ufc_fights = [f for f in fights if is_ufc_event(f.get('event', ''))]

        total_fights = len(fights)
        wins = sum(1 for f in fights if f['result'] == 'W')
        losses = sum(1 for f in fights if f['result'] == 'L')
        ko_w = sum(1 for f in fights if f['result'] == 'W' and f['method'] == 'KO/TKO')
        sub_w = sum(1 for f in fights if f['result'] == 'W' and f['method'] == 'SUB')
        dec_w = sum(1 for f in fights if f['result'] == 'W' and f['method'] == 'DEC')
        ko_l = sum(1 for f in fights if f['result'] == 'L' and f['method'] == 'KO/TKO')
        sub_l = sum(1 for f in fights if f['result'] == 'L' and f['method'] == 'SUB')
        dec_l = sum(1 for f in fights if f['result'] == 'L' and f['method'] == 'DEC')

        # Use UFC record for the model (matches training data)
        ufc_wins = sum(1 for f in ufc_fights if f['result'] == 'W')
        ufc_losses = sum(1 for f in ufc_fights if f['result'] == 'L')

        age = parsed.get('age', 30)
        win_total = max(wins, 1)
        total = max(total_fights, 1)

        streak = 0
        for f in reversed(fights):
            if f['result'] == 'W':
                streak = streak + 1 if streak >= 0 else 1
            elif f['result'] == 'L':
                streak = streak - 1 if streak <= 0 else -1
                break
            else:
                break

        recent = [1 if f['result'] == 'W' else (0 if f['result'] == 'L' else 0.5) for f in reversed(fights)]
        form_3 = float(np.mean(recent[:3])) if recent else 0.5
        form_5 = float(np.mean(recent[:5])) if recent else 0.5

        # UFCStats for striking
        ufc_stats = get_ufcstats_career(name)
        slpm = ufc_stats.get('slpm', 3.5)
        str_acc = ufc_stats.get('str_acc', 0.43)
        sapm = ufc_stats.get('sapm', 3.0)
        str_def = ufc_stats.get('str_def', 0.52)
        td_avg = ufc_stats.get('td_avg', 1.5)
        td_acc = ufc_stats.get('td_acc', 0.35)
        td_def = ufc_stats.get('td_def', 0.55)
        sub_avg = ufc_stats.get('sub_avg', 0.4)

        avg_fight_min = 10
        avg_sig_str_landed = slpm * avg_fight_min

        # Merge global Elo into profile
        elo_data = elo_results.get(name, {})

        profiles[name] = {
            'career_fights': len(ufc_fights) if ufc_fights else total_fights,
            'career_wins': ufc_wins if ufc_fights else wins,
            'losses': ufc_losses if ufc_fights else losses,
            'win_pct': ufc_wins / max(len(ufc_fights), 1) if ufc_fights else wins / total,
            'ko_wins': ko_w, 'sub_wins': sub_w, 'dec_wins': dec_w,
            'ko_losses': ko_l, 'sub_losses': sub_l, 'dec_losses': dec_l,
            'ko_win_rate': ko_w / win_total, 'sub_win_rate': sub_w / win_total,
            'finish_rate': (ko_w + sub_w) / win_total,
            'ko_loss_rate': ko_l / total, 'sub_loss_rate': sub_l / total,
            'been_finished_rate': (ko_l + sub_l) / total,
            'ko_finish_share': ko_w / max(ko_w + sub_w, 1),
            'sub_finish_share': sub_w / max(ko_w + sub_w, 1),
            'ko_loss_share': ko_l / max(ko_l + sub_l, 1),
            'sub_loss_share': sub_l / max(ko_l + sub_l, 1),
            'streak': streak, 'recent_form_3': form_3, 'recent_form_5': form_5,
            'avg_sig_str_landed': avg_sig_str_landed,
            'avg_sig_str_attempts': avg_sig_str_landed / max(str_acc, 0.1),
            'avg_sig_str_accuracy': str_acc,
            'avg_sig_str_received': sapm * avg_fight_min,
            'avg_sig_str_avoided': str_def, 'avg_sig_str_defense': str_def,
            'sig_str_per_min': slpm, 'sig_str_absorbed_per_min': sapm,
            'recent_sig_str_landed': avg_sig_str_landed,
            'recent_sig_str_accuracy': str_acc, 'recent_sig_str_defense': str_def,
            'avg_td_landed': td_avg * avg_fight_min / 15,
            'avg_td_attempts': (td_avg / max(td_acc, 0.1)) * avg_fight_min / 15,
            'avg_td_accuracy': td_acc, 'avg_tds_received': 1.0, 'avg_td_defense': td_def,
            'td_per_min': td_avg / 15,
            'recent_td_landed': td_avg * avg_fight_min / 15,
            'recent_td_accuracy': td_acc, 'recent_td_defense': td_def,
            'avg_sub_attempts': sub_avg * avg_fight_min / 15, 'sub_per_min': sub_avg / 15,
            'avg_kd': ko_w / total * 1.5, 'avg_kds_received': ko_l / total * 1.5,
            'kd_per_fight': ko_w / total * 1.5, 'kd_received_per_fight': ko_l / total * 1.5,
            # GLOBAL ELO — the whole point
            'elo': elo_data.get('elo', 1500),
            'competition_level': elo_data.get('competition_level', 0.5),
            'avg_opp_elo': elo_data.get('avg_opp_elo', 1500),
            'recent_opp_elo': elo_data.get('recent_opp_elo', 1500),
            'best_win_elo': elo_data.get('best_win_elo', 1500),
            'worst_loss_elo': elo_data.get('worst_loss_elo', 1500),
            'strength_of_schedule': elo_data.get('strength_of_schedule', 0),
            'avg_loss_opp_elo': elo_data.get('avg_loss_opp_elo', 1500),
            'losses_to_elite': elo_data.get('losses_to_elite', 0),
            'quality_of_losses': elo_data.get('quality_of_losses', 0),
            'age': age,
            'age_decline_signal': max(0, age - 32) * max(0, -streak) * 0.1,
            'age_weighted_form': form_3 * (1.0 - max(0, age - 30) * 0.05),
            'career_mileage': total_fights * max(0, age - 28) * 0.01,
        }

    return profiles.get(name_a), profiles.get(name_b)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        a, b = sys.argv[1], sys.argv[2]
        pa, pb = build_global_profiles(a, b)
        if pa and pb:
            print(f"\n  {a}: Elo {pa['elo']:.0f} | Avg Opp Elo {pa['avg_opp_elo']:.0f} | SoS {pa['strength_of_schedule']:.2f}")
            print(f"  {b}: Elo {pb['elo']:.0f} | Avg Opp Elo {pb['avg_opp_elo']:.0f} | SoS {pb['strength_of_schedule']:.2f}")
    else:
        print("Usage: python3 global_elo.py 'Fighter A' 'Fighter B'")
