"""
UFC Fighter Profile Builder — ESPN API + UFCStats fallback

Builds a complete fighter profile from ESPN's API (full career including
regional fights) and UFCStats (per-fight striking/TD stats).

Returns a feature dict ready for the ML model.

Usage:
    from profile_builder import build_live_profile
    profile = build_live_profile("Chris Duncan")

    python3 profile_builder.py "Tommy McMillen"
"""

import json
import re
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple

DATA_DIR = Path(__file__).parent / "data"
ESPN_CACHE = DATA_DIR / "espn_ids.json"
ENRICHED_FIGHTERS_PATH = DATA_DIR / "enriched_fighters.csv"
OPP_ELO_CACHE_PATH = DATA_DIR / "opp_elo_cache.json"


# ============================================================
# Training-Elo lookup — fixes the live-Elo bug where
# compute_elo_from_history() starts every opponent at 1500, badly
# under-rating fighters who've faced elite competition (Burns, Costa, etc.)
# ============================================================

_TRAINING_CACHE = None

# Map display names that differ between ESPN and our training data.
_TRAINING_NAME_ALIAS = {
    "Patricio Pitbull": "Patricio Freire",
}

_TRAINING_FEATURES = (
    "elo", "competition_level",
    "avg_opp_elo", "recent_opp_elo",
    "best_win_elo", "worst_loss_elo",
    "strength_of_schedule", "avg_loss_opp_elo",
    "losses_to_elite", "quality_of_losses",
)


def _load_training_cache():
    """Lazy-load the latest training snapshot per fighter, keyed by name."""
    global _TRAINING_CACHE
    if _TRAINING_CACHE is not None:
        return _TRAINING_CACHE
    if not ENRICHED_FIGHTERS_PATH.exists():
        _TRAINING_CACHE = {}
        return _TRAINING_CACHE
    df = pd.read_csv(ENRICHED_FIGHTERS_PATH)
    df["date"] = pd.to_datetime(df["date"])
    latest = df.sort_values("date").groupby("fighter").tail(1)
    _TRAINING_CACHE = latest.set_index("fighter")
    return _TRAINING_CACHE


def _get_training_snapshot(name: str) -> Optional[Dict]:
    """Return the latest training-data values for a fighter, or None if unknown."""
    cache = _load_training_cache()
    if not len(cache):
        return None
    lookup = _TRAINING_NAME_ALIAS.get(name, name)
    # Exact match first.
    if lookup in cache.index:
        row = cache.loc[lookup]
    else:
        # Case-insensitive fallback.
        lower_map = {n.lower(): n for n in cache.index}
        if lookup.lower() in lower_map:
            row = cache.loc[lower_map[lookup.lower()]]
        else:
            return None
    out = {f: float(row[f]) for f in _TRAINING_FEATURES if f in row.index}
    wc = row.get("weight_class")
    if isinstance(wc, str) and wc:
        out["weight_class"] = wc
    return out


# ============================================================
# Opponent Elo lookup — used by compute_elo_from_history to replace
# the default 1500-for-everyone assumption. Prevents debutants from
# inflating Elo off regional scrubs (and vice versa).
# ============================================================

_OPP_ELO_CACHE = None


def _load_opp_elo_cache():
    global _OPP_ELO_CACHE
    if _OPP_ELO_CACHE is not None:
        return _OPP_ELO_CACHE
    if OPP_ELO_CACHE_PATH.exists():
        try:
            _OPP_ELO_CACHE = json.loads(OPP_ELO_CACHE_PATH.read_text())
        except Exception:
            _OPP_ELO_CACHE = {}
    else:
        _OPP_ELO_CACHE = {}
    return _OPP_ELO_CACHE


def _save_opp_elo_cache():
    if _OPP_ELO_CACHE is None:
        return
    OPP_ELO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPP_ELO_CACHE_PATH.write_text(json.dumps(_OPP_ELO_CACHE))


def _estimate_elo_from_record(wins: int, losses: int) -> float:
    """Crude Elo estimate from career W-L. Used when opponent isn't in training data.

    Scales shift by min(total, 20) so a 30-0 fighter doesn't get an absurd rating
    off a small sample of (possibly regional) wins. A 10-2 fighter → ~1567;
    a 0-3 fighter → ~1470.
    """
    total = wins + losses
    if total == 0:
        return 1500.0
    win_pct = wins / total
    effective_n = min(total, 20)
    return 1500.0 + effective_n * (win_pct - 0.5) * 20.0


def get_opponent_elo(name: str, espn_id: Optional[str] = None) -> float:
    """Return estimated Elo for an opponent. Priority: training cache → ESPN fetch → 1500."""
    cache = _load_opp_elo_cache()
    key = str(espn_id) if espn_id else name.lower().strip()
    if key in cache:
        return float(cache[key])

    # Priority 1: this fighter is in our UFC training data — use their real Elo.
    snap = _get_training_snapshot(name)
    if snap is not None:
        elo = float(snap.get("elo", 1500.0))
        cache[key] = elo
        _save_opp_elo_cache()
        return elo

    # Priority 2: fetch ESPN and estimate from record.
    if espn_id:
        data = _curl_json(
            f"https://site.web.api.espn.com/apis/common/v3/sports/mma/athletes/{espn_id}"
        )
        if data and "athlete" in data:
            athlete = data["athlete"]
            stats = {s["name"]: s for s in athlete.get("statsSummary", {}).get("statistics", [])}
            record_str = stats.get("wins-losses-draws", {}).get("displayValue", "0-0-0")
            parts = record_str.split("-")
            try:
                w = int(parts[0]) if len(parts) > 0 else 0
                l = int(parts[1]) if len(parts) > 1 else 0
            except ValueError:
                w = l = 0
            elo = _estimate_elo_from_record(w, l)
            cache[key] = elo
            _save_opp_elo_cache()
            return elo

    # Fallback
    cache[key] = 1500.0
    _save_opp_elo_cache()
    return 1500.0


# ============================================================
# HTTP
# ============================================================

def _curl(url: str) -> str:
    try:
        result = subprocess.run(
            ["curl", "-sk", url], capture_output=True, text=True, timeout=15
        )
        return result.stdout
    except Exception:
        return ""


def _curl_json(url: str) -> Optional[dict]:
    raw = _curl(url)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ============================================================
# ESPN FIGHTER ID LOOKUP
# ============================================================

def _load_espn_cache() -> dict:
    if ESPN_CACHE.exists():
        return json.loads(ESPN_CACHE.read_text())
    return {}


def _save_espn_cache(cache: dict):
    ESPN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ESPN_CACHE.write_text(json.dumps(cache, indent=2))


def find_espn_id(name: str) -> Optional[str]:
    """Find a fighter's ESPN ID by name."""
    cache = _load_espn_cache()
    name_lower = name.lower().strip()

    if name_lower in cache:
        return cache[name_lower]

    # Method 1: ESPN general search API (best — returns structured JSON)
    search_name = name.replace(" ", "+")
    data = _curl_json(
        f"https://site.web.api.espn.com/apis/common/v3/search?query={search_name}&limit=5&type=player"
    )
    if data and data.get("items"):
        for item in data["items"]:
            if item.get("sport") == "mma":
                item_name = item.get("displayName", "").lower()
                if name_lower.split()[-1] in item_name:
                    fid = item["id"]
                    cache[name_lower] = fid
                    _save_espn_cache(cache)
                    return fid
        # Fallback: first MMA result
        for item in data["items"]:
            if item.get("sport") == "mma":
                fid = item["id"]
                cache[name_lower] = fid
                _save_espn_cache(cache)
                return fid

    # Method 2: HTML search fallback
    search_name = name.replace(" ", "%20")
    html = _curl(f"https://search.espn.com/mma/results/_/search/{search_name}")
    if html:
        matches = re.findall(r'/mma/fighter/[^/]*/id/(\d+)/([^"\'>\s]+)', html)
        if matches:
            for fid, slug in matches:
                slug_clean = slug.replace("-", " ").lower()
                if name_lower.split()[-1] in slug_clean:
                    cache[name_lower] = fid
                    _save_espn_cache(cache)
                    return fid

    return None


# ============================================================
# ESPN API DATA FETCHING
# ============================================================

def fetch_espn_profile(espn_id: str) -> Optional[dict]:
    """Fetch full fighter data from ESPN API."""
    return _curl_json(
        f"https://site.web.api.espn.com/apis/common/v3/sports/mma/athletes/{espn_id}"
    )


# Map lowercased weight-class strings to the exact format used in
# enriched_fights.csv / enriched_fighters.csv (see training_data build_training_data.py).
_WC_MAP = {
    "heavyweight": "Men_Heavyweight",
    "light heavyweight": "Men_Light Heavyweight",
    "middleweight": "Men_Middleweight",
    "welterweight": "Men_Welterweight",
    "lightweight": "Men_Lightweight",
    "featherweight": "Men_Featherweight",
    "bantamweight": "Men_Bantamweight",
    "flyweight": "Men_Flyweight",
    "catchweight": "Men_Catchweight",
    "women's strawweight": "Women_Strawweight",
    "women's flyweight": "Women_Flyweight",
    "women's bantamweight": "Women_Bantamweight",
    "women's featherweight": "Women_Featherweight",
    "strawweight": "Women_Strawweight",  # strawweight only exists in women's UFC
}


def _normalize_weight_class(text: str, is_female: bool = False) -> Optional[str]:
    """Normalize an ESPN weight-class string to the training-data format.

    Women's divisions use the `Women_` prefix (e.g. Women_Flyweight); men's use
    `Men_`. Gender comes from athlete.gender since ESPN returns plain division
    names like "Flyweight" regardless of sex.
    """
    if not text:
        return None
    t = str(text).lower().strip()
    t = re.sub(r"^(ufc|mma|\s)+", "", t).strip()
    match = _WC_MAP.get(t)
    if match is None:
        for key, val in _WC_MAP.items():
            if key in t:
                match = val
                break
    if match is None:
        return None
    if is_female and match.startswith("Men_"):
        # Strawweight is already Women_-prefixed and only exists in women's UFC.
        return match.replace("Men_", "Women_", 1)
    return match


def _dict_text(d) -> Optional[str]:
    """Extract a human-readable label from an ESPN subdoc (tries several common keys)."""
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        for k in ("text", "displayName", "name", "shortName", "slug"):
            v = d.get(k)
            if isinstance(v, str) and v:
                return v
    return None


def _extract_weight_class(data: dict) -> Optional[str]:
    """Pull the fighter's weight class from ESPN. Returns training-format string or None."""
    athlete = data.get("athlete", {}) or {}
    gender = str(athlete.get("gender", "")).lower()
    is_female = gender in ("f", "female", "w", "women")

    # Path 1: athlete.weightClass (confirmed ESPN schema — dict with a `text` key)
    for key in ("weightClass", "weight_class"):
        wc = _normalize_weight_class(_dict_text(athlete.get(key)), is_female)
        if wc:
            return wc

    # Path 2: athlete.position (some older ESPN responses)
    wc = _normalize_weight_class(_dict_text(athlete.get("position")), is_female)
    if wc:
        return wc

    # Path 3: most recent event's weight class / division / league label
    events_map = data.get("eventsMap", {}) or {}
    for uid in data.get("events", []):
        ev = events_map.get(uid, {})
        for key in ("weightClass", "division", "league"):
            wc = _normalize_weight_class(_dict_text(ev.get(key)), is_female)
            if wc:
                return wc
        wc = _normalize_weight_class(ev.get("shortName"), is_female)
        if wc:
            return wc
    return None


def parse_espn_data(data: dict) -> dict:
    """Parse ESPN API response into structured fighter data."""
    athlete = data["athlete"]
    name = athlete.get("displayName", "")
    age = athlete.get("age", 30)
    weight_class = _extract_weight_class(data)

    stats = {s["name"]: s for s in athlete.get("statsSummary", {}).get("statistics", [])}

    record_str = stats.get("wins-losses-draws", {}).get("displayValue", "0-0-0")
    parts = record_str.split("-")
    total_wins = int(parts[0]) if len(parts) > 0 else 0
    total_losses = int(parts[1]) if len(parts) > 1 else 0

    tko_str = stats.get("tkos-tkoLosses", {}).get("displayValue", "0-0")
    tko_parts = tko_str.split("-")
    ko_wins = int(tko_parts[0]) if len(tko_parts) > 0 else 0
    ko_losses = int(tko_parts[1]) if len(tko_parts) > 1 else 0

    sub_str = stats.get("submissions-submissionLosses", {}).get("displayValue", "0-0")
    sub_parts = sub_str.split("-")
    sub_wins = int(sub_parts[0]) if len(sub_parts) > 0 else 0
    sub_losses = int(sub_parts[1]) if len(sub_parts) > 1 else 0

    dec_wins = max(0, total_wins - ko_wins - sub_wins)
    dec_losses = max(0, total_losses - ko_losses - sub_losses)

    # Fight history (reverse to get chronological order — oldest first)
    events_order = data.get("events", [])
    events_map = data.get("eventsMap", {})

    fights = []
    for uid in reversed(events_order):
        ev = events_map.get(uid, {})
        if not ev:
            continue

        status = ev.get("status", {})
        result_info = status.get("result", {})
        method_name = result_info.get("name", "")

        if "ko" in method_name.lower() or "tko" in method_name.lower():
            method = "KO/TKO"
        elif "sub" in method_name.lower():
            method = "SUB"
        elif "dec" in method_name.lower():
            method = "DEC"
        else:
            method = "OTHER"

        fights.append({
            "result": ev.get("gameResult", ""),
            "opponent": ev.get("opponent", {}).get("displayName", ""),
            "opponent_id": ev.get("opponent", {}).get("id", ""),
            "method": method,
            "round": status.get("period", 0),
            "time": status.get("displayClock", "0:00"),
            "date": ev.get("gameDate", "")[:10],
            "event": ev.get("shortName", ""),
        })

    return {
        "name": name, "age": age,
        "weight_class": weight_class,
        "total_wins": total_wins, "total_losses": total_losses,
        "ko_wins": ko_wins, "sub_wins": sub_wins, "dec_wins": dec_wins,
        "ko_losses": ko_losses, "sub_losses": sub_losses, "dec_losses": dec_losses,
        "fights": fights,
    }


# ============================================================
# FIGHT-PACE SIGNATURE — 7 features mirroring build_training_data.py
# ============================================================

def _parse_time_to_seconds(time_str: str) -> int:
    """Convert "M:SS" or "MM:SS" round-time string to seconds."""
    try:
        m, s = str(time_str).split(":")
        return int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return 0


def compute_pace_signature(fights: list) -> dict:
    """Mirror of FighterStats._pace_signature_features() in build_training_data.py.

    Computes the same 7 fight-pace features from a live fight history list.
    `fights` is the chronological list returned by parse_espn_data (oldest
    first), each entry containing: result ("W"/"L"/...), method ("KO/TKO" |
    "SUB" | "DEC" | "OTHER"), round (int), time ("M:SS").
    """
    n = len(fights)
    if n == 0:
        return {
            "r1_ending_rate": 0.0,
            "r1_ending_rate_5": 0.0,
            "decision_rate": 0.0,
            "decision_rate_5": 0.0,
            "sub_win_rate_5": 0.0,
            "ko_win_rate_5": 0.0,
            "avg_fight_seconds": 600.0,
            "recent_3_finish_loss_rate": 0.0,
            "recent_5_finish_loss_rate": 0.0,
            "recent_5_r1_loss_rate": 0.0,
        }

    rounds = [int(f.get("round", 3) or 3) for f in fights]
    methods = [str(f.get("method", "OTHER")) for f in fights]
    results = [str(f.get("result", "")).upper() for f in fights]
    fight_seconds = [
        max(0, (rounds[i] - 1) * 300) + _parse_time_to_seconds(f.get("time", "0:00"))
        for i, f in enumerate(fights)
    ]

    # Career
    r1_ending_rate = sum(1 for r in rounds if r == 1) / n
    decision_rate = sum(1 for m in methods if m == "DEC") / n
    avg_fight_seconds = float(sum(fight_seconds)) / n if fight_seconds else 600.0

    # Last 5 — captures current style, not career average
    last5_rounds = rounds[-5:]
    last5_methods = methods[-5:]
    last5_results = results[-5:]
    n5 = len(last5_rounds)
    r1_ending_rate_5 = sum(1 for r in last5_rounds if r == 1) / n5
    decision_rate_5 = sum(1 for m in last5_methods if m == "DEC") / n5
    last5_wins = sum(1 for r in last5_results if r in ("W", "WIN"))
    if last5_wins > 0:
        sub_win_rate_5 = sum(
            1 for i in range(n5)
            if last5_results[i] in ("W", "WIN") and last5_methods[i] == "SUB"
        ) / last5_wins
        ko_win_rate_5 = sum(
            1 for i in range(n5)
            if last5_results[i] in ("W", "WIN") and last5_methods[i] == "KO/TKO"
        ) / last5_wins
    else:
        sub_win_rate_5 = 0.0
        ko_win_rate_5 = 0.0

    def is_loss(i):
        return results[i] in ("L", "LOSS")
    def is_finish_loss(i):
        return is_loss(i) and methods[i] in ("KO/TKO", "SUB")
    def is_r1_loss(i):
        return is_loss(i) and rounds[i] == 1 and methods[i] in ("KO/TKO", "SUB")

    last3_idx = range(max(0, n - 3), n)
    last5_idx = range(max(0, n - 5), n)
    recent_3_finish_loss_rate = sum(1 for i in last3_idx if is_finish_loss(i)) / max(1, len(last3_idx))
    recent_5_finish_loss_rate = sum(1 for i in last5_idx if is_finish_loss(i)) / max(1, len(last5_idx))
    recent_5_r1_loss_rate     = sum(1 for i in last5_idx if is_r1_loss(i))     / max(1, len(last5_idx))

    # ── FIGHT-DENOMINATOR pace stats (added 2026-04-30) ──
    # Existing sub_win_rate_5 / ko_win_rate_5 use LAST-5-WINS as denominator,
    # which over-weights fighters on losing skids who have few but stylistic
    # wins (Morales: 3 UFC losses then 2 regional sub wins → "100% SUB
    # last-5"). The per-FIGHT rates below use n5 (fights, max 5) as
    # denominator regardless of W/L — the honest "is this fighter actually
    # finishing fights at a high rate right now" signal.
    sub_finishes_in_last5 = sum(
        1 for i in range(n5)
        if last5_results[i] in ("W", "WIN") and last5_methods[i] == "SUB"
    )
    ko_finishes_in_last5 = sum(
        1 for i in range(n5)
        if last5_results[i] in ("W", "WIN") and last5_methods[i] == "KO/TKO"
    )
    recent_5_sub_per_fight = sub_finishes_in_last5 / n5
    recent_5_ko_per_fight = ko_finishes_in_last5 / n5
    recent_5_finish_per_fight = (sub_finishes_in_last5 + ko_finishes_in_last5) / n5
    recent_5_r1_finish_per_fight = sum(
        1 for i in range(n5)
        if last5_results[i] in ("W", "WIN")
        and last5_rounds[i] == 1
        and last5_methods[i] in ("KO/TKO", "SUB")
    ) / n5

    return {
        "r1_ending_rate": r1_ending_rate,
        "r1_ending_rate_5": r1_ending_rate_5,
        "decision_rate": decision_rate,
        "decision_rate_5": decision_rate_5,
        "sub_win_rate_5": sub_win_rate_5,        # per-WIN rate — kept for model features
        "ko_win_rate_5": ko_win_rate_5,          # per-WIN rate — kept for model features
        "avg_fight_seconds": avg_fight_seconds,
        "recent_3_finish_loss_rate": recent_3_finish_loss_rate,
        "recent_5_finish_loss_rate": recent_5_finish_loss_rate,
        "recent_5_r1_loss_rate": recent_5_r1_loss_rate,
        # Fight-denominator stats — use these for betting gates / flag thresholds
        # instead of the per-WIN versions above. Captures recent ACTIVITY, not
        # just win-conditional style.
        "recent_5_sub_per_fight": recent_5_sub_per_fight,
        "recent_5_ko_per_fight": recent_5_ko_per_fight,
        "recent_5_finish_per_fight": recent_5_finish_per_fight,
        "recent_5_r1_finish_per_fight": recent_5_r1_finish_per_fight,
        "wins_in_last_5": sum(1 for r in last5_results if r in ("W", "WIN")),
    }


# ============================================================
# ELO COMPUTATION
# ============================================================

def compute_elo_from_history(fights: list, use_opp_lookup: bool = False):
    """Compute Elo by walking full fight history.

    When `use_opp_lookup=True`, each opponent's starting Elo comes from
    `get_opponent_elo()` (training data → ESPN record → 1500). This avoids
    the "everyone starts at 1500" inflation/deflation that caused wrong Elos
    for fighters with rich regional careers (Valentin 11-6 overall → UFC-only
    data saw 0-3 and crashed him to 1266).

    K-factor boost for first-3-fights is 1.2× here (was 2.0× previously) —
    2.0× amplified noise so much that early wins/losses swung Elo by 150+
    points per fight.
    """
    K = 80
    BASE = 1500
    ratings = defaultdict(lambda: BASE)
    fight_counts = defaultdict(int)
    me = "__self__"
    ratings[me] = BASE

    # Pre-seed each unique opponent's starting Elo from real data.
    if use_opp_lookup:
        for fight in fights:
            opp_name = fight.get("opponent", "")
            if not opp_name or opp_name in ratings:
                continue
            opp_id = fight.get("opponent_id", "")
            ratings[opp_name] = get_opponent_elo(opp_name, opp_id)

    opp_elos, win_opp_elos, loss_opp_elos = [], [], []

    for fight in fights:
        opp = fight.get("opponent", "unknown")
        result = fight.get("result", "")
        method = fight.get("method", "DEC")

        opp_elo = ratings[opp]
        opp_elos.append(opp_elo)
        if result == "W":
            win_opp_elos.append(opp_elo)
        elif result == "L":
            loss_opp_elos.append(opp_elo)

        k_mult = 1.4 if method == "KO/TKO" else (1.3 if method == "SUB" else 1.0)

        def get_k(f):
            c = fight_counts[f]
            # Gentler early-career boost: was 2.0/1.5/1.0 which let early
            # wins/losses swing Elo 150+ points per fight (user caught that
            # Leblanc inflated to 1735 off regional wins, Valentin crashed to
            # 1266 off 3 losses). 1.2/1.1/1.0 is still a boost but sane.
            return K * k_mult * (1.2 if c < 3 else (1.1 if c < 6 else 1.0))

        ra, rb = ratings[me], ratings[opp]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

        if result == "W":
            ratings[me] += get_k(me) * (1 - ea)
            ratings[opp] -= get_k(opp) * ea
        elif result == "L":
            ratings[me] -= get_k(me) * ea
            ratings[opp] += get_k(opp) * (1 - ea)

        fight_counts[me] += 1
        fight_counts[opp] += 1

    final_elo = ratings[me]
    comp = max(0.1, min(1.0, (final_elo - 1100) / 800))
    avg_opp = float(np.mean(opp_elos)) if opp_elos else BASE
    best_win = max(win_opp_elos) if win_opp_elos else BASE
    worst_loss = min(loss_opp_elos) if loss_opp_elos else BASE
    recent_opp = float(np.mean(opp_elos[-3:])) if opp_elos else BASE

    return final_elo, comp, opp_elos, avg_opp, recent_opp, best_win, worst_loss


# ============================================================
# UFCSTATS FALLBACK
# ============================================================

def get_ufcstats_career(name: str) -> dict:
    try:
        from scraper import UFCScraper
        s = UFCScraper(delay=0.3)
        data = s.get_fighter(name)
        if data:
            return {
                "slpm": data.slpm, "str_acc": data.str_acc,
                "sapm": data.sapm, "str_def": data.str_def,
                "td_avg": data.td_avg, "td_acc": data.td_acc,
                "td_def": data.td_def, "sub_avg": data.sub_avg,
            }
    except Exception:
        pass
    return {}


# ============================================================
# BUILD FULL PROFILE
# ============================================================

def build_live_profile(fighter_name: str) -> Optional[dict]:
    """Build complete fighter profile from ESPN + UFCStats."""
    espn_id = find_espn_id(fighter_name)
    if not espn_id:
        print(f"  Could not find ESPN ID for {fighter_name}")
        return None

    raw = fetch_espn_profile(espn_id)
    if not raw or "athlete" not in raw:
        print(f"  Could not fetch ESPN data for {fighter_name}")
        return None

    parsed = parse_espn_data(raw)
    fights = parsed["fights"]
    total_fights = len(fights)
    weight_class = parsed.get("weight_class")

    # Always use opponent-aware cascade across the FULL ESPN career (including
    # regional fights). Previously we either fell back to training-data Elo
    # (UFC-only, missed Valentin's 11-3 regional career) or used default-1500
    # opponents (inflated debutants, crashed prospects after 3 losses).
    elo, comp, opp_elos, avg_opp, recent_opp, best_win, worst_loss = (
        compute_elo_from_history(fights, use_opp_lookup=True)
    )

    wins = parsed["total_wins"]
    losses = parsed["total_losses"]
    ko_w, sub_w, dec_w = parsed["ko_wins"], parsed["sub_wins"], parsed["dec_wins"]
    ko_l, sub_l, dec_l = parsed["ko_losses"], parsed["sub_losses"], parsed["dec_losses"]
    age = parsed["age"]
    win_total = max(wins, 1)
    total = max(total_fights, 1)

    # Streak
    streak = 0
    for fight in reversed(fights):
        if fight["result"] == "W":
            streak = streak + 1 if streak >= 0 else 1
        elif fight["result"] == "L":
            if streak <= 0:
                streak -= 1
            else:
                streak = -1
            break
        else:
            break

    # Recent form
    recent = [1 if f["result"] == "W" else (0 if f["result"] == "L" else 0.5) for f in reversed(fights)]
    form_3 = float(np.mean(recent[:3])) if recent else 0.5
    form_5 = float(np.mean(recent[:5])) if recent else 0.5

    # UFCStats fallback for striking/TD stats
    ufc = get_ufcstats_career(fighter_name)
    slpm = ufc.get("slpm", 3.5)
    str_acc = ufc.get("str_acc", 0.43)
    sapm = ufc.get("sapm", 3.0)
    str_def = ufc.get("str_def", 0.52)
    td_avg = ufc.get("td_avg", 1.5)
    td_acc = ufc.get("td_acc", 0.35)
    td_def = ufc.get("td_def", 0.55)
    sub_avg = ufc.get("sub_avg", 0.4)

    avg_fight_min = 10
    avg_sig_str_landed = slpm * avg_fight_min
    sos = (avg_opp - 1500) / 200 if opp_elos else 0

    # Compute opponent-quality features from the full-career cascade above.
    sos = (avg_opp - 1500) / 200 if opp_elos else 0.0
    if losses > 0:
        loss_opp_vals = [e for i, e in enumerate(opp_elos)
                         if i < len(fights) and fights[i]["result"] == "L"]
        avg_loss_elo = float(np.mean(loss_opp_vals)) if loss_opp_vals else 1500.0
        losses_to_elite = sum(1 for e in loss_opp_vals if e > 1650)
        quality_of_losses = (avg_loss_elo - 1500) / 200
    else:
        avg_loss_elo = 1500.0
        losses_to_elite = 0
        quality_of_losses = 0.0

    # The model was trained on UFC-only Elo from enriched_fighters.csv. Feeding
    # it the full-career cascade Elo at predict time is a distribution mismatch
    # (Buchecha's ONE Championship career inflated his cascade Elo by +230 vs
    # training, biasing every prediction). Override Elo features with training
    # values when the fighter is in training data; debutants keep the cascade
    # estimate as a fallback.
    snap = _get_training_snapshot(fighter_name)
    is_ufc_debutant = snap is None
    if snap is not None:
        if not weight_class and "weight_class" in snap:
            weight_class = snap["weight_class"]
        if "elo" in snap: elo = snap["elo"]
        if "competition_level" in snap: comp = snap["competition_level"]
        if "avg_opp_elo" in snap: avg_opp = snap["avg_opp_elo"]
        if "recent_opp_elo" in snap: recent_opp = snap["recent_opp_elo"]
        if "best_win_elo" in snap: best_win = snap["best_win_elo"]
        if "worst_loss_elo" in snap: worst_loss = snap["worst_loss_elo"]
        if "strength_of_schedule" in snap: sos = snap["strength_of_schedule"]
        if "avg_loss_opp_elo" in snap: avg_loss_elo = snap["avg_loss_opp_elo"]
        if "losses_to_elite" in snap: losses_to_elite = int(snap["losses_to_elite"])
        if "quality_of_losses" in snap: quality_of_losses = snap["quality_of_losses"]

    return {
        "weight_class": weight_class,
        "is_ufc_debutant": is_ufc_debutant,
        "career_fights": total_fights,
        "career_wins": wins, "losses": losses,
        "win_pct": wins / total,
        "ko_wins": ko_w, "sub_wins": sub_w, "dec_wins": dec_w,
        "ko_losses": ko_l, "sub_losses": sub_l, "dec_losses": dec_l,
        "ko_win_rate": ko_w / win_total, "sub_win_rate": sub_w / win_total,
        "finish_rate": (ko_w + sub_w) / win_total,
        "ko_loss_rate": ko_l / total, "sub_loss_rate": sub_l / total,
        "been_finished_rate": (ko_l + sub_l) / total,
        "streak": streak, "recent_form_3": form_3, "recent_form_5": form_5,
        "avg_sig_str_landed": avg_sig_str_landed,
        "avg_sig_str_attempts": avg_sig_str_landed / max(str_acc, 0.1),
        "avg_sig_str_accuracy": str_acc,
        "avg_sig_str_received": sapm * avg_fight_min,
        "avg_sig_str_avoided": str_def,
        "avg_sig_str_defense": str_def,
        "sig_str_per_min": slpm, "sig_str_absorbed_per_min": sapm,
        "recent_sig_str_landed": avg_sig_str_landed,
        "recent_sig_str_accuracy": str_acc, "recent_sig_str_defense": str_def,
        "avg_td_landed": td_avg * avg_fight_min / 15,
        "avg_td_attempts": (td_avg / max(td_acc, 0.1)) * avg_fight_min / 15,
        "avg_td_accuracy": td_acc,
        "avg_tds_received": 1.0, "avg_td_defense": td_def,
        "td_per_min": td_avg / 15,
        "recent_td_landed": td_avg * avg_fight_min / 15,
        "recent_td_accuracy": td_acc, "recent_td_defense": td_def,
        "avg_sub_attempts": sub_avg * avg_fight_min / 15,
        "sub_per_min": sub_avg / 15,
        "avg_kd": ko_w / total * 1.5, "avg_kds_received": ko_l / total * 1.5,
        "kd_per_fight": ko_w / total * 1.5, "kd_received_per_fight": ko_l / total * 1.5,
        "elo": elo, "competition_level": comp,
        "avg_opp_elo": avg_opp, "recent_opp_elo": recent_opp,
        "best_win_elo": best_win, "worst_loss_elo": worst_loss,
        "strength_of_schedule": sos,
        "avg_loss_opp_elo": avg_loss_elo if avg_loss_elo is not None else (
            float(np.mean([e for i, e in enumerate(opp_elos) if i < len(fights) and fights[i]["result"] == "L"])) if losses > 0 else 1500
        ),
        "losses_to_elite": losses_to_elite if losses_to_elite is not None else sum(
            1 for i, e in enumerate(opp_elos) if i < len(fights) and fights[i]["result"] == "L" and e > 1650
        ),
        "quality_of_losses": quality_of_losses if quality_of_losses is not None else (
            (float(np.mean([e for i, e in enumerate(opp_elos) if i < len(fights) and fights[i]["result"] == "L"])) - 1500) / 200 if losses > 0 else 0
        ),
        "age": age,
        "age_decline_signal": max(0, age - 32) * max(0, -streak) * 0.1,
        "age_weighted_form": form_3 * (1.0 - max(0, age - 30) * 0.05),
        "career_mileage": total_fights * max(0, age - 28) * 0.01,
        # Fight-pace signature — must mirror build_training_data.py exactly
        # so the model sees the same feature distribution at predict time.
        **compute_pace_signature(fights),
    }


def build_profile_manual(stats: dict) -> dict:
    """Build profile from manually provided stats."""
    defaults = {
        "career_fights": 0, "career_wins": 0, "losses": 0, "win_pct": 0.5,
        "ko_wins": 0, "sub_wins": 0, "dec_wins": 0,
        "ko_losses": 0, "sub_losses": 0, "dec_losses": 0,
        "ko_win_rate": 0, "sub_win_rate": 0, "finish_rate": 0.5,
        "ko_loss_rate": 0, "sub_loss_rate": 0, "been_finished_rate": 0,
        "streak": 0, "recent_form_3": 0.5, "recent_form_5": 0.5,
        "avg_sig_str_landed": 35, "avg_sig_str_attempts": 80, "avg_sig_str_accuracy": 0.43,
        "avg_sig_str_received": 30, "avg_sig_str_avoided": 0.52, "avg_sig_str_defense": 0.52,
        "sig_str_per_min": 3.5, "sig_str_absorbed_per_min": 3.0,
        "recent_sig_str_landed": 35, "recent_sig_str_accuracy": 0.43, "recent_sig_str_defense": 0.52,
        "avg_td_landed": 1.0, "avg_td_attempts": 2.5, "avg_td_accuracy": 0.35,
        "avg_tds_received": 1.0, "avg_td_defense": 0.55, "td_per_min": 0.1,
        "recent_td_landed": 1.0, "recent_td_accuracy": 0.35, "recent_td_defense": 0.55,
        "avg_sub_attempts": 0.3, "sub_per_min": 0.03,
        "avg_kd": 0.2, "avg_kds_received": 0.1, "kd_per_fight": 0.2, "kd_received_per_fight": 0.1,
        "elo": 1500, "competition_level": 0.5,
        "avg_opp_elo": 1500, "recent_opp_elo": 1500, "best_win_elo": 1500, "worst_loss_elo": 1500,
        "strength_of_schedule": 0,
        "avg_loss_opp_elo": 1500, "losses_to_elite": 0, "quality_of_losses": 0,
        "age": 30, "age_decline_signal": 0, "age_weighted_form": 0.5, "career_mileage": 0,
        "weight_class": None,
        "r1_ending_rate": 0.0, "r1_ending_rate_5": 0.0, "decision_rate": 0.0,
        "avg_fight_seconds": 600.0,
        "recent_3_finish_loss_rate": 0.0, "recent_5_finish_loss_rate": 0.0,
        "recent_5_r1_loss_rate": 0.0,
    }
    defaults.update(stats)
    return defaults


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 profile_builder.py 'Fighter Name'")
        sys.exit(1)

    name = " ".join(sys.argv[1:])
    print(f"\n  Building profile for: {name}")
    profile = build_live_profile(name)
    if profile:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        print(f"  Record: {profile['career_wins']}-{profile['losses']} ({profile['career_fights']} fights)")
        print(f"  KO: {profile['ko_wins']}W/{profile['ko_losses']}L | SUB: {profile['sub_wins']}W/{profile['sub_losses']}L | DEC: {profile['dec_wins']}W/{profile['dec_losses']}L")
        print(f"  Finish rate: {profile['finish_rate']:.0%} | Been finished: {profile['been_finished_rate']:.0%}")
        print(f"  Streak: {profile['streak']} | Form (3): {profile['recent_form_3']:.0%} | Form (5): {profile['recent_form_5']:.0%}")
        print(f"  Age: {profile['age']} | Elo: {profile['elo']:.0f} | Comp Level: {profile['competition_level']:.2f} | Division: {profile.get('weight_class') or 'unknown'}")
        print(f"  Avg Opp Elo: {profile['avg_opp_elo']:.0f} | Best Win: {profile['best_win_elo']:.0f} | SoS: {profile['strength_of_schedule']:.2f}")
        print(f"  Striking: {profile['sig_str_per_min']:.1f}/min @ {profile['avg_sig_str_accuracy']:.0%} | Def: {profile['avg_sig_str_defense']:.0%}")
        print(f"{'=' * 60}")
    else:
        print(f"  Failed to build profile for {name}")
