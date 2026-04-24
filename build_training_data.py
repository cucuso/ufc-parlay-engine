"""
Build Enriched Training CSV

Takes the raw fight_data.csv (8,500 fights, 17K rows) and for each
fighter in each fight, computes their FULL PROFILE at that point in time:

  - Elo rating (based on all prior fights, adjusted for opponent quality)
  - Combat profile stats (rolling averages of strikes, TDs, subs, etc.)
  - Career record (W-L, KO/SUB/DEC splits)
  - Finish rates (how often they finish, how often they GET finished)
  - Recent form (last 3 and last 5 fights)
  - Age at fight time (computed from weight class + career length heuristic)
  - Win streak
  - Competition level (derived from opponents' Elo at fight time)

Then pairs fighters into matchups with deltas.

Output: data/enriched_fights.csv — ready for ML training.

Usage:
    python3 build_training_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data"
RAW_PATH = DATA_DIR / "fight_data.csv"
OUTPUT_PATH = DATA_DIR / "enriched_fights.csv"
DOB_PATH = DATA_DIR / "fighter_dobs.csv"
FIGHTERS_RAW_PATH = DATA_DIR / "fighters_raw.csv"

DEFAULT_AGE = 30.0

# Weight class ordinal — heavier = higher value (HW finishes more than FLW)
WEIGHT_ORDINAL = {
    'Women_Strawweight': 1, 'Women_Flyweight': 2, 'Women_Bantamweight': 3,
    'Women_Featherweight': 4,
    'Men_Flyweight': 5, 'Men_Bantamweight': 6, 'Men_Featherweight': 7,
    'Men_Lightweight': 8, 'Men_Welterweight': 9, 'Men_Middleweight': 10,
    'Men_Light Heavyweight': 11, 'Men_Heavyweight': 12,
    'Men_Catchweight': 9,  # default to middle
    'Women_Catchweight': 3,
}

# Average reach by weight class (for imputing missing values)
WEIGHT_CLASS_AVG_REACH = {
    'Women_Strawweight': 64.0, 'Women_Flyweight': 65.0, 'Women_Bantamweight': 67.0,
    'Women_Featherweight': 68.0,
    'Men_Flyweight': 67.0, 'Men_Bantamweight': 68.5, 'Men_Featherweight': 70.0,
    'Men_Lightweight': 72.0, 'Men_Welterweight': 74.0, 'Men_Middleweight': 75.5,
    'Men_Light Heavyweight': 76.5, 'Men_Heavyweight': 77.5,
    'Men_Catchweight': 73.0, 'Women_Catchweight': 66.0,
}

WEIGHT_CLASS_AVG_HEIGHT = {
    'Women_Strawweight': 64.0, 'Women_Flyweight': 65.0, 'Women_Bantamweight': 66.0,
    'Women_Featherweight': 67.0,
    'Men_Flyweight': 66.0, 'Men_Bantamweight': 67.5, 'Men_Featherweight': 69.0,
    'Men_Lightweight': 70.0, 'Men_Welterweight': 71.5, 'Men_Middleweight': 73.0,
    'Men_Light Heavyweight': 74.0, 'Men_Heavyweight': 75.0,
    'Men_Catchweight': 71.0, 'Women_Catchweight': 65.0,
}


def load_fighter_physicals() -> dict:
    """Load height, reach, stance from fighters_raw.csv.

    Returns: {name: {'height_inches': float, 'reach_inches': float, 'stance': int}}
    Stance encoding: 0=Orthodox, 1=Southpaw, 2=Switch
    """
    physicals = {}
    if not FIGHTERS_RAW_PATH.exists():
        print("  WARNING: fighters_raw.csv not found. No physical attributes.")
        return physicals

    df = pd.read_csv(FIGHTERS_RAW_PATH)
    stance_map = {'Orthodox': 0, 'Southpaw': 1, 'Switch': 2, 'Open Stance': 0, 'Sideways': 0}

    for _, row in df.iterrows():
        name = row['name']
        p = {'height_inches': None, 'reach_inches': None, 'stance': 0}

        # Parse height: "5'11\"" -> 71 inches
        h = str(row.get('Height', '--')).strip()
        if h and h != '--':
            try:
                parts = h.replace('"', '').split("'")
                p['height_inches'] = int(parts[0]) * 12 + int(parts[1])
            except (ValueError, IndexError):
                pass

        # Parse reach: "72\"" -> 72.0
        r = str(row.get('Reach', '--')).strip()
        if r and r != '--':
            try:
                p['reach_inches'] = float(r.replace('"', '').strip())
            except ValueError:
                pass

        # Parse stance
        s = str(row.get('STANCE', '')).strip()
        if s in stance_map:
            p['stance'] = stance_map[s]

        physicals[name] = p

    print(f"  Loaded physicals for {len(physicals)} fighters "
          f"({sum(1 for v in physicals.values() if v['reach_inches'])} with reach)")
    return physicals


def load_fighter_dobs() -> dict:
    """Load fighter DOB mapping from fighter_dobs.csv.

    Returns a dict: fighter_name -> datetime (or None if missing).
    """
    dobs = {}
    if not DOB_PATH.exists():
        print("  WARNING: fighter_dobs.csv not found. All ages will default to 30.")
        return dobs

    df = pd.read_csv(DOB_PATH)
    for _, row in df.iterrows():
        name = row['fighter']
        dob_str = str(row['dob']).strip() if pd.notna(row['dob']) else ""
        if dob_str:
            try:
                dobs[name] = datetime.strptime(dob_str, "%b %d, %Y")
            except ValueError:
                # Try alternate format with abbreviated month + period (e.g. "Jul. 19, 1987")
                try:
                    dobs[name] = datetime.strptime(dob_str, "%b. %d, %Y")
                except ValueError:
                    pass
    print(f"  Loaded {len(dobs)} fighter DOBs from {DOB_PATH}")
    return dobs


def compute_age(dobs: dict, fighter: str, fight_date) -> float:
    """Compute fighter age in years at the time of a fight.

    Returns DEFAULT_AGE if DOB is not available.
    """
    if fighter not in dobs:
        return DEFAULT_AGE
    dob = dobs[fighter]
    if isinstance(fight_date, pd.Timestamp):
        fight_date = fight_date.to_pydatetime()
    age = (fight_date - dob).days / 365.25
    return round(age, 2)


def normalize_method(m):
    m = str(m).upper()
    if 'KO' in m or 'TKO' in m:
        return 'KO/TKO'
    if 'SUB' in m:
        return 'SUB'
    if 'DEC' in m:
        return 'DEC'
    return 'OTHER'


def time_to_seconds(t):
    try:
        parts = str(t).split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return 300


def load_raw():
    df = pd.read_csv(RAW_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['method_clean'] = df['method'].apply(normalize_method)
    df['time_seconds'] = df['time'].apply(time_to_seconds)

    # Numeric cleanup
    numeric_cols = ['kd', 'kds_received', 'sig_strike_attempts', 'sig_strike_landed',
                    'sig_strike_percent', 'sig_strikes_avoided', 'sig_strikes_received',
                    'strike_attempts', 'strike_landed', 'strike_percent',
                    'strikes_avoided', 'strikes_received',
                    'td_attempts', 'td_landed', 'td_percent',
                    'tds_defended', 'tds_received', 'sub_attempts',
                    'round_finished', 'rounds', 'pass']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Modern UFC only — pre-2010 was a different sport
    df = df[df['date'] >= '2010-01-01'].copy()

    return df.sort_values('date').reset_index(drop=True)


# ============================================================
# ELO SYSTEM — Opponent-quality-adjusted
# ============================================================

class EloTracker:
    """Track Elo ratings for all fighters across time.

    Key insight: a win over a 1800-rated fighter is worth more than
    a win over a 1200-rated fighter. This naturally captures
    'competition level' — fighters who beat good opponents get
    higher Elo, fighters who beat cans stay low.
    """

    def __init__(self, k_factor=80, base=1500):
        # K=80 (was 40) — MMA needs higher K than chess because:
        # 1. Fighters have few fights (10-30 career) vs chess (hundreds)
        # 2. Skills change rapidly (camps, age, injuries)
        # 3. We need ratings to spread out so the model sees real gaps
        self.ratings = defaultdict(lambda: base)
        self.k = k_factor
        self.base = base
        self.fight_count = defaultdict(int)

    def expected(self, ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, winner, loser, method='DEC', is_ufc=True):
        """Update Elo after a fight. Finishes get a bigger K-factor boost.
        UFC fights weighted 1.5x, matching global_elo.py live predictions."""
        ra = self.ratings[winner]
        rb = self.ratings[loser]

        ea = self.expected(ra, rb)
        eb = self.expected(rb, ra)

        # Finish bonus: KO/SUB wins are more decisive
        if method == 'KO/TKO':
            k_mult = 1.4
        elif method == 'SUB':
            k_mult = 1.3
        else:
            k_mult = 1.0

        # UFC fights count 1.5x, regional 0.7x — matches global_elo.py
        level_mult = 1.5 if is_ufc else 0.7

        # New fighter: K * 2 for first 3 fights, K * 1.5 for fights 4-6
        if self.fight_count[winner] < 3:
            k_w_adj = 2.0
        elif self.fight_count[winner] < 6:
            k_w_adj = 1.5
        else:
            k_w_adj = 1.0

        if self.fight_count[loser] < 3:
            k_l_adj = 2.0
        elif self.fight_count[loser] < 6:
            k_l_adj = 1.5
        else:
            k_l_adj = 1.0

        k_winner = self.k * k_mult * k_w_adj * level_mult
        k_loser = self.k * k_mult * k_l_adj * level_mult

        self.ratings[winner] = ra + k_winner * (1 - ea)
        self.ratings[loser] = rb + k_loser * (0 - eb)

        self.fight_count[winner] += 1
        self.fight_count[loser] += 1

    def get(self, fighter):
        return self.ratings[fighter]

    def get_competition_level(self, fighter):
        """Competition level derived from Elo.
        Maps Elo range to 0-1 scale where:
          1200 or below = 0.2 (regional level)
          1500 = 0.5 (average UFC)
          1700 = 0.75 (ranked)
          1900+ = 1.0 (elite)
        """
        elo = self.ratings[fighter]
        level = (elo - 1100) / 800  # 1100->0.0, 1900->1.0
        return max(0.1, min(1.0, level))


# ============================================================
# ROLLING FIGHTER STATS
# ============================================================

class FighterStats:
    """Track rolling stats for a single fighter."""

    def __init__(self):
        self.fights = []
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.ko_wins = 0
        self.sub_wins = 0
        self.dec_wins = 0
        self.ko_losses = 0
        self.sub_losses = 0
        self.dec_losses = 0
        self.streak = 0
        self.results = []  # list of 1/0/0.5

        # Opponent tracking — strength of schedule
        self.opponent_elos = []        # Elo of each opponent at fight time
        self.win_opponent_elos = []    # Elo of opponents they BEAT
        self.loss_opponent_elos = []   # Elo of opponents they LOST TO

        # Per-fight stat history
        self.sig_str_landed = []
        self.sig_str_attempts = []
        self.sig_str_received = []
        self.sig_str_avoided = []
        self.td_landed = []
        self.td_attempts = []
        self.tds_received = []
        self.tds_defended = []
        self.sub_attempts = []
        self.kd = []
        self.kds_received = []
        self.ctrl_passes = []  # pass = guard passes

        # Per-fight round info (for computing per-minute rates)
        self.total_fight_seconds = []

        # Fight dates (for layoff calculation)
        self.fight_dates = []

        # Physical attributes (set externally from fighters_raw.csv)
        self.height_inches = None
        self.reach_inches = None
        self.stance = 0  # 0=Orthodox, 1=Southpaw, 2=Switch

    def snapshot(self, age: float = DEFAULT_AGE) -> dict:
        """Return current stats as a feature dict (BEFORE current fight)."""
        n = len(self.fights)
        if n == 0:
            return self._empty_snapshot(age=age)

        total = max(n, 1)
        win_total = max(self.wins, 1)

        # Rolling averages
        def avg(lst):
            return np.mean(lst) if lst else 0
        def recent_avg(lst, k=3):
            return np.mean(lst[-k:]) if lst else 0
        def per_min(landed_lst, time_lst):
            """Compute per-minute rate from totals and fight durations."""
            if not landed_lst or not time_lst:
                return 0
            total_landed = sum(landed_lst)
            total_time = sum(time_lst) / 60  # convert to minutes
            return total_landed / max(total_time, 1)

        return {
            # Record
            'career_fights': n,
            'career_wins': self.wins,
            'losses': self.losses,
            'win_pct': self.wins / total,
            'ko_wins': self.ko_wins,
            'sub_wins': self.sub_wins,
            'dec_wins': self.dec_wins,
            'ko_losses': self.ko_losses,
            'sub_losses': self.sub_losses,
            'dec_losses': self.dec_losses,
            # Rates
            'ko_win_rate': self.ko_wins / win_total,
            'sub_win_rate': self.sub_wins / win_total,
            'finish_rate': (self.ko_wins + self.sub_wins) / win_total,
            'ko_loss_rate': self.ko_losses / total,
            'sub_loss_rate': self.sub_losses / total,
            'been_finished_rate': (self.ko_losses + self.sub_losses) / total,
            # Finish method preference — HOW does this fighter finish?
            'ko_finish_share': self.ko_wins / max(self.ko_wins + self.sub_wins, 1),
            'sub_finish_share': self.sub_wins / max(self.ko_wins + self.sub_wins, 1),
            # HOW does this fighter lose when finished?
            'ko_loss_share': self.ko_losses / max(self.ko_losses + self.sub_losses, 1),
            'sub_loss_share': self.sub_losses / max(self.ko_losses + self.sub_losses, 1),
            # Streak & form
            'streak': self.streak,
            'recent_form_3': avg(self.results[-3:]) if self.results else 0.5,
            'recent_form_5': avg(self.results[-5:]) if self.results else 0.5,
            # Striking (career averages)
            'avg_sig_str_landed': avg(self.sig_str_landed),
            'avg_sig_str_attempts': avg(self.sig_str_attempts),
            'avg_sig_str_accuracy': avg(self.sig_str_landed) / max(avg(self.sig_str_attempts), 1),
            'avg_sig_str_received': avg(self.sig_str_received),
            'avg_sig_str_avoided': avg(self.sig_str_avoided),
            'avg_sig_str_defense': avg(self.sig_str_avoided) / max(avg(self.sig_str_avoided) + avg(self.sig_str_received), 1),
            # Per-minute rates
            'sig_str_per_min': per_min(self.sig_str_landed, self.total_fight_seconds),
            'sig_str_absorbed_per_min': per_min(self.sig_str_received, self.total_fight_seconds),
            # Striking (recent 3)
            'recent_sig_str_landed': recent_avg(self.sig_str_landed),
            'recent_sig_str_accuracy': recent_avg(self.sig_str_landed) / max(recent_avg(self.sig_str_attempts), 1),
            'recent_sig_str_defense': recent_avg(self.sig_str_avoided) / max(recent_avg(self.sig_str_avoided) + recent_avg(self.sig_str_received), 1),
            # Takedowns
            'avg_td_landed': avg(self.td_landed),
            'avg_td_attempts': avg(self.td_attempts),
            'avg_td_accuracy': avg(self.td_landed) / max(avg(self.td_attempts), 1),
            'avg_tds_received': avg(self.tds_received),
            'avg_td_defense': avg(self.tds_defended) / max(avg(self.tds_defended) + avg(self.tds_received), 1),
            'td_per_min': per_min(self.td_landed, self.total_fight_seconds),
            # Recent TDs
            'recent_td_landed': recent_avg(self.td_landed),
            'recent_td_accuracy': recent_avg(self.td_landed) / max(recent_avg(self.td_attempts), 1),
            'recent_td_defense': recent_avg(self.tds_defended) / max(recent_avg(self.tds_defended) + recent_avg(self.tds_received), 1),
            # Submissions
            'avg_sub_attempts': avg(self.sub_attempts),
            'sub_per_min': per_min(self.sub_attempts, self.total_fight_seconds),
            # Knockdowns
            'avg_kd': avg(self.kd),
            'avg_kds_received': avg(self.kds_received),
            'kd_per_fight': sum(self.kd) / total,
            'kd_received_per_fight': sum(self.kds_received) / total,
            # Strength of schedule — who have they fought?
            'avg_opp_elo': np.mean(self.opponent_elos) if self.opponent_elos else 1500,
            'recent_opp_elo': np.mean(self.opponent_elos[-3:]) if self.opponent_elos else 1500,
            'best_win_elo': max(self.win_opponent_elos) if self.win_opponent_elos else 1500,
            'worst_loss_elo': min(self.loss_opponent_elos) if self.loss_opponent_elos else 1500,
            'strength_of_schedule': (np.mean(self.opponent_elos) - 1500) / 200 if self.opponent_elos else 0,
            # Quality of losses — losing to elite fighters isn't really declining
            'avg_loss_opp_elo': np.mean(self.loss_opponent_elos) if self.loss_opponent_elos else 1500,
            'losses_to_elite': sum(1 for e in self.loss_opponent_elos if e > 1650) if self.loss_opponent_elos else 0,
            'quality_of_losses': (np.mean(self.loss_opponent_elos) - 1500) / 200 if self.loss_opponent_elos else 0,  # normalized: 0 = avg, +1 = elite opponents
            # Age — only SMART age features that capture actual decline, not just being old
            # Raw age kept for the model to use if needed
            'age': age,
            # These three capture "is the fighter ACTUALLY declining" not just "is he old":
            # age_decline_signal = 0 if winning or young, high if old AND losing
            'age_decline_signal': max(0, age - 32) * max(0, -self.streak) * 0.1,
            # age_weighted_form = form discounted by age (still high if old but winning)
            'age_weighted_form': (avg(self.results[-3:]) if self.results else 0.5) * (1.0 - max(0, age - 30) * 0.05),
            # career_mileage = total wear and tear (many fights + old = worn down)
            'career_mileage': n * max(0, age - 28) * 0.01,
        }

    def _empty_snapshot(self, age: float = DEFAULT_AGE):
        """Features for a fighter with no UFC history."""
        d = {k: 0 for k in [
            'career_fights', 'career_wins', 'losses', 'win_pct',
            'ko_wins', 'sub_wins', 'dec_wins',
            'ko_losses', 'sub_losses', 'dec_losses',
            'ko_win_rate', 'sub_win_rate', 'finish_rate',
            'ko_loss_rate', 'sub_loss_rate', 'been_finished_rate',
            'ko_finish_share', 'sub_finish_share',
            'ko_loss_share', 'sub_loss_share',
            'streak', 'recent_form_3', 'recent_form_5',
            'avg_sig_str_landed', 'avg_sig_str_attempts', 'avg_sig_str_accuracy',
            'avg_sig_str_received', 'avg_sig_str_avoided', 'avg_sig_str_defense',
            'sig_str_per_min', 'sig_str_absorbed_per_min',
            'recent_sig_str_landed', 'recent_sig_str_accuracy', 'recent_sig_str_defense',
            'avg_td_landed', 'avg_td_attempts', 'avg_td_accuracy',
            'avg_tds_received', 'avg_td_defense', 'td_per_min',
            'recent_td_landed', 'recent_td_accuracy', 'recent_td_defense',
            'avg_sub_attempts', 'sub_per_min',
            'avg_kd', 'avg_kds_received', 'kd_per_fight', 'kd_received_per_fight',
        ]}
        d['avg_opp_elo'] = 1500
        d['recent_opp_elo'] = 1500
        d['best_win_elo'] = 1500
        d['worst_loss_elo'] = 1500
        d['strength_of_schedule'] = 0
        d['avg_loss_opp_elo'] = 1500
        d['losses_to_elite'] = 0
        d['quality_of_losses'] = 0
        d['age'] = age
        d['age_decline_signal'] = 0.0
        d['age_weighted_form'] = 0.5 * (1.0 - max(0, age - 30) * 0.05)
        d['career_mileage'] = 0.0
        return d

    def update(self, row, fight_seconds, opponent_elo=1500):
        """Update stats AFTER a fight."""
        self.fights.append(row)
        self.opponent_elos.append(opponent_elo)

        result = row['res']
        method = normalize_method(row['method'])

        if result == 'W':
            self.wins += 1
            self.streak = self.streak + 1 if self.streak >= 0 else 1
            self.results.append(1)
            self.win_opponent_elos.append(opponent_elo)
            if method == 'KO/TKO':
                self.ko_wins += 1
            elif method == 'SUB':
                self.sub_wins += 1
            else:
                self.dec_wins += 1
        elif result == 'L':
            self.losses += 1
            self.streak = self.streak - 1 if self.streak <= 0 else -1
            self.results.append(0)
            self.loss_opponent_elos.append(opponent_elo)
            if method == 'KO/TKO':
                self.ko_losses += 1
            elif method == 'SUB':
                self.sub_losses += 1
            else:
                self.dec_losses += 1
        else:
            self.draws += 1
            self.results.append(0.5)

        self.sig_str_landed.append(row['sig_strike_landed'])
        self.sig_str_attempts.append(row['sig_strike_attempts'])
        self.sig_str_received.append(row['sig_strikes_received'])
        self.sig_str_avoided.append(row['sig_strikes_avoided'])
        self.td_landed.append(row['td_landed'])
        self.td_attempts.append(row['td_attempts'])
        self.tds_received.append(row['tds_received'])
        self.tds_defended.append(row['tds_defended'])
        self.sub_attempts.append(row['sub_attempts'])
        self.kd.append(row['kd'])
        self.kds_received.append(row['kds_received'])
        self.ctrl_passes.append(row.get('pass', 0))
        self.total_fight_seconds.append(fight_seconds)


# ============================================================
# MAIN BUILD PIPELINE
# ============================================================

def build_enriched_csv():
    print("  Loading raw data...")
    df = load_raw()
    print(f"  {len(df)} rows, {df.fight_pk.nunique()} fights, {df.fighter.nunique()} fighters")

    # Load fighter DOBs for age computation
    fighter_dobs = load_fighter_dobs()

    # Sort chronologically
    df = df.sort_values(['date', 'fight_pk']).reset_index(drop=True)

    # Initialize trackers
    elo = EloTracker()
    fighter_stats = defaultdict(FighterStats)

    # Process fights in chronological order
    # Group by fight_pk to process both fighters together
    print("  Computing rolling stats + Elo for every fighter at every fight...")

    fight_features = []  # One row per fighter per fight
    processed = 0

    for fight_pk, fight_group in df.groupby('fight_pk', sort=False):
        if len(fight_group) != 2:
            continue

        fight_date = fight_group.iloc[0]['date']
        round_fin = fight_group.iloc[0]['round_finished']
        time_sec = fight_group.iloc[0]['time_seconds']
        rounds_total = fight_group.iloc[0]['rounds']
        method = normalize_method(fight_group.iloc[0]['method'])

        # Total fight time in seconds
        completed_rounds = max(0, round_fin - 1) * 300  # 5 min rounds
        fight_seconds = completed_rounds + time_sec

        for _, row in fight_group.iterrows():
            fighter = row['fighter']

            # SNAPSHOT: get features BEFORE this fight
            stats = fighter_stats[fighter]
            age = compute_age(fighter_dobs, fighter, fight_date)
            snapshot = stats.snapshot(age=age)

            # Add Elo and competition level BEFORE the fight
            snapshot['elo'] = elo.get(fighter)
            snapshot['competition_level'] = elo.get_competition_level(fighter)

            # Add fight metadata
            snapshot['fighter'] = fighter
            snapshot['fight_pk'] = fight_pk
            snapshot['date'] = fight_date
            snapshot['res'] = row['res']
            snapshot['method'] = method
            snapshot['round_finished'] = round_fin
            snapshot['rounds'] = rounds_total
            snapshot['weight_class'] = row['weight_class']

            fight_features.append(snapshot)

        # UPDATE: apply fight results to Elo and stats AFTER snapshotting
        winner_row = fight_group[fight_group['res'] == 'W']
        loser_row = fight_group[fight_group['res'] == 'L']

        # Get both fighters' Elo BEFORE the update (for opponent tracking)
        fighters_in_fight = [row['fighter'] for _, row in fight_group.iterrows()]
        pre_elos = {f: elo.get(f) for f in fighters_in_fight}

        if len(winner_row) == 1 and len(loser_row) == 1:
            winner = winner_row.iloc[0]['fighter']
            loser = loser_row.iloc[0]['fighter']
            elo.update(winner, loser, method)

        # Update fighter stats with opponent's Elo
        for _, row in fight_group.iterrows():
            fighter = row['fighter']
            # Find opponent's pre-fight Elo
            opponent = [f for f in fighters_in_fight if f != fighter]
            opp_elo = pre_elos[opponent[0]] if opponent else 1500
            fighter_stats[fighter].update(row, fight_seconds, opponent_elo=opp_elo)

        processed += 1
        if processed % 1000 == 0:
            print(f"    Processed {processed} fights...")

    print(f"  Processed {processed} total fights")

    # Build DataFrame
    df_enriched = pd.DataFrame(fight_features)
    print(f"  Enriched dataset: {len(df_enriched)} rows, {len(df_enriched.columns)} columns")

    # Now build paired matchup rows
    print("  Pairing fighters into matchups...")
    matchup_rows = []

    for fight_pk, group in df_enriched.groupby('fight_pk'):
        if len(group) != 2:
            continue

        rows = group.to_dict('records')
        r0, r1 = rows[0], rows[1]

        # Determine winner/loser
        if r0['res'] == 'W':
            winner, loser = r0, r1
        elif r1['res'] == 'W':
            winner, loser = r1, r0
        else:
            continue  # draw/NC

        # Assign A = higher Elo (the "favorite"), B = lower Elo
        # This gives the model a consistent frame:
        # "Given the favorite's stats vs the underdog's stats, does the favorite win?"
        # At prediction time: always put the higher-Elo fighter as A.
        if r0.get('elo', 1500) >= r1.get('elo', 1500):
            a, b = r0, r1
        else:
            a, b = r1, r0
        a_wins = 1 if a['res'] == 'W' else 0

        matchup = {
            'fight_pk': fight_pk,
            'date': a['date'],
            'a_wins': a_wins,
            'method': a['method'] if a['res'] == 'W' else b['method'],
            'round_finished': a['round_finished'],
            'rounds': a['rounds'],
            'weight_class': a['weight_class'],
            'fighter_a': a['fighter'],
            'fighter_b': b['fighter'],
        }

        # Feature columns (everything except metadata)
        meta_cols = {'fighter', 'fight_pk', 'date', 'res', 'method',
                     'round_finished', 'rounds', 'weight_class'}
        feat_cols = [c for c in a.keys() if c not in meta_cols]

        for col in feat_cols:
            matchup[f'a_{col}'] = a[col]
            matchup[f'b_{col}'] = b[col]
            # Delta = A minus B (the matchup interaction)
            try:
                matchup[f'delta_{col}'] = float(a[col]) - float(b[col])
            except (ValueError, TypeError):
                matchup[f'delta_{col}'] = 0

        # ── COMPOSITE SIGNALS ──
        # These combine multiple dimensions into clear signals
        # the model can learn from directly

        # "All arrows point the same way" — Elo + Form + Age all favor A
        a_elo = a.get('elo', 1500)
        b_elo = b.get('elo', 1500)
        a_form = a.get('recent_form_3', 0.5)
        b_form = b.get('recent_form_3', 0.5)
        a_age = a.get('age', 30)
        b_age = b.get('age', 30)

        # Composite advantage score: sum of standardized edges
        elo_edge = (a_elo - b_elo) / 100  # per 100 Elo points
        form_edge = a_form - b_form  # -1 to +1
        age_edge = (b_age - a_age) / 5  # per 5 years younger (positive = A is younger)
        matchup['composite_advantage'] = elo_edge + form_edge + age_edge

        # How many signals agree A should win? (0-3)
        signals_for_a = (
            (1.0 if a_elo > b_elo else 0.0) +
            (1.0 if a_form > b_form else 0.0) +
            (1.0 if a_age < b_age else 0.0)
        )
        matchup['signals_aligned_for_a'] = signals_for_a
        matchup['signals_aligned_for_b'] = 3.0 - signals_for_a

        # "Proven vs unproven" — big experience gap
        a_exp = a.get('career_fights', 0)
        b_exp = b.get('career_fights', 0)
        matchup['experience_gap'] = a_exp - b_exp
        matchup['veteran_vs_newcomer'] = 1.0 if (a_exp > 10 and b_exp < 4) else (-1.0 if (b_exp > 10 and a_exp < 4) else 0.0)

        # "Declining veteran" detector — high Elo + old + losing = trap
        matchup['a_declining_vet'] = max(0, a_elo - 1550) * max(0, a_age - 32) * max(0, -a.get('streak', 0)) * 0.001
        matchup['b_declining_vet'] = max(0, b_elo - 1550) * max(0, b_age - 32) * max(0, -b.get('streak', 0)) * 0.001

        # "Tested vs untested" — strength of schedule gap
        a_sos = a.get('strength_of_schedule', 0)
        b_sos = b.get('strength_of_schedule', 0)
        matchup['schedule_gap'] = a_sos - b_sos

        # ── STYLE & INTERACTION FEATURES ──
        # These need BOTH fighters — can't be computed per-fighter alone.
        # This is where the model learns "grappler vs striker = sub likely"

        # Style ratios — what % of their offense is grappling vs striking
        a_total_offense = (a.get('avg_sig_str_landed', 0) + a.get('avg_td_landed', 0) + a.get('avg_sub_attempts', 0))
        b_total_offense = (b.get('avg_sig_str_landed', 0) + b.get('avg_td_landed', 0) + b.get('avg_sub_attempts', 0))
        a_grapple_ratio = (a.get('avg_td_landed', 0) + a.get('avg_sub_attempts', 0)) / max(a_total_offense, 1)
        b_grapple_ratio = (b.get('avg_td_landed', 0) + b.get('avg_sub_attempts', 0)) / max(b_total_offense, 1)
        matchup['a_grapple_ratio'] = a_grapple_ratio
        matchup['b_grapple_ratio'] = b_grapple_ratio
        matchup['delta_grapple_ratio'] = a_grapple_ratio - b_grapple_ratio

        # WEAPON vs DEFENSE interactions
        # "Can A's takedowns get past B's TDD?"
        a_td_threat = a.get('td_per_min', 0) * (1 - b.get('avg_td_defense', 0.5))
        b_td_threat = b.get('td_per_min', 0) * (1 - a.get('avg_td_defense', 0.5))
        matchup['a_td_threat_vs_b'] = a_td_threat
        matchup['b_td_threat_vs_a'] = b_td_threat
        matchup['delta_td_threat'] = a_td_threat - b_td_threat

        # "Can A's striking get past B's defense?"
        a_strike_threat = a.get('sig_str_per_min', 0) * (1 - b.get('avg_sig_str_defense', 0.5))
        b_strike_threat = b.get('sig_str_per_min', 0) * (1 - a.get('avg_sig_str_defense', 0.5))
        matchup['a_strike_threat_vs_b'] = a_strike_threat
        matchup['b_strike_threat_vs_a'] = b_strike_threat
        matchup['delta_strike_threat'] = a_strike_threat - b_strike_threat

        # "Is A's sub game dangerous against someone who gets finished?"
        a_sub_danger = a.get('sub_per_min', 0) * b.get('been_finished_rate', 0)
        b_sub_danger = b.get('sub_per_min', 0) * a.get('been_finished_rate', 0)
        matchup['a_sub_danger_vs_b'] = a_sub_danger
        matchup['b_sub_danger_vs_a'] = b_sub_danger
        matchup['delta_sub_danger'] = a_sub_danger - b_sub_danger

        # "Is A's KO power dangerous against someone who gets KO'd?"
        a_ko_danger = a.get('kd_per_fight', 0) * b.get('ko_loss_rate', 0)
        b_ko_danger = b.get('kd_per_fight', 0) * a.get('ko_loss_rate', 0)
        matchup['a_ko_danger_vs_b'] = a_ko_danger
        matchup['b_ko_danger_vs_a'] = b_ko_danger
        matchup['delta_ko_danger'] = a_ko_danger - b_ko_danger

        # "Does A's SUB preference match B's SUB vulnerability?"
        # Sub specialist vs someone who gets subbed = SUB likely
        a_sub_match = a.get('sub_finish_share', 0) * b.get('sub_loss_share', 0)
        b_sub_match = b.get('sub_finish_share', 0) * a.get('sub_loss_share', 0)
        matchup['a_sub_style_match'] = a_sub_match
        matchup['b_sub_style_match'] = b_sub_match
        matchup['delta_sub_style_match'] = a_sub_match - b_sub_match

        # "Does A's KO preference match B's KO vulnerability?"
        a_ko_match = a.get('ko_finish_share', 0) * b.get('ko_loss_share', 0)
        b_ko_match = b.get('ko_finish_share', 0) * a.get('ko_loss_share', 0)
        matchup['a_ko_style_match'] = a_ko_match
        matchup['b_ko_style_match'] = b_ko_match
        matchup['delta_ko_style_match'] = a_ko_match - b_ko_match

        # Elo + style = expected method
        # "High elo grappler vs low TDD = sub"
        a_elo = a.get('elo', 1500)
        b_elo = b.get('elo', 1500)
        matchup['a_elite_grappler_threat'] = max(0, a_elo - 1500) * a_grapple_ratio * (1 - b.get('avg_td_defense', 0.5))
        matchup['b_elite_grappler_threat'] = max(0, b_elo - 1500) * b_grapple_ratio * (1 - a.get('avg_td_defense', 0.5))
        matchup['a_elite_striker_threat'] = max(0, a_elo - 1500) * (1 - a_grapple_ratio) * b.get('ko_loss_rate', 0)
        matchup['b_elite_striker_threat'] = max(0, b_elo - 1500) * (1 - b_grapple_ratio) * a.get('ko_loss_rate', 0)

        # Style clash indicator — big difference in grapple ratio = style clash
        matchup['style_clash'] = abs(a_grapple_ratio - b_grapple_ratio)

        matchup_rows.append(matchup)

    df_matchups = pd.DataFrame(matchup_rows)
    print(f"  Matchup dataset: {len(df_matchups)} rows, {len(df_matchups.columns)} columns")

    # Save both
    df_enriched.to_csv(DATA_DIR / "enriched_fighters.csv", index=False)
    df_matchups.to_csv(OUTPUT_PATH, index=False)

    print(f"\n  Saved:")
    print(f"    {DATA_DIR / 'enriched_fighters.csv'} ({len(df_enriched)} rows)")
    print(f"    {OUTPUT_PATH} ({len(df_matchups)} rows)")

    # Quick stats
    print(f"\n  Elo range: {min(elo.ratings.values()):.0f} to {max(elo.ratings.values()):.0f}")
    top_elo = sorted(elo.ratings.items(), key=lambda x: -x[1])[:10]
    print(f"  Top 10 Elo (all-time final):")
    for name, rating in top_elo:
        print(f"    {name:<30} {rating:.0f}")

    print(f"\n  Feature columns per fighter: {len(feat_cols)}")
    print(f"  Total columns in matchup CSV: {len(df_matchups.columns)}")

    return df_enriched, df_matchups


if __name__ == "__main__":
    print("=" * 70)
    print("  BUILDING ENRICHED TRAINING DATA")
    print("=" * 70)
    build_enriched_csv()
    print("\n  Done!")
