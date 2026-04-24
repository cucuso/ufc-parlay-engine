"""
UFC Stats Scraper — Pull fighter data at any point in time.

Scrapes ufcstats.com for:
  - Fighter career stats (SLpM, accuracy, TD rates, etc.)
  - Fight history (W/L, method, round, date)
  - Per-fight stats (strikes, takedowns, control time)
  - Event cards (all fights on a card)

This lets us build a Fighter profile at any point in time for backtesting.

Usage:
    from scraper import UFCScraper
    s = UFCScraper()

    # Get a fighter's full profile
    fighter = s.get_fighter("Jon Jones")

    # Get a fighter's profile as of a specific date
    fighter = s.get_fighter_at_date("Jon Jones", "2011-12-10")

    # Get all fights on a card
    fights = s.get_event("UFC 140")
"""

import re
import time
import subprocess
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime


CACHE_DIR = Path(__file__).parent / ".scraper_cache"


@dataclass
class FightRecord:
    """A single fight in a fighter's history."""
    result: str          # "Win", "Loss", "Draw", "NC"
    opponent: str
    method: str          # "KO/TKO", "SUB", "DEC", "S-DEC", "M-DEC", "U-DEC"
    round: int
    time: str            # "4:18"
    event: str
    date: str            # "Mar 28, 2026"
    # Per-fight stats (if available)
    sig_str_landed: int = 0
    sig_str_attempted: int = 0
    td_landed: int = 0
    td_attempted: int = 0
    sub_att: int = 0
    ctrl_time: str = ""


@dataclass
class FighterData:
    """Raw scraped data for a fighter."""
    name: str
    record: str          # "28-1-0"
    height: str          # "6' 4\""
    weight: str          # "248 lbs."
    reach: str           # "84\""
    stance: str          # "Orthodox"
    dob: str             # "Jul 19, 1987"
    # Career averages
    slpm: float = 0.0    # sig strikes landed per min
    str_acc: float = 0.0 # striking accuracy %
    sapm: float = 0.0    # sig strikes absorbed per min
    str_def: float = 0.0 # striking defense %
    td_avg: float = 0.0  # takedowns per 15 min
    td_acc: float = 0.0  # takedown accuracy %
    td_def: float = 0.0  # takedown defense %
    sub_avg: float = 0.0 # submissions per 15 min
    # Fight history
    fights: List[FightRecord] = field(default_factory=list)
    fighter_url: str = ""


def _curl(url: str) -> str:
    """Fetch a URL using curl (bypasses SSL issues)."""
    try:
        result = subprocess.run(
            ["curl", "-sk", url],
            capture_output=True, text=True, timeout=15
        )
        return result.stdout
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  Error fetching {url}: {e}")
        return ""


def _clean_html(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r'<[^>]+>', '|', html)
    text = re.sub(r'\|+', '|', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class UFCScraper:
    def __init__(self, use_cache: bool = True, delay: float = 0.5):
        self.use_cache = use_cache
        self.delay = delay  # seconds between requests (be nice to the server)
        if use_cache:
            CACHE_DIR.mkdir(exist_ok=True)
        # Fighter URL index: name -> URL
        self._fighter_index: Dict[str, str] = {}

    def _fetch(self, url: str) -> str:
        """Fetch with caching."""
        if self.use_cache:
            cache_key = re.sub(r'[^a-zA-Z0-9]', '_', url)
            cache_path = CACHE_DIR / f"{cache_key}.html"
            if cache_path.exists():
                return cache_path.read_text()

        html = _curl(url)
        time.sleep(self.delay)

        if self.use_cache and html:
            cache_key = re.sub(r'[^a-zA-Z0-9]', '_', url)
            cache_path = CACHE_DIR / f"{cache_key}.html"
            cache_path.write_text(html)

        return html

    # ────────────────────────────────────────────────────
    # FIGHTER SEARCH
    # ────────────────────────────────────────────────────

    def _build_fighter_index(self, first_letter: str):
        """Index all fighters starting with a letter."""
        url = f"http://ufcstats.com/statistics/fighters?char={first_letter}&page=all"
        html = self._fetch(url)

        # Parse the fighter table rows
        rows = re.findall(r'<tr class="b-statistics__table-row"[^>]*>(.*?)</tr>', html, re.DOTALL)
        for row in rows:
            # Get the fighter detail link and name parts
            links = re.findall(r'href="(http://ufcstats.com/fighter-details/[a-f0-9]+)"[^>]*>\s*([^<]+)', row)
            if len(links) >= 2:
                url = links[0][0]
                first = links[0][1].strip()
                last = links[1][1].strip()
                full_name = f"{first} {last}".strip()
                if full_name and url:
                    self._fighter_index[full_name.lower()] = url

    def find_fighter_url(self, name: str) -> Optional[str]:
        """Find a fighter's UFCStats URL by name."""
        name_lower = name.lower().strip()

        # Check if already indexed
        if name_lower in self._fighter_index:
            return self._fighter_index[name_lower]

        # Try to index the right letter page
        last_name = name.split()[-1] if " " in name else name
        letter = last_name[0].lower()
        self._build_fighter_index(letter)

        # Direct match
        if name_lower in self._fighter_index:
            return self._fighter_index[name_lower]

        # Partial match (last name)
        for indexed_name, url in self._fighter_index.items():
            if last_name.lower() in indexed_name:
                return url

        return None

    # ────────────────────────────────────────────────────
    # FIGHTER PROFILE PARSER
    # ────────────────────────────────────────────────────

    def get_fighter(self, name: str) -> Optional[FighterData]:
        """Get a fighter's complete profile and fight history."""
        url = self.find_fighter_url(name)
        if not url:
            print(f"  Fighter not found: {name}")
            return None

        html = self._fetch(url)
        if not html:
            return None

        text = _clean_html(html)

        # Parse career stats
        def extract_stat(label: str) -> str:
            # Stats appear as "Label: | value |" in cleaned text
            pattern = rf'{re.escape(label)}:?\s*\|?\s*([\d.]+%?)'
            m = re.search(pattern, text)
            return m.group(1) if m else ""

        def extract_text_stat(label: str) -> str:
            idx = text.find(label)
            if idx < 0:
                return ""
            chunk = text[idx + len(label):idx + len(label) + 50]
            chunk = chunk.strip().lstrip(': |').split('|')[0].strip()
            return chunk

        record = extract_text_stat("Record:")
        height = extract_text_stat("Height:")
        weight = extract_text_stat("Weight:")
        reach = extract_text_stat("Reach:")
        stance = extract_text_stat("STANCE:")
        dob = extract_text_stat("DOB:")

        def safe_float(val, default=0.0):
            try:
                return float(val.replace('%', '').strip())
            except (ValueError, AttributeError):
                return default

        def safe_pct(val, default=0.0):
            try:
                return float(val.replace('%', '').strip()) / 100
            except (ValueError, AttributeError):
                return default

        slpm = safe_float(extract_stat("SLpM"))
        str_acc = safe_pct(extract_stat("Str. Acc."))
        sapm = safe_float(extract_stat("SApM"))
        str_def = safe_pct(extract_stat("Str. Def"))
        td_avg = safe_float(extract_stat("TD Avg."))
        td_acc = safe_pct(extract_stat("TD Acc."))
        td_def = safe_pct(extract_stat("TD Def."))
        sub_avg = safe_float(extract_stat("Sub. Avg."))

        # Parse fight history
        fights = self._parse_fight_history(html, name)

        return FighterData(
            name=name, record=record, height=height, weight=weight,
            reach=reach, stance=stance, dob=dob,
            slpm=slpm, str_acc=str_acc, sapm=sapm, str_def=str_def,
            td_avg=td_avg, td_acc=td_acc, td_def=td_def, sub_avg=sub_avg,
            fights=fights, fighter_url=url,
        )

    def _parse_fight_history(self, html: str, fighter_name: str) -> List[FightRecord]:
        """Parse the fight history table from a fighter page.
        Row structure: Cell0=Result | Cell1=Fighters | Cell2=KD | Cell3=Str |
                       Cell4=Td | Cell5=Sub | Cell6=Event+Date | Cell7=Method |
                       Cell8=Round | Cell9=Time"""
        fights = []

        rows = re.findall(
            r'<tr class="b-fight-details__table-row b-fight-details__table-row__hover[^"]*"[^>]*>(.*?)</tr>',
            html, re.DOTALL
        )

        for row in rows:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            if len(cells) < 10:
                continue

            # Cell 0: Result
            result_text = _clean_html(cells[0]).lower().strip()
            if "win" in result_text:
                result = "Win"
            elif "loss" in result_text:
                result = "Loss"
            elif "draw" in result_text:
                result = "Draw"
            elif "nc" in result_text:
                result = "NC"
            else:
                continue

            # Cell 1: Fighter names
            names_raw = re.sub(r'<[^>]+>', ' ', cells[1])
            names_raw = re.sub(r'\s+', ' ', names_raw).strip()
            # Split into two names — they're space separated, usually FirstA LastA FirstB LastB
            name_parts = [n.strip() for n in names_raw.split() if n.strip()]
            # Find opponent using the links
            name_links = re.findall(r'<a[^>]*href="[^"]*fighter-details[^"]*"[^>]*>([^<]+)</a>', cells[1])
            name_links = [n.strip() for n in name_links if n.strip()]

            opponent = ""
            if len(name_links) >= 2:
                for n in name_links:
                    name_lower = n.lower()
                    fighter_lower = fighter_name.lower()
                    # Check if this is NOT our fighter
                    if name_lower not in fighter_lower and fighter_lower not in name_lower:
                        # Also check last name
                        if fighter_name.split()[-1].lower() not in name_lower:
                            opponent = n
                            break
                if not opponent:
                    opponent = name_links[1]  # fallback: second name

            # Cell 6: Event + Date
            event = ""
            date = ""
            event_links = re.findall(r'<a[^>]*>([^<]+)</a>', cells[6])
            if event_links:
                event = event_links[0].strip()
            date_matches = re.findall(r'([A-Z][a-z]{2,8}\.?\s+\d{1,2},\s+\d{4})', _clean_html(cells[6]))
            if date_matches:
                date = date_matches[0].strip()

            # Cell 7: Method
            method = self._normalize_method(_clean_html(cells[7]))

            # Cell 8: Round
            try:
                fight_round = int(_clean_html(cells[8]).strip(' |'))
            except ValueError:
                fight_round = 0

            # Cell 9: Time
            time_text = _clean_html(cells[9]).strip(' |')

            # Per-fight stats from cells 2-5
            def parse_stat_pair(cell_html):
                nums = re.findall(r'\d+', _clean_html(cell_html))
                return (int(nums[0]) if nums else 0, int(nums[1]) if len(nums) > 1 else 0)

            kd_a, kd_b = parse_stat_pair(cells[2])
            str_a, str_b = parse_stat_pair(cells[3])
            td_a, td_b = parse_stat_pair(cells[4])
            sub_a, sub_b = parse_stat_pair(cells[5])

            fights.append(FightRecord(
                result=result, opponent=opponent, method=method,
                round=fight_round, time=time_text,
                event=event, date=date,
                sig_str_landed=str_a, sig_str_attempted=str_b,
                td_landed=td_a, td_attempted=td_b,
                sub_att=sub_a,
            ))

        return fights

    def _normalize_method(self, method: str) -> str:
        """Normalize fight method to standard categories."""
        method = method.upper().strip()
        if 'KO' in method or 'TKO' in method:
            return "KO/TKO"
        if 'SUB' in method:
            return "SUB"
        if 'DEC' in method:
            return "DEC"
        if 'DRAW' in method:
            return "DRAW"
        return method

    # ────────────────────────────────────────────────────
    # POINT-IN-TIME PROFILE
    # ────────────────────────────────────────────────────

    def get_fighter_at_date(self, name: str, date_str: str) -> Optional[Dict]:
        """
        Build a fighter profile as of a specific date.

        Uses only fights that happened BEFORE that date to compute:
        - Record (W-L)
        - KO/SUB/DEC wins and losses
        - Win streak
        - Age at that date

        Returns a dict ready to feed into build_fighter().
        """
        data = self.get_fighter(name)
        if not data:
            return None

        target_date = datetime.strptime(date_str, "%Y-%m-%d")

        # Filter fights to only those before the target date
        past_fights = []
        for fight in data.fights:
            if fight.date:
                try:
                    fight_date = datetime.strptime(fight.date.strip(), "%b. %d, %Y")
                    if fight_date < target_date:
                        past_fights.append(fight)
                except ValueError:
                    # Try alternate format
                    try:
                        fight_date = datetime.strptime(fight.date.strip(), "%b %d, %Y")
                        if fight_date < target_date:
                            past_fights.append(fight)
                    except ValueError:
                        pass

        # Compute record from past fights
        wins = sum(1 for f in past_fights if f.result == "Win")
        losses = sum(1 for f in past_fights if f.result == "Loss")
        ko_wins = sum(1 for f in past_fights if f.result == "Win" and f.method == "KO/TKO")
        sub_wins = sum(1 for f in past_fights if f.result == "Win" and f.method == "SUB")
        dec_wins = sum(1 for f in past_fights if f.result == "Win" and f.method == "DEC")
        ko_losses = sum(1 for f in past_fights if f.result == "Loss" and f.method == "KO/TKO")
        sub_losses = sum(1 for f in past_fights if f.result == "Loss" and f.method == "SUB")
        dec_losses = sum(1 for f in past_fights if f.result == "Loss" and f.method == "DEC")

        # Win streak (count consecutive wins from most recent)
        streak = 0
        for fight in past_fights:  # fights are in reverse chronological order
            if fight.result == "Win":
                streak += 1
            elif fight.result == "Loss":
                streak = -1 if streak == 0 else streak
                break
            else:
                break

        # Age at target date
        age = 30  # default
        if data.dob:
            try:
                dob = datetime.strptime(data.dob.strip(), "%b %d, %Y")
                age = (target_date - dob).days // 365
            except ValueError:
                pass

        # Reach in inches
        reach = 70.0  # default
        if data.reach and '"' in data.reach:
            try:
                reach = float(data.reach.replace('"', '').strip())
            except ValueError:
                pass

        # Recent form (last 3 fights)
        last_3 = past_fights[:3]
        if last_3:
            recent_wins = sum(1 for f in last_3 if f.result == "Win")
            form = recent_wins / len(last_3)
        else:
            form = 0.5

        return {
            "name": name,
            "wins": wins, "losses": losses,
            "ko_wins": ko_wins, "sub_wins": sub_wins, "dec_wins": dec_wins,
            "ko_losses": ko_losses, "sub_losses": sub_losses, "dec_losses": dec_losses,
            "slpm": data.slpm, "str_acc": data.str_acc, "str_def": data.str_def,
            "td_avg": data.td_avg, "td_acc": data.td_acc, "td_def": data.td_def,
            "sub_avg": data.sub_avg,
            "reach": reach, "age": age, "streak": streak, "form": form,
            "total_fights": len(past_fights),
            "fights": past_fights,
        }

    # ────────────────────────────────────────────────────
    # EVENT SCRAPER
    # ────────────────────────────────────────────────────

    def get_event_list(self, limit: int = 20) -> List[Tuple[str, str]]:
        """Get list of recent completed events: (name, url)."""
        html = self._fetch("http://ufcstats.com/statistics/events/completed?page=all")
        events = re.findall(
            r'href="(http://ufcstats.com/event-details/[a-f0-9]+)"[^>]*>\s*([^<]+)',
            html
        )
        seen = set()
        result = []
        for url, name in events:
            if url not in seen:
                seen.add(url)
                result.append((name.strip(), url))
        return result[:limit]

    def get_event_fights(self, event_url: str) -> List[Dict]:
        """Get all fights from an event with results."""
        html = self._fetch(event_url)
        text = _clean_html(html)

        # Extract event name and date
        event_name = ""
        name_match = re.search(r'<h2[^>]*class="b-content__title"[^>]*>\s*<a[^>]*>([^<]+)', html)
        if name_match:
            event_name = name_match.group(1).strip()

        # Find fight links
        fight_links = re.findall(r'href="(http://ufcstats.com/fight-details/[a-f0-9]+)"', html)
        fight_links = list(dict.fromkeys(fight_links))  # dedupe preserving order

        # Also extract fighter names from the table
        rows = re.findall(
            r'<tr class="b-fight-details__table-row b-fight-details__table-row__hover[^"]*"[^>]*>(.*?)</tr>',
            html, re.DOTALL
        )

        fights = []
        for row in rows:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            if len(cells) < 9:
                continue

            # Fighter names
            names = re.findall(r'<a[^>]*>\s*<p[^>]*>\s*([^<]+)', cells[1])
            names = [n.strip() for n in names if n.strip()]
            if len(names) < 2:
                names = re.findall(r'>([A-Z][a-z]+ [A-Z][a-z]+)<', cells[1])

            # Method, round, time
            method = self._normalize_method(_clean_html(cells[7]))
            round_text = _clean_html(cells[8]).strip(' |')
            try:
                fight_round = int(round_text)
            except ValueError:
                fight_round = 0

            # Winner (first listed is winner, or check W/L column)
            result_text = _clean_html(cells[0])
            winner_idx = 0 if "Win" in result_text else -1

            if len(names) >= 2:
                fights.append({
                    "fighter_a": names[0],
                    "fighter_b": names[1],
                    "winner": names[winner_idx] if winner_idx >= 0 else names[0],
                    "method": method,
                    "round": fight_round,
                    "event": event_name,
                })

        return fights


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys

    s = UFCScraper()

    if len(sys.argv) > 1:
        name = " ".join(sys.argv[1:])
        print(f"\n  Looking up: {name}")
        data = s.get_fighter(name)
        if data:
            print(f"\n  {data.name}")
            print(f"  Record: {data.record}")
            print(f"  Height: {data.height} | Weight: {data.weight} | Reach: {data.reach}")
            print(f"  Stance: {data.stance} | DOB: {data.dob}")
            print(f"\n  Career Stats:")
            print(f"    SLpM: {data.slpm} | Str Acc: {data.str_acc:.0%} | Str Def: {data.str_def:.0%}")
            print(f"    TD Avg: {data.td_avg} | TD Acc: {data.td_acc:.0%} | TD Def: {data.td_def:.0%}")
            print(f"    Sub Avg: {data.sub_avg}")
            print(f"\n  Fight History ({len(data.fights)} fights):")
            for f in data.fights[:10]:
                print(f"    {f.result:<5} {f.opponent:<25} {f.method:<8} R{f.round} {f.date}")
    else:
        print("\n  Recent UFC Events:")
        events = s.get_event_list(10)
        for name, url in events:
            print(f"    {name:<55} {url}")
        print(f"\n  Usage: python3 scraper.py Jon Jones")
        print(f"         python3 scraper.py Israel Adesanya")
