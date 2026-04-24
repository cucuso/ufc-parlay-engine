"""
Build Fighter DOB Data

Reads all unique fighter names from data/fight_data.csv, scrapes their
DOB from ufcstats.com using the existing UFCScraper, and saves a mapping
file: data/fighter_dobs.csv with columns: fighter, dob

The scraper caches pages in .scraper_cache/ so re-runs skip already-fetched fighters.

Usage:
    python3 build_age_data.py          # full run (all fighters)
    python3 build_age_data.py --test   # test run (first 5 fighters)
"""

import sys
import pandas as pd
from pathlib import Path
from scraper import UFCScraper

DATA_DIR = Path(__file__).parent / "data"
RAW_PATH = DATA_DIR / "fight_data.csv"
DOB_PATH = DATA_DIR / "fighter_dobs.csv"


def build_age_data(test_mode: bool = False):
    # Load unique fighter names
    df = pd.read_csv(RAW_PATH)
    fighters = sorted(df['fighter'].unique())
    total = len(fighters)
    print(f"  Found {total} unique fighters in fight_data.csv")

    if test_mode:
        fighters = fighters[:5]
        print(f"  TEST MODE: only processing first 5 fighters")

    # Load existing DOB file if it exists (to resume partial runs)
    existing_dobs = {}
    if DOB_PATH.exists():
        df_existing = pd.read_csv(DOB_PATH)
        for _, row in df_existing.iterrows():
            existing_dobs[row['fighter']] = row['dob']
        print(f"  Loaded {len(existing_dobs)} existing DOB entries from {DOB_PATH}")

    scraper = UFCScraper(use_cache=True, delay=0.5)

    results = []
    scraped = 0
    not_found = 0

    for i, fighter in enumerate(fighters):
        # Use cached result if we already have it
        if fighter in existing_dobs:
            results.append({'fighter': fighter, 'dob': existing_dobs[fighter]})
            continue

        # Scrape DOB from ufcstats.com
        data = scraper.get_fighter(fighter)

        if data and data.dob and data.dob.strip():
            dob = data.dob.strip()
        else:
            dob = ""
            not_found += 1

        results.append({'fighter': fighter, 'dob': dob})
        scraped += 1

        # Progress reporting every 100 fighters
        if (i + 1) % 100 == 0 or (i + 1) == len(fighters):
            print(f"    Progress: {i + 1}/{len(fighters)} fighters "
                  f"({scraped} scraped, {not_found} DOB not found)")

    # Save results
    df_out = pd.DataFrame(results)
    df_out.to_csv(DOB_PATH, index=False)

    found = sum(1 for r in results if r['dob'])
    missing = sum(1 for r in results if not r['dob'])
    print(f"\n  Done! Saved {len(results)} entries to {DOB_PATH}")
    print(f"  DOB found: {found}, DOB missing: {missing}")


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    print("=" * 60)
    print("  BUILDING FIGHTER DOB DATA")
    if test_mode:
        print("  (TEST MODE)")
    print("=" * 60)
    build_age_data(test_mode=test_mode)
    print("\n  Done!")
