"""
data_processor.py
------------------
Decision Decay: Modeling Late-Game Cognitive Fatigue
AQX Sports Analytics Data Bowl 3.0

This script downloads open-source match event data from the StatsBomb
Open Data GitHub repository, parses the nested event JSON with pandas,
isolates passing events, and engineers the features needed to model
"cognitive fatigue" (the decline in decision quality as the match clock
runs down):

    - Minute            : match minute the pass was attempted in
    - Distance           : Euclidean distance (in StatsBomb pitch units,
                            120x80) between the pass origin and target
    - Pressure           : 1 if the passer was "under_pressure", else 0
    - Outcome            : 1 = pass completed, 0 = incomplete / out /
                            offside / intercepted / unknown

Output: clean_passes.csv
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# StatsBomb Open Data: 2018 FIFA World Cup Final (France vs Croatia, match_id 8658).
# Swap MATCH_IDS to pull additional matches/competitions for a larger sample.
BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
MATCH_IDS = [8658]  # 2018 World Cup Final — France 4-2 Croatia

RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("clean_passes.csv")

# Outcomes that StatsBomb tags on a pass.event when it is NOT completed.
# A pass with no "outcome" key attached is, by StatsBomb convention, complete.
INCOMPLETE_OUTCOMES = {
    "Incomplete",
    "Out",
    "Pass Offside",
    "Injury Clearance",
    "Unknown",
}


def fetch_match_events(match_id: int) -> list:
    """Download raw event JSON for a single match, caching it locally."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"{match_id}.json"

    if cache_path.exists():
        print(f"  [cache] using previously downloaded events for match {match_id}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"{BASE_URL}/events/{match_id}.json"
    print(f"  [fetch] downloading {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    events = resp.json()

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(events, f)

    return events


def euclidean_distance(start: list, end: list) -> float:
    """Distance between two [x, y] StatsBomb pitch coordinates."""
    if not start or not end or len(start) < 2 or len(end) < 2:
        return None
    return math.hypot(end[0] - start[0], end[1] - start[1])


def extract_pass_features(events: list, match_id: int) -> pd.DataFrame:
    """Flatten a match's raw event list down to modeling-ready pass rows."""
    # pandas.json_normalize handles the nested dictionaries StatsBomb ships
    # (event['pass']['end_location'], event['type']['name'], etc.)
    df = pd.json_normalize(events, sep=".")

    if "type.name" not in df.columns:
        return pd.DataFrame()

    passes = df[df["type.name"] == "Pass"].copy()
    if passes.empty:
        return pd.DataFrame()

    # Some optional columns may be entirely absent for a given match export
    for col in ["pass.outcome.name", "under_pressure", "location",
                "pass.end_location", "pass.length", "player.name",
                "team.name", "period"]:
        if col not in passes.columns:
            passes[col] = None

    records = []
    for _, row in passes.iterrows():
        distance = row["pass.length"]
        if pd.isna(distance):
            distance = euclidean_distance(row["location"], row["pass.end_location"])

        outcome_name = row["pass.outcome.name"]
        is_complete = 0 if outcome_name in INCOMPLETE_OUTCOMES else 1

        under_pressure = 1 if row["under_pressure"] is True else 0

        records.append({
            "match_id": match_id,
            "period": row["period"],
            "minute": row["minute"],
            "team": row["team.name"],
            "player": row["player.name"],
            "distance": distance,
            "pressure": under_pressure,
            "outcome": is_complete,
        })

    return pd.DataFrame.from_records(records)


def main():
    all_frames = []

    print("Decision Decay — Data Ingestion")
    print("=" * 40)
    for match_id in MATCH_IDS:
        print(f"Processing match_id={match_id}")
        events = fetch_match_events(match_id)
        match_df = extract_pass_features(events, match_id)
        print(f"  -> {len(match_df)} pass events extracted")
        all_frames.append(match_df)

    full_df = pd.concat(all_frames, ignore_index=True)

    # Drop rows with missing critical fields — a handful of set-piece /
    # broken-play events lack clean coordinates in the raw feed.
    before = len(full_df)
    full_df = full_df.dropna(subset=["minute", "distance", "outcome"])
    dropped = before - len(full_df)
    if dropped:
        print(f"Dropped {dropped} rows with missing minute/distance/outcome")

    full_df["minute"] = full_df["minute"].astype(int)
    full_df["distance"] = full_df["distance"].round(2)
    full_df["pressure"] = full_df["pressure"].astype(int)
    full_df["outcome"] = full_df["outcome"].astype(int)

    full_df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 40)
    print(f"Saved {len(full_df)} rows -> {OUTPUT_PATH}")
    print(f"Overall pass completion rate: {full_df['outcome'].mean():.3f}")
    print(f"Passes under pressure: {full_df['pressure'].sum()} "
          f"({full_df['pressure'].mean():.1%})")


if __name__ == "__main__":
    sys.exit(main())
