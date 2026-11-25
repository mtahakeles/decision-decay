"""
Build a pass-level dataset from StatsBomb Open Data (2018 FIFA World Cup).

For every open-play pass we keep: match minute, pass length, start position,
whether the passer was under pressure, the score state from the passing
team's point of view, and whether the pass was completed.

Usage:
    python src/data_processor.py            # all 64 World Cup matches
    python src/data_processor.py 8658 7525  # only the given match ids

Raw JSON is cached under data/raw/ so re-runs don't hit the network.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITION_ID = 43  # FIFA World Cup
SEASON_ID = 3        # 2018

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "passes.csv"

# StatsBomb leaves pass.outcome empty for completed passes. "Unknown" is
# ambiguous, and "Injury Clearance" is a deliberate ball-out, so we drop both
# instead of counting them as failures.
FAILED = {"Incomplete", "Out", "Pass Offside"}
DROP_OUTCOMES = {"Unknown", "Injury Clearance"}

# Set pieces are planned in advance and have very different completion
# rates, so they say little about in-game decision making.
SET_PIECES = {"Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in"}


def get_json(path: str):
    cache = RAW_DIR / path.replace("/", "_")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    resp = requests.get(f"{BASE_URL}/{path}", timeout=60)
    resp.raise_for_status()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text, encoding="utf-8")
    return resp.json()


def match_ids():
    matches = get_json(f"matches/{COMPETITION_ID}/{SEASON_ID}.json")
    return sorted(m["match_id"] for m in matches)


def is_goal(ev) -> bool:
    if ev["type"]["name"] == "Shot":
        return ev["shot"]["outcome"]["name"] == "Goal"
    return ev["type"]["name"] == "Own Goal For"


def passes_from_match(match_id: int) -> pd.DataFrame:
    events = get_json(f"events/{match_id}.json")
    goals = {}  # team name -> goals so far
    rows = []

    for ev in events:
        team = ev["team"]["name"]

        if ev["type"]["name"] == "Pass":
            p = ev["pass"]
            outcome = p.get("outcome", {}).get("name")
            pass_type = p.get("type", {}).get("name")

            # period 5 = penalty shoot-out
            if ev["period"] < 5 and pass_type not in SET_PIECES and outcome not in DROP_OUTCOMES:
                own = goals.get(team, 0)
                opp = sum(g for t, g in goals.items() if t != team)
                rows.append({
                    "match_id": match_id,
                    "period": ev["period"],
                    "minute": ev["minute"],
                    "team": team,
                    "player": ev["player"]["name"],
                    "distance": round(p["length"], 2),
                    "start_x": ev["location"][0],
                    "pressure": int(ev.get("under_pressure", False)),
                    "height": p["height"]["name"],
                    "goal_diff": own - opp,
                    "outcome": int(outcome not in FAILED),
                })

        # update the score after the event, so a pass is never "aware" of
        # a goal that happens later in the same second
        if is_goal(ev):
            goals[team] = goals.get(team, 0) + 1

    return pd.DataFrame(rows)


def main(argv):
    ids = [int(a) for a in argv] or match_ids()
    frames = []
    for i, mid in enumerate(ids, 1):
        df = passes_from_match(mid)
        frames.append(df)
        print(f"[{i:>2}/{len(ids)}] match {mid}: {len(df)} passes")

    out = pd.concat(frames, ignore_index=True)
    out["score_state"] = pd.cut(
        out["goal_diff"], [-99, -1, 0, 99], labels=["trailing", "level", "leading"]
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{len(out):,} passes from {out['match_id'].nunique()} matches -> {OUTPUT_PATH}")
    print(f"completion rate: {out['outcome'].mean():.3f}")
    print(f"under pressure:  {out['pressure'].mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
