"""Convert CSV extracts from autoresearch/ into .npy binary files for umarace.

NPC data layout (14 columns):
  [spd, sta, pow, gut, wiz,
   dist_short, dist_mile, dist_mid, dist_long,
   surf_turf, surf_dirt,
   strat_apt, strategy, motivation]

Aptitude selection happens at race time, not build time — the caller
picks the correct distance/surface column based on the actual race.

Usage: python -m data.build_data (from umarace/ root)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

AUTORESEARCH = Path(__file__).resolve().parent.parent.parent / "autoresearch"
DATA_DIR = Path(__file__).resolve().parent

NPC_COLS = 14  # 5 stats + 4 dist_apt + 2 surf_apt + strat_apt + strategy + motivation


def _read_csv(filename: str) -> list[dict]:
    path = AUTORESEARCH / filename
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _best_strategy(row: dict) -> int:
    """Pick the strategy with highest aptitude grade for this NPC."""
    strats = {
        1: int(row.get("proper_running_style_nige", 1)),
        2: int(row.get("proper_running_style_senko", 1)),
        3: int(row.get("proper_running_style_sashi", 1)),
        4: int(row.get("proper_running_style_oikomi", 1)),
    }
    return max(strats, key=strats.get)


def _npc_to_array(row: dict) -> list[float]:
    """Convert one NPC row to 14-column array."""
    speed = float(row["speed"])
    stamina = float(row["stamina"])
    power = float(row["pow"])
    guts = float(row["guts"])
    wisdom = float(row["wiz"])

    # All 4 distance aptitudes
    dist_short = int(row.get("proper_distance_short", 1))
    dist_mile = int(row.get("proper_distance_mile", 1))
    dist_mid = int(row.get("proper_distance_middle", 1))
    dist_long = int(row.get("proper_distance_long", 1))

    # Both surface aptitudes
    surf_turf = int(row.get("proper_ground_turf", 1))
    surf_dirt = int(row.get("proper_ground_dirt", 1))

    strategy = _best_strategy(row)

    # Strategy aptitude for chosen strategy
    strat_names = {1: "nige", 2: "senko", 3: "sashi", 4: "oikomi"}
    strat_apt = int(row.get(f"proper_running_style_{strat_names[strategy]}", 7))

    # Motivation: average of min/max
    mot_min = int(row.get("motivation_min", 3))
    mot_max = int(row.get("motivation_max", 3))
    motivation = round((mot_min + mot_max) / 2)

    return [speed, stamina, power, guts, wisdom,
            dist_short, dist_mile, dist_mid, dist_long,
            surf_turf, surf_dirt,
            strat_apt, strategy, motivation]


def build_npc_data():
    """Build main NPC data arrays."""
    rows = _read_csv("single_mode_npc_dump.csv")

    all_data = []
    group_map: dict[int, list[int]] = {}

    for i, row in enumerate(rows):
        arr = _npc_to_array(row)
        all_data.append(arr)
        gid = int(row["npc_group_id"])
        group_map.setdefault(gid, []).append(i)

    npc_data = np.array(all_data, dtype=np.float32)
    np.save(DATA_DIR / "npc_all.npy", npc_data)

    group_arrays = {str(k): np.array(v, dtype=np.uint16) for k, v in group_map.items()}
    np.savez(DATA_DIR / "npc_groups.npz", **group_arrays)

    print(f"NPC data: {npc_data.shape[0]} entries x {npc_data.shape[1]} cols, {len(group_map)} groups")
    for gid, indices in sorted(group_map.items()):
        print(f"  Group {gid}: {len(indices)} NPCs")


def build_climax_data():
    """Build Twinkle Star Climax NPC arrays."""
    rows = _read_csv("climax_npc_dump.csv")

    all_data = []
    group_map: dict[int, list[int]] = {}

    for i, row in enumerate(rows):
        arr = _npc_to_array(row)
        all_data.append(arr)
        gid = int(row["npc_group_id"])
        group_map.setdefault(gid, []).append(i)

    climax_data = np.array(all_data, dtype=np.float32)
    np.save(DATA_DIR / "climax_npc.npy", climax_data)

    group_arrays = {str(k): np.array(v, dtype=np.uint16) for k, v in group_map.items()}
    np.savez(DATA_DIR / "climax_groups.npz", **group_arrays)

    print(f"Climax data: {climax_data.shape[0]} entries x {climax_data.shape[1]} cols, {len(group_map)} groups")


def build_rival_index():
    """Build (player_chara, turn, prog_id) -> [npc_row_idx] mapping.

    Reads rival_assignments.csv (all condition_types).
    Type-2 entries (21 rows, single_mode_npc_id=0) are silently skipped because
    npc_id=0 has no row in single_mode_npc_dump.csv. These need a fallback
    mapping (rival_chara_id * 10 → base Group 0 entry) to be included.
    """
    npc_rows = _read_csv("single_mode_npc_dump.csv")
    npc_id_to_idx = {int(r["id"]): i for i, r in enumerate(npc_rows)}

    rival_rows = _read_csv("rival_assignments.csv")

    entries = []
    skipped = 0
    for row in rival_rows:
        npc_id = int(row["single_mode_npc_id"])
        if npc_id in npc_id_to_idx:
            entries.append([
                int(row["player_chara"]),
                int(row["turn"]),
                int(row["race_program_id"]),
                npc_id_to_idx[npc_id],
            ])
        else:
            skipped += 1

    rival_data = np.array(entries, dtype=np.int32) if entries else np.zeros((0, 4), dtype=np.int32)
    np.save(DATA_DIR / "rival_candidates.npy", rival_data)
    print(f"Rival candidates: {len(entries)} entries ({skipped} skipped — npc_id not found)")


def build_race_group_map():
    """Build race_program_id -> (npc_group_id, entry_num) mapping."""
    rows = _read_csv("race_programs.csv")
    entries = []
    for row in rows:
        entries.append([
            int(row["program_id"]),
            int(row["npc_group_id"]),
            int(row["entry_num"]),
        ])

    race_map = np.array(entries, dtype=np.int32) if entries else np.zeros((0, 3), dtype=np.int32)
    np.save(DATA_DIR / "race_group_map.npy", race_map)
    print(f"Race-group map: {len(entries)} entries")


if __name__ == "__main__":
    build_npc_data()
    build_climax_data()
    build_rival_index()
    build_race_group_map()
    print("Done.")
