"""NPC data loaded as SoA numpy arrays at import time.

Data files are pre-compiled .npy binaries built by data/build_data.py.

NPC data layout (14 columns):
  [0-4]  speed, stamina, power, guts, wisdom
  [5-8]  dist_short, dist_mile, dist_mid, dist_long  (aptitude grade IDs 1-8)
  [9-10] surf_turf, surf_dirt
  [11]   strat_apt (for chosen strategy)
  [12]   strategy ID (1-5)
  [13]   motivation ID (1-5)

Total memory: ~70KB, fits in L2 cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# --- Column indices ---
_STATS = slice(0, 5)       # speed, stamina, power, guts, wisdom
_DIST_SHORT = 5
_DIST_MILE = 6
_DIST_MID = 7
_DIST_LONG = 8
_SURF_TURF = 9
_SURF_DIRT = 10
_STRAT_APT = 11
_STRATEGY = 12
_MOTIVATION = 13

# Distance category -> column index
_DIST_COL = {
    "short": _DIST_SHORT,
    "mile": _DIST_MILE,
    "mid": _DIST_MID,
    "long": _DIST_LONG,
}

# Surface ID -> column index
_SURF_COL = {
    1: _SURF_TURF,   # turf
    2: _SURF_DIRT,    # dirt
}

# --- Load at import time ---

NPC_DATA: np.ndarray = np.load(_DATA_DIR / "npc_all.npy")

_groups_npz = np.load(_DATA_DIR / "npc_groups.npz")
GROUP_INDICES: dict[int, np.ndarray] = {int(k): v for k, v in _groups_npz.items()}

CLIMAX_DATA: np.ndarray = np.load(_DATA_DIR / "climax_npc.npy")
_climax_npz = np.load(_DATA_DIR / "climax_groups.npz")
CLIMAX_GROUP_INDICES: dict[int, np.ndarray] = {int(k): v for k, v in _climax_npz.items()}

_rival_cand_path = _DATA_DIR / "rival_candidates.npy"
_rival_cand_raw = np.load(_rival_cand_path) if _rival_cand_path.exists() else np.zeros((0, 4), dtype=np.int32)
RIVAL_CANDIDATES: dict[tuple[int, int], list[tuple[int, int]]] = {}
for _row in _rival_cand_raw:
    _key = (int(_row[0]), int(_row[1]))
    RIVAL_CANDIDATES.setdefault(_key, []).append((int(_row[2]), int(_row[3])))

# Backward compat: keep old RIVAL_INDEX as best-effort (first entry per key)
RIVAL_INDEX: dict[tuple[int, int], int] = {}
for _key, _entries in RIVAL_CANDIDATES.items():
    RIVAL_INDEX[_key] = _entries[0][1]

_race_map_raw = np.load(_DATA_DIR / "race_group_map.npy")
RACE_GROUP_MAP: dict[int, int] = {int(r[0]): int(r[1]) for r in _race_map_raw}
RACE_ENTRY_MAP: dict[int, int] = (
    {int(r[0]): int(r[2]) for r in _race_map_raw}
    if _race_map_raw.ndim > 1 and _race_map_raw.shape[1] >= 3
    else {}
)

# Per-NPC estimated skill bonus
_npc_skill_path = _DATA_DIR / "npc_skill_bonus.npy"
NPC_SKILL_BONUS: np.ndarray = (
    np.load(_npc_skill_path) if _npc_skill_path.exists()
    else np.zeros(len(NPC_DATA), dtype=np.float32)
)

# Trainee skill lookup: skill_id -> (score_delta, recovery_hp_frac, needs_wisdom)
_skill_table_path = _DATA_DIR / "skill_bonus_table.npy"
if _skill_table_path.exists():
    _skill_table_raw = np.load(_skill_table_path)
    _ncols = _skill_table_raw.shape[1] if _skill_table_raw.ndim > 1 else 2
    if _ncols >= 4:
        SKILL_BONUS_TABLE: dict[int, tuple[float, float, bool]] = {
            int(row[0]): (float(row[1]), float(row[3]), bool(row[2]))
            for row in _skill_table_raw
        }
    else:
        # Backward compat: old 3-column format, recovery defaults to 0.0
        SKILL_BONUS_TABLE: dict[int, tuple[float, float, bool]] = {
            int(row[0]): (float(row[1]), 0.0, bool(row[2]) if _ncols >= 3 else True)
            for row in _skill_table_raw
        }
else:
    SKILL_BONUS_TABLE: dict[int, tuple[float, float, bool]] = {}

# Per-NPC estimated recovery HP fraction
_npc_recovery_path = _DATA_DIR / "npc_recovery_bonus.npy"
NPC_RECOVERY_BONUS: np.ndarray = (
    np.load(_npc_recovery_path) if _npc_recovery_path.exists()
    else np.zeros(len(NPC_DATA), dtype=np.float32)
)


def sum_skill_bonus(skill_ids: list[int], effective_wisdom: float) -> tuple[float, float]:
    """Sum score bonuses and recovery HP fractions for owned skills.

    Returns:
        (static_score_delta, recovery_hp_fraction) — score delta for speed/accel
        skills, and raw HP fraction for recovery skills (evaluated dynamically).
    """
    activation_prob = max(1.0 - 90.0 / max(effective_wisdom, 1.0), 0.20)
    total_delta = 0.0
    total_recovery = 0.0
    for sid in skill_ids:
        entry = SKILL_BONUS_TABLE.get(sid)
        if entry is None:
            continue
        delta, recovery, needs_wisdom = entry
        scale = activation_prob if needs_wisdom else 1.0
        total_delta += delta * scale
        total_recovery += recovery * scale
    return total_delta, total_recovery


DEFAULT_GRADE_GROUPS: dict[str, int] = {
    "G1": 13,
    "G2": 13,
    "G3": 14,
    "OP": 12,
}


def resolve_npc_group_id(group_id: int) -> int:
    """Translate race-level NPC group IDs to single_mode_npc pool IDs."""
    if group_id in GROUP_INDICES or group_id >= 5000:
        return group_id

    hundreds = group_id // 100
    ones = group_id % 10
    mapped = hundreds * 10 + ones
    if mapped in GROUP_INDICES:
        return mapped

    # 199 / 299 do not have a direct NPC pool; use the generic G1 bucket.
    return DEFAULT_GRADE_GROUPS["G1"]


def distance_category(distance: int) -> str:
    """Map race distance to category name.

    Game official categories (from course_data.json DistanceType enum):
      Short:  ≤ 1400m
      Mile:   1401-1800m
      Mid:    1801-2400m
      Long:   ≥ 2401m
    """
    if distance <= 1400:
        return "short"
    if distance <= 1800:
        return "mile"
    if distance <= 2400:
        return "mid"
    return "long"


def sample_field(
    group_id: int,
    count: int,
    rng: np.random.Generator,
    difficulty_rate: float = 1.05,
    distance: int = 2000,
    surface: int = 1,
    rival_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample NPCs and resolve race-specific aptitudes.

    Args:
        group_id: NPC group ID
        count: number of NPCs to sample
        rng: numpy RNG
        difficulty_rate: stat multiplier (1.03-1.13)
        distance: race distance in metres (selects distance aptitude column)
        surface: 1=turf, 2=dirt (selects surface aptitude column)
        rival_idx: if set, inject this NPC row (from NPC_DATA) as the rival,
                   reducing the random field sample by 1

    Returns:
        (npc_batch, skill_bonuses, recovery_bonuses) where npc_batch is (N, 10):
        [spd, sta, pow, gut, wis, dist_apt, surf_apt, strat_apt, strategy, motivation]
        skill_bonuses: (N,) static score deltas per NPC
        recovery_bonuses: (N,) recovery HP fractions per NPC
    """
    source = CLIMAX_DATA if group_id >= 5000 else NPC_DATA
    indices_map = CLIMAX_GROUP_INDICES if group_id >= 5000 else GROUP_INDICES
    bonus_source = NPC_SKILL_BONUS if group_id < 5000 else None
    recovery_source = NPC_RECOVERY_BONUS if group_id < 5000 else None

    available = indices_map.get(group_id)
    if available is None:
        available = GROUP_INDICES[13]
        source = NPC_DATA
        bonus_source = NPC_SKILL_BONUS
        recovery_source = NPC_RECOVERY_BONUS

    effective_count = count - 1 if rival_idx is not None else count
    sample_count = min(effective_count, len(available))
    chosen = rng.choice(available, size=sample_count, replace=False)
    raw = source[chosen]

    # Resolve race-specific aptitudes from the 14-column layout
    dist_cat = distance_category(distance)
    dist_col = _DIST_COL[dist_cat]
    surf_col = _SURF_COL.get(surface, _SURF_TURF)

    # Build 10-column output: [stats(5), dist_apt, surf_apt, strat_apt, strategy, motivation]
    batch = np.zeros((sample_count, 10), dtype=np.float32)
    batch[:, :5] = raw[:, _STATS] * difficulty_rate
    batch[:, 5] = raw[:, dist_col]
    batch[:, 6] = raw[:, surf_col]
    batch[:, 7] = raw[:, _STRAT_APT]
    batch[:, 8] = raw[:, _STRATEGY]
    batch[:, 9] = raw[:, _MOTIVATION]

    if bonus_source is not None and len(bonus_source) > 0:
        bonuses = bonus_source[chosen].copy()
    else:
        bonuses = np.zeros(sample_count, dtype=np.float32)

    if recovery_source is not None and len(recovery_source) > 0:
        recovery = recovery_source[chosen].copy()
    else:
        recovery = np.zeros(sample_count, dtype=np.float32)

    # Inject rival if provided
    if rival_idx is not None:
        # rival_idx always indexes into NPC_DATA (Group 0 career NPCs)
        rival_raw = NPC_DATA[rival_idx:rival_idx + 1]
        rival_row = np.zeros((1, 10), dtype=np.float32)
        rival_row[0, :5] = rival_raw[0, _STATS] * difficulty_rate
        rival_row[0, 5] = rival_raw[0, dist_col]
        rival_row[0, 6] = rival_raw[0, surf_col]
        rival_row[0, 7] = rival_raw[0, _STRAT_APT]
        rival_row[0, 8] = rival_raw[0, _STRATEGY]
        rival_row[0, 9] = rival_raw[0, _MOTIVATION]
        batch = np.vstack([batch, rival_row])

        rival_bonus = (
            NPC_SKILL_BONUS[rival_idx:rival_idx + 1]
            if len(NPC_SKILL_BONUS) > rival_idx
            else np.zeros(1, dtype=np.float32)
        )
        bonuses = np.concatenate([bonuses, rival_bonus])

        rival_recovery = (
            NPC_RECOVERY_BONUS[rival_idx:rival_idx + 1]
            if len(NPC_RECOVERY_BONUS) > rival_idx
            else np.zeros(1, dtype=np.float32)
        )
        recovery = np.concatenate([recovery, rival_recovery])

    return batch, bonuses, recovery


def get_group_for_race(program_id: int) -> int:
    """Map a race program to its NPC group."""
    return resolve_npc_group_id(RACE_GROUP_MAP.get(program_id, 13))


def get_entry_num_for_race(program_id: int) -> int:
    """Map a race program to its entry count (total runners including player)."""
    return RACE_ENTRY_MAP.get(program_id, 18)


def get_rival_npc_idx(chara_id: int, turn: int) -> int | None:
    """Lookup rival row index by (player_chara, turn).

    DEPRECATED: Use _get_max_rival_idx() for race-specific rival selection.
    This returns the first entry found, ignoring race_program_id.
    """
    return RIVAL_INDEX.get((chara_id, turn))


def _get_max_rival_idx(
    player_chara: int,
    turn: int,
    race_program_id: int,
    distance: int,
    surface: int,
    difficulty_rate: float = 1.05,
) -> int | None:
    """Find the strongest rival NPC for this specific race.

    Looks up all candidates for (player_chara, turn) filtered to
    matching race_program_id. Scores each via score_batch on the
    race's distance/surface. Returns npc_row_idx of the strongest,
    or None if no rival exists.
    """
    entries = RIVAL_CANDIDATES.get((player_chara, turn), [])
    candidates = [npc_idx for prog_id, npc_idx in entries
                  if prog_id == race_program_id]

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Score all candidates on this race to find the strongest
    from .batch import score_batch

    N = len(candidates)
    raw = NPC_DATA[candidates]  # (N, 14)

    dist_col = _DIST_COL[distance_category(distance)]
    surf_col = _SURF_COL.get(surface, _SURF_TURF)

    stats = np.zeros((1, N, 5), dtype=np.float32)
    stats[0] = raw[:, _STATS] * difficulty_rate

    apt = np.zeros((1, N, 3), dtype=np.uint8)
    apt[0, :, 0] = raw[:, dist_col].astype(np.uint8)
    apt[0, :, 1] = raw[:, surf_col].astype(np.uint8)
    apt[0, :, 2] = raw[:, _STRAT_APT].astype(np.uint8)

    strat = raw[:, _STRATEGY].astype(np.uint8).reshape(1, N)
    mot = raw[:, _MOTIVATION].astype(np.uint8).reshape(1, N)
    cb = np.full((1, N), 400.0, dtype=np.float32)

    d = np.array([distance], dtype=np.int32)
    sf = np.array([surface], dtype=np.uint8)
    gc = np.array([0], dtype=np.uint8)

    sb = np.zeros((1, N), dtype=np.float32)
    if len(NPC_SKILL_BONUS) > 0:
        sb[0] = NPC_SKILL_BONUS[candidates]

    rec = np.zeros((1, N), dtype=np.float32)
    if len(NPC_RECOVERY_BONUS) > 0:
        rec[0] = NPC_RECOVERY_BONUS[candidates]

    scores = score_batch(stats, apt, strat, mot, cb, d, sf, gc,
                         score_delta=sb, recovery_hp_frac=rec)
    best = int(np.argmax(scores[0]))
    return candidates[best]
