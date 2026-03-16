"""Human-friendly NPC lookup. Returns Runner objects for the convenience API."""

from __future__ import annotations

import numpy as np

from ._npc_data import (
    CLIMAX_DATA,
    CLIMAX_GROUP_INDICES,
    DEFAULT_GRADE_GROUPS,
    NPC_DATA,
    RIVAL_CANDIDATES,
    RIVAL_INDEX,
    _get_max_rival_idx,
    distance_category,
    sample_field,
    _DIST_COL,
    _SURF_COL,
    _STRAT_APT,
    _STRATEGY,
    _MOTIVATION,
    _STATS,
    _SURF_TURF,
)
from ._types import Runner


def _row_to_runner(row: np.ndarray) -> Runner:
    """Convert a (10,) resolved NPC array to a Runner dataclass."""
    return Runner(
        speed=int(row[0]),
        stamina=int(row[1]),
        power=int(row[2]),
        guts=int(row[3]),
        wisdom=int(row[4]),
        dist_aptitude=int(row[5]),
        surf_aptitude=int(row[6]),
        strat_aptitude=int(row[7]),
        strategy=int(row[8]),
        motivation=int(row[9]),
    )


def _resolve_raw_row(raw: np.ndarray, distance: int = 2000, surface: int = 1) -> np.ndarray:
    """Convert a 14-column raw NPC row to 10-column resolved row."""
    dist_col = _DIST_COL[distance_category(distance)]
    surf_col = _SURF_COL.get(surface, _SURF_TURF)
    out = np.zeros(10, dtype=np.float32)
    out[:5] = raw[_STATS]
    out[5] = raw[dist_col]
    out[6] = raw[surf_col]
    out[7] = raw[_STRAT_APT]
    out[8] = raw[_STRATEGY]
    out[9] = raw[_MOTIVATION]
    return out


def get_field_npcs(
    race_grade: str = "G1",
    count: int = 17,
    difficulty_rate: float = 1.05,
    seed: int | None = None,
    distance: int = 2000,
    surface: int = 1,
) -> list[Runner]:
    """Get NPC field opponents for a career race.

    Args:
        race_grade: "G1", "G2", "G3", "OP"
        count: number of NPCs (default 17 for 18-runner race)
        difficulty_rate: stat multiplier (1.03-1.13)
        seed: RNG seed for reproducibility
        distance: race distance in metres
        surface: 1=turf, 2=dirt

    Returns:
        List of Runner dataclasses with race-resolved aptitudes.
    """
    rng = np.random.default_rng(seed)
    group_id = DEFAULT_GRADE_GROUPS.get(race_grade, 13)
    data, _bonuses, _recovery = sample_field(group_id, count, rng, difficulty_rate,
                                              distance=distance, surface=surface)
    return [_row_to_runner(data[i]) for i in range(len(data))]


def get_rival(chara_id: int, turn: int, distance: int = 2000, surface: int = 1) -> Runner | None:
    """Lookup rival by (player_chara_id, turn). Returns None if not found.

    DEPRECATED: Use get_rival_for_race() for race-specific rival selection.
    This returns the first entry found, ignoring race_program_id.
    """
    idx = RIVAL_INDEX.get((chara_id, turn))
    if idx is None:
        return None
    resolved = _resolve_raw_row(NPC_DATA[idx], distance, surface)
    return _row_to_runner(resolved)


def get_rival_for_race(
    player_chara: int,
    turn: int,
    race_program_id: int,
    distance: int = 2000,
    surface: int = 1,
    difficulty_rate: float = 1.05,
) -> Runner | None:
    """Return the strongest rival for this specific race, or None.

    Uses the max-rival model: scores all candidates for this
    (player_chara, turn, race_program_id) combination on the race's
    distance/surface, returns the one with the highest score.
    """
    idx = _get_max_rival_idx(player_chara, turn, race_program_id,
                             distance, surface, difficulty_rate)
    if idx is None:
        return None
    resolved = _resolve_raw_row(NPC_DATA[idx], distance, surface)
    # Apply difficulty_rate to stats
    resolved[:5] *= difficulty_rate
    return _row_to_runner(resolved)


def get_climax_npcs(
    round_num: int,
    surface: str = "turf",
    seed: int | None = None,
    distance: int = 2000,
) -> list[Runner]:
    """Get Twinkle Star Climax NPCs for round 1/2/3."""
    base = 5800 if surface == "turf" else 5830
    group_id = base + round_num
    indices = CLIMAX_GROUP_INDICES.get(group_id)
    if indices is None:
        return []
    surf_id = 1 if surface == "turf" else 2
    data = CLIMAX_DATA[indices]
    return [_row_to_runner(_resolve_raw_row(data[i], distance, surf_id))
            for i in range(len(data))]
