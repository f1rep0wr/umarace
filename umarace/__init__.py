"""umarace — Vectorized race outcome prediction for Umamusume Pretty Derby.

Human API:
    result = umarace.predict(stats=(1000, 800, 900, 600, 700), ...)
    score = umarace.race_power(stats=(1000, 800, 900, 600, 700), ...)

Batch API (for ML training):
    from umarace.batch import score_batch, predict_batch
"""

from __future__ import annotations

import numpy as np

from ._types import (
    Aptitude,
    GroundCondition,
    Motivation,
    PlacementResult,
    RaceConfig,
    Runner,
    Strategy,
    Surface,
    parse_aptitude,
    parse_strategy,
    parse_surface,
)
from .batch import predict_batch, score_batch


def race_power(
    stats: tuple[int, int, int, int, int],
    aptitudes: tuple[str | int, str | int, str | int] = ("A", "A", "A"),
    strategy: str | int = "pace_chaser",
    distance: int = 2000,
    surface: str | int = "turf",
    ground_condition: int = 0,
    motivation: int = 3,
    career_bonus: int = 400,
    skill_ids: list[int] | None = None,
) -> float:
    """Compute race power score for a single runner.

    Args:
        stats: (speed, stamina, power, guts, wisdom) as raw integers
        aptitudes: (distance, surface, strategy) as grade strings or IDs
        strategy: running strategy name or ID (e.g. "pace_chaser", "front_runner",
            "late_surger", "end_closer", "runaway"; Japanese aliases also accepted)
        distance: race distance in metres
        surface: "turf" or "dirt" or Surface ID
        ground_condition: GroundCondition ID (0=good)
        motivation: Motivation ID (3=normal)
        career_bonus: flat stat bonus (400 for career mode)
        skill_ids: list of owned skill IDs for score bonus

    Returns:
        Race power score as float.
    """
    from ._tables import MOTIVATION_RATE, STRAT_RATE

    s = np.array([[stats]], dtype=np.float32)  # (1, 1, 5)
    a = np.array([[[
        parse_aptitude(aptitudes[0]),
        parse_aptitude(aptitudes[1]),
        parse_aptitude(aptitudes[2]),
    ]]], dtype=np.uint8)
    st = np.array([[parse_strategy(strategy)]], dtype=np.uint8)
    mo = np.array([[motivation]], dtype=np.uint8)
    cb = np.array([[career_bonus]], dtype=np.float32)
    d = np.array([distance], dtype=np.int32)
    sf = np.array([parse_surface(surface)], dtype=np.uint8)
    gc = np.array([ground_condition], dtype=np.uint8)

    sd_arr = None
    rh_arr = None
    if skill_ids:
        from ._npc_data import sum_skill_bonus
        eff_wis = float(stats[4])
        if eff_wis > 1200:
            eff_wis = 1200.0 + (eff_wis - 1200.0) * 0.5
        eff_wis *= float(MOTIVATION_RATE[motivation])
        eff_wis += career_bonus
        eff_wis = max(1.0, min(eff_wis, 2000.0))
        eff_wis *= float(STRAT_RATE[parse_aptitude(aptitudes[2])])
        eff_wis = max(1.0, min(eff_wis, 2000.0))
        delta, recovery = sum_skill_bonus(skill_ids, eff_wis)
        sd_arr = np.array([[delta]], dtype=np.float32)
        rh_arr = np.array([[recovery]], dtype=np.float32)

    scores = score_batch(s, a, st, mo, cb, d, sf, gc,
                         score_delta=sd_arr, recovery_hp_frac=rh_arr)
    return float(scores[0, 0])


def predict(
    stats: tuple[int, int, int, int, int],
    aptitudes: tuple[str | int, str | int, str | int] = ("A", "A", "A"),
    strategy: str | int = "pace_chaser",
    distance: int = 2000,
    surface: str | int = "turf",
    ground_condition: int = 0,
    motivation: int = 3,
    career_bonus: int = 400,
    race_grade: str = "G1",
    noise_sigma: float = 0.025,
    npc_seed: int | None = None,
    skill_ids: list[int] | None = None,
    player_chara: int | None = None,
    turn: int | None = None,
    race_program_id: int | None = None,
) -> PlacementResult:
    """Predict race placement for a single runner vs NPC field.

    Args:
        stats: (speed, stamina, power, guts, wisdom)
        aptitudes: (distance, surface, strategy) grade strings or IDs
        strategy: running strategy name or ID
        distance: race distance in metres
        surface: "turf" or "dirt"
        ground_condition: GroundCondition ID
        motivation: Motivation ID
        career_bonus: flat stat bonus
        race_grade: "G1", "G2", "G3", "OP" — determines NPC field
        noise_sigma: placement noise (default 2.5%)
        npc_seed: RNG seed for NPC sampling
        skill_ids: list of owned skill IDs for trainee
        player_chara: player character ID for rival lookup
        turn: current game turn for rival lookup
        race_program_id: specific race program for rival + NPC group + entry_num

    Returns:
        PlacementResult with win probability, expected placement, etc.
    """
    from ._npc_data import (
        DEFAULT_GRADE_GROUPS, RACE_ENTRY_MAP, RACE_GROUP_MAP,
        _get_max_rival_idx, sample_field,
    )

    surf_id = parse_surface(surface)

    # Determine NPC group and field size
    if race_program_id is not None:
        group_id = RACE_GROUP_MAP.get(race_program_id, 13)
        entry_num = RACE_ENTRY_MAP.get(race_program_id, 18)
    else:
        group_id = DEFAULT_GRADE_GROUPS.get(race_grade, 13)
        entry_num = 18

    field_count = entry_num - 1  # player takes one slot

    # Find rival if player context provided
    rival_idx = None
    if player_chara is not None and turn is not None and race_program_id is not None:
        rival_idx = _get_max_rival_idx(
            player_chara, turn, race_program_id,
            distance, surf_id,
        )

    rng = np.random.default_rng(npc_seed)
    npc_data, npc_bonuses, npc_recovery = sample_field(
        group_id, field_count, rng, distance=distance, surface=surf_id,
        rival_idx=rival_idx,
    )

    # Build batch arrays: trainee at index 0, NPCs at 1-17
    N = 1 + len(npc_data)  # 18
    s = np.zeros((1, N, 5), dtype=np.float32)
    s[0, 0] = stats
    s[0, 1:] = npc_data[:, :5]

    a = np.zeros((1, N, 3), dtype=np.uint8)
    a[0, 0] = [parse_aptitude(aptitudes[0]), parse_aptitude(aptitudes[1]),
                parse_aptitude(aptitudes[2])]
    a[0, 1:, 0] = npc_data[:, 5].astype(np.uint8)
    a[0, 1:, 1] = npc_data[:, 6].astype(np.uint8)
    a[0, 1:, 2] = npc_data[:, 7].astype(np.uint8)

    st = np.zeros((1, N), dtype=np.uint8)
    st[0, 0] = parse_strategy(strategy)
    st[0, 1:] = npc_data[:, 8].astype(np.uint8)

    mo = np.zeros((1, N), dtype=np.uint8)
    mo[0, 0] = motivation
    mo[0, 1:] = npc_data[:, 9].astype(np.uint8)

    cb = np.full((1, N), career_bonus, dtype=np.float32)

    # Skill bonuses: trainee from skill_ids, NPCs from data
    sd = np.zeros((1, N), dtype=np.float32)
    rh = np.zeros((1, N), dtype=np.float32)
    if skill_ids:
        from ._npc_data import sum_skill_bonus
        from ._tables import MOTIVATION_RATE, STRAT_RATE
        # Match batch.py preprocessing: halve above 1200, motivation, career bonus, strat apt
        eff_wis = float(stats[4])
        if eff_wis > 1200:
            eff_wis = 1200.0 + (eff_wis - 1200.0) * 0.5
        eff_wis *= float(MOTIVATION_RATE[motivation])
        eff_wis += career_bonus
        eff_wis = max(1.0, min(eff_wis, 2000.0))
        eff_wis *= float(STRAT_RATE[parse_aptitude(aptitudes[2])])
        eff_wis = max(1.0, min(eff_wis, 2000.0))
        delta, recovery = sum_skill_bonus(skill_ids, eff_wis)
        sd[0, 0] = delta
        rh[0, 0] = recovery
    sd[0, 1:] = npc_bonuses
    rh[0, 1:] = npc_recovery

    d = np.array([distance], dtype=np.int32)
    sf = np.array([parse_surface(surface)], dtype=np.uint8)
    gc = np.array([ground_condition], dtype=np.uint8)

    result = predict_batch(s, a, st, mo, cb, d, sf, gc,
                           trainee_idx=0, noise_sigma=noise_sigma,
                           score_delta=sd, recovery_hp_frac=rh)

    return PlacementResult(
        expected_placement=float(result["expected_place"][0]),
        win_probability=float(result["win_prob"][0]),
        top3_probability=float(result["top3_prob"][0]),
        top5_probability=float(result["top5_prob"][0]),
        placement_distribution=result["placement_dist"][0].tolist(),
        score=float(result["scores"][0, 0]),
    )
