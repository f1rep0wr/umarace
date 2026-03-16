"""Batch API — the ML hot path.

All inputs/outputs are numpy arrays. Zero Python objects in the hot path.
Target: <1μs per runner amortized over batch.
"""

from __future__ import annotations

import numpy as np

from ._formulas import (
    acceleration,
    base_speed,
    effective_spurt_speed,
    full_race_hp_consumption,
    last_spurt_speed,
    max_hp,
    preprocess_stats,
    rushing_prob,
    skill_activation,
)
from ._placement import placement_distribution
from ._tables import (
    ACCEL_DIST_PROF,
    DIST_SPEED_RATE,
    GROUND_HP_MOD,
    GROUND_POWER_MOD,
    GROUND_SPEED_MOD,
    STRAT_RATE,
    SURFACE_RATE,
)


def score_batch(
    stats: np.ndarray,
    apt_ids: np.ndarray,
    strategy: np.ndarray,
    motivation: np.ndarray,
    career_bonus: np.ndarray,
    distance: np.ndarray,
    surface: np.ndarray,
    ground_cond: np.ndarray,
    score_delta: np.ndarray | None = None,
    recovery_hp_frac: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized effective_race_power for B races × N runners.

    Args:
        stats: (B, N, 5) float32 — speed, stamina, power, guts, wisdom
        apt_ids: (B, N, 3) uint8 — dist_apt, surf_apt, strat_apt (grade IDs 1-8)
        strategy: (B, N) uint8 — strategy ID 1-5
        motivation: (B, N) uint8 — motivation ID 1-5
        career_bonus: (B, N) float32 — flat bonus (usually 400)
        distance: (B,) int32 — race distance in metres
        surface: (B,) uint8 — surface ID 1-2
        ground_cond: (B,) uint8 — ground condition ID 0-3
        score_delta: (B, N) float32 optional — static speed/accel skill bonus
        recovery_hp_frac: (B, N) float32 optional — recovery HP as fraction of maxHP

    Returns:
        (B, N) float32 — effective race power scores
    """
    # 1. Preprocess stats
    eff = preprocess_stats(stats, motivation, career_bonus)
    spd = eff[..., 0]
    sta = eff[..., 1]
    pwr = eff[..., 2]
    gut = eff[..., 3]
    wit = eff[..., 4]

    # 2. Aptitude multipliers via array indexing
    dist_prof = DIST_SPEED_RATE[apt_ids[..., 0]]
    accel_dist_prof = ACCEL_DIST_PROF[apt_ids[..., 0]]
    surf_prof = SURFACE_RATE[apt_ids[..., 1]]
    strat_prof = STRAT_RATE[apt_ids[..., 2]]

    # Apply strategy proficiency to wisdom
    wit = wit * strat_prof

    # 3. Ground condition modifiers
    gc_idx = (surface - 1).astype(np.int32) * 4 + ground_cond.astype(np.int32)
    spd = spd + GROUND_SPEED_MOD[gc_idx][:, np.newaxis]
    pwr = pwr + GROUND_POWER_MOD[gc_idx][:, np.newaxis]
    ground_hp = GROUND_HP_MOD[gc_idx]

    # Clamp stats
    spd = np.clip(spd, 1.0, 2000.0)
    sta = np.clip(sta, 1.0, 2000.0)
    pwr = np.clip(pwr, 1.0, 2000.0)
    gut = np.clip(gut, 1.0, 2000.0)
    wit = np.clip(wit, 1.0, 2000.0)

    # 4. Core formulas
    bs = base_speed(distance)
    spurt_v = last_spurt_speed(bs, spd, gut, strategy, dist_prof)
    accel = acceleration(pwr, strategy, surf_prof, accel_dist_prof)

    # 5. Stamina → effective spurt speed (the key stamina integration)
    hp = max_hp(sta, distance, strategy)

    # Dynamic recovery: add recovered HP to pool before sustainability check
    if recovery_hp_frac is not None:
        hp = hp + recovery_hp_frac * hp

    hp_consumed = full_race_hp_consumption(
        bs, spurt_v, gut, strategy, distance, ground_hp
    )
    eff_spurt = effective_spurt_speed(
        spurt_v, hp, hp_consumed, gut, sta, bs, distance
    )

    # 6. Skill and rushing
    skill_rate = skill_activation(wit)
    rush = rushing_prob(wit)
    rush_penalty = 1.0 - rush * 0.15

    # 7. Composite score
    score = (eff_spurt * 100.0 + accel * 500.0 + skill_rate * 0.5)
    if score_delta is not None:
        score = score + score_delta
    score = score * rush_penalty

    return score


def predict_batch(
    stats: np.ndarray,
    apt_ids: np.ndarray,
    strategy: np.ndarray,
    motivation: np.ndarray,
    career_bonus: np.ndarray,
    distance: np.ndarray,
    surface: np.ndarray,
    ground_cond: np.ndarray,
    trainee_idx: int = 0,
    noise_sigma: float = 0.025,
    score_delta: np.ndarray | None = None,
    recovery_hp_frac: np.ndarray | None = None,
    fast: bool = False,
) -> dict[str, np.ndarray]:
    """Full placement prediction for batch.

    Calls score_batch(), then computes placement distribution via
    Gauss-Hermite shared-noise integration.

    Args:
        (same as score_batch, plus:)
        trainee_idx: which runner index is the trainee (default 0)
        noise_sigma: calibrated noise sigma (default 2.5%)
        score_delta: (B, N) float32 optional — static speed/accel skill bonus
        recovery_hp_frac: (B, N) float32 optional — recovery HP fraction
        fast: if True, use Q=10 quadrature (~0.1% error, 2x faster)

    Returns:
        dict with 'win_prob', 'top3_prob', 'top5_prob', 'expected_place',
        'placement_dist', 'scores'.
    """
    scores = score_batch(
        stats, apt_ids, strategy, motivation, career_bonus,
        distance, surface, ground_cond,
        score_delta=score_delta,
        recovery_hp_frac=recovery_hp_frac,
    )

    B, N = scores.shape

    # Split trainee vs NPC scores
    trainee_scores = scores[:, trainee_idx]  # (B,)
    npc_mask = np.ones(N, dtype=bool)
    npc_mask[trainee_idx] = False
    npc_scores = scores[:, npc_mask]  # (B, N-1)

    result = placement_distribution(
        trainee_scores, npc_scores, sigma=noise_sigma, fast=fast,
    )
    result["scores"] = scores
    return result
