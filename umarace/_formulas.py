"""Vectorized race physics formulas. All functions operate on numpy arrays.

Source: research-race-model.md (reverse-engineered from game code).
"""

from __future__ import annotations

import numpy as np

from ._tables import (
    ACCEL_DIST_PROF,
    DIST_SPEED_RATE,
    GROUND_HP_MOD,
    GROUND_POWER_MOD,
    GROUND_SPEED_MOD,
    MOTIVATION_RATE,
    STOCHASTIC_HP_COST_PER_SECOND,
    STRAT_ACCEL_FINAL,
    STRAT_HP_COEF,
    STRAT_RATE,
    STRAT_SPEED_FINAL,
    STRAT_SPEED_MIDDLE,
    STRAT_SPEED_OPENING,
    SURFACE_RATE,
)


def preprocess_stats(
    raw: np.ndarray,
    motivation: np.ndarray,
    career_bonus: np.ndarray,
) -> np.ndarray:
    """Raw stats → effective stats: halving above 1200, motivation, career bonus.

    Args:
        raw: (..., 5) float32 — speed, stamina, power, guts, wisdom
        motivation: (...,) uint8 — motivation ID 1-5
        career_bonus: (...,) float32 — flat bonus per runner (usually 400)

    Returns:
        (..., 5) float32 — effective stats clamped to [1, 2000]
    """
    eff = raw.copy()
    mask = eff > 1200
    eff = np.where(mask, 1200.0 + (eff - 1200.0) * 0.5, eff)
    mot = MOTIVATION_RATE[motivation][..., np.newaxis]
    eff *= mot
    eff += career_bonus[..., np.newaxis]
    return np.clip(eff, 1.0, 2000.0)


def base_speed(distance: np.ndarray) -> np.ndarray:
    """Base speed in m/s from race distance.

    Args:
        distance: (B,) int32

    Returns:
        (B,) float32
    """
    return 20.0 - (distance.astype(np.float32) - 2000.0) / 1000.0


def last_spurt_speed(
    bs: np.ndarray,
    speed: np.ndarray,
    guts: np.ndarray,
    strategy: np.ndarray,
    dist_prof: np.ndarray,
) -> np.ndarray:
    """Last Spurt target speed.

    Args:
        bs: (B,) base speed
        speed: (B, N) effective speed stat
        guts: (B, N) effective guts stat
        strategy: (B, N) uint8 strategy ID 1-5
        dist_prof: (B, N) distance proficiency multiplier

    Returns:
        (B, N) float32
    """
    coef = STRAT_SPEED_FINAL[strategy]
    b = bs[:, np.newaxis]
    # speed_term appears twice: once inside baseTargetSpeed (×1.05), once outside
    speed_term = np.sqrt(500.0 * speed) * dist_prof * 0.002
    return (
        (b * coef + speed_term + 0.01 * b) * 1.05
        + speed_term
        + np.power(450.0 * guts, 0.597) * 0.0001
    )


def acceleration(
    power: np.ndarray,
    strategy: np.ndarray,
    surf_prof: np.ndarray,
    dist_prof: np.ndarray,
) -> np.ndarray:
    """Final-phase acceleration.

    Args:
        power: (B, N) effective power stat
        strategy: (B, N) uint8 strategy ID
        surf_prof: (B, N) surface proficiency
        dist_prof: (B, N) distance proficiency

    Returns:
        (B, N) float32
    """
    coef = STRAT_ACCEL_FINAL[strategy]
    return (
        0.0006
        * np.sqrt(500.0 * power)
        * coef
        * surf_prof
        * dist_prof
    )


def max_hp(
    stamina: np.ndarray,
    distance: np.ndarray,
    strategy: np.ndarray,
) -> np.ndarray:
    """Maximum HP from stamina.

    Args:
        stamina: (B, N) effective stamina stat
        distance: (B,) race distance
        strategy: (B, N) uint8 strategy ID

    Returns:
        (B, N) float32
    """
    coef = STRAT_HP_COEF[strategy]
    return 0.8 * coef * stamina + distance[:, np.newaxis].astype(np.float32)


def _hp_drain_at_speed(
    current_speed: np.ndarray,
    bs: np.ndarray,
    ground_hp: np.ndarray,
) -> np.ndarray:
    """Raw HP drain per second at a given speed (before guts modifier).

    Args:
        current_speed: (...) speed in m/s
        bs: (...) base speed (broadcastable)
        ground_hp: (...) ground HP modifier (broadcastable)

    Returns:
        (...) float32
    """
    return 20.0 * np.square(current_speed - bs + 12.0) / 144.0 * ground_hp


def full_race_hp_consumption(
    bs: np.ndarray,
    spurt_speed: np.ndarray,
    guts: np.ndarray,
    strategy: np.ndarray,
    distance: np.ndarray,
    ground_hp: np.ndarray,
) -> np.ndarray:
    """Total HP consumed across all 4 race phases (base running).

    Args:
        bs: (B,) base speed
        spurt_speed: (B, N) last-spurt target speed
        guts: (B, N) effective guts stat
        strategy: (B, N) uint8 strategy ID
        distance: (B,) race distance
        ground_hp: (B,) ground HP modifier

    Returns:
        (B, N) float32 — total HP consumed by base running
    """
    b = bs[:, np.newaxis]
    g = ground_hp[:, np.newaxis]
    dist = distance[:, np.newaxis].astype(np.float32)

    # Phase speeds
    v_open = b * STRAT_SPEED_OPENING[strategy]
    v_mid = b * STRAT_SPEED_MIDDLE[strategy]
    v_final = (v_mid + spurt_speed) / 2.0  # ramp approximation
    v_spurt = spurt_speed

    # Phase distances (fractions of total)
    d_open = dist / 6.0
    d_mid = dist * 3.0 / 6.0
    d_final = dist / 6.0
    d_spurt = dist / 6.0

    # Phase durations
    t_open = d_open / np.maximum(v_open, 1.0)
    t_mid = d_mid / np.maximum(v_mid, 1.0)
    t_final = d_final / np.maximum(v_final, 1.0)
    t_spurt = d_spurt / np.maximum(v_spurt, 1.0)

    # Drain per second per phase
    drain_open = _hp_drain_at_speed(v_open, b, g)
    drain_mid = _hp_drain_at_speed(v_mid, b, g)
    drain_final_raw = _hp_drain_at_speed(v_final, b, g)
    drain_spurt_raw = _hp_drain_at_speed(v_spurt, b, g)

    # Guts modifier: amplifies drain in Final + Spurt (upstream HpPolicy.ts multiplies)
    # Higher guts -> guts_mod closer to 1.0 -> less amplification -> less drain
    guts_mod = 1.0 + 200.0 / np.sqrt(600.0 * np.maximum(guts, 1.0))
    drain_final = drain_final_raw * guts_mod
    drain_spurt = drain_spurt_raw * guts_mod

    return (
        drain_open * t_open
        + drain_mid * t_mid
        + drain_final * t_final
        + drain_spurt * t_spurt
    )


def effective_spurt_speed(
    spurt_speed: np.ndarray,
    hp_pool: np.ndarray,
    hp_consumed: np.ndarray,
    guts: np.ndarray,
    stamina_eff: np.ndarray,
    bs: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    """Modulate last-spurt speed by stamina sustainability.

    Uses HP headroom (hp_pool - hp_consumed) vs estimated stochastic HP costs
    (rushing, Position Keep, skills) that scale with race duration. When headroom
    < stochastic cost, speed degrades toward min_speed.

    Also adds Stamina Limit Break bonus for stamina_eff > 1200 on long courses.

    Args:
        spurt_speed: (B, N) raw last-spurt speed
        hp_pool: (B, N) maximum HP
        hp_consumed: (B, N) HP consumed by base running
        guts: (B, N) effective guts stat
        stamina_eff: (B, N) effective stamina stat
        bs: (B,) base speed
        distance: (B,) race distance

    Returns:
        (B, N) float32 — adjusted spurt speed
    """
    b = bs[:, np.newaxis]
    dist = distance[:, np.newaxis].astype(np.float32)

    # HP headroom
    headroom = hp_pool - hp_consumed

    # Stochastic cost scales with race duration
    race_duration = dist / np.maximum(b, 1.0)
    stoch_cost = race_duration * STOCHASTIC_HP_COST_PER_SECOND

    # Keep a gradient for sub-threshold stamina instead of flattening all
    # negative headroom into the same exhausted state.
    margin = 0.4 * stoch_cost
    sustained = np.clip(
        (headroom + margin) / np.maximum(stoch_cost + margin, 1.0),
        0.0,
        1.0,
    )

    # Min speed when HP depleted (research §2.11)
    min_speed = 0.85 * b + np.sqrt(200.0 * guts) * 0.001

    # Blend
    eff = spurt_speed * sustained + min_speed * (1.0 - sustained)

    # Stamina Limit Break (research §2.12)
    dist_factor = np.where(
        dist < 2101, np.float32(0.0),
        np.where(dist < 2201, np.float32(0.5),
        np.where(dist < 2401, np.float32(1.0),
        np.where(dist < 2601, np.float32(1.2), np.float32(1.5)))),
    )
    excess_sta = np.maximum(stamina_eff - 1200.0, 0.0)
    limit_break = np.sqrt(excess_sta) * 0.0085 * dist_factor * 0.5

    return eff + limit_break


def skill_activation(wisdom: np.ndarray) -> np.ndarray:
    """Skill activation rate from wisdom.

    Args:
        wisdom: (...) effective wisdom stat

    Returns:
        (...) float32 in [0.20, 1.0]
    """
    return np.clip(
        (100.0 - 9000.0 / np.maximum(wisdom, 1.0)) / 100.0, 0.20, 1.0
    )


def rushing_prob(wisdom: np.ndarray) -> np.ndarray:
    """Rushing (掛かり) probability from wisdom.

    Args:
        wisdom: (...) effective wisdom stat

    Returns:
        (...) float32
    """
    raw = np.square(6.5 / np.log10(0.1 * wisdom + 1.0)) / 100.0
    return np.clip(raw, 0.0, 1.0)
