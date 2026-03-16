"""Test vectorized race formulas against known values from research."""

import numpy as np
import pytest

from umarace._formulas import (
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


def test_preprocess_halving():
    raw = np.array([[[1400, 800, 1200, 600, 1000]]], dtype=np.float32)
    mot = np.array([[3]], dtype=np.uint8)  # Normal
    cb = np.array([[0]], dtype=np.float32)  # No career bonus

    result = preprocess_stats(raw, mot, cb)
    # 1400 → 1200 + (1400-1200)*0.5 = 1300
    assert result[0, 0, 0] == pytest.approx(1300.0, abs=1)
    # 800 → 800 (no halving)
    assert result[0, 0, 1] == pytest.approx(800.0, abs=1)
    # 1200 → 1200 (boundary, no halving)
    assert result[0, 0, 2] == pytest.approx(1200.0, abs=1)


def test_preprocess_career_bonus():
    raw = np.array([[[800, 800, 800, 800, 800]]], dtype=np.float32)
    mot = np.array([[3]], dtype=np.uint8)
    cb = np.array([[400]], dtype=np.float32)

    result = preprocess_stats(raw, mot, cb)
    assert result[0, 0, 0] == pytest.approx(1200.0, abs=1)


def test_preprocess_motivation():
    raw = np.array([[[1000, 1000, 1000, 1000, 1000]]], dtype=np.float32)
    # Great = 1.04
    mot = np.array([[5]], dtype=np.uint8)
    cb = np.array([[0]], dtype=np.float32)
    result = preprocess_stats(raw, mot, cb)
    assert result[0, 0, 0] == pytest.approx(1040.0, abs=1)


def test_base_speed():
    d = np.array([1200, 1600, 2000, 2400, 3200], dtype=np.int32)
    bs = base_speed(d)
    assert bs[0] == pytest.approx(20.8, abs=0.01)
    assert bs[2] == pytest.approx(20.0, abs=0.01)
    assert bs[4] == pytest.approx(18.8, abs=0.01)  # 20 - (3200-2000)/1000


def test_skill_activation():
    wit = np.array([100, 200, 600, 1200], dtype=np.float32)
    rates = skill_activation(wit)
    assert rates[0] == pytest.approx(0.20, abs=0.01)  # floor
    assert rates[1] == pytest.approx(0.55, abs=0.01)
    assert rates[2] == pytest.approx(0.85, abs=0.01)
    assert rates[3] == pytest.approx(0.925, abs=0.01)


def test_rushing_prob():
    wit = np.array([300, 600, 900, 1200], dtype=np.float32)
    probs = rushing_prob(wit)
    assert probs[0] == pytest.approx(0.19, abs=0.02)
    assert probs[1] == pytest.approx(0.133, abs=0.02)
    assert probs[2] == pytest.approx(0.11, abs=0.02)
    assert probs[3] == pytest.approx(0.097, abs=0.02)


def test_rushing_prob_clamped_at_low_wisdom():
    """Rushing probability must stay in [0, 1] even at extreme low wisdom."""
    wit = np.array([1.0, 10.0, 30.0], dtype=np.float32)
    probs = rushing_prob(wit)
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


def test_spurt_speed_monotonic_in_speed():
    """Higher speed stat → higher spurt speed."""
    bs = np.array([20.0], dtype=np.float32)
    gut = np.array([[700]], dtype=np.float32)
    strat = np.array([[2]], dtype=np.uint8)  # Senkou
    dp = np.array([[1.0]], dtype=np.float32)

    spd_lo = np.array([[800]], dtype=np.float32)
    spd_hi = np.array([[1200]], dtype=np.float32)

    v_lo = last_spurt_speed(bs, spd_lo, gut, strat, dp)
    v_hi = last_spurt_speed(bs, spd_hi, gut, strat, dp)
    assert v_hi[0, 0] > v_lo[0, 0]


def test_effective_spurt_stamina_matters():
    """Low stamina should reduce effective spurt speed at long distances."""
    bs = np.array([19.6], dtype=np.float32)  # 2400m
    spurt_v = np.array([[22.0]], dtype=np.float32)
    gut = np.array([[700]], dtype=np.float32)
    strat = np.array([[2]], dtype=np.uint8)
    distance = np.array([2400], dtype=np.int32)

    # High stamina
    sta_hi = np.array([[1200]], dtype=np.float32)
    hp_hi = max_hp(sta_hi, distance, strat)
    hp_cons = full_race_hp_consumption(bs, spurt_v, gut, strat, distance,
                                        np.array([1.0], dtype=np.float32))
    eff_hi = effective_spurt_speed(spurt_v, hp_hi, hp_cons, gut, sta_hi, bs, distance)

    # Low stamina
    sta_lo = np.array([[200]], dtype=np.float32)
    hp_lo = max_hp(sta_lo, distance, strat)
    eff_lo = effective_spurt_speed(spurt_v, hp_lo, hp_cons, gut, sta_lo, bs, distance)

    assert eff_hi[0, 0] > eff_lo[0, 0], "High stamina should give higher effective spurt"
