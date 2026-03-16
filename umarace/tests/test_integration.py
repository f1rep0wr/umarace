"""End-to-end integration tests for umarace."""

import numpy as np
import pytest

import umarace


def test_predict_returns_valid_distribution():
    result = umarace.predict(
        stats=(800, 800, 450, 300, 400),
        distance=2000,
        race_grade="G1",
        npc_seed=42,
    )
    assert 0 <= result.win_probability <= 1
    assert 1 <= result.expected_placement <= 18
    assert len(result.placement_distribution) == 18
    assert abs(sum(result.placement_distribution) - 1.0) < 0.01


def test_dominant_trainee_wins():
    result = umarace.predict(
        stats=(1200, 1200, 1200, 1200, 1200),
        distance=2000,
        race_grade="G1",
        npc_seed=42,
    )
    assert result.win_probability > 0.95


def test_balanced_beats_speed_monster_at_2400m():
    score_monster = umarace.race_power(
        stats=(1200, 200, 800, 300, 400), distance=2400)
    score_balanced = umarace.race_power(
        stats=(800, 800, 800, 300, 400), distance=2400)
    assert score_balanced > score_monster


def test_stamina_gradient_exists():
    score_low = umarace.race_power(stats=(800, 200, 800, 300, 400), distance=2400)
    score_mid = umarace.race_power(stats=(800, 600, 800, 300, 400), distance=2400)
    score_hi = umarace.race_power(stats=(800, 800, 800, 300, 400), distance=2400)
    assert score_mid > score_low, "More stamina should improve score"
    assert score_hi > score_mid, "Even more stamina should improve further"


def test_speed_monotonic():
    score_lo = umarace.race_power(stats=(600, 800, 800, 300, 400), distance=2000)
    score_hi = umarace.race_power(stats=(1000, 800, 800, 300, 400), distance=2000)
    assert score_hi > score_lo


def test_strategy_affects_score():
    score_senkou = umarace.race_power(
        stats=(800, 800, 800, 300, 400), distance=2000, strategy="senkou")
    score_oikomi = umarace.race_power(
        stats=(800, 800, 800, 300, 400), distance=2000, strategy="oikomi")
    assert score_senkou != score_oikomi


def test_aptitude_matters():
    score_a = umarace.race_power(
        stats=(800, 800, 800, 300, 400), distance=2000, aptitudes=("A", "A", "A"))
    score_d = umarace.race_power(
        stats=(800, 800, 800, 300, 400), distance=2000, aptitudes=("D", "A", "A"))
    assert score_a > score_d


def test_guts_monotonic():
    """Higher guts should improve score (via reduced HP drain in final phases)."""
    score_lo = umarace.race_power(stats=(800, 800, 800, 200, 400), distance=2000)
    score_hi = umarace.race_power(stats=(800, 800, 800, 800, 400), distance=2000)
    assert score_hi > score_lo


def test_power_monotonic():
    score_lo = umarace.race_power(stats=(800, 800, 400, 300, 400), distance=2000)
    score_hi = umarace.race_power(stats=(800, 800, 1000, 300, 400), distance=2000)
    assert score_hi > score_lo


def test_placement_distribution_sums_to_one():
    """Placement DP should produce a valid probability distribution."""
    import numpy as np
    from umarace._placement import placement_distribution
    trainee = np.array([2200.0], dtype=np.float32)
    npcs = np.array([[2100.0, 2150.0, 2050.0]], dtype=np.float32)
    result = placement_distribution(trainee, npcs)
    dist = result["placement_dist"]
    assert abs(dist.sum() - 1.0) < 0.001
    assert dist.shape == (1, 4)
    assert (dist >= 0).all()


def test_placement_equal_scores_uniform():
    """Equal scores should give roughly uniform placement distribution."""
    import numpy as np
    from umarace._placement import placement_distribution
    B = 1
    M = 3
    trainee = np.array([2000.0], dtype=np.float32)
    npcs = np.full((B, M), 2000.0, dtype=np.float32)
    result = placement_distribution(trainee, npcs)
    dist = result["placement_dist"][0]
    expected = 1.0 / (M + 1)  # 0.25 for 4 runners
    assert all(abs(p - expected) < 0.05 for p in dist)


def test_placement_dominant_wins():
    """Score much higher than all NPCs should give ~100% win."""
    import numpy as np
    from umarace._placement import placement_distribution
    trainee = np.array([3000.0], dtype=np.float32)
    npcs = np.array([[2000.0, 2000.0, 2000.0]], dtype=np.float32)
    result = placement_distribution(trainee, npcs)
    assert result["win_prob"][0] > 0.95


# ---------------------------------------------------------------------------
# Edge-case tests (tests 21-38)
# ---------------------------------------------------------------------------


def _make_batch_arrays(B: int, N: int, strategy_id: int = 2) -> dict:
    """Build minimal valid batch arrays for score_batch / predict_batch."""
    stats = np.full((B, N, 5), 800.0, dtype=np.float32)
    apt_ids = np.full((B, N, 3), 7, dtype=np.uint8)   # A-grade = 7
    strategy = np.full((B, N), strategy_id, dtype=np.uint8)
    motivation = np.full((B, N), 3, dtype=np.uint8)   # NORMAL = 3
    career_bonus = np.full((B, N), 400.0, dtype=np.float32)
    distance = np.full((B,), 2000, dtype=np.int32)
    surface = np.ones((B,), dtype=np.uint8)            # TURF = 1
    ground_cond = np.zeros((B,), dtype=np.uint8)       # GOOD = 0
    return dict(
        stats=stats,
        apt_ids=apt_ids,
        strategy=strategy,
        motivation=motivation,
        career_bonus=career_bonus,
        distance=distance,
        surface=surface,
        ground_cond=ground_cond,
    )


# --- 1. predict_batch direct test with B > 1 ---

def test_predict_batch_b4_output_shapes():
    """predict_batch with B=4, N=18 should return (4,) summary and (4,18) dist."""
    from umarace.batch import predict_batch
    B, N = 4, 18
    arrays = _make_batch_arrays(B, N)
    result = predict_batch(**arrays)
    assert result["win_prob"].shape == (B,)
    assert result["top3_prob"].shape == (B,)
    assert result["top5_prob"].shape == (B,)
    assert result["expected_place"].shape == (B,)
    assert result["placement_dist"].shape == (B, N)
    assert result["scores"].shape == (B, N)


def test_predict_batch_b4_different_inputs_differ():
    """predict_batch with different stats per race should produce different scores."""
    from umarace.batch import predict_batch
    B, N = 4, 18
    arrays = _make_batch_arrays(B, N)
    # Give each race a distinct trainee speed so scores differ
    for b in range(B):
        arrays["stats"][b, 0, 0] = 600.0 + b * 200.0   # 600, 800, 1000, 1200
    result = predict_batch(**arrays)
    scores_trainee = result["scores"][:, 0]  # trainee is at index 0
    # All four trainee scores must be distinct
    assert len(set(scores_trainee.tolist())) == B


def test_predict_batch_b4_distributions_sum_to_one():
    """Each placement distribution row must sum to 1."""
    from umarace.batch import predict_batch
    B, N = 4, 18
    arrays = _make_batch_arrays(B, N)
    result = predict_batch(**arrays)
    for b in range(B):
        total = float(result["placement_dist"][b].sum())
        assert abs(total - 1.0) < 1e-3, f"race {b} dist sums to {total}"


# --- 2. score_batch with B > 1 ---

def test_score_batch_b4_output_shape():
    """score_batch with B=4, N=18 must return (4, 18) array."""
    from umarace.batch import score_batch
    B, N = 4, 18
    arrays = _make_batch_arrays(B, N)
    scores = score_batch(**arrays)
    assert scores.shape == (B, N)


def test_score_batch_identical_inputs_produce_identical_scores():
    """Identical input rows across the batch dimension must yield identical scores."""
    from umarace.batch import score_batch
    B, N = 4, 6
    arrays = _make_batch_arrays(B, N)
    scores = score_batch(**arrays)
    # Every batch row should be equal to the first row
    for b in range(1, B):
        np.testing.assert_array_almost_equal(
            scores[b], scores[0],
            decimal=4,
            err_msg=f"Row {b} differs from row 0 despite identical inputs",
        )


# --- 3. sum_skill_bonus ---

def test_sum_skill_bonus_unknown_ids_return_zero():
    """Unknown skill IDs should contribute nothing to the total."""
    from umarace._npc_data import sum_skill_bonus
    delta, recovery = sum_skill_bonus([999_999_998, 999_999_999], effective_wisdom=1000.0)
    assert delta == 0.0
    assert recovery == 0.0


def test_sum_skill_bonus_wisdom_check_true_scales_with_wisdom():
    """A wisdom-gated skill should give a higher expected bonus at higher wisdom."""
    from umarace._npc_data import SKILL_BONUS_TABLE, sum_skill_bonus
    # Find a skill that requires wisdom check (wisdom_check=True)
    wis_ids = [sid for sid, (delta, recovery, needs_wis) in SKILL_BONUS_TABLE.items()
               if needs_wis and delta > 0]
    if not wis_ids:
        pytest.skip("No wisdom-gated skills present in SKILL_BONUS_TABLE")
    sid = wis_ids[0]
    low_delta, _ = sum_skill_bonus([sid], effective_wisdom=200.0)
    high_delta, _ = sum_skill_bonus([sid], effective_wisdom=1500.0)
    assert high_delta > low_delta, "Wisdom-gated skill should scale with effective wisdom"


def test_sum_skill_bonus_no_wisdom_check_does_not_scale():
    """A non-wisdom skill should return the same bonus regardless of wisdom level."""
    from umarace._npc_data import SKILL_BONUS_TABLE, sum_skill_bonus
    always_ids = [sid for sid, (delta, recovery, needs_wis) in SKILL_BONUS_TABLE.items()
                  if not needs_wis and delta > 0]
    if not always_ids:
        pytest.skip("No always-fire skills present in SKILL_BONUS_TABLE")
    sid = always_ids[0]
    low_delta, _ = sum_skill_bonus([sid], effective_wisdom=200.0)
    high_delta, _ = sum_skill_bonus([sid], effective_wisdom=1500.0)
    assert abs(low_delta - high_delta) < 1e-6, "Always-fire skill must not scale with wisdom"


def test_sum_skill_bonus_activation_floor_at_low_wisdom():
    """Activation probability must be floored at 0.20 even at wisdom=1."""
    from umarace._npc_data import SKILL_BONUS_TABLE, sum_skill_bonus
    wis_ids = [sid for sid, (delta, recovery, needs_wis) in SKILL_BONUS_TABLE.items()
               if needs_wis and delta > 0]
    if not wis_ids:
        pytest.skip("No wisdom-gated skills present in SKILL_BONUS_TABLE")
    sid = wis_ids[0]
    raw_delta = SKILL_BONUS_TABLE[sid][0]
    result_delta, _ = sum_skill_bonus([sid], effective_wisdom=1.0)
    expected_floor = raw_delta * 0.20
    assert abs(result_delta - expected_floor) < 1e-5


# --- 4. sample_field ---

def test_sample_field_valid_group_returns_correct_shapes():
    """sample_field with group 13 must return (count, 10) stats, (count,) bonuses, (count,) recovery."""
    from umarace._npc_data import sample_field
    rng = np.random.default_rng(0)
    count = 17
    stats, bonuses, recovery = sample_field(13, count, rng)
    assert stats.shape == (count, 10)
    assert bonuses.shape == (count,)
    assert recovery.shape == (count,)


def test_sample_field_invalid_group_falls_back_to_group_13():
    """An unknown non-climax group_id should fall back to group 13 without raising.

    The source uses group_id=1 as an example of a missing non-climax group (valid
    non-climax groups are 0, 11-41).  The fallback path returns GROUP_INDICES[13]
    row indices and indexes into NPC_DATA, so the shapes must match a direct
    group-13 call.
    """
    from umarace._npc_data import sample_field
    rng_invalid = np.random.default_rng(1)
    rng_valid = np.random.default_rng(1)
    # group_id=1 is < 5000 and absent from GROUP_INDICES, triggering the fallback
    stats_fallback, _, _ = sample_field(1, 5, rng_invalid)
    stats_group13, _, _ = sample_field(13, 5, rng_valid)
    assert stats_fallback.shape == stats_group13.shape


def test_sample_field_returns_finite_values():
    """Sampled stats must be finite (no NaN/inf from difficulty scaling)."""
    from umarace._npc_data import sample_field
    rng = np.random.default_rng(42)
    stats, bonuses, recovery = sample_field(13, 17, rng)
    assert np.all(np.isfinite(stats))
    assert np.all(np.isfinite(bonuses))
    assert np.all(np.isfinite(recovery))


# --- 5. predict_batch with trainee_idx != 0 ---

def test_predict_batch_trainee_at_position_5():
    """predict_batch should work correctly when trainee is not at index 0.

    All NPCs are set to weak stats (400) so the trainee at index 5 with
    stats (1200) produces a decisively higher score, making win_prob ~1.
    """
    from umarace.batch import predict_batch
    B, N = 2, 18
    arrays = _make_batch_arrays(B, N)
    # Weak NPCs: stats=400 for everyone
    arrays["stats"][:] = 400.0
    # Strong trainee at position 5
    arrays["stats"][:, 5, :] = 1200.0
    result = predict_batch(**arrays, trainee_idx=5)
    assert result["win_prob"].shape == (B,)
    assert result["placement_dist"].shape == (B, N)
    assert result["win_prob"][0] > 0.9


# --- 6. Extreme values ---

def test_score_batch_all_zero_stats_does_not_crash():
    """All-zero stats should not raise and must produce finite scores."""
    from umarace.batch import score_batch
    B, N = 1, 4
    arrays = _make_batch_arrays(B, N)
    arrays["stats"][:] = 0.0
    scores = score_batch(**arrays)
    assert scores.shape == (B, N)
    assert np.all(np.isfinite(scores))


def test_score_batch_all_1200_stats_produces_finite_scores():
    """All stats at 1200 should produce finite, positive scores."""
    from umarace.batch import score_batch
    B, N = 1, 4
    arrays = _make_batch_arrays(B, N)
    arrays["stats"][:] = 1200.0
    scores = score_batch(**arrays)
    assert np.all(np.isfinite(scores))
    assert np.all(scores > 0)


def test_score_batch_strategy_5_oonige_produces_finite_scores():
    """Strategy 5 (oonige / runaway) should produce finite scores."""
    from umarace.batch import score_batch
    B, N = 1, 4
    arrays = _make_batch_arrays(B, N, strategy_id=5)
    scores = score_batch(**arrays)
    assert scores.shape == (B, N)
    assert np.all(np.isfinite(scores))


def test_score_batch_oonige_differs_from_senkou():
    """Oonige (5) and senkou (2) should produce different scores for same stats."""
    from umarace.batch import score_batch
    B, N = 1, 4
    senkou_arrays = _make_batch_arrays(B, N, strategy_id=2)
    oonige_arrays = _make_batch_arrays(B, N, strategy_id=5)
    senkou_scores = score_batch(**senkou_arrays)
    oonige_scores = score_batch(**oonige_arrays)
    assert not np.allclose(senkou_scores, oonige_scores)


# --- 7. Single NPC field (N=2) ---

def test_predict_batch_n2_placement_dist_shape():
    """With N=2 (1 trainee + 1 NPC), placement_dist should have shape (B, 2)."""
    from umarace.batch import predict_batch
    B, N = 3, 2
    arrays = _make_batch_arrays(B, N)
    result = predict_batch(**arrays)
    assert result["placement_dist"].shape == (B, N)


def test_predict_batch_n2_probabilities_sum_to_one():
    """In a 2-runner race each row of placement_dist must sum to 1."""
    from umarace.batch import predict_batch
    B, N = 3, 2
    arrays = _make_batch_arrays(B, N)
    result = predict_batch(**arrays)
    for b in range(B):
        total = float(result["placement_dist"][b].sum())
        assert abs(total - 1.0) < 1e-3


def test_predict_batch_n2_win_plus_loss_equals_one():
    """In a 2-runner race win_prob and (1 - win_prob) must match placement_dist."""
    from umarace.batch import predict_batch
    B, N = 1, 2
    arrays = _make_batch_arrays(B, N)
    arrays["stats"][0, 0, 0] = 1100.0   # trainee faster
    result = predict_batch(**arrays)
    win = float(result["win_prob"][0])
    place1 = float(result["placement_dist"][0, 0])
    assert abs(win - place1) < 1e-4


# --- 8. Dynamic recovery tests ---


def _make_single_runner(raw_stats, distance=2000):
    """Build batch arrays for a single runner (B=1, N=1)."""
    s = np.array([[raw_stats]], dtype=np.float32)
    a = np.array([[[7, 7, 7]]], dtype=np.uint8)
    st = np.array([[2]], dtype=np.uint8)
    mo = np.array([[3]], dtype=np.uint8)
    cb = np.array([[400]], dtype=np.float32)
    d = np.array([distance], dtype=np.int32)
    sf = np.array([1], dtype=np.uint8)
    gc = np.array([0], dtype=np.uint8)
    return s, a, st, mo, cb, d, sf, gc


def test_recovery_surplus_runner_zero_effect():
    """HP-surplus runner (sta=1000, 2000m): 5% recovery has exactly 0 effect.

    sustained=1.0 already (headroom=554.6 > stoch_cost=350.0),
    so extra HP cannot improve eff_spurt.
    """
    from umarace.batch import score_batch
    args = _make_single_runner((600, 1000, 600, 400, 400), distance=2000)
    score_no = score_batch(*args)
    score_with = score_batch(*args, recovery_hp_frac=np.array([[0.05]], dtype=np.float32))
    assert float(score_with[0, 0]) == float(score_no[0, 0])


def test_recovery_borderline_runner_large_effect():
    """Borderline runner (sta=500, 2400m): 5% recovery has large impact.

    sustained=0.1256 without recovery, 0.4804 with → ~220 score delta.
    """
    from umarace.batch import score_batch
    args = _make_single_runner((600, 500, 600, 400, 400), distance=2400)
    score_no = score_batch(*args)
    score_with = score_batch(*args, recovery_hp_frac=np.array([[0.05]], dtype=np.float32))
    delta = float(score_with[0, 0] - score_no[0, 0])
    assert delta > 200.0, f"Borderline runner should gain >200 score from 5% recovery, got {delta:.1f}"


def test_recovery_deeply_starved_zero_effect():
    """Deeply starved runner (sta=150, 2400m): 5% recovery still 0 effect.

    headroom=-195.4, even with 5% recovery headroom=-55.8 → sustained stays 0.0.
    Correct behavior: recovery can't fix fundamentally broken stamina.
    """
    from umarace.batch import score_batch
    args = _make_single_runner((600, 150, 600, 400, 400), distance=2400)
    score_no = score_batch(*args)
    score_with = score_batch(*args, recovery_hp_frac=np.array([[0.05]], dtype=np.float32))
    assert float(score_with[0, 0]) == float(score_no[0, 0])


def test_recovery_zero_frac_identical_to_none():
    """recovery_hp_frac=zeros should produce identical scores to None."""
    from umarace.batch import score_batch
    args = _make_single_runner((600, 500, 600, 400, 400), distance=2400)
    score_none = score_batch(*args)
    score_zero = score_batch(*args, recovery_hp_frac=np.array([[0.0]], dtype=np.float32))
    np.testing.assert_allclose(score_zero, score_none)


def test_recovery_saturation():
    """Recovery saturates when sustained reaches 1.0.

    Borderline runner (sta=500, 2400m): 15% and 20% recovery produce
    identical scores because sustained hits 1.0 cap.
    """
    from umarace.batch import score_batch
    args = _make_single_runner((600, 500, 600, 400, 400), distance=2400)
    score_15 = score_batch(*args, recovery_hp_frac=np.array([[0.15]], dtype=np.float32))
    score_20 = score_batch(*args, recovery_hp_frac=np.array([[0.20]], dtype=np.float32))
    np.testing.assert_allclose(score_15, score_20)


def test_race_power_with_skill_ids():
    """race_power() should produce a different score when skill_ids are provided."""
    from umarace._npc_data import SKILL_BONUS_TABLE
    if not SKILL_BONUS_TABLE:
        pytest.skip("No skills in SKILL_BONUS_TABLE")
    sid = next(iter(SKILL_BONUS_TABLE))
    score_no = umarace.race_power(stats=(800, 800, 800, 400, 800), distance=2000)
    score_with = umarace.race_power(stats=(800, 800, 800, 400, 800), distance=2000,
                                     skill_ids=[sid])
    assert score_with != score_no
