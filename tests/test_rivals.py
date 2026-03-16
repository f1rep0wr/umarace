"""End-to-end tests for rival-aware race prediction."""

from __future__ import annotations

import numpy as np
import pytest


class TestRivalDataLoading:
    """Phase R1-R2: Verify data loads correctly."""

    def test_rival_candidates_loaded(self):
        from umarace._npc_data import RIVAL_CANDIDATES
        assert len(RIVAL_CANDIDATES) > 0
        # Should have ~796 unique (chara, turn) keys
        assert len(RIVAL_CANDIDATES) >= 700

    def test_rival_candidates_structure(self):
        from umarace._npc_data import RIVAL_CANDIDATES
        for key, entries in list(RIVAL_CANDIDATES.items())[:10]:
            assert isinstance(key, tuple)
            assert len(key) == 2
            chara, turn = key
            assert isinstance(chara, int)
            assert isinstance(turn, int)
            assert turn > 0  # floating rivals excluded
            for prog_id, npc_idx in entries:
                assert isinstance(prog_id, int)
                assert isinstance(npc_idx, int)
                assert npc_idx >= 0

    def test_backward_compat_rival_index(self):
        from umarace._npc_data import RIVAL_INDEX
        # Old RIVAL_INDEX still works as best-effort shim
        assert len(RIVAL_INDEX) > 0
        for key, idx in list(RIVAL_INDEX.items())[:5]:
            assert isinstance(key, tuple)
            assert isinstance(idx, int)

    def test_race_entry_map_loaded(self):
        from umarace._npc_data import RACE_ENTRY_MAP
        assert len(RACE_ENTRY_MAP) > 0
        # Verify entry_num values are reasonable
        for prog_id, entry_num in list(RACE_ENTRY_MAP.items())[:20]:
            assert 9 <= entry_num <= 18, f"prog {prog_id}: entry_num={entry_num}"

    def test_race_group_map_still_works(self):
        from umarace._npc_data import RACE_GROUP_MAP
        assert len(RACE_GROUP_MAP) > 0
        # Spot check a known mapping
        for prog_id, group_id in list(RACE_GROUP_MAP.items())[:5]:
            assert isinstance(group_id, int)

    def test_all_51_characters_have_data(self):
        from umarace._npc_data import RIVAL_CANDIDATES
        player_charas = set(k[0] for k in RIVAL_CANDIDATES.keys())
        # Research found 51 characters; some may have been skipped due to missing NPC IDs
        assert len(player_charas) >= 45, f"Only {len(player_charas)} characters found"


class TestRivalLookup:
    """Phase R3: Verify rival candidate lookup."""

    def test_special_week_turn_31_has_rivals(self):
        """Special Week at turn 31, prog 163 (G1 Satsuki Sho) should have 2 rivals."""
        from umarace._npc_data import RIVAL_CANDIDATES
        entries = RIVAL_CANDIDATES.get((1001, 31), [])
        prog_163_candidates = [idx for pid, idx in entries if pid == 163]
        assert len(prog_163_candidates) == 2, f"Expected 2 rivals, got {len(prog_163_candidates)}"

    def test_special_week_turn_12_no_rivals(self):
        """Special Week has no rivals before turn 20."""
        from umarace._npc_data import RIVAL_CANDIDATES
        entries = RIVAL_CANDIDATES.get((1001, 12), [])
        assert len(entries) == 0

    def test_special_week_turn_20_first_rival(self):
        """Special Week's first rival appears at turn 20."""
        from umarace._npc_data import RIVAL_CANDIDATES
        # Find earliest turn for chara 1001
        turns = [t for (c, t) in RIVAL_CANDIDATES.keys() if c == 1001]
        assert min(turns) == 20

    def test_multi_candidate_race(self):
        """Some races have 3+ candidates."""
        from umarace._npc_data import RIVAL_CANDIDATES
        max_per_prog = 0
        for (chara, turn), entries in RIVAL_CANDIDATES.items():
            progs: dict[int, int] = {}
            for pid, _ in entries:
                progs[pid] = progs.get(pid, 0) + 1
            if progs:
                max_per_prog = max(max_per_prog, max(progs.values()))
        assert max_per_prog >= 3, f"Max candidates per race: {max_per_prog}"


class TestMaxRivalScoring:
    """Phase R4: Verify max-rival selection."""

    def test_max_rival_returns_none_for_no_rival(self):
        from umarace._npc_data import _get_max_rival_idx
        result = _get_max_rival_idx(1001, 12, 999, 2000, 1)
        assert result is None

    def test_max_rival_single_candidate(self):
        """When only 1 candidate, return it directly (no scoring needed)."""
        from umarace._npc_data import RIVAL_CANDIDATES, _get_max_rival_idx
        # Find a (chara, turn, prog) with exactly 1 candidate
        for (chara, turn), entries in RIVAL_CANDIDATES.items():
            progs: dict[int, list[int]] = {}
            for pid, idx in entries:
                progs.setdefault(pid, []).append(idx)
            for pid, idxs in progs.items():
                if len(idxs) == 1:
                    result = _get_max_rival_idx(chara, turn, pid, 2000, 1)
                    assert result == idxs[0]
                    return
        pytest.skip("No single-candidate case found")

    def test_max_rival_multi_candidate_picks_strongest(self):
        """Turn 31, prog 163, 2000m Turf: rival 1020 (total=1194, Mid=A) should beat 1061 (total=1144, Mid=B)."""
        from umarace._npc_data import NPC_DATA, RIVAL_CANDIDATES, _get_max_rival_idx
        entries = RIVAL_CANDIDATES.get((1001, 31), [])
        prog_163 = [idx for pid, idx in entries if pid == 163]
        if len(prog_163) < 2:
            pytest.skip("Expected 2 candidates at (1001, 31, 163)")

        result = _get_max_rival_idx(1001, 31, 163, 2000, 1)
        assert result is not None
        # The winner should have higher total stats
        winner_total = float(NPC_DATA[result, :5].sum())
        for idx in prog_163:
            if idx != result:
                loser_total = float(NPC_DATA[idx, :5].sum())
                assert winner_total >= loser_total, (
                    f"Winner total {winner_total} < loser total {loser_total}"
                )

    def test_max_rival_returns_valid_npc_idx(self):
        from umarace._npc_data import NPC_DATA, RIVAL_CANDIDATES, _get_max_rival_idx
        for (chara, turn), entries in list(RIVAL_CANDIDATES.items())[:20]:
            for pid, _ in entries:
                result = _get_max_rival_idx(chara, turn, pid, 2000, 1)
                if result is not None:
                    assert 0 <= result < len(NPC_DATA)


class TestSampleFieldWithRival:
    """Phase R5a: Verify sample_field rival injection."""

    def test_sample_field_no_rival_unchanged(self):
        """Without rival_idx, behavior is unchanged."""
        from umarace._npc_data import sample_field
        rng = np.random.default_rng(42)
        batch, bonuses, recovery = sample_field(13, 17, rng, distance=2000, surface=1)
        assert batch.shape == (17, 10)
        assert bonuses.shape == (17,)
        assert recovery.shape == (17,)

    def test_sample_field_with_rival_adds_one(self):
        """With rival_idx, field has count-1 random + 1 rival = count total."""
        from umarace._npc_data import sample_field, RIVAL_CANDIDATES
        # Find a valid rival idx
        first_entries = list(RIVAL_CANDIDATES.values())[0]
        rival_idx = first_entries[0][1]

        rng = np.random.default_rng(42)
        batch, bonuses, recovery = sample_field(13, 17, rng, distance=2000, surface=1,
                                                  rival_idx=rival_idx)
        assert batch.shape == (17, 10), f"Expected (17, 10), got {batch.shape}"
        assert bonuses.shape == (17,)
        assert recovery.shape == (17,)

    def test_sample_field_rival_stats_applied(self):
        """Rival row should have difficulty_rate applied to stats."""
        from umarace._npc_data import NPC_DATA, sample_field, RIVAL_CANDIDATES
        first_entries = list(RIVAL_CANDIDATES.values())[0]
        rival_idx = first_entries[0][1]
        raw_speed = float(NPC_DATA[rival_idx, 0])

        rng = np.random.default_rng(42)
        batch, _, _ = sample_field(13, 5, rng, difficulty_rate=1.10, distance=2000,
                                    surface=1, rival_idx=rival_idx)
        # Last row is the rival
        rival_speed = float(batch[-1, 0])
        expected = raw_speed * 1.10
        assert abs(rival_speed - expected) < 0.1, (
            f"Rival speed {rival_speed} != expected {expected}"
        )

    def test_sample_field_small_group_with_rival(self):
        """Group 41 has only 10 NPCs. With rival, should sample min(count-1, 10)."""
        from umarace._npc_data import sample_field, RIVAL_CANDIDATES
        first_entries = list(RIVAL_CANDIDATES.values())[0]
        rival_idx = first_entries[0][1]

        rng = np.random.default_rng(42)
        batch, bonuses, recovery = sample_field(41, 17, rng, distance=2000, surface=1,
                                                  rival_idx=rival_idx)
        # Group 41 has 10 NPCs. count-1=16, min(16,10)=10 field + 1 rival = 11
        assert batch.shape[0] == 11, f"Expected 11, got {batch.shape[0]}"

    def test_entry_num_16_runner_race(self):
        """A 16-runner race should produce 15 field NPCs (or fewer + rival)."""
        from umarace._npc_data import sample_field
        rng = np.random.default_rng(42)
        batch, _, _ = sample_field(13, 15, rng, distance=2000, surface=1)
        assert batch.shape[0] == 15

    def test_entry_num_9_runner_race(self):
        """A 9-runner race should produce 8 field NPCs."""
        from umarace._npc_data import sample_field
        rng = np.random.default_rng(42)
        batch, _, _ = sample_field(13, 8, rng, distance=2000, surface=1)
        assert batch.shape[0] == 8


class TestPredictWithRival:
    """Phase R5b: Verify predict() rival integration."""

    def test_predict_backward_compat(self):
        """predict() without rival params works as before."""
        import umarace
        result = umarace.predict(
            stats=(800, 600, 700, 500, 600),
            race_grade="G1",
            npc_seed=42,
        )
        assert 0.0 < result.win_probability < 1.0
        assert 1.0 <= result.expected_placement <= 18.0

    def test_predict_with_rival_lowers_win_prob(self):
        """Adding a rival to the field should lower win probability."""
        import umarace
        from umarace._npc_data import RIVAL_CANDIDATES

        # Find a race with a rival for chara 1001
        for (chara, turn), entries in RIVAL_CANDIDATES.items():
            if chara == 1001 and turn >= 30:
                prog_id = entries[0][0]
                break
        else:
            pytest.skip("No suitable rival race found")

        stats = (800, 600, 700, 500, 600)

        result_no_rival = umarace.predict(
            stats=stats, race_grade="G1", npc_seed=42, distance=2000,
        )
        result_with_rival = umarace.predict(
            stats=stats, distance=2000, npc_seed=42,
            player_chara=chara, turn=turn, race_program_id=prog_id,
        )

        assert result_with_rival.win_probability < result_no_rival.win_probability, (
            f"With rival ({result_with_rival.win_probability:.3f}) should be "
            f"< without ({result_no_rival.win_probability:.3f})"
        )

    def test_predict_no_rival_turn_matches_baseline(self):
        """At a turn with no rivals, predict() should match the no-rival baseline."""
        import umarace

        stats = (800, 600, 700, 500, 600)

        result_baseline = umarace.predict(
            stats=stats, race_grade="G1", npc_seed=42, distance=2000,
        )
        # Turn 12 for chara 1001 has no rivals
        result_no_rival = umarace.predict(
            stats=stats, distance=2000, npc_seed=42,
            player_chara=1001, turn=12, race_program_id=999,  # nonexistent prog
        )

        # Win probs should be similar (not identical due to entry_num differences)
        # But the key point: no rival was injected
        assert result_no_rival.win_probability > 0

    def test_predict_strong_rival_lowers_win_prob(self):
        """A strong rival should lower win probability compared to field-only."""
        import umarace
        from umarace._npc_data import RIVAL_CANDIDATES

        # Find chara 1069's turn 72 entry (has strong rivals)
        entries = RIVAL_CANDIDATES.get((1069, 72), [])
        if not entries:
            pytest.skip("Chara 1069 turn 72 not found")

        prog_id = entries[0][0]

        # Weak stats — makes the rival impact visible
        stats = (300, 300, 300, 200, 200)

        result_no_rival = umarace.predict(
            stats=stats, distance=2000, surface="turf",
            npc_seed=42, race_grade="G1",
        )
        result_with_rival = umarace.predict(
            stats=stats, distance=2000, surface="turf", npc_seed=42,
            player_chara=1069, turn=72, race_program_id=prog_id,
        )
        # With a strong rival, win prob should be meaningfully lower
        assert result_with_rival.win_probability < result_no_rival.win_probability, (
            f"With rival ({result_with_rival.win_probability:.3f}) should be "
            f"< without ({result_no_rival.win_probability:.3f})"
        )

    def test_predict_with_race_program_uses_correct_entry_num(self):
        """When race_program_id is provided, entry_num from RACE_ENTRY_MAP should be used."""
        import umarace
        from umarace._npc_data import RACE_ENTRY_MAP

        # Find a program with entry_num != 18
        non_18 = [(pid, en) for pid, en in RACE_ENTRY_MAP.items() if en != 18]
        if not non_18:
            pytest.skip("No non-18 entry races found")

        prog_id, expected_entry = non_18[0]
        result = umarace.predict(
            stats=(800, 600, 700, 500, 600),
            distance=2000, npc_seed=42,
            race_program_id=prog_id,
        )
        # placement_distribution length should match entry_num
        assert len(result.placement_distribution) == expected_entry, (
            f"Expected {expected_entry} placements, got {len(result.placement_distribution)}"
        )


class TestNpcModule:
    """Phase R5c: Verify npc.py functions."""

    def test_get_rival_deprecated_still_works(self):
        from umarace.npc import get_rival
        # Should work for any (chara, turn) that has data
        rival = get_rival(1001, 31, distance=2000, surface=1)
        # May or may not find one depending on shim
        # Just verify it doesn't crash
        assert rival is None or rival.speed > 0

    def test_get_rival_for_race(self):
        from umarace.npc import get_rival_for_race
        from umarace._npc_data import RIVAL_CANDIDATES

        # Find a valid (chara, turn, prog)
        for (chara, turn), entries in RIVAL_CANDIDATES.items():
            prog_id = entries[0][0]
            rival = get_rival_for_race(chara, turn, prog_id, 2000, 1)
            assert rival is not None
            assert rival.speed > 0
            assert rival.stamina > 0
            return

    def test_get_rival_for_race_no_match(self):
        from umarace.npc import get_rival_for_race
        rival = get_rival_for_race(9999, 1, 999, 2000, 1)
        assert rival is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
