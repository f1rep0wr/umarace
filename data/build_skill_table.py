"""Build skill_id -> (score_delta, recovery_hp_frac) lookup from upstream skill_data.json.

Converts each skill's effects into a net score_delta (speed/accel) and a
recovery HP fraction (recovery), using the umarace composite score formula
with timing multipliers and wisdomCheck flag.

Score mapping (from umarace composite formula):
  score = eff_spurt * 100 + accel * 500 + skill_rate * 0.5

Effect types (from upstream RaceSolver.ts SkillType enum):
  27 = TargetSpeed: +modifier m/s to target speed
  22 = CurrentSpeedWithDecel: +modifier m/s direct displacement (self buff)
  21 = CurrentSpeed: all negative modifiers in data (debuffs)
  31 = Accel: +modifier m/s^2 to acceleration
  9  = Recovery: +modifier fraction of maxHP restored (dynamic at runtime)
  28 = Accel variant (minor, 42 effects)
  1-5 = Permanent stat gain (flat): ignored (marginal, rare on NPCs)

Timing multipliers (weighted average across activation conditions):
  Type 27 (TargetSpeed): ~25% phase-2 (1.0), ~60% mid-race (0.55), ~15% early (0.30)
    -> weighted avg 0.63, with 0.70 accel-to-ceiling discount -> net 0.44
  Type 22 (CurrentSpeed): timing-agnostic -> 1.0 (direct displacement)
  Type 31 (Accel): ~65% phase-2 (1.0), ~25% mid-race (0.10), ~10% early (0.03)
    -> weighted avg 0.68
  Type 9 (Recovery): now dynamic — raw HP fraction stored, evaluated at runtime
    via effective_spurt_speed()'s sustained clamping

Duration scaling: actual_duration = base_duration * (distance / 1000)
Reference distance: 2000m -> duration = base_duration * 2.0

Output: skill_bonus_table.npy — (N, 4) float32 [skill_id, score_delta, wisdom_check, recovery_hp_frac]
  wisdom_check: 1.0 = requires wisdom roll, 0.0 = always activates

Usage: python -m data.build_skill_table
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent
REF_DISTANCE = 2000
# Spurt phase: ~1/6 of race at ~20 m/s base speed
SPURT_DURATION = (REF_DISTANCE / 6.0) / 20.0  # ~16.7s

# Weighted-average timing multipliers per effect type
# Derived from research-skill-timing.md activation condition distributions
TIMING_MULT = {
    27: 0.44,   # TargetSpeed: 0.63 timing × 0.70 accel-to-ceiling
    22: 1.00,   # CurrentSpeed: timing-agnostic (direct displacement)
    21: 1.00,   # CurrentSpeed debuff: same physics
    31: 0.68,   # Accel: mostly phase-2 timed
    28: 0.68,   # Accel variant: same
}


def estimate_score_delta(effects: list[dict], base_duration: float) -> tuple[float, float]:
    """Estimate net score_delta and recovery HP fraction for one skill alternative.

    Args:
        effects: list of {type, modifier, target} dicts (raw ints from JSON)
        base_duration: base duration in seconds (already /10000)

    Returns:
        (score_delta, recovery_hp_fraction) — score_delta for speed/accel,
        recovery_hp_fraction as raw HP fraction for dynamic evaluation.
    """
    duration = base_duration * (REF_DISTANCE / 1000.0)
    delta = 0.0
    recovery = 0.0

    for ef in effects:
        t = ef["type"]
        mod = ef["modifier"] / 10000.0
        target = ef.get("target", 1)

        # Skip opponent-targeted effects for self-score
        if target not in (1, 11):
            continue

        timing = TIMING_MULT.get(t, 0.5)

        if t == 27:  # TargetSpeed
            delta += mod * 100.0 * timing
        elif t in (21, 22):  # CurrentSpeed variants
            if mod > 0:
                delta += mod * 100.0 * timing
        elif t in (31, 28):  # Accel
            if mod > 0:
                frac = min(duration / max(SPURT_DURATION, 1.0), 1.0)
                delta += mod * 500.0 * frac * timing
        elif t == 9:  # Recovery — raw HP fraction, evaluated dynamically
            if mod > 0:
                recovery += mod

    return delta, recovery


def build_skill_table():
    """Build skill_id -> (score_delta, wisdom_check, recovery_hp_frac) lookup."""
    with open(DATA_DIR / "skill_data.json") as f:
        skills = json.load(f)

    entries: list[tuple[int, float, float, float]] = []

    for sid_str, sdata in skills.items():
        sid = int(sid_str)
        alternatives = sdata.get("alternatives", [])
        if not alternatives:
            continue

        alt = alternatives[0]
        base_dur = alt.get("baseDuration", 0) / 10000.0
        effects = alt.get("effects", [])
        wisdom_check = float(sdata.get("wisdomCheck", 1))

        delta, recovery = estimate_score_delta(effects, base_dur)
        if abs(delta) > 0.01 or recovery > 0.0001:
            entries.append((sid, delta, wisdom_check, recovery))

    entries.sort()
    arr = np.array(entries, dtype=np.float32)
    np.save(DATA_DIR / "skill_bonus_table.npy", arr)

    score_deltas = arr[:, 1]
    recovery_fracs = arr[:, 3]
    has_delta = np.abs(score_deltas) > 0.01
    has_recovery = recovery_fracs > 0.0001

    print(f"Skill bonus table: {len(entries)} skills")
    print(f"  With score delta: {has_delta.sum()}")
    print(f"    Score delta range: [{score_deltas[has_delta].min():.1f}, {score_deltas[has_delta].max():.1f}]")
    print(f"    Mean: {score_deltas[has_delta].mean():.1f}, Median: {np.median(score_deltas[has_delta]):.1f}")
    print(f"  With recovery: {has_recovery.sum()}")
    if has_recovery.any():
        print(f"    Recovery HP frac range: [{recovery_fracs[has_recovery].min():.4f}, {recovery_fracs[has_recovery].max():.4f}]")
        print(f"    Mean: {recovery_fracs[has_recovery].mean():.4f}, Median: {np.median(recovery_fracs[has_recovery]):.4f}")

    n_checked = (arr[:, 2] > 0).sum()
    n_always = (arr[:, 2] == 0).sum()
    print(f"  Wisdom-gated: {n_checked}, Always-activate: {n_always}")

    top_idx = np.argsort(score_deltas)[::-1][:10]
    print(f"\nTop 10 highest-impact skills (score delta):")
    for i in top_idx:
        wc = "WC" if arr[i, 2] > 0 else "  "
        rec = f"  rec={arr[i, 3]:.4f}" if arr[i, 3] > 0 else ""
        print(f"  skill_id={int(arr[i, 0]):>8d}  delta={arr[i, 1]:>+7.1f}  {wc}{rec}")


def build_npc_skill_bonus():
    """Build per-NPC expected skill bonus and recovery fraction.

    Emits two arrays:
    - npc_skill_bonus.npy: static score delta per NPC (speed/accel skills)
    - npc_recovery_bonus.npy: recovery HP fraction per NPC

    Estimates are derived from the actual skill table distributions and
    scaled by each NPC's wisdom-based activation probability.
    """
    import csv

    # Load the skill table we just built to get actual distributions
    skill_table_path = DATA_DIR / "skill_bonus_table.npy"
    if skill_table_path.exists():
        st = np.load(skill_table_path)
        positive_deltas = st[:, 1][st[:, 1] > 0]
        positive_recovery = st[:, 3][st[:, 3] > 0]
        # Average per-skill contribution for NPCs with skills
        avg_delta_per_skill = float(positive_deltas.mean()) if len(positive_deltas) > 0 else 5.0
        avg_recovery_per_skill = float(positive_recovery.mean()) if len(positive_recovery) > 0 else 0.0
        n_recovery_skills = len(positive_recovery)
        n_delta_skills = len(positive_deltas)
        recovery_ratio = n_recovery_skills / max(n_delta_skills + n_recovery_skills, 1)
    else:
        avg_delta_per_skill = 5.0
        avg_recovery_per_skill = 0.0
        recovery_ratio = 0.0

    # Try multiple possible locations for the NPC CSV
    candidates = [
        DATA_DIR.parent.parent / "data" / "single_mode_npc_dump.csv",
        DATA_DIR.parent.parent / "autoresearch" / "single_mode_npc_dump.csv",
    ]
    npc_csv = None
    for p in candidates:
        if p.exists():
            npc_csv = p
            break

    if npc_csv is None:
        print("NPC CSV not found, skipping NPC skill bonus")
        return

    with open(npc_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    est_skills = 3  # typical NPC skill count
    npc_bonuses = np.zeros(len(rows), dtype=np.float32)
    npc_recovery = np.zeros(len(rows), dtype=np.float32)

    for i, row in enumerate(rows):
        ssid = int(row.get("skill_set_id", 0))
        if ssid > 0:
            raw_wis = float(row.get("wiz", 150))
            wis_eff = min(raw_wis * 1.05, 1200.0)
            activation = max(1.0 - 90.0 / max(wis_eff, 1.0), 0.20)
            # Static delta: ~3 skills × avg delta × activation
            n_delta = est_skills * (1 - recovery_ratio)
            npc_bonuses[i] = n_delta * avg_delta_per_skill * activation
            # Recovery: ~3 skills × recovery_ratio × avg recovery × activation
            n_recov = est_skills * recovery_ratio
            npc_recovery[i] = n_recov * avg_recovery_per_skill * activation

    np.save(DATA_DIR / "npc_skill_bonus.npy", npc_bonuses)
    np.save(DATA_DIR / "npc_recovery_bonus.npy", npc_recovery)

    nonzero_bonus = npc_bonuses[npc_bonuses > 0]
    nonzero_recov = npc_recovery[npc_recovery > 0]
    from collections import Counter
    ssid_counts = Counter(int(r.get("skill_set_id", 0)) for r in rows)
    n_with = sum(c for s, c in ssid_counts.items() if s > 0)

    print(f"\nNPC skill estimates (data-driven):")
    print(f"  Source: avg_delta/skill={avg_delta_per_skill:.1f}, avg_recovery/skill={avg_recovery_per_skill:.4f}, recovery_ratio={recovery_ratio:.2%}")
    print(f"  With skills: {n_with}/{len(rows)} ({n_with/len(rows):.0%})")
    if len(nonzero_bonus) > 0:
        print(f"  Static bonus range: [{nonzero_bonus.min():.1f}, {nonzero_bonus.max():.1f}], mean={nonzero_bonus.mean():.1f}")
    if len(nonzero_recov) > 0:
        print(f"  Recovery frac range: [{nonzero_recov.min():.6f}, {nonzero_recov.max():.6f}], mean={nonzero_recov.mean():.6f}")
    else:
        print(f"  Recovery: no NPCs with recovery skills")
    print(f"  Saved npc_skill_bonus.npy + npc_recovery_bonus.npy: {len(npc_bonuses)} entries each")


if __name__ == "__main__":
    build_skill_table()
    build_npc_skill_bonus()
    print("\nDone.")
