# umarace

Vectorized race outcome prediction for **Umamusume Pretty Derby**. Predicts placement probabilities from runner stats using reverse-engineered race physics and ground-truth NPC data from `master.mdb`.

Built for two use cases:
- **ML training** — batch-first numpy API for MCTS rollouts (~100K races/sec)
- **Standalone analysis** — human-friendly API: "given these stats, what's my win probability?"

## Install

```
pip install -e .
```

Runtime dependency: **numpy** only. Optional: `scipy` (calibration), `numba` (15x faster placement DP).

## Quick Start

```python
import umarace

# Single race prediction vs NPC field
result = umarace.predict(
    stats=(1100, 900, 1000, 700, 800),  # speed, stamina, power, guts, wisdom
    aptitudes=("A", "A", "A"),           # distance, surface, strategy
    strategy="pace_chaser",
    distance=2000,
    surface="turf",
    race_grade="G1",
)
print(f"Win:   {result.win_probability:.1%}")
print(f"Top 3: {result.top3_probability:.1%}")
print(f"E[place]: {result.expected_placement:.1f}")

# Race-specific prediction with rival + skills
result = umarace.predict(
    stats=(1100, 900, 1000, 700, 800),
    aptitudes=("A", "A", "A"),
    strategy="pace_chaser",
    distance=2000,
    surface="turf",
    player_chara=102,           # Special Week
    turn=31,
    race_program_id=201,        # specific race → NPC group + entry count + rival
    skill_ids=[100101, 100201], # owned skill IDs for activation bonus
)

# Race power score (no NPC field needed)
score = umarace.race_power(
    stats=(1100, 900, 1000, 700, 800),
    strategy="pace_chaser",
    distance=2000,
)
```

## Batch API (ML)

All inputs/outputs are numpy arrays. 
```python
import numpy as np
from umarace.batch import score_batch, predict_batch

B = 1024  # number of races
N = 18    # runners per race

# Struct-of-Arrays layout
stats      = np.random.uniform(400, 1200, (B, N, 5)).astype(np.float32)
apt_ids    = np.full((B, N, 3), 7, dtype=np.uint8)   # all A aptitude
strategy   = np.full((B, N), 2, dtype=np.uint8)       # all Pace Chaser
motivation = np.full((B, N), 3, dtype=np.uint8)       # Normal
career_bonus = np.full((B, N), 400, dtype=np.float32)
distance   = np.full(B, 2000, dtype=np.int32)
surface    = np.ones(B, dtype=np.uint8)                # Turf
ground_cond = np.zeros(B, dtype=np.uint8)              # Good

# Score only (fastest — ~1.8 μs/race)
scores = score_batch(stats, apt_ids, strategy, motivation,
                     career_bonus, distance, surface, ground_cond)

# With skill bonuses (static speed/accel + dynamic recovery)
score_delta = np.zeros((B, N), dtype=np.float32)        # speed/accel skill deltas
recovery_hp_frac = np.zeros((B, N), dtype=np.float32)   # recovery HP fraction
scores = score_batch(stats, apt_ids, strategy, motivation,
                     career_bonus, distance, surface, ground_cond,
                     score_delta=score_delta,
                     recovery_hp_frac=recovery_hp_frac)

# Full placement prediction (~12 μs/race with fast=True)
result = predict_batch(stats, apt_ids, strategy, motivation,
                       career_bonus, distance, surface, ground_cond,
                       trainee_idx=0, fast=True)
# result keys: 'win_prob', 'top3_prob', 'top5_prob',
#              'expected_place', 'placement_dist', 'scores'
```

## NPC Data

1,253 career NPCs + climax variants loaded as contiguous numpy arrays at import (~70KB, fits in L2 cache). Each NPC stores all 4 distance aptitudes (short/mile/mid/long) and both surface aptitudes (turf/dirt), resolved at race time for the specific course.

```python
from umarace.npc import get_field_npcs, get_rival_for_race, get_climax_npcs

# Sample G1 field opponents
npcs = get_field_npcs(race_grade="G1", count=17, seed=42)

# Race-specific rival lookup (max-rival scoring across candidates)
rival = get_rival_for_race(
    player_chara=102, turn=31, race_program_id=201,
    distance=2000, surface=1,
)

# Climax NPCs
climax = get_climax_npcs(round_num=3, surface="turf")
```

45+ playable characters have rival assignments. Multiple candidates per race are scored on the actual distance/surface to select the strongest.

## Skill Bonus System

Skills are split into two channels based on how they affect race outcomes:

- **Speed/Accel skills** (TargetSpeed, CurrentSpeed, Accel) → **static score delta** added to the composite score. These effects are approximately linear: +0.1 m/s target speed ≈ +4.4 score points regardless of the runner's HP state.
- **Recovery skills** → **dynamic HP fraction** injected into the stamina integration path. Recovery adds HP to the pool *before* the sustainability check in `effective_spurt_speed()`, making its value depend on the runner's actual stamina headroom.

This split exists because recovery skills are fundamentally nonlinear — their value depends on the runner's HP state:

| Runner State | Sustained | Recovery Effect |
|---|---|---|
| HP-surplus (headroom > stoch_cost) | 1.0 (capped) | **Zero** — already at max spurt speed |
| Borderline (headroom ≈ stoch_cost) | 0.1–0.9 | **Large** — proportional to HP shortfall |
| Deeply starved (headroom << 0) | 0.0 | **Zero** — recovery can't fix stamina |

A static heuristic would average across all three regimes, giving surplus runners credit they don't deserve and shortchanging borderline runners where recovery actually decides the race.

All skills are wisdom-gated: activation rate = `clamp(1 - 90/wisdom, 0.20, 1.0)`.

```python
# Human API: pass skill_ids to predict() or race_power()
result = umarace.predict(
    stats=(1100, 900, 1000, 700, 800),
    aptitudes=("A", "A", "A"),
    strategy="pace_chaser",
    distance=2000,
    surface="turf",
    skill_ids=[100101, 100201, 200301],
)

score = umarace.race_power(
    stats=(1100, 900, 1000, 700, 800),
    strategy="pace_chaser",
    distance=2000,
    skill_ids=[100101, 100201],
)

# Batch API: pass score_delta + recovery_hp_frac to score_batch() / predict_batch()
score_delta = np.zeros((B, N), dtype=np.float32)
recovery_hp_frac = np.zeros((B, N), dtype=np.float32)
score_delta[:, 0] = 15.0       # trainee speed/accel bonus
recovery_hp_frac[:, 0] = 0.02  # trainee recovery (2% maxHP)
```

NPC skill bonuses (both static and recovery) are pre-computed per NPC and automatically included in field sampling.

## How It Works

### Scoring Pipeline

Each runner gets a composite **race power score** computed from:

1. **Stat preprocessing** — halving above 1200, motivation multiplier, career bonus (current global stat max = 1200)
2. **Aptitude lookup** — distance/surface/strategy proficiency via indexed numpy arrays
3. **Ground condition** — speed/power modifiers and HP drain scaling for turf/dirt × good/heavy/bad
4. **Last Spurt speed** — `(base_speed × strat_coef + speed_term + 0.01 × base_speed) × 1.05 + speed_term + guts_term`
5. **Dynamic recovery** — recovery HP fraction added to HP pool: `hp = max_hp × (1 + recovery_frac)`
6. **Stamina sustainability** — HP pool vs estimated drain across 4 race phases; low stamina degrades spurt speed toward min_speed
7. **Stamina Limit Break** — bonus for effective stamina > 1200 on courses > 2100m
8. **Acceleration** — from power stat, strategy coefficient, surface/distance proficiency
9. **Static skill bonus** — speed/accel skill deltas added to composite score
10. **Rushing penalty** — wisdom-based rushing probability reduces composite score
11. **Composite** — `(effective_spurt × 100 + accel × 500 + skill_rate × 0.5 + score_delta) × rush_penalty`

### Placement Model

Uses **Gauss-Hermite quadrature** (Q=20 default, Q=10 fast mode) to integrate over shared trainee noise, with O(N²) DP per quadrature point for the full placement distribution. 
Validated against Monte Carlo N=100,000: max error 0.14% at Q=20, <0.1% win_prob error at Q=10.

## Performance

Measured on CPU (numpy + optional numba for placement DP):

| Operation | Throughput |
|-----------|-----------|
| `score_batch` (1024×18) | 550K races/sec |
| `predict_batch` Q=20 | 40K races/sec |
| `predict_batch` Q=10 `fast=True` | 80-100K races/sec |

Install `numba` for 15x faster placement DP (auto-detected at import).

## Accuracy

| Metric | Estimate |
|--------|----------|
| Within ±1 placement | 65-72% |
| Within ±2 placement | ~85% |

Primary error sources: unmodeled individual skill effects (±1-2 placements), position keep mechanics (±0.5-1).

## Testing

```
pip install -e ".[dev]"
pytest tests/
```

75 tests across 3 files:
- `umarace/tests/test_formulas.py` — unit tests for individual race physics formulas
- `umarace/tests/test_integration.py` — end-to-end prediction, batch API, edge cases
- `tests/test_rivals.py` — rival lookup, max-rival scoring, field injection, all 51 characters

## Project Structure

```
umarace/
├── __init__.py        # Human API: predict(), race_power()
├── batch.py           # ML API: score_batch(), predict_batch()
├── _formulas.py       # Vectorized race physics (numpy)
├── _placement.py      # Gauss-Hermite placement distribution
├── _tables.py         # Lookup tables as indexed numpy arrays
├── _types.py          # Runner, RaceConfig, PlacementResult, enums
├── _npc_data.py       # NPC data loading, field sampling, rival scoring
├── npc.py             # Human-friendly NPC lookup
└── tests/
    ├── __init__.py
    ├── test_formulas.py   # Formula unit tests
    └── test_integration.py # End-to-end + edge case tests
data/
├── npc_all.npy        # All career NPCs (1253×14 float32)
├── npc_groups.npz     # Group ID → row index mapping
├── climax_npc.npy     # Climax event NPCs (×14 float32)
├── climax_groups.npz  # Climax group → row indices
├── rival_candidates.npy # (player_chara, turn, race_program_id, npc_idx) tuples
├── rival_index.npy    # Backward-compat: (chara, turn) → row index
├── race_group_map.npy # race_program → (group_id, entry_num)
├── npc_skill_bonus.npy # Per-NPC estimated static skill bonus
├── npc_recovery_bonus.npy # Per-NPC estimated recovery HP fraction
├── skill_bonus_table.npy # skill_id → (score_delta, wisdom_check, recovery_hp_frac)
├── skill_data.json    # Upstream skill effect data
├── build_data.py      # CSV → .npy build script
└── build_skill_table.py # Skill bonus table builder
tests/
└── test_rivals.py     # Rival system tests
```

## NPC Data Layout (14 columns)

```
[0-4]   speed, stamina, power, guts, wisdom
[5-8]   distance aptitudes: short, mile, mid, long (grade IDs 1-8)
[9-10]  surface aptitudes: turf, dirt
[11]    strategy aptitude (for chosen strategy)
[12]    strategy ID (1-5)
[13]    motivation ID (1-5)
```

Aptitudes are stored per-NPC across all distance/surface combinations and resolved at race time based on the actual course, enabling efficient field sampling without pre-resolving.

## Formulas Reference

All formulas reverse-engineered from game code, verified against [alpha123/uma-skill-tools](https://github.com/alpha123/uma-skill-tools).

| Formula | Source |
|---------|--------|
| `base_speed = 20 - (dist - 2000) / 1000` | RaceSolver.ts |
| `max_hp = 0.8 × strat_coef × stamina + distance` | HpPolicy.ts |
| `hp_drain = 20 × (v - base + 12)² / 144 × ground_mod` | HpPolicy.ts |
| `guts_mod = 1 + 200 / √(600 × guts)` | HpPolicy.ts |
| `skill_rate = clamp(1 - 90/wisdom, 0.2, 1.0)` | RaceSolverBuilder.ts |
| `rush_prob = (6.5 / log₁₀(0.1w + 1))² / 100` | RaceSolver.ts |
| `limit_break = √(sta_eff - 1200) × 0.0085 × dist_factor × 0.5` | HpPolicy.ts |
| `stochastic_hp_cost = 3.5 HP/s × race_duration` | Calibrated |

## License

MIT
