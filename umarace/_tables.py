"""Lookup tables as indexed numpy arrays. All values from master.mdb ground truth.

Index 0 is a sentinel in all arrays (game IDs start at 1).
"""

import numpy as np

# --- Aptitude multipliers (race_proper_*_rate tables) ---
# Indexed by aptitude grade ID: 1=G, 2=F, 3=E, 4=D, 5=C, 6=B, 7=A, 8=S

DIST_SPEED_RATE = np.array(
    [0, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 1.00, 1.05], dtype=np.float32)

# race_proper_runningstyle_rate → wisdom multiplier
STRAT_RATE = np.array(
    [0, 0.10, 0.20, 0.40, 0.60, 0.75, 0.85, 1.00, 1.10], dtype=np.float32)

# race_proper_ground_rate → acceleration multiplier
SURFACE_RATE = np.array(
    [0, 0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 1.00, 1.05], dtype=np.float32)

# race_motivation_rate
MOTIVATION_RATE = np.array(
    [0, 0.96, 0.98, 1.00, 1.02, 1.04], dtype=np.float32)


# --- Strategy coefficients (indexed by strategy ID 1-5) ---
# 1=Nige, 2=Senkou, 3=Sashi, 4=Oikomi, 5=Oonige

# Target speed multiplier per phase (from upstream RaceSolver.ts Speed.StrategyPhaseCoefficient)
STRAT_SPEED_OPENING = np.array(
    [0, 1.000, 0.978, 0.938, 0.931, 1.063], dtype=np.float32)
STRAT_SPEED_MIDDLE = np.array(
    [0, 0.980, 0.991, 0.998, 1.000, 0.962], dtype=np.float32)
STRAT_SPEED_FINAL = np.array(
    [0, 0.962, 0.975, 0.994, 1.000, 0.950], dtype=np.float32)

# Acceleration multiplier per phase
STRAT_ACCEL_FINAL = np.array(
    [0, 0.996, 0.996, 1.000, 0.997, 0.956], dtype=np.float32)

# Acceleration distance proficiency (separate from speed distance proficiency)
# From upstream Acceleration.DistanceProficiencyModifier (S→G mapped to index 1-8)
ACCEL_DIST_PROF = np.array(
    [0, 0.40, 0.50, 0.60, 1.00, 1.00, 1.00, 1.00, 1.00], dtype=np.float32)

# HP pool coefficient
STRAT_HP_COEF = np.array(
    [0, 0.95, 0.89, 1.00, 0.995, 0.86], dtype=np.float32)


# --- Ground condition modifiers ---
# Indexed by (surface - 1) * 4 + ground_condition
# Layout: [turf_good, turf_slight, turf_heavy, turf_bad,
#          dirt_good, dirt_slight, dirt_heavy, dirt_bad]

GROUND_SPEED_MOD = np.array(
    [0, 0, 0, -50, 0, 0, 0, -50], dtype=np.float32)
GROUND_POWER_MOD = np.array(
    [0, -50, -50, -50, -100, -50, -100, -100], dtype=np.float32)
GROUND_HP_MOD = np.array(
    [1.00, 1.00, 1.02, 1.02, 1.00, 1.00, 1.01, 1.02], dtype=np.float32)


# --- Stochastic HP cost rate (calibrated from community thresholds) ---
# Recalibrated after guts_mod fix (multiply not divide): threshold builds at
# 2000m (sta=800+400, guts=300+400) have headroom ~374, duration ~100s -> 3.7 HP/s
# At 2400m (sta=1000+400): headroom ~363, duration ~122s -> 3.0 HP/s
# Use 3.5 as compromise across distances
STOCHASTIC_HP_COST_PER_SECOND = 3.5
