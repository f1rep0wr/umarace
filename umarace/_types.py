"""Core type definitions for umarace."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Aptitude(IntEnum):
    """Game aptitude grades, stored as 1-8 in master.mdb."""
    G = 1
    F = 2
    E = 3
    D = 4
    C = 5
    B = 6
    A = 7
    S = 8


class Strategy(IntEnum):
    """Running strategy IDs matching game data."""
    NIGE = 1     # Front Runner
    SENKOU = 2   # Pace Chaser
    SASHI = 3    # Late Surger
    OIKOMI = 4   # End Closer
    OONIGE = 5   # Runaway


class Surface(IntEnum):
    TURF = 1
    DIRT = 2


class GroundCondition(IntEnum):
    GOOD = 0
    SLIGHTLY_HEAVY = 1
    HEAVY = 2
    BAD = 3


class Motivation(IntEnum):
    AWFUL = 1
    BAD = 2
    NORMAL = 3
    GOOD = 4
    GREAT = 5


# String-to-enum mappings for human API
_APT_MAP = {"G": 1, "F": 2, "E": 3, "D": 4, "C": 5, "B": 6, "A": 7, "S": 8}
_STRAT_MAP = {
    "nige": 1, "front": 1, "front_runner": 1,
    "senkou": 2, "pace": 2, "pace_chaser": 2,
    "sashi": 3, "late": 3, "late_surger": 3,
    "oikomi": 4, "end": 4, "end_closer": 4,
    "oonige": 5, "runaway": 5,
}
_SURFACE_MAP = {"turf": 1, "dirt": 2}


def parse_aptitude(val: str | int | Aptitude) -> int:
    if isinstance(val, int):
        return val
    return _APT_MAP[val.upper()]


def parse_strategy(val: str | int | Strategy) -> int:
    if isinstance(val, int):
        return val
    return _STRAT_MAP[val.lower()]


def parse_surface(val: str | int | Surface) -> int:
    if isinstance(val, int):
        return val
    return _SURFACE_MAP[val.lower()]


@dataclass(frozen=True, slots=True)
class Runner:
    """A single race participant (human API type)."""
    speed: int
    stamina: int
    power: int
    guts: int
    wisdom: int
    dist_aptitude: int = Aptitude.A
    surf_aptitude: int = Aptitude.A
    strat_aptitude: int = Aptitude.A
    strategy: int = Strategy.SENKOU
    motivation: int = Motivation.NORMAL
    career_bonus: int = 400


@dataclass(frozen=True, slots=True)
class RaceConfig:
    """Race course parameters (human API type)."""
    distance: int
    surface: int = Surface.TURF
    ground_condition: int = GroundCondition.GOOD
    entry_count: int = 18


@dataclass
class PlacementResult:
    """Race outcome prediction (human API return type)."""
    expected_placement: float
    win_probability: float
    top3_probability: float
    top5_probability: float
    placement_distribution: list[float]
    score: float
