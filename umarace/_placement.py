"""Analytic placement distribution with shared-noise Gauss-Hermite integration.

The naive product formula P(win) = prod(P(beat_i)) is wrong by ~5x because the
trainee's noise draw is shared across all comparisons. This module integrates
over the trainee's noise using Gauss-Hermite quadrature, then uses O(N^2) DP
for the full placement distribution.

Validated: max error vs Monte Carlo N=100,000 is 0.14% at Q=20.
"""

from __future__ import annotations

import numpy as np


_SQRT2_INV = np.float32(1.0 / np.sqrt(2.0))

# Resolve erf once at import: scipy (fast C) > pure-numpy Abramowitz-Stegun approx
try:
    from scipy.special import erf as _erf
except ImportError:
    def _erf(x: np.ndarray) -> np.ndarray:
        """Abramowitz & Stegun approximation 7.1.26, max error 1.5e-7."""
        a = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * a)
        t2 = t * t
        y = 1.0 - (0.254829592 * t - 0.284496736 * t2
                    + 1.421413741 * t2 * t - 1.453152027 * t2 * t2
                    + 1.061405429 * t2 * t2 * t) * np.exp(-a * a)
        return np.copysign(y, x)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF. Uses scipy.erf if available, else fast numpy approx."""
    return (0.5 * (1.0 + _erf(x * _SQRT2_INV))).astype(np.float32)


# Precompute Gauss-Hermite nodes/weights for Q=20 and Q=10
def _make_gh(q: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(q)
    nodes = (nodes * np.sqrt(2.0)).astype(np.float32)
    weights = (weights / np.sqrt(np.pi)).astype(np.float32)
    return nodes, weights

_GH20_NODES, _GH20_WEIGHTS = _make_gh(20)
_GH10_NODES, _GH10_WEIGHTS = _make_gh(10)


def _dp_loop_numpy(pwin_flat: np.ndarray, BQ: int, M: int, N: int) -> np.ndarray:
    """Pure-numpy DP fallback."""
    dp0 = np.zeros((BQ, N), dtype=np.float32)
    dp1 = np.empty((BQ, N), dtype=np.float32)
    dp0[:, 0] = 1.0
    for i in range(M):
        p = pwin_flat[:, i : i + 1]
        q = 1.0 - p
        j_end = i + 2
        dp1[:, 0] = dp0[:, 0] * q[:, 0]
        dp1[:, 1:j_end] = dp0[:, 1:j_end] * q + dp0[:, 0 : j_end - 1] * p
        if j_end < N:
            dp1[:, j_end:] = dp0[:, j_end:]
        dp0, dp1 = dp1, dp0
    return dp0


try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _dp_loop_numba(pwin_flat, BQ, M, N):
        dp = np.zeros((BQ, N), np.float32)
        dp[:, 0] = 1.0
        for i in range(M):
            new_dp = np.empty((BQ, N), np.float32)
            for b in prange(BQ):
                p = pwin_flat[b, i]
                q = 1.0 - p
                new_dp[b, 0] = dp[b, 0] * q
                for j in range(1, i + 2):
                    new_dp[b, j] = dp[b, j] * q + dp[b, j - 1] * p
                for j in range(i + 2, N):
                    new_dp[b, j] = dp[b, j]
            dp = new_dp
        return dp

    _dp_loop = _dp_loop_numba
except ImportError:
    _dp_loop = _dp_loop_numpy


def placement_distribution(
    trainee_scores: np.ndarray,
    npc_scores: np.ndarray,
    sigma: float = 0.025,
    fast: bool = False,
) -> dict[str, np.ndarray]:
    """Full placement distribution with shared trainee noise.

    Args:
        trainee_scores: (B,) float32
        npc_scores: (B, M) float32
        sigma: noise sigma (default 2.5%)
        fast: if True, use Q=10 quadrature (~0.1% win_prob error, 2x faster)

    Returns:
        dict with 'win_prob', 'top3_prob', 'top5_prob', 'expected_place',
        'placement_dist'.
    """
    B, M = npc_scores.shape
    N = M + 1

    gh_nodes = _GH10_NODES if fast else _GH20_NODES
    gh_weights = _GH10_WEIGHTS if fast else _GH20_WEIGHTS
    Q = len(gh_nodes)

    # Trainee score at each quadrature node: T * (1 + σ * z)
    t_noisy = trainee_scores[:, np.newaxis] * (
        1.0 + sigma * gh_nodes[np.newaxis, :]
    )

    # P(beat NPC_i | z_t): (B, Q, M)
    t_exp = t_noisy[:, :, np.newaxis]
    npc_exp = npc_scores[:, np.newaxis, :]
    npc_std = sigma * npc_exp + 1e-10
    pwin = _normal_cdf((t_exp - npc_exp) / npc_std)

    # Flatten B×Q into single batch dimension for DP
    BQ = B * Q
    pwin_flat = np.ascontiguousarray(pwin.reshape(BQ, M))

    # DP over M opponents (numba-accelerated if available, else numpy)
    dp = _dp_loop(pwin_flat, BQ, M, N).reshape(B, Q, N)
    dist = np.einsum("bqk,q->bk", dp, gh_weights)

    # Reverse: index 0 = beat all M = 1st place
    dist = dist[:, ::-1].copy()

    # Normalize
    dist /= np.maximum(dist.sum(axis=1, keepdims=True), 1e-10)

    # Summary stats
    positions = np.arange(1, N + 1, dtype=np.float32)
    expected = np.sum(dist * positions[np.newaxis, :], axis=1)

    return {
        "win_prob": dist[:, 0],
        "top3_prob": np.sum(dist[:, :3], axis=1),
        "top5_prob": np.sum(dist[:, :min(5, N)], axis=1),
        "expected_place": expected,
        "placement_dist": dist,
    }
