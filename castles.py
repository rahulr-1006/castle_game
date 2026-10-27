"""Castle allocation game: rules, player types, and solvers.

Both games share one board: 100 soldiers spread over castles worth 1..5 points.
Game 1 scores an allocation head to head against every other applicant.
Game 2 drops it into random groups of four, ranked by castle points, paying 9/4/2/0.
"""

import numpy as np

VALUES = np.array([1, 2, 3, 4, 5], dtype=np.int64)
N_CASTLES = len(VALUES)
TOTAL_SOLDIERS = 100
TOTAL_POINTS = int(VALUES.sum())  # 15

GROUP_SIZE = 4
PAYOUTS = np.array([9.0, 4.0, 2.0, 0.0])

# Game 2 castle points are kept in twelfths so castles split 2, 3 or 4 ways stay
# exact integers. Ranking compares totals for equality, and float thirds would
# turn genuine ties into arbitrary 1st/2nd orderings.
SCALE = 12
_PAYOUT_CUMSUM = np.concatenate([[0.0], np.cumsum(PAYOUTS)])


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def validate(allocs):
    """Return `allocs` as an int64 array, raising if any allocation is illegal.

    Accepts a single allocation (shape (5,)) or any batch (shape (..., 5)).
    """
    a = np.asarray(allocs)
    if a.ndim == 0 or a.shape[-1] != N_CASTLES:
        raise ValueError(f"allocation must have {N_CASTLES} entries, got shape {a.shape}")
    if not np.issubdtype(a.dtype, np.integer):
        if not np.issubdtype(a.dtype, np.floating) or not np.all(a == np.round(a)):
            raise ValueError("soldier counts must be integers")
    a = a.astype(np.int64)
    if (a < 0).any():
        raise ValueError("soldier counts must be non-negative")
    if (a.sum(axis=-1) != TOTAL_SOLDIERS).any():
        raise ValueError(f"each allocation must sum to {TOTAL_SOLDIERS}")
    return a


def game1_points(mine, opponents):
    """Castle points `mine` earns against each opponent in Game 1 (ties split).

    Returns a float array with one entry per opponent, each in [0, 15].
    """
    mine = validate(mine)
    opp = validate(opponents).reshape(-1, N_CASTLES)
    share = (mine > opp) + 0.5 * (mine == opp)
    return share @ VALUES


def game1_score(mine, field):
    """Game 1 score: average castle points against every allocation in `field`."""
    return float(game1_points(mine, field).mean())


def game2_castle_points(groups):
    """Castle points per player in each Game 2 group, in twelfths of a point.

    groups: int array of shape (..., GROUP_SIZE, 5). Returns shape (..., GROUP_SIZE).
    Every castle is awarded, even when all players send zero: a four-way tie at
    zero splits it four ways like any other tie.
    """
    g = np.asarray(groups)
    if g.ndim < 2 or g.shape[-2:] != (GROUP_SIZE, N_CASTLES):
        raise ValueError(f"groups must have shape (..., {GROUP_SIZE}, {N_CASTLES})")
    top = g.max(axis=-2, keepdims=True)
    winners = g == top
    n_winners = winners.sum(axis=-2, keepdims=True)
    # 12 * value is divisible by any winner count in 1..4, so this is exact.
    return (winners * (SCALE * VALUES) // n_winners).sum(axis=-1)


def rank_payouts(points):
    """Payout per player given each player's total points, shape (..., GROUP_SIZE).

    Tied players split the payouts of the places they jointly occupy: a player
    with k opponents strictly ahead and t players (itself included) on its score
    gets mean(PAYOUTS[k : k + t]).
    """
    p = np.asarray(points)
    me, other = p[..., :, None], p[..., None, :]
    ahead = (other > me).sum(axis=-1)
    tied = (other == me).sum(axis=-1)
    return (_PAYOUT_CUMSUM[ahead + tied] - _PAYOUT_CUMSUM[ahead]) / tied


def game2_payouts(groups):
    """Payout per player in each Game 2 group, shape (..., GROUP_SIZE)."""
    return rank_payouts(game2_castle_points(groups))
