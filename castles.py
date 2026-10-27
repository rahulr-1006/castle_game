"""Castle allocation game: rules, player types, and solvers.

Both games share one board: 100 soldiers spread over castles worth 1..5 points.
Game 1 scores an allocation head to head against every other applicant.
Game 2 drops it into random groups of four, ranked by castle points, paying 9/4/2/0.
"""

import functools
import itertools
import math
from typing import NamedTuple

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


# ---------------------------------------------------------------------------
# Allocation space
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def all_allocations():
    """Every legal allocation, C(104, 4) = 4,598,126 rows, for exhaustive checks.

    Stars and bars: 4 bars among 104 slots split the other 100 slots into 5 castles.
    """
    slots = TOTAL_SOLDIERS + N_CASTLES - 1
    bars = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(slots), N_CASTLES - 1)),
        dtype=np.int16,
        count=math.comb(slots, N_CASTLES - 1) * (N_CASTLES - 1),
    ).reshape(-1, N_CASTLES - 1)
    edges = np.pad(bars, ((0, 0), (1, 1)), constant_values=(-1, slots))
    out = (np.diff(edges, axis=1) - 1).astype(np.int8)
    out.flags.writeable = False  # cached and shared between callers
    return out


def round_to_total(shares, step, total=TOTAL_SOLDIERS):
    """Round rows of shares (each summing to 1) to multiples of `step` adding up to `total`.

    Largest-remainder rounding, so the result is always a legal allocation.
    `total` may be a scalar or one value per row; it must be a multiple of `step`.
    """
    shares = np.atleast_2d(shares)
    units = np.broadcast_to(np.asarray(total) // step, (len(shares),))
    raw = shares * units[:, None]
    base = np.floor(raw).astype(np.int64)
    short = units - base.sum(axis=1)
    remainder_rank = np.argsort(np.argsort(base - raw, axis=1), axis=1)
    return (base + (remainder_rank < short[:, None])) * step


# ---------------------------------------------------------------------------
# Game 1 analysis
#
# Fact 1: Game 1 is separable. Against a field, sending s soldiers to castle i is
# worth w[i, s] = v_i * (P(opponent sends < s) + 0.5 * P(opponent sends s)) in
# expectation, and an allocation scores sum_i w[i, a_i]. Only each castle's
# marginal distribution in the field matters, never how opponents correlate
# castles. So a whole field reduces to a (5, 101) table of marginals.
# ---------------------------------------------------------------------------

def marginals(allocs, weights=None):
    """Per-castle distribution of soldier counts in a field, shape (5, 101)."""
    a = np.asarray(allocs).reshape(-1, N_CASTLES)
    weights = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
    counts = np.stack([
        np.bincount(a[:, i], weights=weights, minlength=TOTAL_SOLDIERS + 1)
        for i in range(N_CASTLES)
    ])
    return counts / weights.sum()


def win_value(marg):
    """w[i, s]: expected Game 1 points from sending s soldiers to castle i."""
    below = np.cumsum(marg, axis=1) - marg
    return VALUES[:, None] * (below + 0.5 * marg)


def expected_score(w, allocs):
    """Expected Game 1 score of each allocation in `allocs` given a win-value table."""
    a = np.asarray(allocs)
    return w[np.arange(w.shape[0]), a].sum(axis=-1)


@functools.lru_cache(maxsize=4)
def _knapsack_index(size):
    # rest[t, s] = t - s: soldiers left for the earlier castles when this castle
    # gets s of the t placed so far. Clipped to 0 where s > t and masked out by `ok`.
    t, s = np.ogrid[:size, :size]
    return np.maximum(t - s, 0), s <= t


def best_response(w):
    """Exact maximizer of sum_i w[i, a_i] over allocations; returns (alloc, value).

    Knapsack DP over castles: best[t] is the top value from the castles seen so
    far using exactly t soldiers. Works for any table shape (castles, soldiers + 1),
    which is what lets the tests check it against brute force on small boards.
    """
    n_castles, size = w.shape
    rest, ok = _knapsack_index(size)
    best = w[0]
    picks = []
    for row in w[1:]:
        table = np.where(ok, best[rest] + row, -np.inf)
        pick = table.argmax(axis=1)
        best = table[np.arange(size), pick]
        picks.append(pick)
    alloc = np.empty(n_castles, dtype=np.int64)
    left = size - 1
    for i in range(n_castles - 1, 0, -1):
        alloc[i] = picks[i - 1][left]
        left -= alloc[i]
    alloc[0] = left
    return alloc, float(best[-1])


def _logsumexp(x, axis=-1):
    top = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(top, axis) + np.log(np.sum(np.exp(x - top), axis=axis))


def _log_partition(logits):
    """Same DP as best_response with (max, +) swapped for (logsumexp, +).

    out[k][t] = log of sum over placements of exactly t soldiers on castles 0..k
    of exp(sum of their logits).
    """
    rest, ok = _knapsack_index(logits.shape[1])
    out = [logits[0]]
    for row in logits[1:]:
        out.append(_logsumexp(np.where(ok, out[-1][rest] + row, -np.inf)))
    return out


def logit_marginals(w, lam):
    """Exact per-castle marginals of the noisy best response p(a) ∝ exp(lam * score(a)).

    lam is an inverse temperature in 1/points: lam = 0 is uniform over all
    allocations, lam -> inf approaches the exact best response.
    """
    logits = lam * np.asarray(w, dtype=float)
    out = np.empty_like(logits)
    for i in range(len(logits)):
        others = _log_partition(np.delete(logits, i, axis=0))[-1]
        # Sending s here leaves exactly (soldiers - s) for the other castles.
        log_p = logits[i] + others[::-1]
        out[i] = np.exp(log_p - _logsumexp(log_p))
    return out


def logit_sample(w, lam, n, rng):
    """Draw n allocations from the noisy best response p(a) ∝ exp(lam * score(a)).

    Exact, without enumerating allocations: run the log-partition DP forward,
    then sample castles backward, each conditional on the soldiers still unplaced.
    """
    logits = lam * np.asarray(w, dtype=float)
    n_castles, size = logits.shape
    rest, ok = _knapsack_index(size)
    log_z = _log_partition(logits)
    out = np.empty((n, n_castles), dtype=np.int64)
    left = np.full(n, size - 1)
    for i in range(n_castles - 1, 0, -1):
        # Row t: P(castle i gets s | t soldiers left for castles 0..i).
        log_p = np.where(ok, logits[i] + log_z[i - 1][rest], -np.inf)
        p = np.exp(log_p - _logsumexp(log_p)[:, None])
        cdf = np.where(ok, np.cumsum(p, axis=1), 1.0)
        # Inverse-CDF sampling for every row at once: offsetting row t by 2t keeps
        # the flattened table sorted, so one searchsorted serves all samples.
        keys = (cdf + 2 * np.arange(size)[:, None]).ravel()
        u = rng.random(n)
        out[:, i] = np.searchsorted(keys, u + 2 * left, side="right") - left * size
        left = left - out[:, i]
    out[:, 0] = left
    return out


def noisy_response(w, lam):
    """Marginals of a (noisy) best response; lam = inf means the exact best response."""
    if np.isinf(lam):
        return marginals(best_response(w)[0])
    return logit_marginals(w, lam)


class FictitiousPlay(NamedTuple):
    marginals: np.ndarray       # (5, 101) marginals of the average strategy
    plays: np.ndarray           # (n_iter + 1, 5) the start plus every best response
    exploitability: np.ndarray  # (n_iter,) best-response gain over 7.5 at each round


def fictitious_play(n_iter, start=(20, 20, 20, 20, 20)):
    """Symmetric fictitious play for Game 1 (fact 2).

    Each round best-responds to the average of all earlier rounds. Game 1 is
    symmetric and constant-sum (value 7.5 each), so the average is an
    epsilon-equilibrium with epsilon = (best response score - 7.5), and
    Robinson (1951) guarantees epsilon -> 0. Only marginals are needed per round.
    """
    plays = [validate(start)]
    marg = marginals(plays[0])
    gaps = np.empty(n_iter)
    for t in range(n_iter):
        alloc, value = best_response(win_value(marg))
        gaps[t] = value - TOTAL_POINTS / 2
        plays.append(alloc)
        marg += (marginals(alloc) - marg) / (t + 2)
    return FictitiousPlay(marg, np.array(plays), gaps)


def mean_allocation(marg):
    """Average soldiers per castle under `marg`, rounded to a legal allocation."""
    mean = marg @ np.arange(marg.shape[1])
    return round_to_total(mean / mean.sum(), 1)[0]


# ---------------------------------------------------------------------------
# Player types
#
# Level 0 players follow habits, not reasoning. Strategic players (cognitive
# hierarchy levels, meta exploiters) play a noisy best response to what they
# believe the field does. Equilibrium players sample fictitious play's average.
# ---------------------------------------------------------------------------

EQUAL_SPLIT = np.array([20, 20, 20, 20, 20])
PROPORTIONAL = np.array([7, 13, 20, 27, 33])  # 100 * v / 15, rounded


def level0_round_numbers(n, rng):
    """A loose 'more for bigger castles' split, rounded to multiples of 5 or 10.

    Mean shares 10/15/20/25/30%; the Dirichlet concentration (12) gives each castle a
    spread of roughly +-10 soldiers, so outputs range from 20-20-20-20-20 to 0-0-20-30-50.
    """
    shares = rng.dirichlet(12 * np.array([0.10, 0.15, 0.20, 0.25, 0.30]), size=n)
    tens = rng.random(n) < 0.5
    return np.where(tens[:, None], round_to_total(shares, 10), round_to_total(shares, 5))


def level0_stack_five(n, rng):
    """'Stack the 5': 40, 50 or 60 on castle 5, the rest roughly by value in 5s."""
    top = rng.choice([40, 50, 60], p=[0.4, 0.4, 0.2], size=n)
    rest = round_to_total(rng.dirichlet(4 * VALUES[:4] / 10, size=n), 5, TOTAL_SOLDIERS - top)
    return np.column_stack([rest, top])


def level0_uniform(n, rng):
    """Uniform over all 4.6M allocations (the textbook level-0 assumption)."""
    return logit_sample(np.zeros((N_CASTLES, TOTAL_SOLDIERS + 1)), 0.0, n, rng)


LEVEL0_HABITS = {
    "equal": lambda n, rng: np.tile(EQUAL_SPLIT, (n, 1)),
    "proportional": lambda n, rng: np.tile(PROPORTIONAL, (n, 1)),
    "round": level0_round_numbers,
    "stack5": level0_stack_five,
    "uniform": level0_uniform,
}

# Two level-0 populations: humans with round-number habits, and the textbook
# uniform-random level 0. Shares in "focal" are judgment calls (see README).
LEVEL0_FLAVORS = {
    "focal": {"equal": 0.15, "proportional": 0.15, "round": 0.45, "stack5": 0.15, "uniform": 0.10},
    "random": {"uniform": 1.0},
}


def level0_sample(flavor, n, rng):
    """n allocations from a level-0 flavor, habits mixed in their listed shares."""
    habits = LEVEL0_FLAVORS[flavor]
    counts = rng.multinomial(n, list(habits.values()))
    out = np.concatenate([LEVEL0_HABITS[h](k, rng) for h, k in zip(habits, counts)])
    return out[rng.permutation(n)]


def poisson_levels(tau, max_level):
    """Shares of thinking levels 0..max_level, Poisson(tau), tail pooled into the top level."""
    k = np.arange(max_level + 1)
    shares = np.exp(-tau) * tau**k / np.array([math.factorial(i) for i in k])
    shares[-1] += 1 - shares.sum()
    return shares


def cognitive_hierarchy(level0_marg, tau, lam, max_level):
    """Game 1 marginals of levels 0..max_level, shape (max_level + 1, 5, 101).

    Level k noisily best-responds to levels 0..k-1 mixed in their Poisson(tau)
    proportions (Camerer, Ho and Chong 2004). The top level's share pools the tail,
    but its beliefs use only the true Poisson weights of the levels below it.
    """
    shares = poisson_levels(tau, max_level)
    levels = [level0_marg]
    for k in range(1, max_level + 1):
        belief = np.tensordot(shares[:k] / shares[:k].sum(), np.array(levels), axes=1)
        levels.append(noisy_response(win_value(belief), lam))
    return np.array(levels)
