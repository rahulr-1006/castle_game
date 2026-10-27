from fractions import Fraction

import numpy as np
import pytest

import castles as c


# ---------------------------------------------------------------------------
# Slow, obviously-correct reference implementations of the written rules.
# The vectorized code in castles.py is checked against these on random inputs.
# ---------------------------------------------------------------------------

def ref_castle_points(group):
    pts = [Fraction(0)] * len(group)
    for castle, value in enumerate(c.VALUES):
        sent = [alloc[castle] for alloc in group]
        winners = [i for i, s in enumerate(sent) if s == max(sent)]
        for i in winners:
            pts[i] += Fraction(int(value), len(winners))
    return pts


def ref_rank_payouts(points):
    order = sorted(points, reverse=True)
    out = []
    for p in points:
        places = [i for i, q in enumerate(order) if q == p]
        out.append(sum(c.PAYOUTS[i] for i in places) / len(places))
    return out


def random_allocs(rng, n, max_cluster=None):
    """Random legal allocations; `max_cluster` coarsens them so ties are common."""
    if max_cluster is None:
        cuts = np.sort(rng.integers(0, c.TOTAL_SOLDIERS + 1, size=(n, c.N_CASTLES - 1)), axis=1)
        edges = np.concatenate([np.zeros((n, 1), int), cuts, np.full((n, 1), c.TOTAL_SOLDIERS)], axis=1)
        return np.diff(edges, axis=1)
    # Multiples of 100 / max_cluster, so equal castle counts show up constantly.
    units = random_allocs(rng, n) * max_cluster // c.TOTAL_SOLDIERS
    units[:, -1] += max_cluster - units.sum(axis=1)
    return units * (c.TOTAL_SOLDIERS // max_cluster)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    [20, 20, 20, 20, 19],          # sums to 99
    [20, 20, 20, 20, 21],          # sums to 101
    [-1, 21, 20, 20, 40],          # negative
    [20.5, 19.5, 20, 20, 20],      # non-integer
    [25, 25, 25, 25],              # wrong length
])
def test_validate_rejects_illegal(bad):
    with pytest.raises(ValueError):
        c.validate(bad)


def test_validate_accepts_integral_floats_and_batches():
    out = c.validate([[20.0, 20, 20, 20, 20], [0, 0, 0, 0, 100]])
    assert out.dtype == np.int64 and out.shape == (2, 5)


# ---------------------------------------------------------------------------
# Game 1
# ---------------------------------------------------------------------------

def test_game1_sweep_and_mirror():
    a = [10, 15, 20, 25, 30]
    b = [9, 14, 19, 24, 34]  # a wins castles 1-4, loses castle 5
    assert c.game1_points(a, b)[0] == 10
    assert c.game1_points(a, a)[0] == 7.5


def test_game1_castle_tie_splits_value():
    a = [20, 20, 20, 20, 20]
    b = [20, 10, 30, 40, 0]  # tie on 1, win 2, lose 3, lose 4, win 5
    assert c.game1_points(a, b)[0] == 0.5 + 2 + 5


def test_game1_is_constant_sum():
    rng = np.random.default_rng(0)
    allocs = random_allocs(rng, 400, max_cluster=10)
    for a, b in zip(allocs[::2], allocs[1::2]):
        assert c.game1_points(a, b)[0] + c.game1_points(b, a)[0] == c.TOTAL_POINTS


def test_game1_score_averages_over_field():
    me = [20, 20, 20, 20, 20]
    field = [[20, 20, 20, 20, 20], [0, 0, 0, 0, 100], [100, 0, 0, 0, 0]]
    assert c.game1_score(me, field) == pytest.approx((7.5 + 10 + 14) / 3)


# ---------------------------------------------------------------------------
# Game 2: castle ties
# ---------------------------------------------------------------------------

def points(group):
    return c.game2_castle_points(np.array(group)) / c.SCALE


def test_game2_two_way_castle_tie():
    group = [[0, 0, 0, 0, 100],
             [0, 0, 0, 0, 100],
             [50, 50, 0, 0, 0],
             [0, 0, 50, 50, 0]]
    # Castle 5 splits 2.5 / 2.5. Castles 1, 2 to player 2; castles 3, 4 to player 3.
    np.testing.assert_allclose(points(group), [2.5, 2.5, 3, 7])


def test_game2_three_way_castle_tie_is_exact():
    group = [[0, 0, 0, 0, 100],
             [0, 0, 0, 0, 100],
             [0, 0, 0, 0, 100],
             [10, 20, 30, 40, 0]]
    pts = c.game2_castle_points(np.array(group))
    # 5/3 each: equal in integer twelfths, so the ranking sees an exact 3-way tie.
    assert pts[0] == pts[1] == pts[2] == 20
    assert pts[3] == 10 * c.SCALE
    np.testing.assert_allclose(c.rank_payouts(pts), [2, 2, 2, 9])


def test_game2_four_way_castle_tie_including_zeros():
    group = [[20, 20, 20, 20, 20]] * 4
    np.testing.assert_allclose(points(group), [3.75] * 4)
    # Nobody sends anything to castles 1-2: those still split four ways at zero.
    group = [[0, 0, 100, 0, 0], [0, 0, 0, 100, 0], [0, 0, 0, 0, 100], [0, 0, 0, 0, 100]]
    np.testing.assert_allclose(points(group), [0.75 + 3, 0.75 + 4, 0.75 + 2.5, 0.75 + 2.5])


def test_game2_castle_points_always_total_15():
    rng = np.random.default_rng(1)
    groups = random_allocs(rng, 4000, max_cluster=10).reshape(-1, 4, 5)
    assert (c.game2_castle_points(groups).sum(axis=-1) == c.TOTAL_POINTS * c.SCALE).all()


# ---------------------------------------------------------------------------
# Game 2: ranking ties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pts, expected", [
    ([4, 3, 2, 1], [9, 4, 2, 0]),        # strict order
    ([1, 4, 2, 3], [0, 9, 2, 4]),        # order independent of seat
    ([5, 5, 3, 2], [6.5, 6.5, 2, 0]),    # 2-way tie for 1st
    ([6, 4, 4, 1], [9, 3, 3, 0]),        # 2-way tie for 2nd
    ([6, 5, 2, 2], [9, 4, 1, 1]),        # 2-way tie for 3rd
    ([5, 5, 2, 2], [6.5, 6.5, 1, 1]),    # two pairs
    ([5, 5, 5, 0], [5, 5, 5, 0]),        # 3-way tie for 1st
    ([9, 2, 2, 2], [9, 2, 2, 2]),        # 3-way tie for 2nd
    ([3, 3, 3, 3], [3.75] * 4),          # 4-way tie
])
def test_rank_payouts_ties(pts, expected):
    np.testing.assert_allclose(c.rank_payouts(np.array(pts)), expected)


def test_payouts_always_total_15():
    rng = np.random.default_rng(2)
    groups = random_allocs(rng, 4000, max_cluster=5).reshape(-1, 4, 5)
    np.testing.assert_allclose(c.game2_payouts(groups).sum(axis=-1), c.PAYOUTS.sum())


# ---------------------------------------------------------------------------
# Vectorized rules == reference rules on random groups
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_cluster", [None, 20, 10, 5, 4])
def test_game2_matches_reference(max_cluster):
    rng = np.random.default_rng(3)
    groups = random_allocs(rng, 2000, max_cluster).reshape(-1, 4, 5)
    pts = c.game2_castle_points(groups)
    pay = c.game2_payouts(groups)
    for g, p, y in zip(groups, pts, pay):
        ref = ref_castle_points(g.tolist())
        assert [Fraction(int(x), c.SCALE) for x in p] == ref
        np.testing.assert_allclose(y, ref_rank_payouts(ref))


def test_game1_matches_reference():
    rng = np.random.default_rng(4)
    me = random_allocs(rng, 1, max_cluster=10)[0]
    field = random_allocs(rng, 500, max_cluster=10)
    got = c.game1_points(me, field)
    for opp, score in zip(field, got):
        # Game 1 is a 2-player Game 2 castle split; reuse the reference.
        assert score == float(ref_castle_points([me.tolist(), opp.tolist()])[0])
