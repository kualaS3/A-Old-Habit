"""The C split engines (blackjack/split_exact.c), via the loader.

Two engines: split_decomp_ev is the live one (the sequential decomposition,
near-exact), split_ev is the joint recursion (exact, the verification tool).
Both take the same arguments and return the total EV of splitting a pair,
per unit staked. All of these tests skip when the helper is not built
(run python build.py).

The numbers below are the measured values, not aspirations: the small-pool
values are the one-split (default cap) values, the full-deck values are the
same spots with a full shoe unseen, and the three-split small-pool values
are the resplit cases the joint can still verify.
"""
import pytest

from blackjack import split_exact
from blackjack.rules import Rules

SMALL_POOL = (1, 1, 1, 1, 1, 1, 1, 1, 1, 4)  # one of each value, four tens

# Full deck minus the pair and the upcard (49 unseen cards).
FULL_88_VS_6 = (4, 4, 4, 4, 4, 3, 4, 2, 4, 16)
FULL_AA_VS_5 = (2, 4, 4, 4, 3, 4, 4, 4, 4, 16)
FULL_88_VS_10 = (4, 4, 4, 4, 4, 4, 4, 2, 4, 15)


@pytest.fixture(autouse=True)
def _needs_c():
    if not split_exact.available():
        pytest.skip("C helper not built (run python build.py)")


# ---- the decomposition (live engine): small pool, one split ----------------


def test_decomp_small_pool_one_split():
    for single, up, excl, want in (
        (8, 6, None, 0.3522),
        (1, 5, None, 0.6709),
        (8, 10, 1, -0.4364),
    ):
        value = split_exact.split_decomp_ev(
            SMALL_POOL, single, single == 1, up, up == 1, excl, Rules()
        )
        assert value == pytest.approx(want, abs=0.001)


def test_decomp_small_pool_three_splits():
    # With resplits the 8,8 vs 6 value rises and the 8,8 vs 10 value falls.
    assert split_exact.split_decomp_ev(
        SMALL_POOL, 8, False, 6, False, None, Rules(max_splits=3)
    ) == pytest.approx(0.3868, abs=0.001)
    assert split_exact.split_decomp_ev(
        SMALL_POOL, 1, True, 5, False, None, Rules(max_splits=3)
    ) == pytest.approx(0.6709, abs=0.001)
    assert split_exact.split_decomp_ev(
        SMALL_POOL, 8, False, 10, False, 1, Rules(max_splits=3)
    ) == pytest.approx(-0.4280, abs=0.001)


# ---- the joint (exact): small pool, three splits ----------------------------


def test_joint_small_pool_three_splits():
    # The exact values the decomposition is checked against. The 8,8 vs 6
    # gap (joint +0.4440, decomposition +0.3868) is the one spot where the
    # second-order coupling shows, on a pool where one card is a twelfth of
    # the shoe. At full-deck scale the gap is far smaller.
    assert split_exact.split_ev(
        SMALL_POOL, 8, False, 6, False, None, Rules(max_splits=3)
    ) == pytest.approx(0.4440, abs=0.001)
    assert split_exact.split_ev(
        SMALL_POOL, 1, True, 5, False, None, Rules(max_splits=3)
    ) == pytest.approx(0.8138, abs=0.001)
    assert split_exact.split_ev(
        SMALL_POOL, 8, False, 10, False, 1, Rules(max_splits=3)
    ) == pytest.approx(-0.4193, abs=0.001)


# ---- full deck, one split: the decomposition matches the joint exactly ------


def test_decomp_matches_joint_on_a_full_deck():
    # At one split the coupling the decomposition ignores is worth nothing
    # on these spots: both engines give the same value to four decimals.
    # (8,8 vs 6 takes ~3s through the decomposition, ~1s through the joint.)
    for counts, single, up, excl, want in (
        (FULL_88_VS_6, 8, 6, None, 0.3420),
        (FULL_AA_VS_5, 1, 5, None, 0.7322),
        (FULL_88_VS_10, 8, 10, 1, -0.4521),
    ):
        decomp = split_exact.split_decomp_ev(
            counts, single, single == 1, up, up == 1, excl, Rules()
        )
        joint = split_exact.split_ev(
            counts, single, single == 1, up, up == 1, excl, Rules()
        )
        assert decomp == pytest.approx(want, abs=0.001)
        assert joint == pytest.approx(want, abs=0.001)
        assert decomp == pytest.approx(joint, abs=0.0005)


def test_decomp_state_counts_are_small_at_one_split():
    # The whole point of the decomposition: its state space stays in the
    # hundreds of thousands at one split, where the joint is already at
    # 389k and climbing steeply. (The dealer cache is shared by both engines.)
    split_exact.split_decomp_ev(
        FULL_88_VS_6, 8, False, 6, False, None, Rules()
    )
    assert 0 < split_exact.decomp_states() < 1_000_000
    assert split_exact.decomp_hand_states() > 0
    assert split_exact.decomp_tail_states() > 0
    assert split_exact.decomp_first_states() > 0
    assert split_exact.dealer_cache_hit_rate() > 0.5


def test_joint_state_count_at_one_split():
    split_exact.split_ev(FULL_88_VS_6, 8, False, 6, False, None, Rules())
    assert split_exact.joint_states() == pytest.approx(389226, rel=0.01)
    assert split_exact.dealer_cache_hit_rate() > 0.5
