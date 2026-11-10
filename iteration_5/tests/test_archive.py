"""The archived Python decomposition (blackjack/archive/split_sequential.py).

It is NOT imported by the game -- the live engine is the C port of this same
algorithm -- but it is kept to show the attempt, and it must still work: it is
the reference the C port was checked against. These tests run on small pools,
where the Python version takes about a second a spot; on a full deck it takes
minutes, which is why it is archived.

The archived entry point sets up the calculator's split caches itself (the
live OddsCalculator no longer carries them), but _hand_after called on its
own needs them set up by the caller, so the test does that.
"""
import pytest

from blackjack.archive import split_sequential as arch
from blackjack.cards import Card, Rank, Suit
from blackjack.hand import Hand
from blackjack.odds import OddsCalculator
from blackjack.rules import Rules

SMALL_POOL = (1, 1, 1, 1, 1, 1, 1, 1, 1, 4)  # one of each value, four tens


def hand(*ranks: Rank, **kwargs) -> Hand:
    h = Hand(**kwargs)
    for r in ranks:
        h.add(Card(r, Suit.SPADES))
    return h


def C(rank: Rank, suit: Suit = Suit.HEARTS) -> Card:
    return Card(rank, suit)


def _arch_calc(upcard: Card, rules: Rules | None = None) -> OddsCalculator:
    """A calculator with the upcard and the split caches the archive needs."""
    calc = OddsCalculator(rules or Rules())
    calc._up_value = upcard.rank.blackjack_value
    calc._up_is_ace = upcard.rank.is_ace
    calc._split_cache = {}
    calc._remaining_cache = {}
    calc._hand_cache = {}
    calc._split_dealer_cache = {}
    return calc


def test_hand_after_agrees_with_optimal_play():
    pool = (2, 2, 2, 2, 2, 2, 2, 2, 2, 8)
    calc = OddsCalculator(Rules())
    a = calc.analyze(hand(Rank.EIGHT, Rank.TWO), C(Rank.SIX), pool)
    # Analyze first: _hand_after reads the upcard set up by it; the caches
    # are the archive's own, set up here.
    calc._split_cache = {}
    calc._remaining_cache = {}
    calc._hand_cache = {}
    calc._split_dealer_cache = {}
    own, _ = arch._hand_after(calc, pool, 10, False, 2, False)
    assert own == pytest.approx(a.odds(a.best()).ev)


def test_archive_split_odds_on_the_small_pool():
    # The one-split values (the default cap): the C port reproduces them
    # exactly, and the joint agrees on them too -- see test_split_engine.py.
    for single, up_rank, want in (
        (8, Rank.SIX, 0.3522),
        (1, Rank.FIVE, 0.6709),
        (8, Rank.TEN, -0.4364),
    ):
        calc = _arch_calc(C(up_rank))
        odds = arch._split_odds(calc, SMALL_POOL, single, single == 1)
        assert odds.ev == pytest.approx(want, abs=0.001)


def test_archive_split_odds_at_three_splits():
    # With resplits allowed the 8,8 vs 6 value rises by the resplit, and the
    # 8,8 vs 10 value falls: the resplit hands are losers against a ten, and
    # the archive values them the same way the C port does.
    calc = _arch_calc(C(Rank.SIX), Rules(max_splits=3))
    assert arch._split_odds(calc, SMALL_POOL, 8, False).ev == pytest.approx(
        0.3868, abs=0.001
    )
    calc = _arch_calc(C(Rank.TEN), Rules(max_splits=3))
    assert arch._split_odds(calc, SMALL_POOL, 8, False).ev == pytest.approx(
        -0.4280, abs=0.001
    )
