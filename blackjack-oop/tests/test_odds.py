import pytest

from blackjack.cards import Card, Rank, Suit
from blackjack.hand import Hand
from blackjack.odds import Action, OddsCalculator
from blackjack.rules import Rules


def hand(*ranks: Rank) -> Hand:
    h = Hand()
    for r in ranks:
        h.add(Card(r, Suit.SPADES))
    return h


def card(rank: Rank) -> Card:
    return Card(rank, Suit.HEARTS)


def deck_minus(*seen: Rank) -> tuple[int, ...]:
    """A full single deck minus the given ranks, as blackjack-value counts."""
    counts = [4] * 9 + [16]
    for r in seen:
        counts[r.blackjack_value - 1] -= 1
    return tuple(counts)


@pytest.fixture
def calc() -> OddsCalculator:
    return OddsCalculator(Rules())


def test_probabilities_sum_to_one(calc):
    a = calc.analyze(
        hand(Rank.SEVEN, Rank.FIVE), card(Rank.KING), deck_minus(Rank.SEVEN, Rank.FIVE, Rank.KING)
    )
    for action in Action:
        o = a.odds(action)
        assert o.win + o.push + o.lose == pytest.approx(1.0)


def test_deterministic_shoe(calc):
    # Only two tens are unseen. Player has 10 vs dealer 7.
    # Stand: dealer's hole card is a 10 -> 17, so the player's 10 loses.
    # Hit: player draws a 10 -> 20, dealer gets the other 10 -> 17, player wins.
    unseen = (0,) * 9 + (2,)
    a = calc.analyze(hand(Rank.SIX, Rank.FOUR), card(Rank.SEVEN), unseen)
    assert a.odds(Action.STAND).ev == pytest.approx(-1.0)
    assert a.odds(Action.HIT).ev == pytest.approx(1.0)
    assert a.best() is Action.HIT


def test_obvious_decisions(calc):
    twenty = calc.analyze(
        hand(Rank.KING, Rank.QUEEN), card(Rank.SIX), deck_minus(Rank.KING, Rank.QUEEN, Rank.SIX)
    )
    assert twenty.best() is Action.STAND

    eight = calc.analyze(
        hand(Rank.FIVE, Rank.THREE), card(Rank.SIX), deck_minus(Rank.FIVE, Rank.THREE, Rank.SIX)
    )
    assert eight.best() is Action.HIT


def test_known_values_cross_checked_against_monte_carlo(calc):
    # 7+5 vs dealer King, unseen = deck minus those three cards.
    a = calc.analyze(
        hand(Rank.SEVEN, Rank.FIVE), card(Rank.KING), deck_minus(Rank.SEVEN, Rank.FIVE, Rank.KING)
    )
    stand = a.odds(Action.STAND)
    assert stand.win == pytest.approx(0.2175, abs=1e-3)
    assert stand.lose == pytest.approx(0.7825, abs=1e-3)


def test_cannot_hit_when_nothing_is_unseen(calc):
    a = calc.analyze(hand(Rank.KING, Rank.QUEEN), card(Rank.SIX), (0,) * 10)
    assert not a.offers(Action.HIT)
