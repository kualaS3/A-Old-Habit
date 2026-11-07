from blackjack.cards import Card, Rank, Suit
from blackjack.hand import Hand


def hand(*ranks: Rank) -> Hand:
    h = Hand()
    for r in ranks:
        h.add(Card(r, Suit.SPADES))
    return h


def test_soft_hand_counts_ace_as_eleven():
    h = hand(Rank.ACE, Rank.SIX)
    assert h.total == 17
    assert h.is_soft


def test_ace_drops_to_one_when_it_would_bust():
    h = hand(Rank.ACE, Rank.SIX, Rank.KING)
    assert h.total == 17
    assert not h.is_soft


def test_two_aces_are_soft_twelve():
    assert hand(Rank.ACE, Rank.ACE).total == 12


def test_blackjack_needs_exactly_two_cards():
    assert hand(Rank.ACE, Rank.QUEEN).is_blackjack
    assert not hand(Rank.SEVEN, Rank.SEVEN, Rank.SEVEN).is_blackjack


def test_bust_is_detected():
    assert hand(Rank.KING, Rank.QUEEN, Rank.TWO).is_bust
