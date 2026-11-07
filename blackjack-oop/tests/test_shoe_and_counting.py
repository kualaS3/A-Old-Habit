import random

from blackjack.cards import Rank
from blackjack.counting import CardCounter, HiLo
from blackjack.shoe import Shoe


def test_single_deck_composition():
    shoe = Shoe(1, random.Random(1))
    assert shoe.cards_remaining == 52
    counts = shoe.rank_counts()
    assert counts[:9] == (4,) * 9
    assert counts[9] == 16


def test_drawing_removes_cards():
    shoe = Shoe(1, random.Random(1))
    shoe.draw()
    assert shoe.cards_remaining == 51


def test_hi_lo_sums_to_zero_over_a_full_deck():
    shoe = Shoe(1, random.Random(2))
    counter = CardCounter(HiLo())
    while shoe.cards_remaining:
        counter.observe(shoe.draw())
    assert counter.running_count == 0


def test_hi_lo_values():
    s = HiLo()
    assert s.value(Rank.FIVE) == 1
    assert s.value(Rank.EIGHT) == 0
    assert s.value(Rank.ACE) == -1
    assert s.value(Rank.JACK) == -1
