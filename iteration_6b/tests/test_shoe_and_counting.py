import random

from blackjack.cards import Rank
from blackjack.counting import CardCounter, HiLo
from blackjack.shoe import Shoe


def test_single_deck_composition():
    # No burn card here, so the full deck is in the shoe.
    shoe = Shoe(1, random.Random(1), burn_card=False)
    assert shoe.cards_remaining == 52
    counts = shoe.rank_counts()
    assert counts[:9] == (4,) * 9
    assert counts[9] == 16


def test_drawing_removes_cards():
    shoe = Shoe(1, random.Random(1), burn_card=False)
    shoe.draw()
    assert shoe.cards_remaining == 51


def test_hi_lo_sums_to_zero_over_a_full_deck():
    shoe = Shoe(1, random.Random(2), burn_card=False)
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


# ---- iteration 5: the burn card ------------------------------------------------------


def test_burn_card_is_removed_unseen_and_uncounted():
    shoe = Shoe(1, random.Random(1))  # burn on by default
    assert shoe.burnt is not None
    assert shoe.cards_remaining == 51
    assert sum(shoe.rank_counts()) == 51
    # The burnt card is gone from the composition: its value has one fewer card.
    counts = shoe.rank_counts()
    assert counts[shoe.burnt.rank.blackjack_value - 1] == (
        3 if shoe.burnt.rank.blackjack_value < 10 else 15
    )


def test_burn_card_off_leaves_the_full_deck():
    shoe = Shoe(1, random.Random(1), burn_card=False)
    assert shoe.burnt is None
    assert shoe.cards_remaining == 52


def test_hi_lo_over_a_burnt_shoe_is_minus_the_burnt_card():
    shoe = Shoe(1, random.Random(2))
    counter = CardCounter(HiLo())
    while shoe.cards_remaining:
        counter.observe(shoe.draw())
    # Every card was counted except the burnt one, so the count is the
    # negative of the burnt card's Hi-Lo value.
    assert counter.running_count == -HiLo().value(shoe.burnt.rank)
