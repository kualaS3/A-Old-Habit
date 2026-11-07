import random

from blackjack.counting import CardCounter, HiLo
from blackjack.round import Outcome, Round
from blackjack.rules import Rules
from blackjack.shoe import Shoe


def test_hole_card_is_counted_only_when_revealed():
    shoe = Shoe(1, random.Random(5))
    counter = CardCounter(HiLo())
    rnd = Round(shoe, counter, Rules())
    rnd.deal()
    # 3 visible cards so far: both player cards and the dealer upcard.
    expected = sum(HiLo().value(c.rank) for c in (*rnd.player.cards, rnd.dealer_upcard))
    assert counter.running_count == expected
    rnd.dealer_play()
    assert rnd.hole_revealed


def test_unseen_counts_include_hole_card_and_total_correctly():
    shoe = Shoe(1, random.Random(6))
    rnd = Round(shoe, CardCounter(HiLo()), Rules())
    rnd.deal()
    # 52 cards minus the 3 visible ones (hole card is still unseen to the player).
    assert sum(rnd.unseen_counts()) == 49


def test_a_full_round_always_produces_an_outcome():
    for seed in range(200):
        shoe = Shoe(1, random.Random(seed))
        rnd = Round(shoe, CardCounter(HiLo()), Rules())
        rnd.deal()
        rnd.dealer_play()
        assert isinstance(rnd.outcome, Outcome)
