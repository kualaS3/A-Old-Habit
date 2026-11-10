import random

from blackjack.bankroll import Bankroll
from blackjack.counting import CardCounter, HiLo
from blackjack.player import Player
from blackjack.round import Outcome, Round
from blackjack.rules import Rules
from blackjack.shoe import Shoe


def make_round(shoe: Shoe) -> Round:
    return Round(shoe, CardCounter(HiLo()), Rules(), Player(Bankroll()))


def test_hole_card_is_counted_only_when_revealed():
    shoe = Shoe(1, random.Random(5))
    counter = CardCounter(HiLo())
    rnd = Round(shoe, counter, Rules(), Player(Bankroll()))
    rnd.deal(1)
    # 3 visible cards so far: both player cards and the dealer upcard.
    expected = sum(HiLo().value(c.rank) for c in (*rnd.hands[0].cards, rnd.dealer_upcard))
    assert counter.running_count == expected
    rnd.dealer_play()
    assert rnd.hole_revealed


def test_unseen_counts_include_hole_card_and_total_correctly():
    shoe = Shoe(1, random.Random(6))
    rnd = make_round(shoe)
    rnd.deal(1)
    # 51 cards in the shoe after the burn, minus the 3 visible ones dealt
    # (the hole card is still unseen to the player).
    assert sum(rnd.unseen_counts()) == 48


def test_a_full_round_always_produces_an_outcome():
    for seed in range(200):
        shoe = Shoe(1, random.Random(seed))
        rnd = make_round(shoe)
        rnd.deal(1)
        rnd.dealer_play()
        assert isinstance(rnd.hand_outcome(rnd.hands[0]), Outcome)
