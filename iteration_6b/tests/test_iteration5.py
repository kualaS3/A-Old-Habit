"""Iteration 5: multi-deck shoes and the cut card, insurance and even money,
table money in config, 6:5, and the split-rule toggles."""
import random

import pytest

from blackjack import split_exact
from blackjack.bankroll import Bankroll
from blackjack.cards import Card, Rank, Suit
from blackjack.counting import CardCounter, HiLo
from blackjack.hand import Hand
from blackjack.odds import Action, OddsCalculator
from blackjack.player import Player
from blackjack.round import Outcome, Round
from blackjack.rules import Rules
from blackjack.shoe import Shoe


def _require_c() -> None:
    if not split_exact.available():
        pytest.skip("C helper not built (run python build.py)")

H = Suit.HEARTS
SMALL_POOL = (1, 1, 1, 1, 1, 1, 1, 1, 1, 4)  # one of each value, four tens


def hand(*ranks: Rank, **kwargs) -> Hand:
    h = Hand(**kwargs)
    for r in ranks:
        h.add(Card(r, Suit.SPADES))
    return h


def C(rank: Rank, suit: Suit = H) -> Card:
    return Card(rank, suit)


def stacked_shoe(draw_order: list[Card]) -> Shoe:
    """A shoe that deals exactly the given cards, in the given order."""
    shoe = Shoe(1, random.Random(0))
    shoe._cards = list(reversed(draw_order))
    return shoe


def stacked_round(
    draw_order: list[Card],
    rules: Rules | None = None,
    balance: float = 100,
    bet: int = 10,
) -> Round:
    rnd = Round(
        stacked_shoe(draw_order),
        CardCounter(HiLo()),
        rules or Rules(),
        Player(Bankroll(balance)),
    )
    rnd.deal(bet)
    return rnd


# ---- Rules: the whole table config -------------------------------------------------


def test_rules_defaults_are_the_table_timothy_described():
    r = Rules()
    assert r.num_decks == 1
    assert r.cut_cards is None
    assert r.cut_fraction == 0.75
    assert r.burn_card is True
    assert r.starting_balance == 1000
    assert r.min_bet == 10
    assert r.max_bet == 1000
    assert r.max_splits == 1  # one split, no resplits, by default (see README)
    assert r.double_after_split is True
    assert r.resplit_aces is True
    assert r.hit_split_aces is False
    assert r.insurance_allowed is True


def test_rules_rejects_impossible_tables():
    with pytest.raises(ValueError):
        Rules(num_decks=0)
    with pytest.raises(ValueError):
        Rules(num_decks=9)
    with pytest.raises(ValueError):
        Rules(cut_fraction=0.6)
    with pytest.raises(ValueError):
        Rules(cut_cards=50)  # a 1-deck shoe must leave 10 cards after the cut
    with pytest.raises(ValueError):
        Rules(cut_cards=0)
    with pytest.raises(ValueError):
        Rules(min_bet=0)
    with pytest.raises(ValueError):
        Rules(max_bet=5)  # below the minimum bet
    with pytest.raises(ValueError):
        Rules(starting_balance=5)  # cannot cover one minimum bet
    with pytest.raises(ValueError):
        Rules(blackjack_payout=(5, 4))
    with pytest.raises(ValueError):
        Rules(max_splits=-1)


def test_rules_accepts_the_full_config_range():
    assert Rules(num_decks=8).total_cards == 416
    assert Rules(cut_cards=42).cut_cards == 42  # the deepest legal 1-deck cut
    assert Rules(cut_fraction=2 / 3).cut_fraction == 2 / 3
    assert Rules(cut_fraction=0.5).cut_fraction == 0.5
    assert Rules(blackjack_payout=(6, 5)).blackjack_payout_ratio == pytest.approx(1.2)


# ---- the shoe: decks, cut, burn ------------------------------------------------------


def test_six_deck_shoe_has_312_cards():
    shoe = Shoe(6, random.Random(0), burn_card=False)
    assert shoe.cards_remaining == 312
    assert shoe.rank_counts() == (24,) * 9 + (96,)


def test_exact_cut_is_placed_where_config_says():
    shoe = Shoe(1, random.Random(0), cut_cards=39)
    assert shoe.cut_position == 39
    assert not shoe.cut_reached
    for _ in range(37):
        shoe.draw()
    assert not shoe.cut_reached  # 38 cards gone including the burn
    shoe.draw()
    shoe.draw()
    assert shoe.cut_reached
    shoe.reshuffle()
    assert shoe.cut_position == 39  # an exact cut does not move


def test_random_cut_lands_within_five_cards_of_the_fraction():
    for fraction, centre in ((0.75, 39), (2 / 3, 35), (0.5, 26)):
        shoe = Shoe(1, random.Random(7), cut_fraction=fraction)
        assert centre - 5 <= shoe.cut_position <= centre + 5
        assert 1 <= shoe.cut_position <= 42  # never past the last ten cards


def test_random_cut_is_re_rolled_on_every_shuffle():
    cuts = {
        Shoe(1, random.Random(seed), cut_fraction=0.75).cut_position
        for seed in range(20)
    }
    assert len(cuts) > 1  # the jitter actually varies the cut


# ---- insurance and even money --------------------------------------------------------


def test_insurance_wins_when_the_dealer_has_blackjack():
    rnd = stacked_round([C(Rank.TEN), C(Rank.ACE), C(Rank.SEVEN), C(Rank.TEN)])
    assert rnd.can_insure()
    rnd.take_insurance()
    assert rnd.insurance_bet == 5.0
    assert rnd.peek() is True
    assert rnd.insurance_net() == 10.0  # 2:1 on the $5 insurance bet
    assert rnd.hand_net(rnd.hands[0]) == -10.0
    assert rnd.net_dollars() == 0.0  # the insurance exactly covers the hand


def test_insurance_loses_when_the_dealer_does_not_have_blackjack():
    rnd = stacked_round([C(Rank.TEN), C(Rank.ACE), C(Rank.SEVEN), C(Rank.NINE)])
    rnd.take_insurance()
    assert rnd.peek() is False
    rnd.dealer_play()  # dealer stands on ace-nine
    assert rnd.insurance_net() == -5.0
    assert rnd.net_dollars() == -15.0  # lost the hand and the insurance


def test_insurance_is_not_offered_without_an_ace_upcard():
    rnd = stacked_round([C(Rank.TEN), C(Rank.SIX), C(Rank.SEVEN), C(Rank.NINE)])
    assert not rnd.can_insure()


def test_insurance_is_not_offered_when_the_balance_cannot_cover_it():
    rnd = stacked_round(
        [C(Rank.TEN), C(Rank.ACE), C(Rank.SEVEN), C(Rank.NINE)], balance=10
    )
    # $10 is already at risk on the hand. Half the bet ($5) more is not covered.
    assert not rnd.can_insure()


def test_even_money_takes_one_to_one_and_ends_the_round():
    rnd = stacked_round([C(Rank.ACE), C(Rank.ACE), C(Rank.KING), C(Rank.NINE)])
    assert rnd.can_take_even_money()
    assert not rnd.can_insure()  # a blackjack takes even money, never insurance
    rnd.take_even_money()
    assert rnd.is_over
    assert rnd.hand_outcome(rnd.hands[0]) is Outcome.EVEN_MONEY
    assert rnd.hand_net(rnd.hands[0]) == 10.0
    rnd.dealer_play()
    assert rnd.dealer.total == 20  # the dealer never drew
    assert rnd.net_dollars() == 10.0


def test_declining_even_money_risks_the_push():
    rnd = stacked_round([C(Rank.ACE), C(Rank.ACE), C(Rank.KING), C(Rank.TEN)])
    assert rnd.can_take_even_money()
    # Decline: the peek finds the dealer blackjack, so the natural pushes.
    assert rnd.peek() is True
    assert rnd.hand_outcome(rnd.hands[0]) is Outcome.PUSH
    assert rnd.net_dollars() == 0.0


def test_insurance_ev_and_even_money_ev_from_the_unseen_pool():
    calc = OddsCalculator(Rules())
    full = (4,) * 9 + (16,)
    # P(hole is a ten) = 16/52, so insurance EV = 3p - 1.
    assert calc.insurance_ev(full) == pytest.approx(3 * 16 / 52 - 1)
    # Declining even money at 3:2 is worth 1.5 * (1 - p) - just above 1.
    assert calc.even_money_ev(full) == pytest.approx(1.5 * 36 / 52)
    assert calc.even_money_ev(full) > 1.0
    # On a 6:5 table declining even money is a loser: 1.2 * (1 - p) < 1.
    calc65 = OddsCalculator(Rules(blackjack_payout=(6, 5)))
    assert calc65.even_money_ev(full) == pytest.approx(1.2 * 36 / 52)
    assert calc65.even_money_ev(full) < 1.0


# ---- 6:5 payout ------------------------------------------------------------------------


def test_blackjack_pays_six_to_five_when_configured():
    calc = OddsCalculator(Rules(blackjack_payout=(6, 5)))
    a = calc.analyze(
        hand(Rank.ACE, Rank.KING),
        C(Rank.SIX),
        (3, 3, 3, 3, 3, 3, 3, 3, 3, 15),
    )
    # The full 6:5: against a 6 upcard the dealer cannot have blackjack
    # (that needs an ace or a ten showing), so the natural always pays.
    assert a.odds(Action.STAND).ev == pytest.approx(1.2)


def test_six_to_five_blackjack_settles_for_six_on_a_five_bet():
    rnd = stacked_round(
        [C(Rank.ACE), C(Rank.SIX), C(Rank.KING), C(Rank.NINE), C(Rank.TWO)],
        rules=Rules(blackjack_payout=(6, 5)),
        bet=5,
    )
    assert rnd.peek() is False
    rnd.dealer_play()  # dealer 15 draws the two to 17
    assert rnd.hand_outcome(rnd.hands[0]) is Outcome.PLAYER_BLACKJACK
    assert rnd.hand_net(rnd.hands[0]) == 6.0


# ---- the split-rule toggles --------------------------------------------------------------


def test_double_after_split_off_blocks_doubling_a_split_hand():
    rnd = stacked_round(
        [C(Rank.EIGHT), C(Rank.SIX), C(Rank.EIGHT), C(Rank.FIVE),
         C(Rank.THREE), C(Rank.TWO)],
        rules=Rules(double_after_split=False),
    )
    first, _ = rnd.split_hand(rnd.hands[0])
    assert first.total == 11
    assert not rnd.can_double(first)


def test_double_after_split_on_still_allows_it():
    rnd = stacked_round(
        [C(Rank.EIGHT), C(Rank.SIX), C(Rank.EIGHT), C(Rank.FIVE),
         C(Rank.THREE), C(Rank.TWO)],
    )
    first, _ = rnd.split_hand(rnd.hands[0])
    assert rnd.can_double(first)


def test_resplit_aces_off_blocks_a_second_split_of_aces():
    # The cap is raised explicitly: with the default of one split the resplit
    # is blocked by the cap, and this test would not test the toggle at all.
    rnd = stacked_round(
        [C(Rank.ACE), C(Rank.SIX), C(Rank.ACE), C(Rank.FIVE),
         C(Rank.ACE), C(Rank.TWO)],
        rules=Rules(max_splits=3, resplit_aces=False),
    )
    first, _ = rnd.split_hand(rnd.hands[0])
    assert first.restricted
    assert not rnd.can_split(first)


def test_hit_split_aces_on_lets_a_split_ace_play():
    rnd = stacked_round(
        [C(Rank.ACE), C(Rank.SIX), C(Rank.ACE), C(Rank.FIVE),
         C(Rank.FIVE), C(Rank.TWO)],
        rules=Rules(hit_split_aces=True),
    )
    first, _ = rnd.split_hand(rnd.hands[0])
    assert not first.restricted
    assert rnd.hand_has_decision(first)  # ace-five can hit


def test_double_after_split_off_removes_double_from_the_analysis():
    calc = OddsCalculator(Rules(double_after_split=False))
    a = calc.analyze(
        hand(Rank.EIGHT, Rank.THREE, from_split=True),
        C(Rank.SIX),
        (4, 4, 4, 4, 4, 4, 4, 4, 4, 16),
    )
    assert not a.offers(Action.DOUBLE)


# ---- the money in config ------------------------------------------------------------------


def test_starting_balance_comes_from_config():
    assert Bankroll(Rules().starting_balance).balance == 1000.0


# ---- the split engine: the C decomposition, one split by default --------------------
#
# _hand_after (the single-hand check) moved with the Python decomposition to
# the archive. See tests/test_archive.py. The value below is the one-split
# value on the small pool: at three splits it was -0.4280 instead, and the
# joint reads -0.4193 there.


def test_splitting_eights_against_ten_is_negative_on_the_small_pool():
    _require_c()
    calc = OddsCalculator(Rules())
    a = calc.analyze(
        hand(Rank.EIGHT, Rank.EIGHT), C(Rank.TEN), SMALL_POOL, allow_split=True
    )
    assert a.odds(Action.SPLIT).ev == pytest.approx(-0.4364, abs=0.001)
