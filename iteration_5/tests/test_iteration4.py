"""Iteration 4: surrender, the blackjack payout, and S17/H17 as table options."""
import random

import pytest

from blackjack.bankroll import Bankroll
from blackjack.cards import Card, Rank, Suit
from blackjack.counting import CardCounter, HiLo
from blackjack.hand import Hand
from blackjack.odds import Action, OddsCalculator
from blackjack.player import Player
from blackjack.round import Outcome, Round
from blackjack.rules import Rules
from blackjack.shoe import Shoe


# ---- helpers (deck/hand style follows test_odds.py) ---------------------------------------


def hand(*ranks: Rank, **kwargs) -> Hand:
    h = Hand(**kwargs)
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


def stacked_shoe(draw_order: list[Card]) -> Shoe:
    shoe = Shoe(1, random.Random(0))
    shoe._cards = list(reversed(draw_order))
    return shoe


def stacked_round(
    draw_order: list[Card], rules: Rules | None = None, balance: float = 100, bet: int = 1
) -> Round:
    rnd = Round(
        stacked_shoe(draw_order),
        CardCounter(HiLo()),
        rules or Rules(),
        Player(Bankroll(balance)),
    )
    rnd.deal(bet)
    return rnd


C = Card
S, H, D, CL = Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS


# ---- the config ----------------------------------------------------------------------------


def test_defaults_are_surrender_on_blackjack_three_to_two_stand_on_soft_17():
    r = Rules()
    assert r.surrender_allowed is True
    assert r.blackjack_payout == (3, 2)
    assert r.blackjack_payout_ratio == pytest.approx(1.5)
    assert r.blackjack_payout_label == "3:2"
    assert r.dealer_hits_soft_17 is False
    assert r.dealer_rule_label.startswith("S17")


def test_blackjack_payout_only_allows_three_to_two_or_two_to_one():
    # Iteration 5 added 6:5 as a third option; anything else still raises.
    assert Rules(blackjack_payout=(2, 1)).blackjack_payout_ratio == pytest.approx(2.0)
    assert Rules(blackjack_payout=(6, 5)).blackjack_payout_ratio == pytest.approx(1.2)
    with pytest.raises(ValueError):
        Rules(blackjack_payout=(1, 1))
    with pytest.raises(ValueError):
        Rules(blackjack_payout=(5, 4))


# ---- surrender in the game ---------------------------------------------------------------------


def test_surrender_loses_half_the_stake_and_the_dealer_does_not_play():
    rnd = stacked_round(
        [C(Rank.TEN, S), C(Rank.TEN, H), C(Rank.SIX, S), C(Rank.NINE, D)],
        bet=5,
    )
    h = rnd.hands[0]
    assert rnd.can_surrender(h)
    rnd.player_surrender(h)
    assert h.surrendered
    assert not rnd.hand_has_decision(h)
    assert rnd.hand_outcome(h) is Outcome.SURRENDER
    assert rnd.hand_net(h) == -2.5
    assert rnd.net_dollars() == -2.5
    rnd.dealer_play()  # nothing live: the dealer does not draw
    assert rnd.dealer.total == 19  # upcard ten + hole nine, no draw


def test_a_surrendered_hand_does_not_stop_the_dealer_when_another_hand_is_live():
    # A surrendered hand never makes the dealer play on its own - but it does
    # not stop the dealer while another hand is still live. A split hand cannot
    # surrender through the game, so the surrender here is set directly: this
    # tests the dealer predicate itself, which is what multi-player will rely on.
    rnd = stacked_round(
        [C(Rank.EIGHT, S), C(Rank.SIX, H), C(Rank.EIGHT, H), C(Rank.TEN, D),
         C(Rank.THREE, CL), C(Rank.KING, D), C(Rank.NINE, CL)],
        bet=2,
    )
    first, second = rnd.split_hand(rnd.hands[0])  # 11 and 18
    first.surrendered = True
    rnd.dealer_play()  # dealer 16 draws the 9 and busts - the live hand made him play
    assert rnd.dealer.is_bust
    assert rnd.hand_net(first) == -1  # half of the $2 stake
    assert rnd.hand_net(second) == 2
    assert rnd.net_dollars() == 1

    both_out = stacked_round(
        [C(Rank.EIGHT, S), C(Rank.SIX, H), C(Rank.EIGHT, H), C(Rank.TEN, D),
         C(Rank.THREE, CL), C(Rank.KING, D), C(Rank.NINE, CL)],
        bet=2,
    )
    first, second = both_out.split_hand(both_out.hands[0])
    first.surrendered = True
    second.surrendered = True
    both_out.dealer_play()  # no live hand: the dealer does not draw
    assert both_out.dealer.total == 16
    assert both_out.net_dollars() == -2


def test_surrender_gives_back_half_on_an_odd_bet():
    rnd = stacked_round(
        [C(Rank.TEN, S), C(Rank.TEN, H), C(Rank.SIX, S), C(Rank.NINE, D)],
        bet=1,
    )
    rnd.player_surrender(rnd.hands[0])
    assert rnd.hand_net(rnd.hands[0]) == -0.5


def test_surrender_is_only_the_first_decision_on_the_dealt_hand():
    rnd = stacked_round(
        [C(Rank.TEN, S), C(Rank.SIX, H), C(Rank.SIX, S), C(Rank.TEN, D),
         C(Rank.FIVE, CL)],  # the hit card
    )
    h = rnd.hands[0]
    assert rnd.can_surrender(h)
    rnd.player_hit(h)  # 16 -> 21, three cards now
    assert not rnd.can_surrender(h)
    assert len(h.cards) == 3


def test_surrender_is_not_offered_after_a_split_or_when_the_table_says_no():
    rnd = stacked_round(
        [C(Rank.EIGHT, S), C(Rank.SIX, H), C(Rank.EIGHT, H), C(Rank.TEN, D),
         C(Rank.THREE, CL), C(Rank.KING, D)],
        bet=2,
    )
    first, _ = rnd.split_hand(rnd.hands[0])
    assert not rnd.can_surrender(first)  # a split hand can never surrender

    no_surrender = stacked_round(
        [C(Rank.TEN, S), C(Rank.TEN, H), C(Rank.SIX, S), C(Rank.NINE, D)],
        rules=Rules(surrender_allowed=False),
    )
    assert not no_surrender.can_surrender(no_surrender.hands[0])


def test_surrender_is_not_offered_on_a_blackjack():
    rnd = stacked_round(
        [C(Rank.ACE, S), C(Rank.SIX, H), C(Rank.KING, S), C(Rank.NINE, D)]
    )
    assert rnd.is_over  # the natural already ended the round
    assert not rnd.can_surrender(rnd.hands[0])


# ---- surrender in the odds -----------------------------------------------------------------------


@pytest.fixture
def calc() -> OddsCalculator:
    return OddsCalculator(Rules())


def test_surrender_ev_is_exactly_minus_a_half(calc):
    a = calc.analyze(
        hand(Rank.TEN, Rank.SIX), card(Rank.TEN), deck_minus(Rank.TEN, Rank.SIX, Rank.TEN)
    )
    assert a.offers(Action.SURRENDER)
    assert a.odds(Action.SURRENDER).ev == pytest.approx(-0.5)


def test_surrender_is_best_on_sixteen_against_ten(calc):
    # The classic late-surrender spot: standing loses ~0.543, hitting ~0.507,
    # surrendering exactly 0.5.
    a = calc.analyze(
        hand(Rank.TEN, Rank.SIX), card(Rank.TEN), deck_minus(Rank.TEN, Rank.SIX, Rank.TEN)
    )
    assert a.best() is Action.SURRENDER
    assert a.odds(Action.STAND).ev == pytest.approx(-0.543, abs=1e-3)
    assert a.odds(Action.HIT).ev == pytest.approx(-0.507, abs=1e-3)


def test_surrender_is_not_best_on_sixteen_against_nine(calc):
    a = calc.analyze(
        hand(Rank.TEN, Rank.SIX), card(Rank.NINE), deck_minus(Rank.TEN, Rank.SIX, Rank.NINE)
    )
    assert a.best() is Action.HIT  # hitting loses ~0.479, surrendering 0.5
    assert a.odds(Action.HIT).ev == pytest.approx(-0.479, abs=1e-3)


def test_surrender_not_offered_when_the_table_says_no_or_after_a_hit():
    calc_off = OddsCalculator(Rules(surrender_allowed=False))
    a = calc_off.analyze(
        hand(Rank.TEN, Rank.SIX), card(Rank.TEN), deck_minus(Rank.TEN, Rank.SIX, Rank.TEN)
    )
    assert not a.offers(Action.SURRENDER)

    calc_on = OddsCalculator(Rules())
    three = calc_on.analyze(
        hand(Rank.TEN, Rank.FIVE, Rank.ACE),
        card(Rank.TEN),
        deck_minus(Rank.TEN, Rank.FIVE, Rank.ACE, Rank.TEN),
    )
    assert not three.offers(Action.SURRENDER)  # three cards: the moment has passed


# ---- the blackjack payout in the game and the odds -------------------------------------------------


def test_blackjack_pays_three_to_two_by_default_and_two_to_one_as_the_option():
    default = stacked_round(
        [C(Rank.ACE, S), C(Rank.SIX, H), C(Rank.KING, S), C(Rank.NINE, D)],
        bet=2,
    )
    assert default.net_dollars() == 3.0  # $2 bet, $3 profit

    two_to_one = stacked_round(
        [C(Rank.ACE, S), C(Rank.SIX, H), C(Rank.KING, S), C(Rank.NINE, D)],
        rules=Rules(blackjack_payout=(2, 1)),
        bet=2,
    )
    assert two_to_one.net_dollars() == 4.0  # $2 bet, $4 profit


def test_a_split_hand_twenty_one_still_pays_even_money():
    rnd = stacked_round(
        [C(Rank.ACE, S), C(Rank.SIX, H), C(Rank.ACE, H), C(Rank.TEN, D),
         C(Rank.KING, CL), C(Rank.FIVE, CL), C(Rank.NINE, CL)],
        bet=3,
    )
    first, _ = rnd.split_hand(rnd.hands[0])  # split A+K is an ordinary 21
    rnd.dealer_play()  # dealer 16 draws the 9 and busts
    assert rnd.hand_net(first) == 3  # plain stake - no blackjack bonus here


def test_blackjack_stand_ev_is_the_payout(calc):
    a = calc.analyze(
        hand(Rank.ACE, Rank.KING), card(Rank.SIX), deck_minus(Rank.ACE, Rank.KING, Rank.SIX)
    )
    assert list(a.by_action) == [Action.STAND]  # a natural has no other decision
    stand = a.odds(Action.STAND)
    assert stand.blackjack_win == pytest.approx(1.0)
    assert stand.win + stand.push + stand.lose + stand.blackjack_win == pytest.approx(1.0)
    assert stand.ev == pytest.approx(1.5)

    calc_2_to_1 = OddsCalculator(Rules(blackjack_payout=(2, 1)))
    b = calc_2_to_1.analyze(
        hand(Rank.ACE, Rank.KING), card(Rank.SIX), deck_minus(Rank.ACE, Rank.KING, Rank.SIX)
    )
    assert b.odds(Action.STAND).ev == pytest.approx(2.0)


# ---- S17 vs H17 ------------------------------------------------------------------------------------


def test_dealer_stands_on_soft_seventeen_by_default_and_hits_when_configured():
    # Only a 2 is unseen. Dealer holds soft 17 (ace + 6).
    unseen = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    s17 = OddsCalculator(Rules())._dealer_distribution(unseen, 7, True)
    assert s17[17] == pytest.approx(1.0)  # stands on soft 17
    h17 = OddsCalculator(Rules(dealer_hits_soft_17=True))._dealer_distribution(unseen, 7, True)
    assert h17[19] == pytest.approx(1.0)  # hits, draws the 2, stands on soft 19


def test_hitting_soft_seventeen_costs_the_player_on_eighteen_against_an_ace(calc):
    # Standing on 18 vs an ace: under S17 the dealer standing on 17 is a certain
    # win for the 18; under H17 the dealer can improve to 18-21, so the EV falls.
    unseen = deck_minus(Rank.TEN, Rank.EIGHT, Rank.ACE)
    s17 = calc.analyze(hand(Rank.TEN, Rank.EIGHT), card(Rank.ACE), unseen)
    h17 = OddsCalculator(Rules(dealer_hits_soft_17=True)).analyze(
        hand(Rank.TEN, Rank.EIGHT), card(Rank.ACE), unseen
    )
    assert s17.odds(Action.STAND).ev == pytest.approx(-0.082, abs=1e-3)
    assert h17.odds(Action.STAND).ev == pytest.approx(-0.211, abs=1e-3)
    assert h17.odds(Action.STAND).ev < s17.odds(Action.STAND).ev


def test_dealer_hits_soft_seventeen_in_a_round():
    # Dealer shows an ace, hole card a 6 (soft 17), then a 2 in the shoe.
    rnd = stacked_round(
        [C(Rank.TEN, S), C(Rank.ACE, H), C(Rank.EIGHT, S), C(Rank.SIX, D),
         C(Rank.TWO, CL)],
        rules=Rules(dealer_hits_soft_17=True),
    )
    rnd.dealer_play()  # H17: dealer draws the 2 and stands on soft 19
    assert rnd.dealer.total == 19


# ---- money in halves ---------------------------------------------------------------------------------


def test_bankroll_holds_halves():
    b = Bankroll()
    assert b.balance == 100
    assert b.settle(1.5) == 101.5
    assert b.settle(-0.5) == 101.0
