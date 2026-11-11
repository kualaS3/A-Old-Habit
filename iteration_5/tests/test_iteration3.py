"""Iteration 3: variable bets, the Player, and splitting (game + odds).

The split odds were an independent-hands approximation in iteration 3. Iteration 5
replaced them with the C decomposition engine (see odds.py and split_exact.py).
The split tests below run on a small unseen pool: the engine takes a few seconds
on a full deck, and about a second on this pool. The default split cap is one
(Rules.max_splits = 1), so the values below are the one-split values. The
resplit cases are tested with the cap raised explicitly.
"""
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


# ---- helpers ---------------------------------------------------------------------------------


def hand(*ranks: Rank, **kwargs) -> Hand:
    h = Hand(**kwargs)
    for r in ranks:
        h.add(Card(r, Suit.SPADES))
    return h


def deck_minus(*seen: Rank) -> tuple[int, ...]:
    counts = [4] * 9 + [16]
    for r in seen:
        counts[r.blackjack_value - 1] -= 1
    return tuple(counts)


def stacked_shoe(draw_order: list[Card]) -> Shoe:
    shoe = Shoe(1, random.Random(0))
    shoe._cards = list(reversed(draw_order))
    return shoe


def stacked_round(
    draw_order: list[Card],
    balance: int = 100,
    bet: int = 1,
    rules: Rules | None = None,
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


# ---- the bankroll and the player ---------------------------------------------------------------


def test_bankroll_is_just_a_pot_now():
    b = Bankroll()
    assert b.balance == 100
    assert b.settle(37) == 137
    assert b.settle(-40) == 97
    assert b.settle(0) == 97
    with pytest.raises(ValueError):
        Bankroll(-1)


def test_variable_bet_scales_the_result():
    rnd = stacked_round(
        [C(Rank.NINE, S), C(Rank.TEN, H), C(Rank.EIGHT, S), C(Rank.SEVEN, D)],
        bet=5,
    )
    rnd.dealer_play()  # 17 vs 17... dealer has 10+7=17, player 9+8=17: push
    assert rnd.net_dollars() == 0
    rnd2 = stacked_round(
        [C(Rank.TEN, S), C(Rank.SIX, H), C(Rank.EIGHT, S), C(Rank.TEN, D),
         C(Rank.NINE, CL)],
        bet=5,
    )
    rnd2.dealer_play()  # dealer 16 draws 9 and busts. Player 18 wins the $5 bet
    assert rnd2.net_dollars() == 5


def test_exposure_and_cover_track_stakes():
    player = Player(Bankroll(10))
    h = Hand(bet=4)
    player.hands = [h]
    assert player.exposure == 4
    assert player.can_cover(4)  # a split or double needs 4 more: 8 <= 10
    h.doubled = True
    assert player.exposure == 8
    assert not player.can_cover(4)


# ---- pairs and the natural rule ------------------------------------------------------------------


def test_pair_means_equal_rank_not_equal_value():
    assert hand(Rank.JACK, Rank.JACK).is_pair
    assert hand(Rank.TEN, Rank.TEN).is_pair
    assert not hand(Rank.JACK, Rank.QUEEN).is_pair
    assert not hand(Rank.JACK, Rank.KING).is_pair


def test_split_hand_21_is_not_a_blackjack():
    split = hand(Rank.ACE, Rank.KING, from_split=True)
    assert split.total == 21
    assert not split.is_blackjack
    assert hand(Rank.ACE, Rank.KING).is_blackjack  # the dealt version still is


# ---- splitting in the game -----------------------------------------------------------------------


def test_split_makes_two_hands_with_a_card_each():
    rnd = stacked_round(
        [C(Rank.EIGHT, S), C(Rank.SIX, H), C(Rank.EIGHT, H), C(Rank.TEN, D),
         C(Rank.THREE, CL), C(Rank.KING, D), C(Rank.NINE, CL), C(Rank.TWO, CL)],
        bet=2,
    )
    pair = rnd.hands[0]
    assert rnd.can_split(pair)
    first, second = rnd.split_hand(pair)
    assert len(rnd.hands) == 2
    assert (first.total, second.total) == (11, 18)
    assert first.bet == second.bet == 2  # each hand carries the original bet
    assert first.from_split and second.from_split
    assert rnd.splits_used == 1

    assert rnd.can_double(first)  # double after split is allowed
    rnd.player_double(first)  # draws the 9 -> 20 on a doubled stake of $4
    rnd.dealer_play()  # dealer 16 draws the 2 -> 18
    assert rnd.hand_net(first) == 4  # 20 beats 18, on the doubled stake
    assert rnd.hand_net(second) == 0  # 18 pushes
    assert rnd.net_dollars() == 4


def test_split_needs_the_balance_to_cover_the_extra_stake():
    poor = stacked_round(
        [C(Rank.EIGHT, S), C(Rank.SIX, H), C(Rank.EIGHT, H), C(Rank.TEN, D)],
        balance=3, bet=2,
    )
    assert not poor.can_split(poor.hands[0])  # 2 at risk + 2 more > 3
    rich = stacked_round(
        [C(Rank.EIGHT, S), C(Rank.SIX, H), C(Rank.EIGHT, H), C(Rank.TEN, D)],
        balance=4, bet=2,
    )
    assert rich.can_split(rich.hands[0])


def test_jack_queen_cannot_split():
    rnd = stacked_round(
        [C(Rank.JACK, S), C(Rank.SIX, H), C(Rank.QUEEN, S), C(Rank.TEN, D)]
    )
    assert not rnd.can_split(rnd.hands[0])


def test_resplitting_stops_at_the_cap():
    # A synthetic stack with more 8s than a real deck holds. Stacked shoes allow it.
    # The cap is raised explicitly here: the default is one split (see below).
    rnd = stacked_round(
        [C(Rank.EIGHT, S), C(Rank.SIX, H), C(Rank.EIGHT, H), C(Rank.TEN, D),
         C(Rank.EIGHT, D), C(Rank.EIGHT, CL),   # split 1: both hands pair up again
         C(Rank.EIGHT, S), C(Rank.TWO, CL),     # split 2 (first hand): pair + an 8,2
         C(Rank.THREE, CL), C(Rank.FOUR, CL)],  # split 3 (the new pair)
        rules=Rules(max_splits=3),
    )
    rnd.split_hand(rnd.hands[0])
    rnd.split_hand(rnd.hands[0])
    rnd.split_hand(rnd.hands[0])
    assert rnd.splits_used == 3
    assert len(rnd.hands) == 4  # 3 splits = at most 4 hands
    remaining_pair = rnd.hands[3]
    assert remaining_pair.is_pair
    assert not rnd.can_split(remaining_pair)  # the cap, not the cards, says no


def test_default_cap_of_one_blocks_a_resplit():
    # Same stack as the cap test, but with the default table: one split and
    # the pair that comes up again cannot be split a second time. This is
    # the deliberate limitation (see Rules.max_splits and the README).
    rnd = stacked_round(
        [C(Rank.EIGHT, S), C(Rank.SIX, H), C(Rank.EIGHT, H), C(Rank.TEN, D),
         C(Rank.EIGHT, D), C(Rank.EIGHT, CL),
         C(Rank.EIGHT, S), C(Rank.TWO, CL),
         C(Rank.THREE, CL), C(Rank.FOUR, CL)],
    )
    assert Rules().max_splits == 1
    rnd.split_hand(rnd.hands[0])
    assert rnd.splits_used == 1
    assert rnd.hands[0].is_pair
    assert not rnd.can_split(rnd.hands[0])  # the cap, not the cards, says no


def test_no_splits_allowed_when_the_cap_is_zero():
    rnd = stacked_round(
        [C(Rank.EIGHT, S), C(Rank.SIX, H), C(Rank.EIGHT, H), C(Rank.TEN, D)],
        rules=Rules(max_splits=0),
    )
    assert not rnd.can_split(rnd.hands[0])


def test_split_aces_take_one_card_and_no_decisions():
    rnd = stacked_round(
        [C(Rank.ACE, S), C(Rank.SIX, H), C(Rank.ACE, H), C(Rank.TEN, D),
         C(Rank.KING, CL), C(Rank.FIVE, CL), C(Rank.NINE, CL)],
        bet=3,
    )
    first, second = rnd.split_hand(rnd.hands[0])
    assert first.restricted and second.restricted
    assert (first.total, second.total) == (21, 16)
    assert not first.is_blackjack  # split A+K is an ordinary 21
    assert not rnd.hand_has_decision(first)
    assert not rnd.hand_has_decision(second)
    assert not rnd.can_split(second)  # A+5 is not a pair: nothing to resplit
    rnd.dealer_play()  # dealer 16 draws the 9 and busts
    assert rnd.hand_net(first) == 3  # plain stake back - no blackjack bonus here
    assert rnd.net_dollars() == 6


def test_split_aces_can_resplit_on_another_ace():
    # The cap is raised explicitly: the default of one split allows no resplit.
    rnd = stacked_round(
        [C(Rank.ACE, S), C(Rank.SIX, H), C(Rank.ACE, H), C(Rank.TEN, D),
         C(Rank.ACE, D), C(Rank.SEVEN, CL),   # split 1: A+A again, and A+7
         C(Rank.FOUR, CL), C(Rank.QUEEN, CL)],  # the resplit's two cards
        rules=Rules(max_splits=3),
    )
    first, second = rnd.split_hand(rnd.hands[0])
    assert first.is_pair and rnd.can_split(first)  # the exception to "no decisions"
    new_first, new_second = rnd.split_hand(first)
    assert len(rnd.hands) == 3
    assert rnd.splits_used == 2
    assert (new_first.total, new_second.total) == (15, 21)
    assert all(h.restricted for h in rnd.hands)


def test_default_cap_blocks_the_ace_resplit_too():
    rnd = stacked_round(
        [C(Rank.ACE, S), C(Rank.SIX, H), C(Rank.ACE, H), C(Rank.TEN, D),
         C(Rank.ACE, D), C(Rank.SEVEN, CL),
         C(Rank.FOUR, CL), C(Rank.QUEEN, CL)]
    )
    first, second = rnd.split_hand(rnd.hands[0])
    assert first.is_pair  # A+A again, and the table would allow it...
    assert not rnd.can_split(first)  # ...but the default cap of one says no


def test_dealer_plays_once_for_all_hands():
    rnd = stacked_round(
        [C(Rank.EIGHT, S), C(Rank.SIX, H), C(Rank.EIGHT, H), C(Rank.SIX, D),
         C(Rank.TEN, CL), C(Rank.NINE, CL), C(Rank.FIVE, CL)],
    )
    first, second = rnd.split_hand(rnd.hands[0])  # 18 and 17
    rnd.dealer_play()  # dealer 12 draws the 5 -> 17, once
    assert rnd.dealer.total == 17
    assert rnd.hand_outcome(first) is Outcome.WIN
    assert rnd.hand_outcome(second) is Outcome.PUSH
    assert rnd.net_dollars() == 1


# ---- splitting in the odds (C decomposition engine, iteration 5) -------------------------
#
# The split EV is near-exact up to the second-order coupling between the hands
# (see odds.py), and it is computed by the C decomposition engine -- so these
# tests skip when the helper is not built (run python build.py). They run on a
# 13-card unseen pool, where it takes about a second. The values below are the
# one-split values on that pool (the default cap): the decomposition, the joint
# and the archived Python version all agree on them exactly. At three splits
# the 8,8 vs 6 value is +0.3868 instead -- the resplit is worth that much on
# this pool -- and the joint reads +0.4440 there, the one spot where the
# coupling shows, because one card is a twelfth of the shoe.


SMALL_POOL = (1, 1, 1, 1, 1, 1, 1, 1, 1, 4)  # one of each value, four tens


@pytest.fixture
def calc() -> OddsCalculator:
    if not split_exact.available():
        pytest.skip("C helper not built (run python build.py)")
    return OddsCalculator(Rules())


def test_split_only_offered_when_allowed(calc):
    pair = hand(Rank.EIGHT, Rank.EIGHT)
    assert not calc.analyze(pair, C(Rank.SIX, H), SMALL_POOL).offers(Action.SPLIT)
    a = calc.analyze(pair, C(Rank.SIX, H), SMALL_POOL, allow_split=True)
    assert a.offers(Action.SPLIT)


def test_split_not_offered_for_jack_queen(calc):
    a = calc.analyze(
        hand(Rank.JACK, Rank.QUEEN),
        C(Rank.SIX, H),
        deck_minus(Rank.JACK, Rank.QUEEN, Rank.SIX),
        allow_split=True,
    )
    assert not a.offers(Action.SPLIT)


def test_splitting_eights_against_six_is_best_and_positive(calc):
    a = calc.analyze(
        hand(Rank.EIGHT, Rank.EIGHT),
        C(Rank.SIX, H),
        SMALL_POOL,
        allow_split=True,
    )
    split = a.odds(Action.SPLIT)
    assert a.best() is Action.SPLIT
    assert split.ev == pytest.approx(0.3522, abs=0.001)
    # The SPLIT row carries only its combined EV now: no per-hand
    # win/push/lose split is tracked (the hands share one dealer hand).
    assert (split.win, split.push, split.lose) == (0.0, 0.0, 0.0)


def test_never_split_fives(calc):
    a = calc.analyze(
        hand(Rank.FIVE, Rank.FIVE),
        C(Rank.SIX, H),
        SMALL_POOL,
        allow_split=True,
    )
    assert a.best() is not Action.SPLIT  # doubling the 10 beats two hands from 5


def test_splitting_aces_against_five_is_best(calc):
    a = calc.analyze(
        hand(Rank.ACE, Rank.ACE),
        C(Rank.FIVE, H),
        SMALL_POOL,
        allow_split=True,
    )
    assert a.best() is Action.SPLIT
    assert a.odds(Action.SPLIT).ev == pytest.approx(0.6709, abs=0.001)
