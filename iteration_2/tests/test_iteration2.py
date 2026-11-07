"""Iteration 2: double down, the dealer's peek, and the bankroll."""
import random

import pytest

from blackjack.bankroll import Bankroll
from blackjack.cards import Card, Rank, Suit
from blackjack.counting import CardCounter, HiLo
from blackjack.hand import Hand
from blackjack.odds import Action, OddsCalculator
from blackjack.round import Outcome, Round
from blackjack.rules import Rules
from blackjack.shoe import Shoe


# ---- helpers (deck/hand style follows test_odds.py) ---------------------------------------


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


def stacked_shoe(draw_order: list[Card]) -> Shoe:
    """A shoe that deals exactly the given cards, in the given order."""
    shoe = Shoe(1, random.Random(0))
    shoe._cards = list(reversed(draw_order))
    return shoe


def stacked_round(draw_order: list[Card]) -> Round:
    rnd = Round(stacked_shoe(draw_order), CardCounter(HiLo()), Rules())
    rnd.deal()
    return rnd


@pytest.fixture
def calc() -> OddsCalculator:
    return OddsCalculator(Rules())


# ---- double in the odds -------------------------------------------------------------------


def test_double_offered_only_on_two_card_hands(calc):
    two = calc.analyze(
        hand(Rank.FIVE, Rank.SIX), card(Rank.SIX), deck_minus(Rank.FIVE, Rank.SIX, Rank.SIX)
    )
    assert two.offers(Action.DOUBLE)

    three = calc.analyze(
        hand(Rank.TWO, Rank.THREE, Rank.SIX),
        card(Rank.SIX),
        deck_minus(Rank.TWO, Rank.THREE, Rank.SIX, Rank.SIX),
    )
    assert not three.offers(Action.DOUBLE)
    assert three.offers(Action.HIT)


def test_double_probabilities_sum_to_one_but_ev_is_on_the_doubled_stake(calc):
    a = calc.analyze(
        hand(Rank.FIVE, Rank.SIX), card(Rank.SIX), deck_minus(Rank.FIVE, Rank.SIX, Rank.SIX)
    )
    d = a.odds(Action.DOUBLE)
    assert d.win + d.push + d.lose == pytest.approx(1.0)
    assert d.stake == 2.0
    assert d.ev == pytest.approx(2.0 * (d.win - d.lose))
    s = a.odds(Action.STAND)
    assert s.stake == 1.0
    assert s.ev == pytest.approx(s.win - s.lose)


def test_deterministic_double(calc):
    # Only two tens are unseen. Player has 10 vs dealer 7.
    # Double: player draws a ten -> 20, dealer gets the other ten -> 17, player wins
    # on a doubled stake, so the EV is +2 rather than the +1 that hitting gives.
    unseen = (0,) * 9 + (2,)
    a = calc.analyze(hand(Rank.SIX, Rank.FOUR), card(Rank.SEVEN), unseen)
    d = a.odds(Action.DOUBLE)
    assert d.win == pytest.approx(1.0)
    assert d.ev == pytest.approx(2.0)
    assert a.best() is Action.DOUBLE


# ---- the peek in the odds -----------------------------------------------------------------


def test_peek_excludes_the_blackjack_hole_card_ace_up(calc):
    # Player 20 vs an ace. The unseen pool is one 6 and two tens.
    # Unconditioned, the hole card is usually a ten and the dealer has blackjack.
    # After the peek the hole card cannot be a ten, so it is the 6: the dealer
    # stands on soft 17 and the player's 20 always wins.
    unseen = (0, 0, 0, 0, 0, 1, 0, 0, 0, 2)
    a = calc.analyze(hand(Rank.KING, Rank.QUEEN), card(Rank.ACE), unseen)
    assert a.odds(Action.STAND).win == pytest.approx(1.0)


def test_peek_excludes_the_blackjack_hole_card_ten_up(calc):
    # Player 20 vs a king. The unseen pool is one ace and two 6s.
    # After the peek the hole card cannot be the ace, so it is a 6: the dealer
    # has 16, must draw from {ace, 6}, and ends on 17 or bust - the 20 always wins.
    unseen = (1, 0, 0, 0, 0, 2, 0, 0, 0, 0)
    a = calc.analyze(hand(Rank.KING, Rank.QUEEN), card(Rank.KING), unseen)
    assert a.odds(Action.STAND).win == pytest.approx(1.0)


def test_peek_conditioning_changes_stand_odds_vs_ace(calc):
    unseen = deck_minus(Rank.KING, Rank.QUEEN, Rank.ACE)
    a = calc.analyze(hand(Rank.KING, Rank.QUEEN), card(Rank.ACE), unseen)
    # The old unconditioned dealer distribution (private, used here just for the
    # contrast) gives the dealer a large chance of finishing on 21 - mostly the
    # blackjack the peek has already ruled out.
    unconditioned = calc._dealer_distribution(unseen, 1, True)
    unconditioned_win = unconditioned[22] + sum(unconditioned[:20])
    assert unconditioned[21] > 0.2
    assert a.odds(Action.STAND).win > unconditioned_win


# ---- the peek and double in a round --------------------------------------------------------


def test_peek_finds_dealer_blackjack_under_an_ace():
    rnd = stacked_round(
        [
            Card(Rank.FIVE, Suit.SPADES),  # player
            Card(Rank.ACE, Suit.HEARTS),  # dealer upcard
            Card(Rank.SIX, Suit.SPADES),  # player
            Card(Rank.KING, Suit.DIAMONDS),  # hole card
        ]
    )
    assert rnd.peek() is True
    assert rnd.peeked
    assert rnd.is_over
    assert not rnd.player_has_decision
    assert rnd.outcome is Outcome.DEALER_BLACKJACK
    assert rnd.net_units() == -1


def test_peek_finds_dealer_blackjack_under_a_ten():
    rnd = stacked_round(
        [
            Card(Rank.NINE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.SPADES),
            Card(Rank.ACE, Suit.DIAMONDS),
        ]
    )
    assert rnd.peek() is True
    assert rnd.peeked
    assert rnd.outcome is Outcome.DEALER_BLACKJACK
    assert rnd.net_units() == -1


def test_peek_without_blackjack_lets_the_hand_continue():
    rnd = stacked_round(
        [
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.SIX, Suit.SPADES),
            Card(Rank.SIX, Suit.DIAMONDS),
        ]
    )
    assert rnd.peek() is False
    assert rnd.peeked
    assert rnd.player_has_decision


def test_no_peek_when_the_upcard_cannot_make_blackjack():
    rnd = stacked_round(
        [
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SEVEN, Suit.HEARTS),
            Card(Rank.SIX, Suit.SPADES),
            Card(Rank.KING, Suit.DIAMONDS),
        ]
    )
    assert rnd.peek() is False
    assert not rnd.peeked


def test_player_double_draws_one_card_and_ends_the_turn():
    rnd = stacked_round(
        [
            Card(Rank.FIVE, Suit.SPADES),  # player
            Card(Rank.SIX, Suit.HEARTS),  # dealer upcard
            Card(Rank.SIX, Suit.SPADES),  # player - 11 in total
            Card(Rank.TEN, Suit.DIAMONDS),  # hole card
            Card(Rank.KING, Suit.SPADES),  # the double card
            Card(Rank.NINE, Suit.CLUBS),  # the dealer's draw
        ]
    )
    assert rnd.player_has_decision
    drawn = rnd.player_double()
    assert drawn.rank is Rank.KING
    assert len(rnd.player.cards) == 3
    assert rnd.doubled
    assert not rnd.player_has_decision

    rnd.dealer_play()  # dealer has 16, draws the 9 and busts
    assert rnd.outcome is Outcome.WIN
    assert rnd.net_units() == 2


def test_doubled_hand_loses_two_units():
    rnd = stacked_round(
        [
            Card(Rank.FIVE, Suit.SPADES),  # player
            Card(Rank.TEN, Suit.HEARTS),  # dealer upcard
            Card(Rank.SIX, Suit.SPADES),  # player - 11 in total
            Card(Rank.NINE, Suit.DIAMONDS),  # hole card - dealer has 19
            Card(Rank.TWO, Suit.SPADES),  # the double card - player ends on 13
        ]
    )
    rnd.player_double()
    rnd.dealer_play()
    assert rnd.outcome is Outcome.LOSE
    assert rnd.net_units() == -2


def test_player_blackjack_pays_one_unit_and_is_never_doubled():
    rnd = stacked_round(
        [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.SIX, Suit.HEARTS),
            Card(Rank.KING, Suit.SPADES),
            Card(Rank.NINE, Suit.DIAMONDS),
        ]
    )
    assert rnd.outcome is Outcome.PLAYER_BLACKJACK
    assert rnd.net_units() == 1


# ---- the bankroll --------------------------------------------------------------------------


def test_bankroll_starts_at_100_with_a_flat_1_bet():
    b = Bankroll()
    assert b.balance == 100
    assert b.bet == 1
    assert b.can_play()
    assert b.can_double()
    assert b.settle(1) == 101
    assert b.settle(-2) == 99
    assert b.settle(0) == 99
    assert b.balance == 99


def test_bankroll_edges():
    low = Bankroll(starting_balance=1)
    assert low.can_play()
    assert not low.can_double()
    assert low.settle(-1) == 0
    assert not low.can_play()

    assert Bankroll(starting_balance=2).can_double()

    bigger_bet = Bankroll(starting_balance=3, bet=2)
    assert bigger_bet.can_play()
    assert not bigger_bet.can_double()
    assert bigger_bet.settle(2) == 7  # 2 units at $2 a unit
