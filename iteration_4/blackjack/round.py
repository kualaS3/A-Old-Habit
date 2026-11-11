"""One round of blackjack: deal, player decisions (including splits), dealer play.

Iteration 3: the round works on the Player's list of hands instead of a single hand.
A split replaces a pair with two hands in place. The new hands are then played one at
a time, in order, against the same dealer hand. The dealer plays once, at the end,
and only if at least one player hand is still live.
"""
from __future__ import annotations

from enum import Enum

from .cards import Card
from .counting import CardCounter
from .hand import Hand
from .player import Player
from .rules import Rules
from .shoe import Shoe


class Outcome(Enum):
    PLAYER_BLACKJACK = "PLAYER_BLACKJACK"
    DEALER_BLACKJACK = "DEALER_BLACKJACK"
    PUSH = "PUSH"
    WIN = "WIN"
    LOSE = "LOSE"
    SURRENDER = "SURRENDER"


class Round:
    def __init__(self, shoe: Shoe, counter: CardCounter, rules: Rules, player: Player) -> None:
        self._shoe = shoe
        self._counter = counter
        self._rules = rules
        self.player = player
        self.dealer = Hand()
        self._hole_card: Card | None = None
        self.hole_revealed = False
        self.peeked = False
        self.splits_used = 0

    @property
    def hands(self) -> list[Hand]:
        return self.player.hands

    def deal(self, bet: int) -> None:
        """Two cards each. The dealer's second card (the hole card) stays face down
        and uncounted. The opening player hand carries the chosen bet."""
        hand = Hand(bet=bet)
        self.player.hands = [hand]
        hand.add(self._draw_visible())
        self.dealer.add(self._draw_visible())
        hand.add(self._draw_visible())
        self._hole_card = self._shoe.draw()
        self.dealer.add(self._hole_card)

    @property
    def dealer_upcard(self) -> Card:
        return self.dealer.cards[0]

    def peek(self) -> bool:
        """The dealer checks the hole card when showing an ace or a ten-value card.

        Records that the check happened in self.peeked and returns True when the
        dealer has blackjack (the round is then over before any player decision).
        With any other upcard there is nothing to check and this returns False.
        """
        up = self.dealer_upcard
        if up.rank.is_ace or up.rank.blackjack_value == 10:
            self.peeked = True
            return self.dealer.is_blackjack
        return False

    @property
    def is_over(self) -> bool:
        """True when a natural ends the round before any decision (dealer assumed
        to peek). Only the dealt hand can be a natural, so it is the only one to
        check on the player side."""
        return self.hands[0].is_blackjack or self.dealer.is_blackjack

    # ---- player decisions -------------------------------------------------------

    def hand_has_decision(self, hand: Hand) -> bool:
        """True while a normal hand can still be hit, stood on or doubled.

        Restricted hands (from split aces) never have decisions - they took their
        one card when they were created. Their only possible action is a resplit,
        which can_split covers separately.
        """
        return (
            not hand.restricted
            and not hand.doubled
            and not hand.surrendered
            and not hand.is_bust
            and hand.total < 21
            and not self.is_over
        )

    def can_double(self, hand: Hand) -> bool:
        """Double = exactly two cards, a live decision, and the balance covers one
        more stake of this hand's bet on top of everything already at risk."""
        return (
            self.hand_has_decision(hand)
            and len(hand.cards) == 2
            and self.player.can_cover(hand.bet)
        )

    def can_split(self, hand: Hand) -> bool:
        """Split = an equal-rank pair, under the split cap, and the balance covers
        the extra hand's stake. This is also the only action a restricted ace hand
        can ever take, when its one card was another ace."""
        return (
            hand.is_pair
            and not self.is_over
            and self.splits_used < self._rules.max_splits
            and self.player.can_cover(hand.bet)
        )

    def can_surrender(self, hand: Hand) -> bool:
        """Late surrender: the first decision only, on the dealt hand.

        The table must allow it, the hand must still hold its original two cards
        (no hit, double or split yet), and the round must still be live - after
        the peek, so a dealer blackjack or a player natural already ended it.
        Split hands can never surrender: they were created by a decision.
        """
        return (
            self._rules.surrender_allowed
            and len(hand.cards) == 2
            and not hand.from_split
            and not hand.restricted
            and not hand.doubled
            and not hand.surrendered
            and not self.is_over
        )

    def player_surrender(self, hand: Hand) -> None:
        """Give up the hand: half the stake is lost, the hand is finished.

        The dealer never plays for this hand - dealer_play skips a round where
        every hand is bust or surrendered - and settlement gives back the other
        half (see hand_net).
        """
        hand.surrendered = True

    def player_hit(self, hand: Hand) -> Card:
        card = self._draw_visible()
        hand.add(card)
        return card

    def player_double(self, hand: Hand) -> Card:
        """Double down: exactly one more card, on a doubled stake, then the turn ends."""
        card = self._draw_visible()
        hand.add(card)
        hand.doubled = True
        return card

    def split_hand(self, hand: Hand) -> tuple[Hand, Hand]:
        """Replace a pair with its two split hands, each dealt one more card.

        The new hands take the pair's place in the player's list, in order. Cards
        are dealt the casino way: one to the first hand, then one to the second.
        Split aces come back restricted - the card just dealt is their only card.
        """
        index = self.hands.index(hand)
        first, second = hand.split_hands()
        first.add(self._draw_visible())
        second.add(self._draw_visible())
        self.hands[index : index + 1] = [first, second]
        self.splits_used += 1
        return first, second

    # ---- dealer and settlement ----------------------------------------------------

    def dealer_play(self) -> None:
        """Reveal the hole card. The dealer draws only if some player hand is live.

        A hand is live when it is neither bust nor surrendered. A surrendered
        hand never makes the dealer play on its own - but it does not stop the
        dealer either, as long as another hand is still live. That is the rule
        that carries over unchanged when more players sit down: the dealer plays
        iff at least one hand at the table is live.
        """
        self.hole_revealed = True
        self._counter.observe(self._hole_card)
        if self.is_over or all(h.is_bust or h.surrendered for h in self.hands):
            return
        while self._rules.dealer_must_hit(self.dealer.total, self.dealer.is_soft):
            self.dealer.add(self._draw_visible())

    def hand_outcome(self, hand: Hand) -> Outcome:
        if hand.surrendered:
            return Outcome.SURRENDER
        d = self.dealer
        if hand.is_blackjack and d.is_blackjack:
            return Outcome.PUSH
        if hand.is_blackjack:
            return Outcome.PLAYER_BLACKJACK
        if d.is_blackjack:
            return Outcome.DEALER_BLACKJACK
        if hand.is_bust:
            return Outcome.LOSE
        if d.is_bust or hand.total > d.total:
            return Outcome.WIN
        if hand.total < d.total:
            return Outcome.LOSE
        return Outcome.PUSH

    def hand_net(self, hand: Hand) -> float:
        """One hand's result in dollars: its stake won, lost, or 0 on a push.

        A natural pays the table's blackjack payout on its stake - a dealt
        blackjack can never have been doubled, since it ends the round before any
        decision, so stake == bet. A surrender loses exactly half the stake.
        The result is a float because 3:2 on an odd bet and a surrendered half
        stake are both fifty-cent amounts.
        """
        o = self.hand_outcome(hand)
        if o is Outcome.PLAYER_BLACKJACK:
            return hand.stake * self._rules.blackjack_payout_ratio
        if o is Outcome.WIN:
            return float(hand.stake)
        if o is Outcome.PUSH:
            return 0.0
        if o is Outcome.SURRENDER:
            return -hand.stake / 2
        return -float(hand.stake)

    def net_dollars(self) -> float:
        """The round's total result in dollars, summed over all the player's hands."""
        return sum(self.hand_net(h) for h in self.hands)

    def unseen_counts(self) -> tuple[int, ...]:
        """What the player has not seen: the shoe plus the face-down hole card."""
        counts = list(self._shoe.rank_counts())
        if not self.hole_revealed:
            counts[self._hole_card.rank.blackjack_value - 1] += 1
        return tuple(counts)

    def _draw_visible(self) -> Card:
        card = self._shoe.draw()
        self._counter.observe(card)
        return card
