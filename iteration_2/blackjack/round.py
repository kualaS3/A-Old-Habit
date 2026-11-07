"""One round of blackjack: deal, player decisions, dealer play, result."""
from __future__ import annotations

from enum import Enum

from .cards import Card
from .counting import CardCounter
from .hand import Hand
from .rules import Rules
from .shoe import Shoe


class Outcome(Enum):
    PLAYER_BLACKJACK = "PLAYER_BLACKJACK"
    DEALER_BLACKJACK = "DEALER_BLACKJACK"
    PUSH = "PUSH"
    WIN = "WIN"
    LOSE = "LOSE"


class Round:
    def __init__(self, shoe: Shoe, counter: CardCounter, rules: Rules) -> None:
        self._shoe = shoe
        self._counter = counter
        self._rules = rules
        self.player = Hand()
        self.dealer = Hand()
        self._hole_card: Card | None = None
        self.hole_revealed = False
        self.peeked = False
        self.doubled = False

    def deal(self) -> None:
        """Two cards each. The dealer's second card (the hole card) stays face down and uncounted."""
        self.player.add(self._draw_visible())
        self.dealer.add(self._draw_visible())
        self.player.add(self._draw_visible())
        self._hole_card = self._shoe.draw()
        self.dealer.add(self._hole_card)

    @property
    def dealer_upcard(self) -> Card:
        return self.dealer.cards[0]

    def peek(self) -> bool:
        """The dealer checks the hole card when showing an ace or a ten-value card.

        Records that the check happened in self.peeked and returns True when the
        dealer has blackjack (the hand is then over before any player decision).
        With any other upcard there is nothing to check and this returns False.
        """
        up = self.dealer_upcard
        if up.rank.is_ace or up.rank.blackjack_value == 10:
            self.peeked = True
            return self.dealer.is_blackjack
        return False

    @property
    def is_over(self) -> bool:
        """True when a natural ends the round before any decision (dealer assumed to peek)."""
        return self.player.is_blackjack or self.dealer.is_blackjack

    @property
    def player_has_decision(self) -> bool:
        return (
            not self.doubled
            and not self.player.is_bust
            and self.player.total < 21
            and not self.is_over
        )

    def unseen_counts(self) -> tuple[int, ...]:
        """What the player has not seen: the shoe plus the face-down hole card."""
        counts = list(self._shoe.rank_counts())
        if not self.hole_revealed:
            counts[self._hole_card.rank.blackjack_value - 1] += 1
        return tuple(counts)

    def player_hit(self) -> Card:
        card = self._draw_visible()
        self.player.add(card)
        return card

    def player_double(self) -> Card:
        """Double down: exactly one more card, on a doubled stake, then the turn ends."""
        card = self._draw_visible()
        self.player.add(card)
        self.doubled = True
        return card

    def dealer_play(self) -> None:
        """Reveal the hole card; the dealer draws only if the player is still in the hand."""
        self.hole_revealed = True
        self._counter.observe(self._hole_card)
        if self.player.is_bust or self.is_over:
            return
        while self._rules.dealer_must_hit(self.dealer.total, self.dealer.is_soft):
            self.dealer.add(self._draw_visible())

    @property
    def outcome(self) -> Outcome:
        p, d = self.player, self.dealer
        if p.is_blackjack and d.is_blackjack:
            return Outcome.PUSH
        if p.is_blackjack:
            return Outcome.PLAYER_BLACKJACK
        if d.is_blackjack:
            return Outcome.DEALER_BLACKJACK
        if p.is_bust:
            return Outcome.LOSE
        if d.is_bust or p.total > d.total:
            return Outcome.WIN
        if p.total < d.total:
            return Outcome.LOSE
        return Outcome.PUSH

    def net_units(self) -> int:
        """The hand's result in bet units: +stake won, -stake lost, 0 on a push.

        The stake is 2 on a doubled hand and 1 otherwise. A player blackjack is
        never doubled (it ends the hand before any decision), so it pays +1.
        """
        stake = 2 if self.doubled else 1
        o = self.outcome
        if o is Outcome.PLAYER_BLACKJACK:
            return 1
        if o is Outcome.WIN:
            return stake
        if o is Outcome.PUSH:
            return 0
        return -stake

    def _draw_visible(self) -> Card:
        card = self._shoe.draw()
        self._counter.observe(card)
        return card
