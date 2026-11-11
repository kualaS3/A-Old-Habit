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

    @property
    def is_over(self) -> bool:
        """True when a natural ends the round before any decision (dealer assumed to peek)."""
        return self.player.is_blackjack or self.dealer.is_blackjack

    @property
    def player_has_decision(self) -> bool:
        return not self.player.is_bust and self.player.total < 21 and not self.is_over

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

    def dealer_play(self) -> None:
        """Reveal the hole card. The dealer draws only if the player is still in the hand."""
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

    def _draw_visible(self) -> Card:
        card = self._shoe.draw()
        self._counter.observe(card)
        return card
