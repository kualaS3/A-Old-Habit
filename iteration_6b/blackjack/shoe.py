"""The shoe: one or more shuffled decks, with a cut card and a burn card.

Iteration 1 uses one deck, but num_decks is already a parameter so later
iterations need no redesign. Pass a seeded random.Random for reproducible shuffles.

Iteration 5: the cut card and the burn card.
- The cut is a number of cards dealt, set on every reshuffle: an exact
  cut_cards when the table config gives one, otherwise cut_fraction of the shoe
  plus a random jitter of up to 5 cards either way (re-rolled per shuffle, so
  the penetration is realistic rather than exact). cut_reached is True once that
  many cards have been dealt - the round in progress always finishes first, and
  the caller reshuffles between rounds.
- The burn card is the first card off the top after a shuffle. It is discarded
  unseen and uncounted: it is simply not in the shoe any more, so it never
  appears in rank_counts and never enters the odds' unseen pool - which is
  exactly right, because it can never be drawn.
"""
from __future__ import annotations

import random

from .cards import Card, Rank, Suit
from .rules import CUT_JITTER, MIN_CARDS_AFTER_CUT

CARDS_PER_DECK = 52


class Shoe:
    """Holds, shuffles and deals cards."""

    def __init__(
        self,
        num_decks: int = 1,
        rng: random.Random | None = None,
        *,
        cut_cards: int | None = None,
        cut_fraction: float = 0.75,
        burn_card: bool = True,
    ) -> None:
        if num_decks < 1:
            raise ValueError("num_decks must be >= 1")
        self.num_decks = num_decks
        self._rng = rng or random.Random()
        self._cut_cards = cut_cards
        self._cut_fraction = cut_fraction
        self._burn_card = burn_card
        self._cards: list[Card] = []
        self.cut_position: int = 0  # cards dealt before the cut, set by reshuffle
        self.burnt: Card | None = None  # the burnt card, if any (unseen, uncounted)
        self.reshuffle()

    @property
    def total_cards(self) -> int:
        return self.num_decks * CARDS_PER_DECK

    def reshuffle(self) -> None:
        """Rebuild every deck, shuffle, place the cut, burn the first card."""
        self._cards = [
            Card(rank, suit)
            for _ in range(self.num_decks)
            for suit in Suit
            for rank in Rank
        ]
        self._rng.shuffle(self._cards)
        if self._cut_cards is not None:
            self.cut_position = self._cut_cards
        else:
            jitter = self._rng.randint(-CUT_JITTER, CUT_JITTER)
            cut = round(self._cut_fraction * self.total_cards) + jitter
            self.cut_position = max(
                1, min(cut, self.total_cards - MIN_CARDS_AFTER_CUT)
            )
        self.burnt = self._cards.pop() if self._burn_card else None

    @property
    def cut_reached(self) -> bool:
        """True once the cut position has been dealt through.

        Measured in cards remaining: the cut sits cut_position cards into the
        shoe, so it is reached when that many cards are gone (the burn card
        counts as gone too - it was dealt off the top at the shuffle).
        """
        return self.total_cards - len(self._cards) >= self.cut_position

    def draw(self) -> Card:
        if not self._cards:
            raise RuntimeError("Shoe is empty")
        return self._cards.pop()

    @property
    def cards_remaining(self) -> int:
        return len(self._cards)

    @property
    def decks_remaining(self) -> float:
        return len(self._cards) / CARDS_PER_DECK

    def rank_counts(self) -> tuple[int, ...]:
        """Composition of the remaining cards, indexed by blackjack value - 1.

        counts[v - 1] = number of remaining cards worth v, so index 9 holds all the
        10/J/Q/K.
        """
        counts = [0] * 10
        for card in self._cards:
            counts[card.rank.blackjack_value - 1] += 1
        return tuple(counts)
