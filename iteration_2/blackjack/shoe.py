"""The shoe: one or more shuffled decks."""
from __future__ import annotations

import random

from .cards import Card, Rank, Suit

CARDS_PER_DECK = 52


class Shoe:
    """Holds, shuffles and deals cards.

    Iteration 1 uses one deck, but num_decks is already a parameter so later
    iterations need no redesign. Pass a seeded random.Random for reproducible shuffles.
    """

    def __init__(self, num_decks: int = 1, rng: random.Random | None = None) -> None:
        if num_decks < 1:
            raise ValueError("num_decks must be >= 1")
        self.num_decks = num_decks
        self._rng = rng or random.Random()
        self._cards: list[Card] = []
        self.reshuffle()

    def reshuffle(self) -> None:
        """Rebuild every deck and shuffle."""
        self._cards = [
            Card(rank, suit)
            for _ in range(self.num_decks)
            for suit in Suit
            for rank in Rank
        ]
        self._rng.shuffle(self._cards)

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
