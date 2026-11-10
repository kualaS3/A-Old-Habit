"""Card counting: a strategy interface, Hi-Lo, and the counter that tracks the count."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .cards import Card, Rank


class CountingSystem(ABC):
    """A counting system assigns a running-count value to each rank."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def value(self, rank: Rank) -> int: ...


class HiLo(CountingSystem):
    """Hi-Lo: 2-6 are +1, 7-9 are 0, tens and aces are -1."""

    @property
    def name(self) -> str:
        return "Hi-Lo"

    def value(self, rank: Rank) -> int:
        v = rank.blackjack_value
        if rank.is_ace or v == 10:
            return -1
        if 2 <= v <= 6:
            return 1
        return 0


class CardCounter:
    """Tracks the running count for whatever CountingSystem it is given."""

    def __init__(self, system: CountingSystem) -> None:
        self._system = system
        self.running_count = 0

    @property
    def system_name(self) -> str:
        return self._system.name

    def observe(self, card: Card) -> None:
        """Call once for every card that becomes visible."""
        self.running_count += self._system.value(card.rank)

    def reset(self) -> None:
        self.running_count = 0

    def true_count(self, decks_remaining: float) -> float:
        """Running count per deck left. Meaningful mainly with several decks."""
        return 0.0 if decks_remaining <= 0 else self.running_count / decks_remaining
