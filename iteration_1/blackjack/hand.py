"""A blackjack hand (player or dealer)."""
from __future__ import annotations

from .cards import Card


class Hand:
    def __init__(self) -> None:
        self._cards: list[Card] = []

    def add(self, card: Card) -> None:
        self._cards.append(card)

    @property
    def cards(self) -> tuple[Card, ...]:
        return tuple(self._cards)

    @property
    def hard_total(self) -> int:
        """Total counting every ace as 1."""
        return sum(c.rank.blackjack_value for c in self._cards)

    @property
    def has_ace(self) -> bool:
        return any(c.rank.is_ace for c in self._cards)

    @property
    def is_soft(self) -> bool:
        """True if an ace is currently counted as 11."""
        return self.has_ace and self.hard_total + 10 <= 21

    @property
    def total(self) -> int:
        """Best total: one ace is worth 11 whenever that does not bust the hand."""
        return self.hard_total + 10 if self.is_soft else self.hard_total

    @property
    def is_bust(self) -> bool:
        return self.hard_total > 21

    @property
    def is_blackjack(self) -> bool:
        return len(self._cards) == 2 and self.total == 21

    def __str__(self) -> str:
        cards = " ".join(str(c) for c in self._cards)
        soft = "soft " if self.is_soft else ""
        return f"{cards} ({soft}{self.total})"
