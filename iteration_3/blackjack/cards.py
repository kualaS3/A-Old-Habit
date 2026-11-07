"""Card data: ranks, suits and the immutable Card."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Rank(Enum):
    """A card rank. Ace is valued 1 here; Hand promotes it to 11 when that is safe."""

    ACE = ("A", 1)
    TWO = ("2", 2)
    THREE = ("3", 3)
    FOUR = ("4", 4)
    FIVE = ("5", 5)
    SIX = ("6", 6)
    SEVEN = ("7", 7)
    EIGHT = ("8", 8)
    NINE = ("9", 9)
    TEN = ("10", 10)
    JACK = ("J", 10)
    QUEEN = ("Q", 10)
    KING = ("K", 10)

    @property
    def symbol(self) -> str:
        return self.value[0]

    @property
    def blackjack_value(self) -> int:
        return self.value[1]

    @property
    def is_ace(self) -> bool:
        return self is Rank.ACE


class Suit(Enum):
    """Suits only matter for display; the maths never looks at them."""

    SPADES = "S"
    HEARTS = "H"
    DIAMONDS = "D"
    CLUBS = "C"


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank.symbol}{self.suit.value}"
