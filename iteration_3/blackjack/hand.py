"""A blackjack hand (player or dealer).

Iteration 3: a hand now carries its own stake (the bet placed on it, in dollars,
doubled by a double down) and remembers its origin. A natural only counts from the
original deal: a hand created by a split can never be a blackjack, however the cards
land. Hands made by splitting aces are "restricted" - they take exactly one card and
allow no decisions (beyond a resplit if the card is another ace).
"""
from __future__ import annotations

from .cards import Card


class Hand:
    def __init__(
        self, bet: int = 1, *, from_split: bool = False, restricted: bool = False
    ) -> None:
        self._cards: list[Card] = []
        self.bet = bet
        self.doubled = False
        self.from_split = from_split
        self.restricted = restricted

    def add(self, card: Card) -> None:
        self._cards.append(card)

    @property
    def cards(self) -> tuple[Card, ...]:
        return tuple(self._cards)

    @property
    def stake(self) -> int:
        """Dollars at risk on this hand right now: the bet, doubled on a double."""
        return self.bet * (2 if self.doubled else 1)

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
        """A natural: exactly two cards totalling 21 on a hand from the original deal.

        A split hand that lands on 21 with two cards is just 21 - it was created by a
        decision, after the deal, so it pays and compares as an ordinary 21.
        """
        return not self.from_split and len(self._cards) == 2 and self.total == 21

    @property
    def is_pair(self) -> bool:
        """Two cards of equal rank - the only hands that can be split (J+Q cannot)."""
        return len(self._cards) == 2 and self._cards[0].rank is self._cards[1].rank

    def split_hands(self) -> tuple["Hand", "Hand"]:
        """Separate a pair into two new hands, each holding one of the cards.

        Both inherit this hand's bet. Splitting aces makes both hands restricted.
        The caller deals each new hand its extra card.
        """
        first, second = self._cards
        aces = first.rank.is_ace
        h1 = Hand(self.bet, from_split=True, restricted=aces)
        h2 = Hand(self.bet, from_split=True, restricted=aces)
        h1.add(first)
        h2.add(second)
        return h1, h2

    def __str__(self) -> str:
        cards = " ".join(str(c) for c in self._cards)
        soft = "soft " if self.is_soft else ""
        return f"{cards} ({soft}{self.total})"
