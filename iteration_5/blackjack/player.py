"""The player: a bankroll plus the hands they are playing this round.

Money has two homes, deliberately. The pot lives here on the Player (it survives
every hand and every round); each wager lives on its Hand, because a double changes
that hand's stake and a split creates new wagers. This is also the shape that lets
more players sit down later: each gets their own Player, hands and bankroll alike.
"""
from __future__ import annotations

from .bankroll import Bankroll
from .hand import Hand


class Player:
    def __init__(self, bankroll: Bankroll) -> None:
        self.bankroll = bankroll
        self.hands: list[Hand] = []

    @property
    def exposure(self) -> int:
        """Total dollars at risk across the current hands (doubles counted twice)."""
        return sum(h.stake for h in self.hands)

    def can_cover(self, extra: int) -> bool:
        """True if one more stake of `extra` dollars stays within the balance.

        The balance only moves at settlement, so during a round every gate is
        checked against exposure: you can never end up owing more than you have.
        """
        return self.exposure + extra <= self.bankroll.balance
