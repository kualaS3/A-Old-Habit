"""The player's money: a balance that settles each round's net result.

Iteration 3: bet sizing is variable, so the bet no longer lives here - it is chosen
per hand and carried by the Hand itself. The bankroll is just the pot: it only moves
when a round settles, by the round's net dollars (won positive, lost negative).
"""
from __future__ import annotations


class Bankroll:
    """Tracks the balance across a session."""

    def __init__(self, starting_balance: int = 100) -> None:
        if starting_balance < 0:
            raise ValueError("starting_balance must be >= 0")
        self._balance = starting_balance

    @property
    def balance(self) -> int:
        return self._balance

    def settle(self, net_dollars: int) -> int:
        """Apply a round's net result in dollars; returns the new balance."""
        self._balance += net_dollars
        return self._balance
