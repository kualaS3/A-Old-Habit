"""The player's money: a balance that settles each round's net result.

Iteration 3: bet sizing is variable, so the bet no longer lives here - it is chosen
per hand and carried by the Hand itself. The bankroll is just the pot: it only moves
when a round settles, by the round's net dollars (won positive, lost negative).

Iteration 4: the balance is a float, not an int. A 3:2 blackjack on an odd bet
pays $1.50, and a surrender gives back half a stake - both are fifty-cent amounts,
so whole dollars can no longer hold the balance exactly. Bets are still whole
dollars. Only the pot can hold halves.
"""
from __future__ import annotations


class Bankroll:
    """Tracks the balance across a session."""

    def __init__(self, starting_balance: float = 100) -> None:
        if starting_balance < 0:
            raise ValueError("starting_balance must be >= 0")
        self._balance = float(starting_balance)

    @property
    def balance(self) -> float:
        return self._balance

    def settle(self, net_dollars: float) -> float:
        """Apply a round's net result in dollars. Returns the new balance."""
        self._balance += net_dollars
        return self._balance
