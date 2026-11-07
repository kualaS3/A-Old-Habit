"""The player's money: a starting balance and one flat bet, hand after hand."""
from __future__ import annotations


class Bankroll:
    """Tracks the balance across a session. Every hand bets the same flat amount;
    a double down needs the balance to cover twice that bet."""

    def __init__(self, starting_balance: int = 100, bet: int = 1) -> None:
        if bet < 1:
            raise ValueError("bet must be >= 1")
        if starting_balance < 0:
            raise ValueError("starting_balance must be >= 0")
        self._balance = starting_balance
        self._bet = bet

    @property
    def balance(self) -> int:
        return self._balance

    @property
    def bet(self) -> int:
        return self._bet

    def can_play(self) -> bool:
        """True while the balance covers one more flat bet."""
        return self._balance >= self._bet

    def can_double(self) -> bool:
        """True while the balance covers a doubled stake."""
        return self._balance >= 2 * self._bet

    def settle(self, net_units: int) -> int:
        """Apply a hand's result (in bet units) to the balance; returns the balance."""
        self._balance += net_units * self._bet
        return self._balance
