"""Table rules.

Iteration 1 only needs the dealer's soft-17 behaviour, but keeping the rules in one
object gives later iterations (payouts, doubling, decks, ...) an obvious home.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    dealer_hits_soft_17: bool = False

    def dealer_must_hit(self, best_total: int, soft: bool) -> bool:
        """Single source of truth for dealer drawing, used by Round and OddsCalculator."""
        return best_total < 17 or (best_total == 17 and soft and self.dealer_hits_soft_17)
