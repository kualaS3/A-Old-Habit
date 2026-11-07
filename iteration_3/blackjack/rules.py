"""Table rules.

Iteration 1 only needed the dealer's soft-17 behaviour, but keeping the rules in one
object gives later iterations (payouts, doubling, decks, ...) an obvious home.
Iteration 3 adds the split cap alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    dealer_hits_soft_17: bool = False
    max_splits: int = 3  # splits per round, so at most max_splits + 1 hands

    def dealer_must_hit(self, best_total: int, soft: bool) -> bool:
        """Single source of truth for dealer drawing, used by Round and OddsCalculator."""
        return best_total < 17 or (best_total == 17 and soft and self.dealer_hits_soft_17)
