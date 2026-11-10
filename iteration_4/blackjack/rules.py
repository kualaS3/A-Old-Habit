"""Table rules.

Iteration 1 only needed the dealer's soft-17 behaviour, but keeping the rules in one
object gives later iterations (payouts, doubling, decks, ...) an obvious home.
Iteration 3 adds the split cap alongside it.

Iteration 4: three more table options, all on this object because it is the config:
- surrender_allowed: late surrender on the first decision, on by default.
- blackjack_payout: what a natural pays, 3:2 or 2:1, default 3:2. Only those two
  are allowed - anything else is a table that does not exist, so it is rejected.
- dealer_hits_soft_17: S17 (stand) by default, H17 when True. This field already
  existed; iteration 4 wires it through the game and the odds and tests it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    dealer_hits_soft_17: bool = False
    max_splits: int = 3  # splits per round, so at most max_splits + 1 hands
    surrender_allowed: bool = True
    blackjack_payout: tuple[int, int] = (3, 2)  # profit per unit staked on a natural

    def __post_init__(self) -> None:
        if self.blackjack_payout not in ((3, 2), (2, 1)):
            raise ValueError(
                f"blackjack_payout must be (3, 2) or (2, 1), got {self.blackjack_payout}"
            )

    @property
    def blackjack_payout_ratio(self) -> float:
        """Profit on a natural, per unit staked: 1.5 for 3:2, 2.0 for 2:1."""
        num, den = self.blackjack_payout
        return num / den

    @property
    def blackjack_payout_label(self) -> str:
        num, den = self.blackjack_payout
        return f"{num}:{den}"

    @property
    def dealer_rule_label(self) -> str:
        return "H17 (dealer hits soft 17)" if self.dealer_hits_soft_17 else "S17 (dealer stands on soft 17)"

    def dealer_must_hit(self, best_total: int, soft: bool) -> bool:
        """Single source of truth for dealer drawing, used by Round and OddsCalculator."""
        return best_total < 17 or (best_total == 17 and soft and self.dealer_hits_soft_17)
