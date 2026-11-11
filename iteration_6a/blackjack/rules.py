"""Table rules.

Iteration 1 only needed the dealer's soft-17 behaviour, but keeping the rules in one
object gives later iterations (payouts, doubling, decks, ...) an obvious home.
Iteration 3 adds the split cap alongside it. Iteration 5 caps it at one split
by default (see max_splits): the exact split value cannot be computed with
resplits -- the joint recursion's state space explodes (see the iteration 5
README for the numbers) -- so the default is the case the engine can stand
behind, and raising it is a deliberate choice, not an accident.

Iteration 4: three more table options, all on this object because it is the config:
- surrender_allowed: late surrender on the first decision, on by default.
- blackjack_payout: what a natural pays, 3:2 or 2:1, default 3:2. Only those two
  are allowed - anything else is a table that does not exist, so it is rejected.
- dealer_hits_soft_17: S17 (stand) by default, H17 when True. This field already
  existed. Iteration 4 wires it through the game and the odds and tests it.

Iteration 5: Rules is now the whole table config - the shoe, the money and the
rule toggles live here too, so one object describes the table you are sitting at:
- num_decks: 1-8, default 1. The shoe is built from this.
- cut_cards / cut_fraction: where the cut card goes, in cards dealt. An exact
  cut_cards wins when set. Otherwise the cut is placed at cut_fraction of the
  shoe (3/4, 2/3 or 1/2 - the realistic penetrations) plus a random jitter of
  up to 5 cards either way, re-rolled on every shuffle. A round in progress
  always finishes before the reshuffle.
- burn_card: burn the first card after a shuffle, on by default. The burnt card
  is unseen and uncounted, and it never comes back into play.
- starting_balance / min_bet / max_bet: the player's starting pot and the table
  limits. Defaults $1000, $10 and $1000.
- blackjack_payout: 6:5 joins the options (a real, bad table - the EV drop is
  the point of including it).
- double_after_split / resplit_aces / hit_split_aces: the three split-rule
  toggles that vary by casino. Defaults are the player-friendly ones: double
  after split allowed, resplitting aces allowed, split aces take one card only.
- insurance_allowed: insurance (and even money) on by default.
"""

from __future__ import annotations

from dataclasses import dataclass

CUT_FRACTIONS = (0.75, 2 / 3, 0.5)
CUT_JITTER = 5  # cards either way around the fractional cut, re-rolled per shuffle
MIN_CARDS_AFTER_CUT = 10  # the cut must leave at least this many cards unplayed


@dataclass(frozen=True)
class Rules:
    dealer_hits_soft_17: bool = False
    # Splits per round, so at most max_splits + 1 hands. Default 1: one split,
    # no resplits. That is a deliberate limitation, not an oversight -- the
    # exact split value cannot be computed with resplits (the joint
    # recursion runs out of memory at three splits, and the live
    # decomposition engine takes ~18s at two splits and runs out of memory
    # at three, see the iteration 5 README). One split is the case the
    # engine can verify exactly, in a few seconds. Raise it only knowing that.
    max_splits: int = 1
    surrender_allowed: bool = True
    blackjack_payout: tuple[int, int] = (3, 2)  # profit per unit staked on a natural
    # -- iteration 5: the shoe --
    num_decks: int = 1
    cut_cards: int | None = None  # exact cut, in cards dealt. None = use cut_fraction
    cut_fraction: float = 0.75  # penetration when cut_cards is None
    burn_card: bool = True
    # -- iteration 5: the money --
    starting_balance: float = 1000
    min_bet: int = 10
    max_bet: int = 1000
    # -- iteration 5: the rule toggles --
    double_after_split: bool = True
    resplit_aces: bool = True
    hit_split_aces: bool = False
    insurance_allowed: bool = True

    def __post_init__(self) -> None:
        if self.blackjack_payout not in ((3, 2), (2, 1), (6, 5)):
            raise ValueError(
                "blackjack_payout must be (3, 2), (2, 1) or (6, 5), "
                f"got {self.blackjack_payout}"
            )
        if not 1 <= self.num_decks <= 8:
            raise ValueError(f"num_decks must be 1..8, got {self.num_decks}")
        if self.cut_fraction not in CUT_FRACTIONS:
            raise ValueError(
                f"cut_fraction must be one of {CUT_FRACTIONS}, got {self.cut_fraction}"
            )
        total = self.num_decks * 52
        if self.cut_cards is not None and not (
            1 <= self.cut_cards <= total - MIN_CARDS_AFTER_CUT
        ):
            raise ValueError(
                "cut_cards must leave at least "
                f"{MIN_CARDS_AFTER_CUT} cards unplayed in a {self.num_decks}-deck "
                f"shoe (1..{total - MIN_CARDS_AFTER_CUT}), got {self.cut_cards}"
            )
        if self.min_bet < 1:
            raise ValueError(f"min_bet must be >= 1, got {self.min_bet}")
        if self.max_bet < self.min_bet:
            raise ValueError(
                f"max_bet ({self.max_bet}) must be >= min_bet ({self.min_bet})"
            )
        if self.starting_balance < self.min_bet:
            raise ValueError(
                "starting_balance "
                f"({self.starting_balance}) must cover at least one minimum bet "
                f"({self.min_bet})"
            )
        if self.max_splits < 0:
            raise ValueError(f"max_splits must be >= 0, got {self.max_splits}")

    @property
    def blackjack_payout_ratio(self) -> float:
        """Profit on a natural, per unit staked: 1.5 for 3:2, 2.0 for 2:1, 1.2 for 6:5."""
        num, den = self.blackjack_payout
        return num / den

    @property
    def blackjack_payout_label(self) -> str:
        num, den = self.blackjack_payout
        return f"{num}:{den}"

    @property
    def dealer_rule_label(self) -> str:
        return "H17 (dealer hits soft 17)" if self.dealer_hits_soft_17 else "S17 (dealer stands on soft 17)"

    @property
    def total_cards(self) -> int:
        return self.num_decks * 52

    def dealer_must_hit(self, best_total: int, soft: bool) -> bool:
        """Single source of truth for dealer drawing, used by Round and OddsCalculator."""
        return best_total < 17 or (best_total == 17 and soft and self.dealer_hits_soft_17)
