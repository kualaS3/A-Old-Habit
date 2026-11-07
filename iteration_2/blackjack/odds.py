"""Exact odds by recursion over the composition of the unseen cards (no simulation).

Model (the player's point of view): "unseen" = the shoe plus the dealer's hole card.
Cards are exchangeable, so drawing the player's next card and the dealer's hole card from
the same unseen pool gives the right probabilities.

STAND = stop now. HIT = take one card, then play the better of stand/hit from there.
DOUBLE = take exactly one card, then stand, on a doubled stake.

The odds are computed after the dealer's peek: when the upcard is an ace or a ten, the
dealer has already checked and does not have blackjack, so the hole card is conditioned
on not being the blackjack-completing card (not a ten under an ace, not an ace under a
ten). Player draws keep the exchangeable treatment above.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .cards import Card
from .hand import Hand
from .rules import Rules

_BUST = 22  # dealer distribution: index 0..21 = final total, 22 = bust


class Action(Enum):
    HIT = "HIT"
    STAND = "STAND"
    DOUBLE = "DOUBLE"


@dataclass(frozen=True)
class OutcomeOdds:
    """Win/push/lose probabilities. EV is per 1 unit staked, even-money payout."""

    win: float
    push: float
    lose: float
    stake: float = 1.0

    @property
    def ev(self) -> float:
        return (self.win - self.lose) * self.stake

    def scaled(self, factor: float) -> OutcomeOdds:
        return OutcomeOdds(
            self.win * factor, self.push * factor, self.lose * factor, self.stake
        )

    def __add__(self, other: OutcomeOdds) -> OutcomeOdds:
        return OutcomeOdds(
            self.win + other.win,
            self.push + other.push,
            self.lose + other.lose,
            self.stake,
        )


NO_ODDS = OutcomeOdds(0.0, 0.0, 0.0)
LOSS = OutcomeOdds(0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Analysis:
    """The odds for every available action at one decision point."""

    by_action: dict[Action, OutcomeOdds]

    def odds(self, action: Action) -> OutcomeOdds:
        return self.by_action[action]

    def offers(self, action: Action) -> bool:
        return action in self.by_action

    def best(self) -> Action:
        """Highest-EV action; STAND wins ties."""
        best = Action.STAND
        for action, o in self.by_action.items():
            if o.ev > self.by_action[best].ev + 1e-12:
                best = action
        return best


def _best_total(hard: int, has_ace: bool) -> int:
    return hard + 10 if has_ace and hard + 10 <= 21 else hard


class OddsCalculator:
    """Not thread-safe: analyze() reuses internal caches."""

    def __init__(self, rules: Rules) -> None:
        self._rules = rules
        self._up_value = 0
        self._up_is_ace = False
        self._dealer_cache: dict[tuple, list[float]] = {}
        self._player_cache: dict[tuple, OutcomeOdds] = {}

    def analyze(self, player: Hand, dealer_upcard: Card, unseen: tuple[int, ...]) -> Analysis:
        """unseen: composition of unseen cards, indexed by blackjack value - 1
        (shoe contents plus the dealer's hole card)."""
        self._dealer_cache.clear()
        self._player_cache.clear()
        self._up_value = dealer_upcard.rank.blackjack_value
        self._up_is_ace = dealer_upcard.rank.is_ace

        hard, ace = player.hard_total, player.has_ace
        result = {Action.STAND: self._stand_odds(unseen, _best_total(hard, ace))}
        if sum(unseen) > 0:
            result[Action.HIT] = self._hit_odds(unseen, hard, ace)
            if len(player.cards) == 2:
                result[Action.DOUBLE] = self._double_odds(unseen, hard, ace)
        return Analysis(result)

    # ---- player side ---------------------------------------------------------------------

    def _peek_excluded_value(self) -> int | None:
        """The hole-card value the peek rules out, if the upcard can make blackjack."""
        if self._up_is_ace:
            return 10
        if self._up_value == 10:
            return 1
        return None

    def _stand_odds(self, counts: tuple[int, ...], player_total: int) -> OutcomeOdds:
        d = self._dealer_distribution(
            counts, self._up_value, self._up_is_ace, exclude_first=self._peek_excluded_value()
        )
        win = d[_BUST] + sum(d[:player_total])
        push = d[player_total]
        lose = sum(d[player_total + 1 : _BUST])
        return OutcomeOdds(win, push, lose)

    def _hit_odds(self, counts: tuple[int, ...], hard: int, ace: bool) -> OutcomeOdds:
        """Take exactly one card, then continue optimally. Caller guarantees cards remain."""
        n = sum(counts)
        acc = NO_ODDS
        for v in range(1, 11):
            c = counts[v - 1]
            if c == 0:
                continue
            nxt = counts[: v - 1] + (c - 1,) + counts[v:]
            new_hard = hard + v
            if new_hard > 21:
                outcome = LOSS
            else:
                outcome = self._optimal_play(nxt, new_hard, ace or v == 1)
            acc = acc + outcome.scaled(c / n)
        return acc

    def _double_odds(self, counts: tuple[int, ...], hard: int, ace: bool) -> OutcomeOdds:
        """Take exactly one card, then stand, with the stake doubled.

        The win/push/lose figures are the same probabilities as standing after one
        card; only the EV doubles, via stake=2.
        """
        n = sum(counts)
        acc = NO_ODDS
        for v in range(1, 11):
            c = counts[v - 1]
            if c == 0:
                continue
            nxt = counts[: v - 1] + (c - 1,) + counts[v:]
            new_hard = hard + v
            if new_hard > 21:
                outcome = LOSS
            else:
                outcome = self._stand_odds(nxt, _best_total(new_hard, ace or v == 1))
            acc = acc + outcome.scaled(c / n)
        return OutcomeOdds(acc.win, acc.push, acc.lose, stake=2.0)

    def _optimal_play(self, counts: tuple[int, ...], hard: int, ace: bool) -> OutcomeOdds:
        """Best of standing or hitting again from this state."""
        key = (counts, hard, ace)
        cached = self._player_cache.get(key)
        if cached is not None:
            return cached
        best = _best_total(hard, ace)
        result = self._stand_odds(counts, best)
        if best < 21 and sum(counts) > 0:
            hit = self._hit_odds(counts, hard, ace)
            if hit.ev > result.ev:
                result = hit
        self._player_cache[key] = result
        return result

    # ---- dealer side ---------------------------------------------------------------------

    def _dealer_distribution(
        self,
        counts: tuple[int, ...],
        hard: int,
        ace: bool,
        exclude_first: int | None = None,
    ) -> list[float]:
        """Distribution of the dealer's final result. Returned lists are read-only.

        exclude_first: on this (outermost) call only, the first card drawn — which
        models the hole card — may not have this blackjack value, and the draw is
        renormalised without it. That is the peek condition. Deeper draws are
        unconditioned. If every unseen card has the excluded value there is nothing
        to condition on and the draw falls back to unconditioned.
        """
        out = [0.0] * (_BUST + 1)
        if hard > 21:
            out[_BUST] = 1.0
            return out
        best = _best_total(hard, ace)
        n = sum(counts)
        if n == 0 or not self._rules.dealer_must_hit(best, best != hard):
            out[best] = 1.0
            return out

        if exclude_first is not None and n - counts[exclude_first - 1] == 0:
            exclude_first = None
        if exclude_first is None:
            key = (counts, hard, ace)
            cached = self._dealer_cache.get(key)
            if cached is not None:
                return cached
        denom = n if exclude_first is None else n - counts[exclude_first - 1]
        for v in range(1, 11):
            if v == exclude_first:
                continue
            c = counts[v - 1]
            if c == 0:
                continue
            nxt = counts[: v - 1] + (c - 1,) + counts[v:]
            sub = self._dealer_distribution(nxt, hard + v, ace or v == 1)
            p = c / denom
            for i, s in enumerate(sub):
                if s:
                    out[i] += p * s
        if exclude_first is None:
            self._dealer_cache[key] = out
        return out
