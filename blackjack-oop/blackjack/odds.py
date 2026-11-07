"""Exact odds by recursion over the composition of the unseen cards (no simulation).

Model (the player's point of view): "unseen" = the shoe plus the dealer's hole card.
Cards are exchangeable, so drawing the player's next card and the dealer's hole card from
the same unseen pool gives the right probabilities.

STAND = stop now. HIT = take one card, then play the better of stand/hit from there.
Known simplification for iteration 1: the dealer's blackjack peek is not conditioned on.
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


@dataclass(frozen=True)
class OutcomeOdds:
    """Win/push/lose probabilities. EV is per 1 unit staked, even-money payout."""

    win: float
    push: float
    lose: float

    @property
    def ev(self) -> float:
        return self.win - self.lose

    def scaled(self, factor: float) -> OutcomeOdds:
        return OutcomeOdds(self.win * factor, self.push * factor, self.lose * factor)

    def __add__(self, other: OutcomeOdds) -> OutcomeOdds:
        return OutcomeOdds(self.win + other.win, self.push + other.push, self.lose + other.lose)


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
        return Analysis(result)

    # ---- player side ---------------------------------------------------------------------

    def _stand_odds(self, counts: tuple[int, ...], player_total: int) -> OutcomeOdds:
        d = self._dealer_distribution(counts, self._up_value, self._up_is_ace)
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

    def _dealer_distribution(self, counts: tuple[int, ...], hard: int, ace: bool) -> list[float]:
        """Distribution of the dealer's final result. Returned lists are read-only."""
        out = [0.0] * (_BUST + 1)
        if hard > 21:
            out[_BUST] = 1.0
            return out
        best = _best_total(hard, ace)
        n = sum(counts)
        if n == 0 or not self._rules.dealer_must_hit(best, best != hard):
            out[best] = 1.0
            return out

        key = (counts, hard, ace)
        cached = self._dealer_cache.get(key)
        if cached is not None:
            return cached
        for v in range(1, 11):
            c = counts[v - 1]
            if c == 0:
                continue
            nxt = counts[: v - 1] + (c - 1,) + counts[v:]
            sub = self._dealer_distribution(nxt, hard + v, ace or v == 1)
            p = c / n
            for i, s in enumerate(sub):
                if s:
                    out[i] += p * s
        self._dealer_cache[key] = out
        return out
