"""Bet sizing: the Kelly policy and the flat-betting baseline.

Iteration 6b. The Kelly criterion says the bet that maximises the
growth of the bankroll is the edge divided by the variance, times the
bankroll: bet = edge / variance * bankroll. Both inputs are measured,
not guessed: the edge comes from the 6a edge-vs-true-count curve, and
the variance is the 6a round variance. The policy bets a fraction of
that (Rules.kelly_fraction, half Kelly by default), because full Kelly
swings hard enough that most people bet a fraction of it, and the
simulator is the way to see what each fraction costs in growth and
saves in drawdown.

The edge curve is a list of (true-count centre, edge, variance) points,
one per bucket of the 6a curve, and the policy interpolates linearly
between centres. Outside the measured range it holds the nearest end:
below the lowest centre the edge is the lowest measured one, above the
highest it is the highest. A negative edge bets the table minimum, and
every bet is clamped to the table limits and to whole dollars. A bet
can never exceed the bankroll in front of it: the simulator caps it
there, and a bankroll under the table minimum is ruin, not a bet.

FlatBetPolicy is the baseline the Kelly numbers have to beat: the same
amount every hand, whatever the count. It exists so a sizing rule is
only interesting if it beats betting the same amount every hand.
"""

from __future__ import annotations

from .rules import Rules

# One point of the edge curve: (true-count centre, edge per unit bet,
# round variance per unit bet squared).
CurvePoint = tuple[float, float, float]


def _interpolate(points: list[CurvePoint], tc: float, index: int) -> float:
    """Linear interpolation of one column of the curve at this true count.

    index 1 is the edge, index 2 the variance. Outside the measured
    range the nearest end is held, not extrapolated: the edge does not
    keep rising past the highest count measured.
    """
    if tc <= points[0][0]:
        return points[0][index]
    if tc >= points[-1][0]:
        return points[-1][index]
    for (x0, *rest0), (x1, *rest1) in zip(points, points[1:]):
        if x0 <= tc <= x1:
            v0 = rest0[index - 1]
            v1 = rest1[index - 1]
            return v0 + (v1 - v0) * (tc - x0) / (x1 - x0)
    raise AssertionError("interpolation fell through the curve")


def edge_at(curve: list[CurvePoint], tc: float) -> float:
    """The edge per unit bet at this true count, from the curve."""
    return _interpolate(curve, tc, 1)


def variance_at(curve: list[CurvePoint], tc: float) -> float:
    """The round variance per unit bet squared at this true count."""
    return _interpolate(curve, tc, 2)


class KellyPolicy:
    """Bet a fraction of Kelly, from the edge curve and the bankroll.

    bet = kelly_fraction * edge(true_count) / variance(true_count)
    * bankroll, clamped to [min_bet, max_bet] and to whole dollars.
    A negative edge bets the table minimum: the count says the table
    has the edge, so the smallest bet the table allows is the most the
    policy will risk on it.
    """

    def __init__(self, rules: Rules, curve: list[CurvePoint]) -> None:
        self._rules = rules
        self._curve = sorted(curve)

    def bet(self, bankroll: float, true_count: float) -> int:
        """The bet in whole dollars for this bankroll at this count."""
        if bankroll < self._rules.min_bet:
            return 0
        edge = edge_at(self._curve, true_count)
        if edge <= 0.0:
            return self._rules.min_bet
        variance = variance_at(self._curve, true_count)
        if variance <= 0.0:
            return self._rules.min_bet
        raw = self._rules.kelly_fraction * edge / variance * bankroll
        bet = int(round(raw))
        bet = max(self._rules.min_bet, min(self._rules.max_bet, bet))
        return min(bet, int(bankroll))


class FlatBetPolicy:
    """The same bet every hand, whatever the count: the baseline."""

    def __init__(self, rules: Rules, bet: int | None = None) -> None:
        self._rules = rules
        self._bet = bet if bet is not None else rules.min_bet

    def bet(self, bankroll: float, true_count: float) -> int:
        """The fixed bet, or 0 when the bankroll cannot cover the minimum."""
        if bankroll < self._rules.min_bet:
            return 0
        return min(self._bet, int(bankroll))
