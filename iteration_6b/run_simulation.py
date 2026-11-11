"""Run the batch simulator: Kelly vs flat betting, and print the numbers.

Usage, from the iteration_6b folder:
    python run_simulation.py [--bankroll 10000] [--rounds 2000]
        [--sessions 50] [--seed 20261111] [--kelly-fraction 0.5]
        [--curve curve.json]

Runs the same seeded sessions twice, once with the Kelly policy and
once with flat betting, and prints the risk of ruin, the mean growth,
the mean max drawdown and the mean ending bankroll for each, plus the
textbook analytic risk of ruin as a cross-check. The edge curve comes
from the 6a measurement (blackjack/edge_curve.py, MEASURED_EDGE_CURVE)
unless --curve gives a JSON file of [true-count centre, edge,
variance] points instead.

A session is 2000 rounds by default, which is a long sitting at a
table. 50 sessions is enough for the risk of ruin to settle to a few
percent. The run takes about a minute.
"""

from __future__ import annotations

import argparse
import json
import random

from blackjack.basic_strategy import BasicStrategy
from blackjack.edge_curve import MEASURED_EDGE_CURVE
from blackjack.kelly import FlatBetPolicy, KellyPolicy
from blackjack.rules import Rules
from blackjack.simulator import (
    SessionConfig,
    analytic_risk_of_ruin,
    run_sessions,
)


def _mean_edge(curve, lo: float = -3.0, hi: float = 5.0) -> float:
    """The curve's mean edge over the measured range, for the analytic check."""
    edges = [e for tc, e, _v in curve if lo <= tc <= hi]
    return sum(edges) / len(edges) if edges else 0.0


def _mean_variance(curve, lo: float = -3.0, hi: float = 5.0) -> float:
    varis = [v for tc, _e, v in curve if lo <= tc <= hi]
    return sum(varis) / len(varis) if varis else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bankroll", type=float, default=10000)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--sessions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--kelly-fraction", type=float, default=0.5)
    parser.add_argument("--curve", type=str, default=None)
    args = parser.parse_args()
    if args.curve:
        with open(args.curve, encoding="utf-8") as f:
            curve = [tuple(p) for p in json.load(f)]
    else:
        curve = MEASURED_EDGE_CURVE
    if curve is None:
        raise SystemExit(
            "no edge curve: pass --curve curve.json, or set MEASURED_EDGE_CURVE "
            "in blackjack/edge_curve.py from the 6a measurement"
        )
    rules = Rules(
        num_decks=6,
        starting_balance=args.bankroll,
        kelly_fraction=args.kelly_fraction,
    )
    strategy = BasicStrategy(rules)
    session = SessionConfig(rounds=args.rounds)
    kelly = run_sessions(
        rules, strategy, KellyPolicy(rules, curve), session, args.sessions, args.seed
    )
    flat = run_sessions(
        rules, strategy, FlatBetPolicy(rules, rules.min_bet), session, args.sessions, args.seed
    )
    print(f"# {args.sessions} sessions x {args.rounds} rounds, 6 decks, "
          f"S17 DAS LS 3:2, bankroll ${args.bankroll:,.0f}, "
          f"half Kelly ({args.kelly_fraction}) vs flat ${rules.min_bet}, "
          f"seed {args.seed}")
    print(f"# {'policy':>8} {'risk_of_ruin':>13} {'mean_growth':>11} "
          f"{'mean_drawdown':>14} {'mean_ending':>11}")
    for name, out in (("Kelly", kelly), ("flat", flat)):
        print(f"# {name:>8} {out['risk_of_ruin']:>13.3f} "
              f"{out['mean_growth']:>+11.3f} {out['mean_max_drawdown']:>14.1f} "
              f"{out['mean_ending_bankroll']:>11.1f}")
    mean_edge = _mean_edge(curve)
    mean_var = _mean_variance(curve)
    print(f"# analytic risk of ruin at the curve mean edge {mean_edge:+.4f} "
          f"and variance {mean_var:.3f}, flat ${rules.min_bet} bet: "
          f"{analytic_risk_of_ruin(args.bankroll, mean_edge, mean_var, rules.min_bet):.3f} "
          f"(cross-check only: the edge and the bet are not constant)")


if __name__ == "__main__":
    main()
