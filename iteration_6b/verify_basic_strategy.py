"""Verify the hardcoded basic strategy table against the exact engine.

Usage, from the iteration_6b folder:
    python verify_basic_strategy.py [--decks 6] [--keep-cache]

Generates the (hard, soft, pairs) tables from the exact engine, one
representative hand per cell on a full shoe, and prints every cell
where the generated table disagrees with the hardcoded one in
blackjack/basic_strategy.py. A disagreement is usually a close cell:
the engine's answer for one representative hand can differ from the
published table when the two best actions are within a hundredth, or
when the composition of the representative hand matters. Each one is
worth a look, not an automatic correction: the table is the published
strategy for the game, and the engine is the cross-check. Without
--keep-cache each cell gets a fresh calculator, which is slower (about
three minutes on a six-deck shoe) but keeps the memory small enough to
run alongside the edge-curve measurement.
"""

from __future__ import annotations

import argparse
import time

from blackjack.basic_strategy import HARD, PAIRS, SOFT, diff_tables, generate_table
from blackjack.odds import OddsCalculator
from blackjack.rules import Rules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decks", type=int, default=6)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    rules = Rules(num_decks=args.decks)
    calc = OddsCalculator(rules, keep_dealer_cache=args.keep_cache)
    started = time.time()
    generated = generate_table(rules, calc)
    diffs = diff_tables((HARD, SOFT, PAIRS), generated)
    cells = sum(len(row) for t in (HARD, SOFT, PAIRS) for row in t.values())
    print(f"# {cells} cells checked, {len(diffs)} disagreements, "
          f"{time.time() - started:.1f}s")
    for section, total_value, up_value, table_action, engine_action in diffs:
        print(f"# {section} {total_value} vs {up_value}: "
              f"table {table_action.name}, engine {engine_action.name}")


if __name__ == "__main__":
    main()
