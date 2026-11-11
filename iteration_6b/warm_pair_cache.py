"""Warm the slow-pair premium cache for the edge curve (iteration 6a).

Usage, from the iteration_6a folder:
    python warm_pair_cache.py [--decks 6] [--out /tmp/pair_cache_6d.json]

Measures the split premium and split variance of every low pair (2
through 6) against every upcard, on a full shoe, and writes the cache
as JSON. The low pairs are the slow ones in the C moments engine -- a
pair of 2s against a 6 takes about 47 seconds -- so the curve script
values them by premium instead of exactly, while every other pair
stays exact. Fifty C moments calls, about eighteen minutes on a
six-deck shoe, most of it the 2s and 3s.

Also prints a bias check: the premium of 2,2 against a 6 on a high
count composition (TC +3 to +5), against the full-shoe premium. The
difference is the bias the premium shortcut puts on that pair in an
extreme bucket, and the README records it.
"""

from __future__ import annotations

import argparse
import os
import random
import time

from blackjack import split_exact
from blackjack.edge_curve import (
    SLOW_PAIR_VALUES,
    load_pair_cache,
    sample_composition,
    save_pair_cache,
    warm_slow_pair_cache,
)
from blackjack.moments import _split_premium_and_var
from blackjack.odds import OddsCalculator
from blackjack.rules import Rules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decks", type=int, default=6)
    parser.add_argument("--out", type=str, default="/tmp/pair_cache_6d.json")
    args = parser.parse_args()
    rules = Rules(num_decks=args.decks)
    calc = OddsCalculator(rules)
    started = time.time()
    # Resume from a partial cache if one is already on disk, and save
    # after each rank, so an interrupted run picks up where it left off
    # instead of starting over: the whole warm-up is about eighteen
    # minutes, and a restart halfway would otherwise lose it all.
    cache = load_pair_cache(args.out) if os.path.exists(args.out) else {}
    for rank_value in SLOW_PAIR_VALUES:
        cache = warm_slow_pair_cache(rules, calc, cache, ranks=(rank_value,))
        save_pair_cache(cache, args.out)
        done = sorted({k[0] for k in cache})
        print(f"# ranks done: {done} ({len(cache)} keys, "
              f"{time.time() - started:.1f}s)", flush=True)
    print(f"# warmed {len(cache)} slow-pair premiums in "
          f"{time.time() - started:.1f}s, wrote {args.out}", flush=True)
    # Bias check: the 2,2 vs 6 premium on a high-count composition.
    rng = random.Random(22)
    comp, _running = sample_composition(rng, (3.0, 5.0), args.decks)
    counts = list(comp)
    counts[1] -= 2
    counts[5] -= 1
    unseen = tuple(counts)
    from blackjack.cards import Card, Rank, Suit
    from blackjack.hand import Hand

    hand = Hand()
    hand.add(Card(Rank.TWO, Suit.SPADES))
    hand.add(Card(Rank.TWO, Suit.HEARTS))
    upcard = Card(Rank.SIX, Suit.CLUBS)
    analysis = calc.analyze(hand, upcard, unseen, allow_split=False)
    best_nonsplit = max(o.ev for o in analysis.by_action.values())
    ev_split, _var = split_exact.split_moments(
        unseen, 2, False, 6, False, None, rules
    )
    premium_high = ev_split - best_nonsplit
    premium_full, _var_full = cache[(2, 6)]
    print(f"# bias check 2,2 vs 6: premium full shoe {premium_full:+.4f}, "
          f"high count {premium_high:+.4f}, move {premium_high - premium_full:+.4f}",
          flush=True)
    print(f"# {time.time() - started:.1f}s total")


if __name__ == "__main__":
    main()
