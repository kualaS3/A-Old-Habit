"""Measure the edge vs true count curve (iteration 6a) and print the table.

Usage, from the iteration_6a folder:
    python measure_edge_curve.py [--decks 6] [--compositions 1] [--seed 20261111]
    [--buckets 0,1,2,3,4,5,6] [--pair-cache /tmp/pair_cache_6d.json]

Each bucket is measured by sampling a shoe composition at its centre
and enumerating the fresh-round edge exactly over every (player hand,
upcard) combination weighted by its deal probability, with optimal play,
the peek and the insurance choice included. Pairs are valued exactly by
the C joint moments engine, except the slow low pairs (2 through 6),
which are valued by their full-shoe premium when --pair-cache is given
(warm it first with python warm_pair_cache.py). Without it every pair
is exact and a bucket takes about seventeen minutes on a six-deck shoe,
against about a minute and a half with the cache. The table it prints is the
one recorded in the README: mean EV per unit bet, the spread across
compositions when more than one is used, the mean conditional round
variance per unit bet squared, and the mean total round variance, which
adds the spread of the hand EVs across deals and is the number Kelly
sizing reads. A full seven-bucket run takes about twelve minutes on
a six-deck shoe with the cache, so --buckets can select a subset by
index (0 is the lowest bucket).
"""

from __future__ import annotations

import argparse
import random
import time

from blackjack.edge_curve import EDGE_BUCKETS, load_pair_cache, measure_bucket
from blackjack.rules import Rules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decks", type=int, default=6)
    parser.add_argument("--compositions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--buckets", type=str, default=None,
                        help="comma-separated bucket indices, default all")
    parser.add_argument("--pair-cache", type=str, default=None,
                        help="JSON premium cache from warm_pair_cache.py")
    args = parser.parse_args()
    rules = Rules(num_decks=args.decks)
    rng = random.Random(args.seed)
    pair_cache = load_pair_cache(args.pair_cache) if args.pair_cache else None
    if args.buckets is None:
        indices = list(range(len(EDGE_BUCKETS)))
    else:
        indices = [int(i) for i in args.buckets.split(",")]
    started = time.time()
    print(f"# edge curve: {args.decks} decks, S17, DAS, late surrender, "
          f"{rules.blackjack_payout_label}, seed {args.seed}, "
          f"{args.compositions} composition(s) per bucket")
    print(f"# {'bucket':>10} {'n':>4} {'mean_ev':>9} {'se_ev':>7} {'mean_var':>9} {'mean_total_var':>14}")
    for i in indices:
        low, high = EDGE_BUCKETS[i]
        row = measure_bucket(rng, EDGE_BUCKETS[i], args.decks, rules,
                             compositions=args.compositions,
                             pair_cache=pair_cache)
        print(f"# [{low:+.0f},{high:+.0f}) {row['n']:>4} "
              f"{row['mean_ev']:>+9.4f} {row['se_ev']:>7.4f} {row['mean_var']:>9.4f} "
              f"{row['mean_total_var']:>14.4f}",
              flush=True)
    print(f"# {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
