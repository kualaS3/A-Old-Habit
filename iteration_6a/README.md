## Iteration 6a

Kelly sizing needs two numbers for every round, the edge and the variance. So far this project only has one of them. Iteration 5 made the edge exact. This iteration measures the other one.

It also measures how the edge moves with the count. The edge is not one number for the whole shoe. It is negative when the count is low and positive when the count is high, and the Kelly bet has to follow it. This iteration maps that curve, using the exact engine, so the sizing rule in the next iteration has real inputs instead of a guessed edge.

- **Round variance, exact.** The variance of a round, per unit bet, under optimal play. A plain hand gets it from the win, push, lose and blackjack probabilities already in `OutcomeOdds`: E[X^2] = win + lose + blackjack_win * payout^2, times the stake squared, and the variance is that minus the EV squared. It is exposed on the odds layer as `second_moment` and `variance`, so every action carries both numbers now. Surrender is a certain -0.5, so its second moment is 0.25 and its variance is 0. A double puts two units at risk, so its second moment is four times win + lose. The measured values, on one deck:

  | Hand | EV | Variance |
  |---|---|---|
  | Stand 20 vs 6 | +0.6974 | 0.4105 |
  | Natural vs 6 | +1.5000 | 0.0000 |
  | Double 11 vs 6 (stake 2) | +0.7614 | 3.1749 |

  Splits are the hard part, because the two hands are correlated through the one dealer hand: there is no honest per-hand split of the variance. The C joint engine, which has both hands and the dealer in one recursion, now carries second moments alongside the EV (`split_moments_exact` in `blackjack/split_exact.c`, loaded through `blackjack/split_exact.py`). It is a measurement tool here, not the live per-decision engine: the joint is what can see the cross term between the two hands, and its cost at one split on a full deck is about a second, which is fine for measuring and far too slow for playing. The joint moments EV agrees with the joint EV exactly (to 2e-16 on every spot checked), and the A,A vs 5 variance is checked twice over: once by an independent enumeration of the split in Python, once by Monte Carlo simulation of the real game. Measured split variances, one deck, one split:

  | Split | EV | Round variance |
  |---|---|---|
  | 8,8 vs 6 | +0.3420 | 4.7060 |
  | 8,8 vs 10 (peek) | -0.4521 | 3.0735 |
  | A,A vs 5 | +0.7322 | 2.2955 |

  Splitting roughly triples the round variance against a plain hand (about 1.3), because two stakes move together against one dealer hand. The small-pool numbers (one of each value, four tens, three splits) are higher still: 8,8 vs 6 has variance 5.4091 there, A,A vs 5 has 2.4821. The full-deck three-split moments run out of memory, the same wall as the joint EV in iteration 5, so three splits are never measured on a full deck. The small pool is the only place they are.

- **The whole round, not just the hand.** `fresh_round_moments` in `blackjack/moments.py` is the EV and the variance of one round from the deal, per unit bet, under optimal play, including everything that moves the money: the peek branch (the hand is only played when the dealer has no blackjack, and loses its single stake on the blackjack branch), doubles (the stake changes mid-hand), splits (the joint moments engine, above), insurance (taken only when its EV is positive, which on a full shoe it never is) and even money (taken only when a certain +1 beats declining, which at 3:2 it does not). A few spots, all checked against hand computations:

  | Round | EV | Variance |
  |---|---|---|
  | Blackjack vs 6 | +1.5000 | 0.0000 |
  | Blackjack vs ace (declining even money) | +1.0408 | 0.4779 |
  | 20 vs 6 | +0.6974 | 0.4105 |
  | 16 vs 10 (peek branch) | -0.5408 | 0.0187 |
  | 8,8 vs 6 (split) | +0.3420 | 4.7060 |
  | A,A vs 5 (split) | +0.7322 | 2.2955 |

  The 16 vs 10 variance is tiny because the round is nearly decided before it starts: the dealer has blackjack about 8% of the time (and the hand loses), and the rest of the time the hand is a heavy underdog that mostly loses by one unit either way.

- **The edge vs true count curve.** The method, and why it is an enumeration and not a sample: the first version dealt random rounds and averaged them, and the numbers came out as pure noise. The edge differences across buckets are a few hundredths of a unit, while one round has a standard deviation near 0.4, so resolving the curve needed about two thousand samples a bucket at half a second a sample, about an hour and a half. A fixed panel of situations was tried as well, and it barely moves with the count: the rise comes mostly from being dealt better hands, which a fixed panel excludes by construction, so a panel understates the slope. The measurement that works is an exact enumeration: for a sampled shoe composition at a true count, every (player hand, upcard) combination is weighted by its deal probability and valued with `fresh_round_moments`, so the bucket value is the exact expectation over all deals from that composition, with no sampling noise in the dealing at all. The weights sum to one, checked in the tests, and the enumeration agrees with dealing random rounds and averaging them on a small shoe. The curve is the edge of optimal play, which is a touch better than basic strategy: the composition-dependent edge is the honest input for sizing, and the simulator in iteration 6b plays basic strategy, so its realised edge runs a fraction under this curve.

  The validation for the whole curve is the true-count-0 bucket: on a six-deck S17 double-after-split late-surrender 3:2 table it must land near the known house edge, about -0.4%. It does: **-0.0038, with a mean conditional round variance of 1.0478 and a total round variance of 1.287**. That is the anchor the rest of the curve is measured against, and it is also a test (`test_tc0_anchor_is_near_the_house_edge`, gated behind AOH_SLOW=1 because the full enumeration takes about twenty minutes).

  Two honest limits on the curve. First, the low pairs (2 through 6) are too slow to value exactly in every bucket: a pair of 2s against a 6 visits 10.9 million moments states and takes about 47 seconds on a six-deck shoe, against 0.8 million states and under 3 seconds for 8,8, because a low pair can draw many cards without busting. Valuing them exactly in every bucket would cost about twelve minutes a bucket in pairs alone, so the curve values them by their full-shoe split premium instead (warmed once with `python warm_pair_cache.py`, about eighteen minutes, most of it the 2s and 3s), while every other pair is exact. The premium of a pair moves with the composition by a few hundredths at most, and the slow pairs carry about three hundredths of the deal weight between them, so the bias this puts on the edge is about a thousandth and a half. The anchor bucket was measured exactly, and its premium-valued rerun is the check on that bias (see the table). Second, the curve is one sampled composition per bucket: the edge varies a little between shoes at the same count, so the bucket values carry that composition noise, a few thousandths, on top of the premium bias.

  The bucket table, six decks, S17, double after split, late surrender, 3:2, seed 20261111. The conditional variance is the variance of the round once the dealt hand is known, averaged over deals. The total variance adds the spread of the hand EVs across deals, and it is the number Kelly sizing reads.

  | True count | Edge | Conditional round variance | Total round variance |
  |---|---|---|---|
  | -3 to -2 | -0.0153 | 1.0899 | 1.3029 |
  | -2 to -1 | -0.0089 | 1.0787 | 1.3053 |
  | -1 to 0 | -0.0047 | 1.0520 | 1.2891 |
  | 0 to +1 | -0.0038 | 1.0478 | 1.2865 |
  | +1 to +2 | +0.0023 | 1.0450 | 1.2940 |
  | +2 to +3 | +0.0071 | 1.0114 | 1.2716 |
  | +3 to +5 | +0.0141 | 1.0061 | 1.2754 |

  The slope is about 0.45% per true count (from -1.53% at a true count of -2.5 to +1.41% at +4), which is the textbook number, and the edge crosses zero between a true count of +0.5 and +1.5. The conditional round variance falls gently with the count, from 1.09 at the bottom to 1.01 at the top. The total round variance stays near 1.29 across the curve, from 1.30 at the bottom to 1.28 at the top. The fixed-panel check behind the slope is in the tests: the same 24 hands are worth 0.0047 more on a high-count one-deck composition than on a low-count one, which is small because the panel excludes the dealing shift, but it is positive and deterministic.

- **Correction: what the round variance is.** The variance this iteration first published was the wrong one for Kelly sizing. The 1.05 at a true count of 0 is the mean conditional variance: the variance of the round once the dealt hand is known, averaged over deals. The Kelly bet is placed before the cards are dealt, so the hand you are dealt is part of the round's risk. The variance the Kelly formula needs is the total round variance: the conditional variance plus the spread of the hand EVs across deals. At a true count of 0 that total is 1.287, not 1.048. The curve now records the total in its third column, and the conditional values stay in the table above for comparison. The 6b Kelly bets made with the old number were about 23% larger than true Kelly, so the half-Kelly default behaved like about 0.61 Kelly. The simulator results stand as measured outcomes of the bets actually placed.

- **Literature cross-check.** The published numbers agree with the measurement. The Wizard of Odds house edge calculator (Michael Shackleford) puts the reference game, six decks, dealer stands on soft 17, double after split, no surrender, 3:2, at a 0.4026% house edge. Late surrender is worth about 0.07 points, so this project's rules come out near 0.33 to 0.36%. The measured true-count-0 edge is -0.38%, inside that band. The rule prices from the same tables are:

  | Rule change | House edge change |
  |---|---|
  | Dealer hits soft 17 | +0.21 points |
  | Blackjack pays 6:5 | +1.39 points |
  | No double after split | +0.14 points |
  | Late surrender | -0.07 points |
  | Eight decks instead of six | +0.02 points |

  The edge rises about 0.5% per true count in the counting literature (QFIT GameMaster, Wattenberger), and this curve measures 0.45%. The Wizard publishes a round variance of 1.303 for a six-deck S17 game, and 1.3344 for its Appendix 4 rules (six decks, S17, double after split, no surrender, split to four hands, no resplit aces). This project's Monte Carlo reproduces the Appendix 4 figure at 1.3376, and the total round variance at a true count of 0 is 1.287, against the published 1.303. The books behind the numbers are Thorp's Beat the Dealer, Griffin's The Theory of Blackjack, Wong's Professional Blackjack, and Schlesinger's Blackjack Attack.

- **How to run it.** Build the C helper first (`python build.py`, needs gcc or cc), then `python measure_edge_curve.py --decks 6 --pair-cache /tmp/pair_cache_6d.json` for the full curve (about twelve minutes with the warmed cache), or `--buckets 3` for the anchor bucket alone. Warm the slow-pair cache first with `python warm_pair_cache.py --decks 6 --out /tmp/pair_cache_6d.json`. The tests run with `pytest tests/` (131 passed, 2 skipped, about 50 seconds). The skipped tests are the TC-0 anchor and the every-bucket total-variance check, both gated behind AOH_SLOW=1 because the full enumerations take minutes.

- **What changed, file by file.** `blackjack/odds.py`: `OutcomeOdds` carries `e2_override`, `second_moment` and `variance`. `scaled` and `__add__` carry the second moment. Surrender sets e2_override 0.25. `OddsCalculator` takes a `keep_dealer_cache` switch for measurement (off by default, it roughly halves the cost of valuing many situations on nearly the same shoe, at the price of a cache that grows to about a gigabyte on a six-deck enumeration). `blackjack/split_exact.c`: the joint moments engine (`split_moments_exact`, `split_moments_states`) alongside the joint EV engine. `blackjack/split_exact.py`: the `split_moments` loader. `blackjack/moments.py`: `fresh_round_moments` and the split premium helper. A pair whose key is pre-populated in `pair_cache` is valued by premium, every other pair exactly. `blackjack/edge_curve.py`: the enumeration (`enumerate_fresh_round_edge`), the sampler (`sample_composition`), the bucket measurement (`measure_bucket`, `measure_curve`), and the slow-pair cache (`warm_slow_pair_cache`, `save_pair_cache`, `load_pair_cache`). `measure_edge_curve.py` and `warm_pair_cache.py`: the two scripts. `tests/test_iteration6a.py`: the variance, moments, fresh-round, premium, enumeration and curve tests.

Correction (total round variance): `blackjack/edge_curve.py` now returns the total round variance alongside the mean conditional variance (`enumerate_fresh_round_edge` also accumulates E[ev^2] across deals), `measure_bucket` reports both `mean_var` and `mean_total_var`, and `MEASURED_EDGE_CURVE` carries the total in its third column. `measure_edge_curve.py` prints both columns. `tests/test_iteration6a.py` checks the identity on the small pool, the stored curve against 1.287 at a true count of 0, the total against the conditional in every bucket, and the anchor test checks both variances.

Still not in: bet sizing (that is iteration 6b, which reads this curve), and one player only.
