"""The edge of a fresh round as a function of the true count.

Iteration 6a. The edge is not one number for the whole shoe: it is
negative when the count is low and positive when the count is high, and
the Kelly bet in the next iteration has to follow it. This module maps
that curve, using the exact engine, so the sizing rule has real inputs
instead of a guessed edge.

Method. For each true-count bucket, sample a shoe composition at that
count (sample_composition) and enumerate the fresh-round edge exactly:
every (player hand, upcard) combination is weighted by its deal
probability from that composition and valued with fresh_round_moments,
which plays optimally and includes the peek, the insurance choice and
the even-money choice. The bucket value is the exact expectation over
all deals from that composition, so there is no sampling noise in the
dealing at all. The mean round variance comes along the same way, and
so does the total round variance: the Kelly formula in iteration 6b
needs the total, because the bet is placed before the cards are dealt.

Why enumerate instead of dealing random rounds. The first version of
this module dealt random rounds and averaged them, and the numbers came
out as pure noise: the edge differences across buckets are a few
hundredths of a unit, while one round has a standard deviation near
0.4, so resolving the curve needed about two thousand samples a bucket
at half a second a sample, about an hour and a half. The enumeration
costs about three minutes a composition and is exact. A fixed panel of
situations was tried as well, and it barely moves with the count: the
rise comes mostly from being dealt better hands, which a fixed panel
excludes by construction, so a panel understates the slope. The
enumeration is the honest measurement.

Two things to know about the numbers. First, the curve is the edge of
OPTIMAL play (the exact engine chooses every action), which is a touch
better than basic strategy: the composition-dependent edge is the honest
input for sizing, and the simulator in iteration 6b plays basic
strategy, so its realised edge runs a fraction under this curve. Second,
a (10, 10) hand is a pair only when the two cards share a rank, about
one time in four in a fresh shoe: the enumeration splits that hand into
a pair part and a non-pair part by that probability, using the fresh
shoe formula, since the counts are tracked by value and do not record
the rank split among tens. The error from that is under a thousandth
of a unit on the edge.

Pairs are valued exactly by the C joint moments engine when pair_cache
is None, which is the default and what the anchor bucket uses. The low
pairs (2 through 6) are too slow for that in every bucket: a pair of 2s
against a 6 visits 10.9 million moments states and takes about 47
seconds on a six-deck shoe, against 0.8 million states and under 3
seconds for 8,8, because a low pair can draw many cards without
busting, so valuing them exactly in every bucket would cost about
twelve minutes a bucket in pairs alone. The curve script therefore
warms a cache of their full-shoe premiums once (warm_slow_pair_cache)
and passes it to the enumeration: slow pairs are valued by premium,
every other pair exactly. The premium of a pair moves with the
composition by a few hundredths at most, and the slow pairs carry about
three hundredths of the deal weight between them, so the bias this
puts on the edge is about a thousandth and a half, and the README
records the comparison of the exact anchor bucket against its
premium-valued rerun as the check.

Speed. Valuing 550 situations per composition is the slow part, and
almost all of the time goes into the dealer distribution. The odds
calculator is therefore run with keep_dealer_cache, which shares dealer
states between situations on nearly the same shoe and roughly halves
the cost. A fresh calculator is made per bucket, since the cache grows
to a few hundred thousand entries per composition and a long-lived one
would hold close to a gigabyte. deal_fresh_round stays in the module
as an independent cross-check: sampled rounds from a composition must
average to the enumerated edge, and the tests check that on a small
shoe.
"""

from __future__ import annotations

import json
import random

from .cards import Card, Rank, Suit
from .hand import Hand
from .moments import _split_premium_and_var, fresh_round_moments
from .odds import OddsCalculator
from .rules import Rules

# True-count buckets: (low, high), one point wide except the last, which
# is open to +5. The centre of each bucket is the target count used when
# sampling compositions for it.
EDGE_BUCKETS: tuple[tuple[float, float], ...] = (
    (-3.0, -2.0),
    (-2.0, -1.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 5.0),
)


# The measured edge curve: one (true-count centre, edge, total round
# variance) point per bucket of EDGE_BUCKETS, from the enumeration in
# measure_edge_curve.py. Six decks, S17, double after split, late
# surrender, 3:2, seed 20261111. The variance here is the TOTAL round
# variance: the mean conditional variance given the dealt hand plus the
# spread of the hand EVs across deals. Kelly sizing needs the total,
# because the bet is placed before the cards are dealt. The TC 0
# bucket was measured exactly and again with the slow-pair premium
# cache: both give -0.0038 edge and 1.0478 conditional variance, so
# the premium bias is below the fourth decimal there. The conditional
# variances are still printed by the measurement script alongside the
# totals. The Kelly policy in 6b reads it from here.
MEASURED_EDGE_CURVE: list[tuple[float, float, float]] | None = [
    (-2.5, -0.0153, 1.3029),
    (-1.5, -0.0089, 1.3053),
    (-0.5, -0.0047, 1.2891),
    (0.5, -0.0038, 1.2865),
    (1.5, 0.0023, 1.2940),
    (2.5, 0.0071, 1.2716),
    (4.0, 0.0141, 1.2754),
]


def bucket_centre(bucket: tuple[float, float]) -> float:
    """The middle of a bucket, the count its compositions are sampled at."""
    return (bucket[0] + bucket[1]) / 2.0


def hi_lo_value(value: int) -> int:
    """Hi-Lo value of a blackjack value: low cards +1, tens and aces -1."""
    if value == 1 or value == 10:
        return -1
    if 2 <= value <= 6:
        return 1
    return 0


def true_count(counts: tuple[int, ...], running_count: int) -> float:
    """Running count per deck left in the shoe."""
    decks_left = sum(counts) / 52.0
    if decks_left <= 0:
        return 0.0
    return running_count / decks_left


def sample_composition(
    rng: random.Random, bucket: tuple[float, float], num_decks: int
) -> tuple[tuple[int, ...], int] | None:
    """A shoe composition inside this true-count bucket, or None.

    Starts from a fresh shoe and removes cards one at a time, biased
    toward the direction the count needs to move, until the true count
    lands inside the bucket. The second value returned is the running
    count of the removed cards. None when the shoe runs too low first:
    a composition with almost no cards left cannot deal a round from it.
    """
    target = bucket_centre(bucket)
    counts = [4 * num_decks if i < 9 else 16 * num_decks for i in range(10)]
    running = 0
    for _ in range(400):
        tc = true_count(tuple(counts), running)
        if bucket[0] <= tc < bucket[1]:
            return tuple(counts), running
        if sum(counts) <= 20:
            return None
        rising = tc < target
        weights = []
        for i in range(10):
            v = hi_lo_value(i + 1)
            if rising:
                bias = 3.0 if v == 1 else (1.0 if v == 0 else 0.25)
            else:
                bias = 3.0 if v == -1 else (1.0 if v == 0 else 0.25)
            weights.append(counts[i] * bias)
        total = sum(weights)
        pick = rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if pick <= acc:
                counts[i] -= 1
                running += hi_lo_value(i + 1)
                break
    return None


_RANK_BY_VALUE = {
    2: Rank.TWO,
    3: Rank.THREE,
    4: Rank.FOUR,
    5: Rank.FIVE,
    6: Rank.SIX,
    7: Rank.SEVEN,
    8: Rank.EIGHT,
    9: Rank.NINE,
}

_TEN_RANKS = (Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING)


def _rank_for_value(rng: random.Random, value: int) -> Rank:
    """A rank worth this blackjack value. Ten-value cards are one of the
    four ten ranks, picked at random, so two of them are a pair about
    one time in four, as in a real shoe."""
    if value == 1:
        return Rank.ACE
    if value == 10:
        return rng.choice(list(_TEN_RANKS))
    return _RANK_BY_VALUE[value]


def _draw_value(rng: random.Random, counts: list[int]) -> int:
    """Draw one card value from the counts, without replacement."""
    total = sum(counts)
    pick = rng.randrange(total)
    acc = 0
    for i, c in enumerate(counts):
        acc += c
        if pick < acc:
            counts[i] -= 1
            return i + 1
    raise AssertionError("draw fell through an empty shoe")


def deal_fresh_round(
    rng: random.Random, counts: tuple[int, ...]
) -> tuple[Hand, Card, tuple[int, ...]]:
    """Deal a fresh round from a composition: two cards to the player, one
    upcard to the dealer. Returns the player hand, the upcard, and the
    unseen composition (everything left, which the odds engine reads as
    the shoe plus the dealer's hole card). Kept as the independent
    cross-check for the enumeration: sampled rounds from a composition
    must average to its enumerated edge."""
    left = list(counts)
    v1 = _draw_value(rng, left)
    up = _draw_value(rng, left)
    v2 = _draw_value(rng, left)
    hand = Hand()
    hand.add(Card(_rank_for_value(rng, v1), Suit.SPADES))
    hand.add(Card(_rank_for_value(rng, v2), Suit.HEARTS))
    upcard = Card(_rank_for_value(rng, up), Suit.CLUBS)
    return hand, upcard, tuple(left)


def _ordered_deal_probability(
    first: int, upcard: int, second: int, counts: tuple[int, ...]
) -> float:
    """P(deal first to the player, upcard to the dealer, second to the
    player), in that order, from these counts, without replacement."""
    left = list(counts)
    n = sum(left)
    p = 1.0
    for value in (first, upcard, second):
        if left[value - 1] <= 0:
            return 0.0
        p *= left[value - 1] / n
        left[value - 1] -= 1
        n -= 1
    return p


def _hand_for_values(v1: int, v2: int, *, pair: bool) -> Hand:
    """A hand with these two blackjack values. A ten-value pair is two
    tens. A ten-value non-pair is ten plus jack."""
    hand = Hand()
    if v1 == 1:
        hand.add(Card(Rank.ACE, Suit.SPADES))
    elif v1 == 10:
        hand.add(Card(Rank.TEN, Suit.SPADES))
    else:
        hand.add(Card(_RANK_BY_VALUE[v1], Suit.SPADES))
    if v2 == 1:
        hand.add(Card(Rank.ACE, Suit.HEARTS))
    elif v2 == 10:
        hand.add(Card(Rank.TEN if pair else Rank.JACK, Suit.HEARTS))
    else:
        hand.add(Card(_RANK_BY_VALUE[v2], Suit.HEARTS))
    return hand


def _deal_situations(
    counts: tuple[int, ...], num_decks: int
) -> list[tuple[Hand, Card, tuple[int, ...], float]]:
    """Every (player hand, upcard) that can be dealt from these counts,
    with the unseen composition after the deal and the deal probability.

    One entry per (value pair, upcard), except (10, 10), which splits
    into a pair part and a non-pair part by the fresh-shoe probability
    that two ten-value cards share a rank, (4d-1)/(16d-1) for d decks.
    The weights over all entries sum to one. Zero-weight entries are
    left out, so a nearly empty shoe yields only the situations it can
    still deal.
    """
    n = sum(counts)
    if n < 3:
        return []
    pair_prob_tens = (4 * num_decks - 1) / (16 * num_decks - 1)
    situations: list[tuple[Hand, Card, tuple[int, ...], float]] = []
    for v1 in range(1, 11):
        for v2 in range(v1, 11):
            for up in range(1, 11):
                if v1 != v2:
                    weight = _ordered_deal_probability(
                        v1, up, v2, counts
                    ) + _ordered_deal_probability(v2, up, v1, counts)
                else:
                    weight = _ordered_deal_probability(v1, up, v2, counts)
                if weight <= 0.0:
                    continue
                left = list(counts)
                for value in (v1, up, v2):
                    left[value - 1] -= 1
                unseen = tuple(left)
                upcard = Card(
                    Rank.ACE
                    if up == 1
                    else Rank.TEN
                    if up == 10
                    else _RANK_BY_VALUE[up],
                    Suit.CLUBS,
                )
                if v1 == 10 and v2 == 10:
                    situations.append(
                        (_hand_for_values(10, 10, pair=True), upcard, unseen, weight * pair_prob_tens)
                    )
                    situations.append(
                        (_hand_for_values(10, 10, pair=False), upcard, unseen, weight * (1.0 - pair_prob_tens))
                    )
                else:
                    situations.append(
                        (_hand_for_values(v1, v2, pair=v1 == v2), upcard, unseen, weight)
                    )
    return situations


def enumerate_fresh_round_edge(
    counts: tuple[int, ...],
    rules: Rules,
    calc: OddsCalculator | None = None,
    pair_cache: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> tuple[float, float, float]:
    """Exact EV, mean conditional variance and total round variance of a
    fresh round from this shoe.

    Enumerates every (player hand, upcard) combination with its deal
    probability and values each with fresh_round_moments, so the result
    is the exact expectation over all deals from this composition, with
    no sampling noise. Pairs go through the exact C joint moments engine
    when pair_cache is None. Pass a cache to use the full-shoe premium
    shortcut instead. A calculator with keep_dealer_cache is used when
    none is passed: it roughly halves the cost, since most dealer states
    are shared between situations on nearly the same shoe.

    The three numbers are the mean EV, the mean conditional variance
    (the variance of the round once the dealt hand is known, averaged
    over deals), and the total round variance. The total adds the spread
    of the hand EVs themselves: total = mean conditional var + Var(ev),
    where Var(ev) = E[ev^2] - edge^2 over the deal distribution. The
    total is the one the Kelly formula needs, because the bet is placed
    before the cards are dealt: the hand you are dealt is part of the
    round's risk, not something known when you size the bet.
    """
    calc = calc or OddsCalculator(rules, keep_dealer_cache=True)
    edge = 0.0
    mean_var = 0.0
    mean_e2 = 0.0
    for hand, upcard, unseen, weight in _deal_situations(counts, rules.num_decks):
        ev, var = fresh_round_moments(hand, upcard, unseen, rules, calc, pair_cache)
        edge += weight * ev
        mean_var += weight * var
        mean_e2 += weight * ev * ev
    total_var = mean_var + (mean_e2 - edge * edge)
    return edge, mean_var, total_var


def measure_bucket(
    rng: random.Random,
    bucket: tuple[float, float],
    num_decks: int,
    rules: Rules,
    *,
    compositions: int = 1,
    pair_cache: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> dict[str, float | int]:
    """Exact edge, conditional variance and total round variance at one
    true count.

    Samples `compositions` shoe compositions inside the bucket and
    enumerates the fresh-round edge of each, then averages. With one
    composition the value is exact for that composition. More
    compositions average out how much the edge varies between shoes at
    the same count. A fresh odds calculator is made per bucket, so the
    dealer cache it keeps stays bounded. Returns the bucket, the number
    of compositions, the mean EV per unit bet, the spread of the
    enumerated edges across compositions (zero with one), the mean
    conditional round variance per unit bet squared, and the mean total
    round variance, which is the conditional variance plus the spread
    of the hand EVs across deals. The total is what Kelly sizing reads.
    """
    edges: list[float] = []
    varis: list[float] = []
    total_varis: list[float] = []
    attempts = 0
    while len(edges) < compositions and attempts < compositions * 10:
        attempts += 1
        sampled = sample_composition(rng, bucket, num_decks)
        if sampled is None:
            continue
        counts, _running = sampled
        calc = OddsCalculator(rules, keep_dealer_cache=True)
        ev, var, total_var = enumerate_fresh_round_edge(counts, rules, calc, pair_cache)
        edges.append(ev)
        varis.append(var)
        total_varis.append(total_var)
    n = len(edges)
    mean_ev = sum(edges) / n if n else 0.0
    mean_var = sum(varis) / n if n else 0.0
    mean_total_var = sum(total_varis) / n if n else 0.0
    if n > 1:
        se = (sum((x - mean_ev) ** 2 for x in edges) / (n * (n - 1))) ** 0.5
    else:
        se = 0.0
    return {
        "low": bucket[0],
        "high": bucket[1],
        "target_tc": bucket_centre(bucket),
        "n": n,
        "mean_ev": mean_ev,
        "se_ev": se,
        "mean_var": mean_var,
        "mean_total_var": mean_total_var,
    }


def measure_curve(
    rng: random.Random,
    num_decks: int,
    rules: Rules,
    *,
    compositions_per_bucket: int = 1,
    pair_cache: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> list[dict[str, float | int]]:
    """The edge curve: one enumerated bucket per entry of EDGE_BUCKETS.

    Each bucket is enumerated exactly, so this is slow: about seventeen
    minutes a bucket on a six-deck shoe with every pair exact, about
    a minute and a half with a warmed slow-pair cache. Run it as a script,
    not in a test. pair_cache stays None by default, so pairs are valued
    exactly. Pass a warmed cache to value the slow low pairs by premium
    and every other pair exactly.
    """
    return [
        measure_bucket(
            rng,
            bucket,
            num_decks,
            rules,
            compositions=compositions_per_bucket,
            pair_cache=pair_cache,
        )
        for bucket in EDGE_BUCKETS
    ]


# The low pairs (2 through 6) are the slow ones in the C moments engine:
# a pair of 2s against a 6 visits 10.9 million states and takes about 47
# seconds on a six-deck shoe, against 0.8 million states and under 3
# seconds for 8,8, because a low pair can draw many cards without
# busting. Valuing them exactly in every bucket would cost about twelve
# minutes a bucket in pairs alone. They are the pairs the premium cache
# is for.
SLOW_PAIR_VALUES: tuple[int, ...] = (2, 3, 4, 5, 6)


def warm_slow_pair_cache(
    rules: Rules,
    calc: OddsCalculator | None = None,
    cache: dict[tuple[int, int], tuple[float, float]] | None = None,
    ranks: tuple[int, ...] | None = None,
) -> dict[tuple[int, int], tuple[float, float]]:
    """(Premium, variance) of splitting each slow pair, on a full shoe.

    One C moments call per (pair, upcard): fifty calls, about eighteen
    minutes for a six-deck shoe, most of it the 2s and 3s. The result is
    the cache the curve script passes to the enumeration, so slow pairs
    are valued by their full-shoe premium and every other pair exactly.
    Keys already in `cache` are skipped, and `ranks` limits the work to
    some of the slow values, so a run interrupted halfway can pick up
    where it left off by passing the partial cache back in.
    """
    calc = calc or OddsCalculator(rules)
    cache = cache if cache is not None else {}
    for rank_value in ranks if ranks is not None else SLOW_PAIR_VALUES:
        for upcard_value in range(1, 11):
            key = (rank_value, upcard_value)
            if key in cache:
                continue
            cache[key] = _split_premium_and_var(key, rules, calc)
    return cache


def save_pair_cache(
    cache: dict[tuple[int, int], tuple[float, float]], path: str
) -> None:
    """Write a premium cache to JSON, keys as "pair,upcard"."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {f"{k[0]},{k[1]}": [v[0], v[1]] for k, v in cache.items()},
            f,
            indent=1,
        )


def load_pair_cache(path: str) -> dict[tuple[int, int], tuple[float, float]]:
    """Read a premium cache written by save_pair_cache."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        (int(k.split(",")[0]), int(k.split(",")[1])): (v[0], v[1])
        for k, v in raw.items()
    }
