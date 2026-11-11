"""Iteration 6a: round variance and the edge vs true count curve.

The variance of a round, per unit bet, under optimal play: plain hands
from the win, push and lose probabilities already in OutcomeOdds, splits
from the C joint moments engine (which carries a second moment through
the joint recursion), surrender as a certain -0.5. The fresh-round
moments add the peek branch, insurance and even money on top. The edge
curve maps the exact fresh-round EV against the true count.

The numbers below are measured values, not aspirations. The C moments
EVs agree with the joint EVs exactly (the same recursion, the same
policy), and the A,A vs 5 variance is checked twice over: once by an
independent enumeration of the split in Python, once by Monte Carlo
simulation of the real game.
"""
import math
import os
import random

import pytest

from blackjack import split_exact
from blackjack.cards import Card, Rank, Suit
from blackjack.edge_curve import (
    EDGE_BUCKETS,
    _deal_situations,
    deal_fresh_round,
    enumerate_fresh_round_edge,
    measure_bucket,
    sample_composition,
    true_count,
)
from blackjack.hand import Hand
from blackjack.moments import _split_premium_and_var, fresh_round_moments
from blackjack.odds import Action, OddsCalculator, OutcomeOdds
from blackjack.rules import Rules

SMALL_POOL = (1, 1, 1, 1, 1, 1, 1, 1, 1, 4)  # one of each value, four tens

# Full deck minus the pair and the upcard (49 unseen cards).
FULL_88_VS_6 = (4, 4, 4, 4, 4, 3, 4, 2, 4, 16)
FULL_AA_VS_5 = (2, 4, 4, 4, 3, 4, 4, 4, 4, 16)
FULL_88_VS_10 = (4, 4, 4, 4, 4, 4, 4, 2, 4, 15)


def hand(*ranks: Rank) -> Hand:
    h = Hand()
    for r in ranks:
        h.add(Card(r, Suit.SPADES))
    return h


def card(rank: Rank) -> Card:
    return Card(rank, Suit.HEARTS)


def deck_minus(*seen: Rank) -> tuple[int, ...]:
    """A full single deck minus the given ranks, as blackjack-value counts."""
    counts = [4] * 9 + [16]
    for r in seen:
        counts[r.blackjack_value - 1] -= 1
    return tuple(counts)


@pytest.fixture
def calc() -> OddsCalculator:
    return OddsCalculator(Rules())


def _best_total(hard: int, ace: bool) -> int:
    return hard + 10 if ace and hard + 10 <= 21 else hard


# ---- OutcomeOdds second moment: plain hands ------------------------------


def test_stand_variance_from_the_probabilities(calc):
    # 20 vs 6 on a full deck: the stand odds carry win, push, lose, and
    # the variance is E[X^2] - EV^2 with E[X^2] = win + lose on a
    # single stake. The recorded values pin the numbers, not just the
    # formula.
    a = calc.analyze(
        hand(Rank.KING, Rank.QUEEN),
        card(Rank.SIX),
        deck_minus(Rank.KING, Rank.QUEEN, Rank.SIX),
    )
    stand = a.odds(Action.STAND)
    assert stand.ev == pytest.approx(0.6974, abs=0.001)
    assert stand.second_moment == pytest.approx(stand.win + stand.lose, abs=1e-9)
    assert stand.variance == pytest.approx(0.4105, abs=0.001)


def test_natural_variance_is_zero(calc):
    # A natural pays a certain 3:2: E[X^2] is 2.25 and the variance is 0.
    a = calc.analyze(
        hand(Rank.ACE, Rank.KING),
        card(Rank.SIX),
        deck_minus(Rank.ACE, Rank.KING, Rank.SIX),
    )
    stand = a.odds(Action.STAND)
    assert stand.blackjack_win == pytest.approx(1.0)
    assert stand.ev == pytest.approx(1.5)
    assert stand.second_moment == pytest.approx(2.25)
    assert stand.variance == pytest.approx(0.0)


def test_double_variance_uses_the_doubled_stake(calc):
    # Doubling puts two units at risk, so E[X^2] is four times win + lose.
    a = calc.analyze(
        hand(Rank.SIX, Rank.FIVE),
        card(Rank.SIX),
        deck_minus(Rank.SIX, Rank.FIVE, Rank.SIX),
    )
    double = a.odds(Action.DOUBLE)
    assert double.stake == 2.0
    assert double.second_moment == pytest.approx(
        4.0 * (double.win + double.lose), abs=1e-9
    )
    assert double.variance == pytest.approx(
        double.second_moment - double.ev**2, abs=1e-9
    )


def test_surrender_second_moment(calc):
    # Surrender is a certain -0.5: E[X^2] is 0.25 and the variance is 0.
    a = calc.analyze(
        hand(Rank.TEN, Rank.SIX),
        card(Rank.TEN),
        deck_minus(Rank.TEN, Rank.SIX, Rank.TEN),
    )
    surrender = a.odds(Action.SURRENDER)
    assert surrender.ev == pytest.approx(-0.5)
    assert surrender.second_moment == pytest.approx(0.25)
    assert surrender.variance == pytest.approx(0.0)


def test_split_odds_carry_no_second_moment(calc):
    # The live split engine returns an EV and no variance: the moments
    # report that as missing, never invent one.
    a = calc.analyze(
        hand(Rank.EIGHT, Rank.EIGHT),
        card(Rank.SIX),
        deck_minus(Rank.EIGHT, Rank.EIGHT, Rank.SIX),
        allow_split=True,
    )
    split = a.odds(Action.SPLIT)
    assert split.ev == pytest.approx(0.3420, abs=0.001)
    assert split.second_moment is None
    assert split.variance is None


def test_scaled_and_added_odds_carry_the_second_moment():
    # Scaling by a probability weight scales E[X^2] by the same weight,
    # and adding mixtures adds the second moments, like the EVs.
    o = OutcomeOdds(0.6, 0.1, 0.3)
    assert o.second_moment == pytest.approx(0.9)
    assert o.scaled(0.5).second_moment == pytest.approx(0.45)
    assert (o.scaled(0.5) + o.scaled(0.5)).second_moment == pytest.approx(0.9)
    bj = OutcomeOdds(0.0, 0.0, 0.0, blackjack_win=1.0, blackjack_payout=1.5)
    assert bj.second_moment == pytest.approx(2.25)


# ---- the C joint moments engine --------------------------------------------


@pytest.fixture(autouse=True)
def _needs_c():
    if not split_exact.available():
        pytest.skip("C helper not built (run python build.py)")


def test_moments_ev_matches_the_joint_ev():
    # The moments recursion makes the same EV-maximising choices as the
    # joint recursion, so the EVs agree to floating point noise.
    for counts, single, up, excl, rules in (
        (FULL_88_VS_6, 8, 6, None, Rules()),
        (FULL_88_VS_10, 8, 10, 1, Rules()),
        (FULL_AA_VS_5, 1, 5, None, Rules()),
        (SMALL_POOL, 8, 6, None, Rules(max_splits=3)),
        (SMALL_POOL, 1, 5, None, Rules(max_splits=3)),
    ):
        joint = split_exact.split_ev(
            counts, single, single == 1, up, up == 1, excl, rules
        )
        ev, _var = split_exact.split_moments(
            counts, single, single == 1, up, up == 1, excl, rules
        )
        assert ev == pytest.approx(joint, abs=1e-9)


def test_moments_variance_values():
    # Splitting puts two stakes in play against one dealer hand, and the
    # hands move together, so the round variance is far above a plain
    # hand (about 1.3): splitting 8s against a 6 is the swingiest spot
    # here, and resplits (small pool, three splits) swing harder still.
    cases = (
        (FULL_88_VS_6, 8, 6, None, Rules(), 0.3420, 4.7060),
        (FULL_88_VS_10, 8, 10, 1, Rules(), -0.4521, 3.0735),
        (FULL_AA_VS_5, 1, 5, None, Rules(), 0.7322, 2.2955),
        (SMALL_POOL, 8, 6, None, Rules(max_splits=3), 0.4430, 5.4091),
        (SMALL_POOL, 1, 5, None, Rules(max_splits=3), 0.8138, 2.4821),
    )
    for counts, single, up, excl, rules, want_ev, want_var in cases:
        ev, var = split_exact.split_moments(
            counts, single, single == 1, up, up == 1, excl, rules
        )
        assert ev == pytest.approx(want_ev, abs=0.001)
        assert var == pytest.approx(want_var, abs=0.001)
        assert 0.0 < var < 16.0


def test_moments_states_are_counted():
    split_exact.split_moments(FULL_88_VS_6, 8, False, 6, False, None, Rules())
    assert split_exact.moments_states() == pytest.approx(389226, rel=0.01)


def _enumerate_split_aces(counts, calc):
    """(EV, variance) of splitting aces, by direct enumeration in Python.

    With the split cap at one, a split ace takes exactly one card and
    stands: deal the two second cards, then settle both hands against
    the dealer distribution from what is left. No recursion, no memo,
    so this checks the C leaf -- the one place the correlation between
    the two hands lives -- independently.
    """
    n = sum(counts)
    ev = 0.0
    e2 = 0.0
    for v1 in range(1, 11):
        c1 = counts[v1 - 1]
        if c1 == 0:
            continue
        mid = list(counts)
        mid[v1 - 1] -= 1
        for v2 in range(1, 11):
            c2 = mid[v2 - 1]
            if c2 == 0:
                continue
            left = list(mid)
            left[v2 - 1] -= 1
            dist = calc._dealer_distribution(tuple(left), 5, False)
            t1 = 11 + v1
            t2 = 11 + v2
            mean = 0.0
            sq = 0.0
            # The distribution is indexed by final total (17..21, then 22
            # for a bust. Totals below 17 never occur.
            for j in range(6):
                p = dist[22] if j == 5 else dist[17 + j]
                if p == 0.0:
                    continue
                dealer = 22 if j == 5 else 17 + j
                y1 = 1.0 if dealer == 22 or dealer < t1 else (0.0 if dealer == t1 else -1.0)
                y2 = 1.0 if dealer == 22 or dealer < t2 else (0.0 if dealer == t2 else -1.0)
                x = y1 + y2
                mean += p * x
                sq += p * x * x
            p = (c1 / n) * (c2 / (n - 1))
            ev += p * mean
            e2 += p * sq
    return ev, e2 - ev * ev


def test_moments_aces_match_python_enumeration(calc):
    # The enumeration and the C engine agree on both the EV and the
    # variance, on a full deck and on the small pool.
    for counts in (FULL_AA_VS_5, SMALL_POOL):
        want_ev, want_var = _enumerate_split_aces(counts, calc)
        ev, var = split_exact.split_moments(
            counts, 1, True, 5, False, None, Rules()
        )
        assert ev == pytest.approx(want_ev, abs=1e-9)
        assert var == pytest.approx(want_var, abs=1e-9)


def _dealer_final(counts, up_value, up_is_ace, rng, rules):
    """Simulate the dealer from these counts: hole card, then draws to the rule.

    Returns the final total, or 22 for a bust.
    """
    left = list(counts)

    def draw():
        total = sum(left)
        pick = rng.randrange(total)
        acc = 0
        for i, c in enumerate(left):
            acc += c
            if pick < acc:
                left[i] -= 1
                return i + 1
        raise AssertionError("draw fell through an empty shoe")

    hard = up_value
    ace = up_is_ace
    hole = draw()
    hard += hole
    ace = ace or hole == 1
    while rules.dealer_must_hit(_best_total(hard, ace), _best_total(hard, ace) != hard):
        v = draw()
        hard += v
        ace = ace or v == 1
    return _best_total(hard, ace) if hard <= 21 else 22


def _payoff(total: int, dealer: int) -> float:
    if dealer == 22 or dealer < total:
        return 1.0
    if dealer == total:
        return 0.0
    return -1.0


def test_moments_aces_match_monte_carlo():
    # Simulate split aces against a 5, both hands standing on their one
    # card, and compare the sample mean and variance with the C engine.
    # 20000 rounds puts three standard errors at about 0.03 on the EV
    # and 0.07 on the variance.
    rng = random.Random(20261111)
    rules = Rules()
    counts = FULL_AA_VS_5
    total = 0.0
    total_sq = 0.0
    n = 20000
    for _ in range(n):
        left = list(counts)

        def draw_from(pool):
            tot = sum(pool)
            pick = rng.randrange(tot)
            acc = 0
            for i, c in enumerate(pool):
                acc += c
                if pick < acc:
                    pool[i] -= 1
                    return i + 1
            raise AssertionError("draw fell through")

        v1 = draw_from(left)
        v2 = draw_from(left)
        dealer = _dealer_final(tuple(left), 5, False, rng, rules)
        x = _payoff(11 + v1, dealer) + _payoff(11 + v2, dealer)
        total += x
        total_sq += x * x
    mean = total / n
    var = total_sq / n - mean * mean
    ev, want_var = split_exact.split_moments(counts, 1, True, 5, False, None, rules)
    assert mean == pytest.approx(ev, abs=0.032)
    assert var == pytest.approx(want_var, abs=0.07)


def test_stand_variance_matches_monte_carlo(calc):
    # The same check for a plain hand: stand on 20 vs 6, simulated.
    rng = random.Random(20261111)
    rules = Rules()
    counts = deck_minus(Rank.KING, Rank.QUEEN, Rank.SIX)
    a = calc.analyze(
        hand(Rank.KING, Rank.QUEEN), card(Rank.SIX), counts
    )
    stand = a.odds(Action.STAND)
    total = 0.0
    total_sq = 0.0
    n = 20000
    for _ in range(n):
        dealer = _dealer_final(counts, 6, False, rng, rules)
        x = _payoff(20, dealer)
        total += x
        total_sq += x * x
    mean = total / n
    var = total_sq / n - mean * mean
    assert mean == pytest.approx(stand.ev, abs=0.02)
    assert var == pytest.approx(stand.variance, abs=0.03)


# ---- fresh-round moments ---------------------------------------------------


def test_fresh_round_blackjack_spots(calc):
    rules = Rules()
    # A natural vs 6 pays a certain 3:2.
    ev, var = fresh_round_moments(
        hand(Rank.ACE, Rank.KING),
        card(Rank.SIX),
        deck_minus(Rank.ACE, Rank.KING, Rank.SIX),
        rules,
        calc,
    )
    assert ev == pytest.approx(1.5)
    assert var == pytest.approx(0.0)
    # A natural vs an ace: declining even money wins the payout unless
    # the dealer has blackjack (15 tens in 49 unseen), which beats the
    # certain +1, so declining is the play and the variance is the push.
    unseen = deck_minus(Rank.ACE, Rank.KING, Rank.ACE)
    ev, var = fresh_round_moments(
        hand(Rank.ACE, Rank.KING), card(Rank.ACE), unseen, rules, calc
    )
    p = 15 / 49
    want_ev = 1.5 * (1.0 - p)
    assert want_ev > 1.0
    assert ev == pytest.approx(want_ev, abs=1e-9)
    assert var == pytest.approx(2.25 * (1.0 - p) - want_ev**2, abs=1e-9)


def test_fresh_round_matches_best_odds_on_a_plain_hand(calc):
    # With no ace or ten up, there is no peek branch and no insurance, so
    # the fresh round is just the best action's odds: 20 vs 6 stands.
    rules = Rules()
    counts = deck_minus(Rank.KING, Rank.QUEEN, Rank.SIX)
    a = calc.analyze(hand(Rank.KING, Rank.QUEEN), card(Rank.SIX), counts)
    stand = a.odds(Action.STAND)
    ev, var = fresh_round_moments(
        hand(Rank.KING, Rank.QUEEN), card(Rank.SIX), counts, rules, calc
    )
    assert ev == pytest.approx(stand.ev, abs=1e-9)
    assert var == pytest.approx(stand.variance, abs=1e-9)


def test_fresh_round_sixteen_vs_ten_includes_the_peek(calc):
    # 16 vs 10: the peek ends the round on a dealer blackjack (4 aces in
    # 49 unseen), so the fresh-round EV is below the -0.5 surrender value
    # the post-peek odds show, and the variance is the peek itself.
    rules = Rules()
    counts = deck_minus(Rank.TEN, Rank.SIX, Rank.TEN)
    ev, var = fresh_round_moments(
        hand(Rank.TEN, Rank.SIX), card(Rank.TEN), counts, rules, calc
    )
    p = 4 / 49
    want_ev = p * (-1.0) + (1.0 - p) * (-0.5)
    want_e2 = p * 1.0 + (1.0 - p) * 0.25
    assert ev == pytest.approx(want_ev, abs=1e-9)
    assert var == pytest.approx(want_e2 - want_ev**2, abs=1e-9)


def test_fresh_round_split_spots(calc):
    # The split spots agree with the C joint moments engine exactly.
    rules = Rules()
    ev, var = fresh_round_moments(
        hand(Rank.EIGHT, Rank.EIGHT),
        card(Rank.SIX),
        deck_minus(Rank.EIGHT, Rank.EIGHT, Rank.SIX),
        rules,
        calc,
    )
    assert ev == pytest.approx(0.3420, abs=0.001)
    assert var == pytest.approx(4.7060, abs=0.001)
    ev, var = fresh_round_moments(
        hand(Rank.ACE, Rank.ACE),
        card(Rank.FIVE),
        deck_minus(Rank.ACE, Rank.ACE, Rank.FIVE),
        rules,
        calc,
    )
    assert ev == pytest.approx(0.7322, abs=0.001)
    assert var == pytest.approx(2.2955, abs=0.001)


def test_fresh_round_insurance_when_it_is_profitable(calc):
    # A synthetic tens-rich shoe (1 ace, 20 tens unseen) against an ace:
    # the dealer has blackjack 20 times in 21, so insurance is very
    # profitable and must be taken. The expected values are built from
    # the post-peek hand odds by hand, so this checks the wiring of the
    # peek branch, not just the formula.
    rules = Rules()
    unseen = (1, 0, 0, 0, 0, 0, 0, 0, 0, 20)
    ev, var = fresh_round_moments(
        hand(Rank.KING, Rank.QUEEN), card(Rank.ACE), unseen, rules, calc
    )
    a = calc.analyze(hand(Rank.KING, Rank.QUEEN), card(Rank.ACE), unseen)
    hand_odds = a.odds(a.best())
    p = 20 / 21
    ev_ins = p * 1.0 + (1.0 - p) * (-0.5)
    e2_ins = p * 1.0 + (1.0 - p) * 0.25
    e_ins_hand = p * (-1.0) + (1.0 - p) * (-0.5) * hand_odds.ev
    ev_h = p * (-1.0) + (1.0 - p) * hand_odds.ev
    e2_h = p * 1.0 + (1.0 - p) * hand_odds.second_moment
    want_ev = ev_ins + ev_h
    want_e2 = e2_ins + 2.0 * e_ins_hand + e2_h
    assert ev == pytest.approx(want_ev, abs=1e-9)
    assert var == pytest.approx(want_e2 - want_ev**2, abs=1e-9)
    assert ev > ev_h  # taking the insurance is worth having here


def test_fresh_round_insurance_declined_when_it_loses(calc):
    # On a full deck against an ace, insurance loses money, so the fresh
    # round is the peek branch with no side bet at all.
    rules = Rules()
    counts = deck_minus(Rank.KING, Rank.QUEEN, Rank.ACE)
    ev, var = fresh_round_moments(
        hand(Rank.KING, Rank.QUEEN), card(Rank.ACE), counts, rules, calc
    )
    a = calc.analyze(hand(Rank.KING, Rank.QUEEN), card(Rank.ACE), counts)
    hand_odds = a.odds(a.best())
    # Fourteen tens are left unseen: the player's king and queen are
    # ten-value cards already out of the shoe, so it is 14 in 49, not 15.
    p = 14 / 49
    want_ev = p * (-1.0) + (1.0 - p) * hand_odds.ev
    want_e2 = p * 1.0 + (1.0 - p) * hand_odds.second_moment
    assert ev == pytest.approx(want_ev, abs=1e-9)
    assert var == pytest.approx(want_e2 - want_ev**2, abs=1e-9)


# ---- the edge curve --------------------------------------------------------


def test_sample_composition_lands_in_its_bucket():
    rng = random.Random(20261111)
    for bucket in EDGE_BUCKETS:
        sampled = sample_composition(rng, bucket, 6)
        assert sampled is not None
        counts, running = sampled
        tc = true_count(counts, running)
        assert bucket[0] <= tc < bucket[1]
        assert sum(counts) > 100


def test_split_premium_is_stable_across_counts(calc):
    # The premium of splitting over the best non-split action, measured
    # on a full shoe and on a high-count composition: it moves, but by
    # under a tenth, so reusing the full-shoe premium across the sampled
    # compositions of a curve biases the edge by under half a percent.
    rules = Rules(num_decks=6)
    for key in ((8, 6), (1, 5)):
        premium_full, _var = _split_premium_and_var(key, rules, calc)
        rng = random.Random(99)
        comp, _running = sample_composition(rng, (3.0, 5.0), 6)
        counts = list(comp)
        counts[key[0] - 1] -= 2
        counts[key[1] - 1] -= 1
        unseen = tuple(counts)
        rank_of = {
            1: Rank.ACE, 2: Rank.TWO, 3: Rank.THREE, 4: Rank.FOUR, 5: Rank.FIVE,
            6: Rank.SIX, 7: Rank.SEVEN, 8: Rank.EIGHT, 9: Rank.NINE, 10: Rank.TEN,
        }
        rank = rank_of[key[0]]
        upcard = Card(rank_of[key[1]], Suit.CLUBS)
        h = hand(rank, rank)
        a = calc.analyze(h, upcard, unseen, allow_split=False)
        best = max(o.ev for o in a.by_action.values())
        ev_split, _var_split = split_exact.split_moments(
            unseen, key[0], key[0] == 1, key[1], key[1] == 1,
            10 if key[1] == 1 else (1 if key[1] == 10 else None), rules,
        )
        assert abs((ev_split - best) - premium_full) < 0.10


def test_pair_cache_is_used_only_for_the_keys_it_holds(calc):
    # A pre-populated cache values its own pairs by premium and every
    # other pair exactly. Here the cache holds only (8, 6): 8,8 vs 6
    # must come out as the best non-split action plus the premium, and
    # 9,9 vs 6 must come out as the exact C moments value, untouched by
    # the cache.
    rules = Rules(num_decks=6)
    cache = {(8, 6): _split_premium_and_var((8, 6), rules, calc)}
    counts = list(tuple(4 * 6 if i < 9 else 16 * 6 for i in range(10)))
    counts[7] -= 2
    counts[5] -= 1
    unseen88 = tuple(counts)
    a = calc.analyze(hand(Rank.EIGHT, Rank.EIGHT), card(Rank.SIX), unseen88,
                     allow_split=False)
    best = max(o.ev for o in a.by_action.values())
    ev, _var = fresh_round_moments(
        hand(Rank.EIGHT, Rank.EIGHT), card(Rank.SIX), unseen88, rules, calc, cache
    )
    assert ev == pytest.approx(best + cache[(8, 6)][0], abs=1e-9)
    counts = list(tuple(4 * 6 if i < 9 else 16 * 6 for i in range(10)))
    counts[8] -= 2
    counts[5] -= 1
    unseen99 = tuple(counts)
    ev_exact, var_exact = split_exact.split_moments(
        unseen99, 9, False, 6, False, None, rules
    )
    ev, var = fresh_round_moments(
        hand(Rank.NINE, Rank.NINE), card(Rank.SIX), unseen99, rules, calc, cache
    )
    assert ev == pytest.approx(ev_exact, abs=1e-9)
    assert var == pytest.approx(var_exact, abs=1e-9)


def test_enumeration_weights_sum_to_one():
    # The deal probabilities over every (hand, upcard) combination are
    # a partition of the deal space: they must sum to one, on a full
    # shoe and on a depleted one.
    full6 = tuple(4 * 6 if i < 9 else 16 * 6 for i in range(10))
    assert sum(w for *_, w in _deal_situations(full6, 6)) == pytest.approx(1.0, abs=1e-9)
    assert sum(w for *_, w in _deal_situations(SMALL_POOL, 1)) == pytest.approx(1.0, abs=1e-9)


def test_enumeration_matches_sampled_deals(calc):
    # On a small shoe the enumeration must agree with dealing random
    # rounds and averaging them: the two are the same expectation,
    # computed two ways. Loose tolerance, since 300 deals still carry
    # a standard error near two hundredths.
    rules = Rules(num_decks=1)
    ev, _var = enumerate_fresh_round_edge(SMALL_POOL, rules, calc)
    rng = random.Random(20261111)
    evs = []
    for _ in range(300):
        h, upcard, unseen = deal_fresh_round(rng, SMALL_POOL)
        evs.append(fresh_round_moments(h, upcard, unseen, rules, calc)[0])
    assert sum(evs) / len(evs) == pytest.approx(ev, abs=0.08)


def test_edge_curve_rises_with_the_count():
    # A fixed set of 24 non-pair situations, valued on a low-count and
    # a high-count one-deck composition: the same hands are worth more
    # in the rich shoe, so the mean EV rises. Fixed situations, not
    # random deals, so the dealing noise cancels and the comparison is
    # deterministic. The rise on fixed hands is small -- about half a
    # percent over five true-count points, since the bigger part of the
    # rise comes from being dealt better hands, which a fixed panel
    # excludes -- but it is positive, and the recorded rise in the
    # README is 0.0047 here.
    rules = Rules(num_decks=1)
    calc = OddsCalculator(rules, keep_dealer_cache=True)
    low, _lrc = sample_composition(random.Random(11), (-3.0, -2.0), 1)
    high, _hrc = sample_composition(random.Random(22), (3.0, 5.0), 1)
    rank = {
        1: Rank.ACE, 2: Rank.TWO, 3: Rank.THREE, 4: Rank.FOUR, 5: Rank.FIVE,
        6: Rank.SIX, 7: Rank.SEVEN, 8: Rank.EIGHT, 9: Rank.NINE, 10: Rank.TEN,
    }

    def unseen_from(comp, cards):
        counts = list(comp)
        for c in cards:
            counts[c.rank.blackjack_value - 1] -= 1
        return tuple(counts)

    hands = [(10, 10), (10, 6), (10, 2), (6, 5), (1, 7), (1, 2), (10, 7), (10, 5)]
    ups = [6, 10, 1]
    evs_low = []
    evs_high = []
    for v1, v2 in hands:
        for up in ups:
            h = hand(rank[v1], rank[v2])
            upcard = card(rank[up])
            cards = list(h.cards) + [upcard]
            evs_low.append(
                fresh_round_moments(h, upcard, unseen_from(low, cards), rules, calc)[0]
            )
            evs_high.append(
                fresh_round_moments(h, upcard, unseen_from(high, cards), rules, calc)[0]
            )
    mean_low = sum(evs_low) / len(evs_low)
    mean_high = sum(evs_high) / len(evs_high)
    assert mean_high > mean_low
    assert mean_high - mean_low == pytest.approx(0.0047, abs=0.001)


@pytest.mark.skipif(
    not os.environ.get("AOH_SLOW"),
    reason="full six-deck enumeration takes about twenty minutes. Run with AOH_SLOW=1",
)
def test_tc0_anchor_is_near_the_house_edge():
    # The validation for the whole curve: the enumerated TC 0 bucket on
    # a six-deck S17 DAS late-surrender 3:2 table must land near the
    # known house edge, about -0.4%, with a round variance near 1.2.
    rng = random.Random(20261111)
    rules = Rules(num_decks=6)
    row = measure_bucket(rng, (0.0, 1.0), 6, rules)
    assert row["n"] == 1
    assert -0.010 < row["mean_ev"] < 0.0
    assert 0.8 < row["mean_var"] < 1.6
