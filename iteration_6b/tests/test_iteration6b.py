"""Iteration 6b: Kelly bet sizing and the batch simulator.

The Kelly policy sizes the bet from the edge curve and the bankroll,
and the simulator plays basic strategy for a thousand rounds and
reports what happens to the money. The tests check the sizing maths
exactly (a synthetic curve with a clear slope, so the numbers are not
hostage to the measured curve), the famous basic strategy cells, and
the simulator's behaviour: deterministic on a fixed seed, ruin falling
as the bankroll rises, Kelly beating flat betting over long sessions
with loose bounds, and the textbook ruin formula as a cross-check.
"""
import random

import pytest

from blackjack.basic_strategy import BasicStrategy, is_standard_game
from blackjack.cards import Card, Rank, Suit
from blackjack.hand import Hand
from blackjack.kelly import FlatBetPolicy, KellyPolicy, edge_at, variance_at
from blackjack.odds import Action
from blackjack.rules import Rules
from blackjack.simulator import (
    SessionConfig,
    analytic_risk_of_ruin,
    run_sessions,
    simulate,
)

# A synthetic edge curve with a clear slope, for testing the sizing and
# the simulator without depending on the measured curve: the edge rises
# from -1% at a true count of -2.5 to +1.5% at +4, and the variance is
# near 1. The centres are the 6a bucket centres.
TEST_CURVE = [
    (-2.5, -0.010, 1.0),
    (-1.5, -0.008, 1.0),
    (-0.5, -0.005, 1.0),
    (0.5, -0.002, 1.0),
    (1.5, 0.003, 1.1),
    (2.5, 0.008, 1.1),
    (4.0, 0.015, 1.2),
]


def hand(*ranks: Rank) -> Hand:
    h = Hand()
    for r in ranks:
        h.add(Card(r, Suit.SPADES))
    return h


def card(rank: Rank) -> Card:
    return Card(rank, Suit.HEARTS)


# ---- Kelly sizing --------------------------------------------------------


def test_kelly_bet_grows_with_the_edge():
    rules = Rules(num_decks=6, kelly_fraction=0.5)
    policy = KellyPolicy(rules, TEST_CURVE)
    low = policy.bet(20000, -2.5)
    mid = policy.bet(20000, 1.5)
    high = policy.bet(20000, 4.0)
    assert low == rules.min_bet  # negative edge: the table minimum
    assert mid > low
    assert high > mid
    # Half Kelly on $20000 at edge 0.015, variance 1.2: 0.5 * 0.015 / 1.2
    # * 20000 = $125.
    assert high == pytest.approx(125, abs=1)


def test_kelly_bet_shrinks_with_variance():
    rules = Rules(num_decks=6, kelly_fraction=0.5)
    steep = KellyPolicy(rules, [(-2.5, -0.01, 1.0), (4.0, 0.015, 1.0)])
    flat = KellyPolicy(rules, [(-2.5, -0.01, 4.0), (4.0, 0.015, 4.0)])
    assert steep.bet(5000, 4.0) > flat.bet(5000, 4.0)
    assert flat.bet(5000, 4.0) == pytest.approx(9, abs=1)  # 0.5*0.015/4*5000


def test_kelly_fraction_scales_the_bet():
    half = KellyPolicy(Rules(num_decks=6, kelly_fraction=0.5), TEST_CURVE)
    full = KellyPolicy(Rules(num_decks=6, kelly_fraction=1.0), TEST_CURVE)
    assert full.bet(5000, 4.0) == pytest.approx(2 * half.bet(5000, 4.0), abs=1)


def test_kelly_bet_respects_the_table_limits():
    rules = Rules(num_decks=6, kelly_fraction=1.0, max_bet=100)
    policy = KellyPolicy(rules, TEST_CURVE)
    assert policy.bet(1_000_000, 4.0) == 100  # huge bankroll: capped
    assert policy.bet(5000, -2.5) == rules.min_bet  # negative edge: floor
    assert policy.bet(5000, 0.5) == rules.min_bet  # edge -0.002: floor


def test_kelly_bet_never_exceeds_the_bankroll():
    # A raw Kelly bet bigger than the bankroll is capped at it: with an
    # edge of 2.0 the raw bet on $25 is $50, so the bet is $25. And a
    # bankroll under the minimum bets nothing at all.
    rules = Rules(num_decks=6, kelly_fraction=1.0)
    steep = KellyPolicy(rules, [(-2.5, -0.01, 1.0), (4.0, 2.0, 1.0)])
    assert steep.bet(25, 4.0) == 25
    policy = KellyPolicy(rules, TEST_CURVE)
    assert policy.bet(25, 4.0) == rules.min_bet  # raw bet $0.31: the floor
    assert policy.bet(5, 4.0) == 0  # under the minimum: no bet, ruin


def test_flat_bet_is_the_same_every_hand():
    rules = Rules(num_decks=6)
    policy = FlatBetPolicy(rules, 25)
    assert policy.bet(1000, -2.5) == 25
    assert policy.bet(1000, 4.0) == 25
    assert policy.bet(5, 4.0) == 0


def test_edge_curve_interpolation():
    assert edge_at(TEST_CURVE, -2.5) == pytest.approx(-0.010)
    assert edge_at(TEST_CURVE, 4.0) == pytest.approx(0.015)
    assert edge_at(TEST_CURVE, 0.0) == pytest.approx(-0.0035, abs=1e-9)
    assert edge_at(TEST_CURVE, -10.0) == pytest.approx(-0.010)  # held
    assert edge_at(TEST_CURVE, 10.0) == pytest.approx(0.015)  # held
    assert variance_at(TEST_CURVE, 4.0) == pytest.approx(1.2)


# ---- basic strategy ------------------------------------------------------


def test_basic_strategy_famous_cells():
    # The cells every basic strategy player knows: always split aces and
    # 8s, never split 10s, double 11 against anything but an ace, stand
    # on 17, surrender 16 against 9, 10 and ace, and soft 18 doubles
    # against 3-6 and stands against 2, 7 and 8.
    rules = Rules(num_decks=6)
    assert is_standard_game(rules)
    strategy = BasicStrategy(rules)
    ace = card(Rank.ACE)
    six = card(Rank.SIX)
    ten = card(Rank.TEN)
    assert strategy.decide(hand(Rank.ACE, Rank.ACE), six) is Action.SPLIT
    assert strategy.decide(hand(Rank.EIGHT, Rank.EIGHT), six) is Action.SPLIT
    # A pair of 8s is split even against an ace: the split is worth
    # -0.3852 on a six-deck shoe, better than surrendering at -0.5.
    # The table first had surrender here, and the verification caught it.
    assert strategy.decide(hand(Rank.EIGHT, Rank.EIGHT), ace) is Action.SPLIT
    assert strategy.decide(hand(Rank.TEN, Rank.TEN), six) is Action.STAND
    assert strategy.decide(hand(Rank.NINE, Rank.TWO), six) is Action.DOUBLE
    assert strategy.decide(hand(Rank.NINE, Rank.TWO), ace) is Action.HIT
    assert strategy.decide(hand(Rank.TEN, Rank.SEVEN), ace) is Action.STAND
    assert strategy.decide(hand(Rank.TEN, Rank.SIX), ten) is Action.SURRENDER
    assert strategy.decide(hand(Rank.TEN, Rank.SIX), ace) is Action.SURRENDER
    assert strategy.decide(hand(Rank.TEN, Rank.FIVE), ten) is Action.SURRENDER
    assert strategy.decide(hand(Rank.ACE, Rank.SEVEN), card(Rank.FOUR)) is Action.DOUBLE
    assert strategy.decide(hand(Rank.ACE, Rank.SEVEN), card(Rank.TWO)) is Action.STAND
    assert strategy.decide(hand(Rank.ACE, Rank.SEVEN), card(Rank.NINE)) is Action.HIT
    # Soft 19 always stands on a multi-deck S17 shoe: doubling it
    # against a 6 is the H17 play, but standing beats it by about a
    # hundredth here, and the verification script caught the difference.
    assert strategy.decide(hand(Rank.ACE, Rank.EIGHT), card(Rank.SIX)) is Action.STAND
    assert strategy.decide(hand(Rank.FIVE, Rank.FIVE), six) is Action.DOUBLE
    assert strategy.decide(hand(Rank.FOUR, Rank.FOUR), card(Rank.FIVE)) is Action.SPLIT
    assert strategy.decide(hand(Rank.FOUR, Rank.FOUR), six) is Action.SPLIT
    assert strategy.decide(hand(Rank.FOUR, Rank.FOUR), card(Rank.SEVEN)) is Action.HIT


def test_basic_strategy_split_fallback_is_the_total():
    # When the table says split and the round cannot, the hand is played
    # as its total: 8,8 against a 6 is a split, and as a hard 16 against
    # a 6 it stands.
    rules = Rules(num_decks=6)
    strategy = BasicStrategy(rules)
    assert strategy.decide_total(16, False, card(Rank.SIX)) is Action.STAND
    assert strategy.decide_total(16, False, card(Rank.TEN)) is Action.SURRENDER


def test_basic_strategy_generates_for_non_standard_rules():
    # Under H17 the 11 against an ace is a double, where the S17 table
    # says hit: the generated table follows the rules it is given, not
    # the hardcoded one.
    rules = Rules(num_decks=1, dealer_hits_soft_17=True)
    assert not is_standard_game(rules)
    strategy = BasicStrategy(rules)
    assert strategy.decide(hand(Rank.NINE, Rank.TWO), card(Rank.ACE)) is Action.DOUBLE


# ---- the simulator -------------------------------------------------------


def _session(rounds=500, **kwargs):
    return SessionConfig(rounds=rounds, **kwargs)


def test_simulator_is_deterministic_on_a_fixed_seed():
    rules = Rules(num_decks=6)
    strategy = BasicStrategy(rules)
    policy = KellyPolicy(rules, TEST_CURVE)
    session = _session(500)
    a = simulate(rules, strategy, policy, session, random.Random(7))
    b = simulate(rules, strategy, policy, session, random.Random(7))
    assert a.bankroll_history == b.bankroll_history
    assert a.rounds_played == b.rounds_played == 500
    assert a.ending_bankroll == b.ending_bankroll
    c = simulate(rules, strategy, policy, session, random.Random(8))
    assert c.bankroll_history != a.bankroll_history


def test_simulator_plays_a_full_session_without_ruin():
    # A normal session on a $1000 bankroll: it plays every round, the
    # bankroll moves, and the drawdown and growth are sane numbers.
    rules = Rules(num_decks=6)
    strategy = BasicStrategy(rules)
    policy = FlatBetPolicy(rules, 10)
    result = simulate(rules, strategy, policy, _session(500), random.Random(11))
    assert result.rounds_played == 500
    assert not result.ruined
    assert result.stopped_by == "rounds"
    assert 0 < result.max_drawdown < 1000
    assert -1.0 < result.growth < 1.0
    assert len(result.bankroll_history) == 501
    assert result.peak >= result.starting_bankroll


def test_simulator_ruin_falls_as_the_bankroll_rises():
    # The same flat $10 sessions from a $100 bankroll ruin often and
    # from a $5000 bankroll (nearly) never: ruin is a bankroll problem,
    # and more bankroll is more safety.
    rules = Rules(num_decks=6, starting_balance=100)
    strategy = BasicStrategy(rules)
    policy = FlatBetPolicy(rules, 10)
    session = _session(200)
    small = run_sessions(rules, strategy, policy, session, 30, seed=101)
    # Re-run from a $5000 bankroll: run_sessions uses
    # rules.starting_balance, so a richer rules object is needed.
    rich_rules = Rules(num_decks=6, starting_balance=5000)
    big = run_sessions(rich_rules, strategy, policy, session, 30, seed=101)
    assert small["risk_of_ruin"] > 0.0
    assert big["risk_of_ruin"] < small["risk_of_ruin"]


def test_kelly_growth_beats_flat_over_long_sessions():
    # Over long seeded sessions on a curve with a real slope, Kelly bets
    # more when the count favours the player and less when it does not,
    # so its mean growth beats flat betting. Both policies run the same
    # seeds, so the comparison is paired: they face the same cards, and
    # only the bet sizing differs. Loose bounds: this is a Monte Carlo
    # claim, not an exact number. The bankroll is $10000 so the Kelly
    # bet is above the table minimum at a positive count and the sizing
    # actually varies.
    rules = Rules(num_decks=6, kelly_fraction=0.5, starting_balance=10000)
    strategy = BasicStrategy(rules)
    session = _session(2000)
    kelly = run_sessions(rules, strategy, KellyPolicy(rules, TEST_CURVE), session, 30, seed=202)
    flat = run_sessions(rules, strategy, FlatBetPolicy(rules, 10), session, 30, seed=202)
    assert kelly["mean_growth"] > flat["mean_growth"]


def test_simulator_stops_on_win_goal_and_stop_loss():
    rules = Rules(num_decks=6)
    strategy = BasicStrategy(rules)
    policy = FlatBetPolicy(rules, 10)
    win = simulate(rules, strategy, policy, _session(1000, win_goal=50), random.Random(3))
    assert win.stopped_by in ("win_goal", "rounds")
    if win.stopped_by == "win_goal":
        assert win.ending_bankroll >= 1050
    loss = simulate(rules, strategy, policy, _session(1000, stop_loss=50), random.Random(3))
    assert loss.stopped_by in ("stop_loss", "rounds")
    if loss.stopped_by == "stop_loss":
        assert loss.ending_bankroll <= 950


def test_simulator_survives_a_bankroll_that_cannot_cover_a_double():
    # A $15 bankroll betting the $10 minimum cannot cover a double or a
    # split: the round must fall back to hitting and finish, not crash.
    rules = Rules(num_decks=6, starting_balance=15)
    strategy = BasicStrategy(rules)
    policy = FlatBetPolicy(rules, 10)
    result = simulate(rules, strategy, policy, _session(50), random.Random(5))
    assert result.rounds_played > 0
    assert result.ending_bankroll >= 0


# ---- the textbook ruin formula -------------------------------------------


def test_analytic_ruin_formula_sanity():
    # More bankroll is safer, a bigger bet is riskier, a better edge is
    # safer, and a non-positive edge means ruin is certain in the long run.
    assert analytic_risk_of_ruin(1000, 0.01, 1.0, 10) < analytic_risk_of_ruin(100, 0.01, 1.0, 10)
    assert analytic_risk_of_ruin(1000, 0.01, 1.0, 100) > analytic_risk_of_ruin(1000, 0.01, 1.0, 10)
    assert analytic_risk_of_ruin(1000, 0.02, 1.0, 10) < analytic_risk_of_ruin(1000, 0.01, 1.0, 10)
    assert analytic_risk_of_ruin(1000, 0.0, 1.0, 10) == 1.0
    assert analytic_risk_of_ruin(1000, -0.01, 1.0, 10) == 1.0
    assert 0.0 < analytic_risk_of_ruin(1000, 0.01, 1.0, 10) < 1.0
