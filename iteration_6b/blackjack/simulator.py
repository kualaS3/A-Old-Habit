"""The batch simulator: play many rounds and watch the bankroll.

Iteration 6b. The simulator answers a different question from the
exact engine: not what the best move is for this hand, but what
happens to your money if you play that way for a thousand rounds. It
deals shoe after shoe, follows basic strategy for every decision,
sizes each bet by the policy, and records the bankroll after every
round. It never calls the exact engine while it plays: that would take
seconds on a split hand, and a thousand-round session would take days.
Basic strategy is the fixed table of best moves, and the exact engine
is what proves the table is right. The simulator trusts the table and
spends its time on the thing the engine cannot do: playing a lot of
rounds.

The count drives two decisions. The bet is sized from the true count
at the start of the round, before any cards are dealt, because that is
when the bet is placed. Insurance and even money are decided after the
deal, on the count including the cards just dealt, and only at a true
count of +3 or better: that is the Hi-Lo index where the dealer is
likely enough to have blackjack for the side bet to pay. Basic
strategy on its own never insures. The simulator is a counting player,
so it takes the side bet exactly where the count says it pays.

A session ends when it runs out of rounds, runs out of shoes, hits the
win goal or the stop loss, or is ruined: a bankroll under the table
minimum cannot place the next bet. The risk of ruin is the share of
sessions that end ruined, over many seeded sessions, not a formula:
the textbook formula is in here as a cross-check, and it assumes every
round has the same edge and the same variance, which real rounds do
not, because the edge moves with the count and the bet moves with the
edge. A split or a double that the bankroll cannot cover falls back to
the next best move (double becomes hit, split becomes the total), which
is what a short bankroll forces at a real table.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .bankroll import Bankroll
from .basic_strategy import BasicStrategy
from .counting import CardCounter, HiLo
from .odds import Action
from .player import Player
from .round import Round
from .rules import Rules
from .shoe import Shoe

# The Hi-Lo index for insurance and even money: the dealer has blackjack
# often enough for the side bet to pay at a true count of +3 or better.
INSURANCE_INDEX = 3.0


@dataclass(frozen=True)
class SessionConfig:
    """One session: how long it runs and when it stops early.

    rounds is the maximum number of rounds. max_shoes caps the number
    of shoes (None means no cap). win_goal stops the session once the
    bankroll is that far above where it started, stop_loss once it is
    that far below. Both are deltas in dollars, not absolute balances.
    """

    rounds: int = 1000
    max_shoes: int | None = None
    win_goal: float | None = None
    stop_loss: float | None = None


@dataclass
class SessionResult:
    """One session, played out: the bankroll path and how it ended."""

    rounds_played: int
    shoes_played: int
    starting_bankroll: float
    ending_bankroll: float
    bankroll_history: list[float] = field(default_factory=list)
    ruined: bool = False
    stopped_by: str = "rounds"

    @property
    def growth(self) -> float:
        """Ending bankroll as a fraction of the starting one, minus one."""
        if self.starting_bankroll == 0:
            return 0.0
        return self.ending_bankroll / self.starting_bankroll - 1.0

    @property
    def peak(self) -> float:
        """The highest the bankroll ever reached."""
        return max(self.bankroll_history) if self.bankroll_history else self.starting_bankroll

    @property
    def max_drawdown(self) -> float:
        """The worst fall from a peak, in dollars, over the session."""
        peak = self.bankroll_history[0] if self.bankroll_history else self.starting_bankroll
        worst = 0.0
        for balance in self.bankroll_history:
            peak = max(peak, balance)
            worst = max(worst, peak - balance)
        return worst


def _play_hand(rnd: Round, hand, strategy: BasicStrategy) -> None:
    """Play one hand to completion, following basic strategy.

    A split replaces the hand with two hands, which are played in
    order, in place: the recursion handles resplits the same way. A
    restricted hand (split aces) can only resplit. A double or a split
    the bankroll cannot cover falls back: double becomes hit, split
    becomes the hand played as its total.
    """
    if hand.restricted:
        if rnd.can_split(hand) and strategy.decide(hand, rnd.dealer_upcard) is Action.SPLIT:
            first, second = rnd.split_hand(hand)
            _play_hand(rnd, first, strategy)
            _play_hand(rnd, second, strategy)
        return
    while rnd.hand_has_decision(hand):
        action = strategy.decide(hand, rnd.dealer_upcard)
        if action is Action.SPLIT:
            if rnd.can_split(hand):
                first, second = rnd.split_hand(hand)
                _play_hand(rnd, first, strategy)
                _play_hand(rnd, second, strategy)
                return
            action = strategy.decide_total(hand.total, hand.is_soft, rnd.dealer_upcard)
        if action is Action.DOUBLE:
            if rnd.can_double(hand):
                rnd.player_double(hand)
                return
            action = Action.HIT
        if action is Action.SURRENDER:
            if rnd.can_surrender(hand):
                rnd.player_surrender(hand)
                return
            action = Action.HIT
        if action is Action.HIT:
            rnd.player_hit(hand)
        elif action is Action.STAND:
            return


def simulate(
    rules: Rules,
    strategy: BasicStrategy,
    policy,
    session: SessionConfig,
    rng: random.Random,
    starting_bankroll: float | None = None,
) -> SessionResult:
    """Play one session and return the bankroll path and how it ended.

    policy is any object with a bet(bankroll, true_count) method in
    whole dollars (KellyPolicy or FlatBetPolicy). rng drives the shoe:
    the same seed reproduces the same session exactly.
    """
    start = starting_bankroll if starting_bankroll is not None else rules.starting_balance
    bankroll = Bankroll(start)
    player = Player(bankroll)
    shoe = Shoe(
        rules.num_decks,
        rng,
        cut_cards=rules.cut_cards,
        cut_fraction=rules.cut_fraction,
        burn_card=rules.burn_card,
    )
    counter = CardCounter(HiLo())
    history = [bankroll.balance]
    shoes_played = 1
    stopped_by = "rounds"
    ruined = False
    rounds_played = 0
    for _ in range(session.rounds):
        if bankroll.balance < rules.min_bet:
            ruined = True
            stopped_by = "ruin"
            break
        if shoe.cut_reached:
            if session.max_shoes is not None and shoes_played >= session.max_shoes:
                stopped_by = "shoes"
                break
            shoe.reshuffle()
            counter.reset()
            shoes_played += 1
        bet = policy.bet(bankroll.balance, counter.true_count(shoe.decks_remaining))
        if bet <= 0:
            ruined = True
            stopped_by = "ruin"
            break
        rnd = Round(shoe, counter, rules, player)
        rnd.deal(bet)
        tc_after_deal = counter.true_count(shoe.decks_remaining)
        if rnd.can_take_even_money():
            if tc_after_deal >= INSURANCE_INDEX:
                rnd.take_even_money()
        elif rnd.can_insure():
            if tc_after_deal >= INSURANCE_INDEX:
                rnd.take_insurance()
            else:
                rnd.decline_insurance()
        rnd.peek()
        if not rnd.is_over:
            _play_hand(rnd, rnd.hands[0], strategy)
        rnd.dealer_play()
        bankroll.settle(rnd.net_dollars())
        history.append(bankroll.balance)
        rounds_played += 1
        if session.win_goal is not None and bankroll.balance >= start + session.win_goal:
            stopped_by = "win_goal"
            break
        if session.stop_loss is not None and bankroll.balance <= start - session.stop_loss:
            stopped_by = "stop_loss"
            break
    return SessionResult(
        rounds_played=rounds_played,
        shoes_played=shoes_played,
        starting_bankroll=start,
        ending_bankroll=bankroll.balance,
        bankroll_history=history,
        ruined=ruined,
        stopped_by=stopped_by,
    )


def run_sessions(
    rules: Rules,
    strategy: BasicStrategy,
    policy,
    session: SessionConfig,
    n: int,
    seed: int,
) -> dict[str, object]:
    """Run n sessions and aggregate them: risk of ruin, growth, drawdown.

    Each session gets its own seed, derived from the base seed, so the
    whole batch is reproducible from the one number. Returns the
    aggregates and the per-session results.
    """
    results = [
        simulate(rules, strategy, policy, session, random.Random(seed + i))
        for i in range(n)
    ]
    ruined = sum(1 for r in results if r.ruined)
    endings = [r.ending_bankroll for r in results]
    growths = [r.growth for r in results]
    drawdowns = [r.max_drawdown for r in results]
    return {
        "n": n,
        "risk_of_ruin": ruined / n if n else 0.0,
        "sessions_ruined": ruined,
        "mean_ending_bankroll": sum(endings) / n if n else 0.0,
        "mean_growth": sum(growths) / n if n else 0.0,
        "mean_max_drawdown": sum(drawdowns) / n if n else 0.0,
        "min_ending_bankroll": min(endings) if endings else 0.0,
        "max_ending_bankroll": max(endings) if endings else 0.0,
        "mean_rounds_played": sum(r.rounds_played for r in results) / n if n else 0.0,
        "results": results,
    }


def analytic_risk_of_ruin(
    bankroll: float, edge: float, variance: float, bet: float
) -> float:
    """The textbook risk of ruin, as a cross-check on the simulator.

    The diffusion approximation for a fixed bet and a constant edge and
    variance: exp(-2 * edge * bankroll / (variance * bet)). It is the
    probability that a random walk with drift edge * bet per round and
    variance variance * bet^2 per round ever hits zero from this
    bankroll. It is a check, not the answer: real rounds do not have a
    constant edge or variance, and the bet does not stay fixed, so the
    simulated ruin is the number that counts and this one only has to be
    in the same neighbourhood. A non-positive edge means ruin is certain
    in the long run, so it returns 1.0.
    """
    if edge <= 0.0 or variance <= 0.0 or bet <= 0.0 or bankroll <= 0.0:
        return 1.0
    return min(1.0, math.exp(-2.0 * edge * bankroll / (variance * bet)))
