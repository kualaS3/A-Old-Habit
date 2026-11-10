"""Exact odds by recursion over the composition of the unseen cards (no simulation).

Model (the player's point of view): "unseen" = the shoe plus the dealer's hole card.
Cards are exchangeable, so drawing the player's next card and the dealer's hole card from
the same unseen pool gives the right probabilities.

STAND = stop now. HIT = take one card, then play the better of stand/hit from there.
DOUBLE = take exactly one card, then stand, on a doubled stake.
SPLIT = separate a pair into two hands. Iteration 5 makes this near-exact,
and the story of why is worth telling. The two hands share one shrinking
shoe and one dealer hand, so they cannot be valued independently. The exact
formulation is a single joint recursion over both hands and the dealer --
but its state space explodes with resplits (389k states at one split, 27M
at two, out of memory at three), so it cannot be the live engine. The live
engine is a sequential decomposition instead, in the order the game actually
plays the hands: the hand being split is played to maximise the TOTAL ev of
every hand, and the hands still to come are each played to maximise their
own ev against the expected dealer distribution after them -- so the first
hand's decisions do account for what its cards do to the second hand and to
the dealer. The one thing the later hands do not account for is the reverse:
a later hand's draws also move the dealer distribution the earlier hands are
settled against, and its decisions are not re-optimised for that. That effect
is second-order -- a card removed moves the earlier hands' ev by about a
hundredth, and it only flips a later hand's decision when that decision is
already within a hundredth of a tie -- so the split ev is exact up to well
under a hundredth at one split, far inside simulation noise. At one split
the decomposition agrees with the joint exactly on the spots checked (8,8
vs 6: both +0.3420 on a full deck). With resplits the gap grows (8,8 vs 6
at two splits: joint +0.3848, decomposition +0.3598) and the decomposition
itself runs out of memory at three splits -- which is why the table caps
splits at one by default (Rules.max_splits). The old independent-hands
approximation read about 0.05 low against a 6, 0.08 low splitting aces
against a 5, and overstated splitting by about 0.18 against a ten (8,8 vs
10: approx -0.453, true -0.633); the new values agree with a simulation of
the true game within its noise. One honest limit remains: the per-hand
HIT/STAND odds shown while you are playing a split hand still value that
hand alone - the other split hand's future draws are in the unseen pool it
draws the dealer from, when in the real game they are gone before the dealer
plays. And the SPLIT row carries only its combined ev now: the per-hand
win/push/lose split is not tracked, because the hands' outcomes are correlated
through the one dealer hand and cannot be summarised per hand.

INSURANCE is not an action in the decision table: it is a side bet offered once,
before the peek, when the dealer shows an ace. insurance_ev gives its EV per unit
staked, from the unseen composition: 2 * P(hole is a ten) - P(hole is not). Even
money is the same bet on a player blackjack: taking it is worth exactly +1, and
declining it is worth the payout times the chance the dealer does not have
blackjack, so even_money_ev gives the declining side.

SURRENDER = give the hand up on the first decision and lose half the stake, so
its EV is exactly -0.5 per unit staked. It is only offered at the top level, never
after a hit: surrender is a first-decision action, and _optimal_play (the
post-hit machinery) never considers it.

A natural is not an even-money win: standing on a blackjack wins the table's
payout (3:2 by default, 2:1 as the other option) per unit staked. The win
probability for a natural lives in OutcomeOdds.blackjack_win, separate from
ordinary wins, so the EV picks up the payout and the probabilities still sum to
one. Only a dealt hand can be a natural - a split hand's 21 is an ordinary win -
and the peek has already ruled out a dealer blackjack, so a natural stands to
win against every dealer outcome the distribution can produce.

The dealer rule comes from Rules: S17 (stand on soft 17) by default, H17 when
dealer_hits_soft_17 is set. It flows through _dealer_distribution, so every EV
in this file already answers to it.

The odds are computed after the dealer's peek: when the upcard is an ace or a ten, the
dealer has already checked and does not have blackjack, so the hole card is conditioned
on not being the blackjack-completing card (not a ten under an ace, not an ace under a
ten). Player draws keep the exchangeable treatment above.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from . import split_exact
from .cards import Card
from .hand import Hand
from .rules import Rules

_BUST = 22  # dealer distribution: index 0..21 = final total, 22 = bust


class Action(Enum):
    HIT = "HIT"
    STAND = "STAND"
    DOUBLE = "DOUBLE"
    SPLIT = "SPLIT"
    SURRENDER = "SURRENDER"


@dataclass(frozen=True)
class OutcomeOdds:
    """Win/push/lose probabilities. EV is per 1 unit staked.

    blackjack_win is the probability of winning with a natural, kept apart from
    ordinary wins because it pays the table's blackjack payout, not even money:
    EV = (win - lose + blackjack_win * blackjack_payout) * stake. For every
    hand that is not a natural, blackjack_win is 0 and the formula is the old
    (win - lose) * stake. win + push + lose + blackjack_win always sums to 1.

    ev_override exists for SPLIT and SURRENDER: the total EV of several hands,
    each possibly played on a doubled stake, cannot be summarised by one
    (win - lose) * stake, so the exact total EV is carried explicitly. For
    SURRENDER the win/push/lose split is the certain loss (0/0/1); for SPLIT
    no per-hand probabilities are tracked at all (0/0/0) - the hands' outcomes
    are correlated through the one dealer hand, so a per-hand split would be
    a fiction. Surrender's EV is exactly -0.5 per unit staked, which no
    win/push/lose split can express either.
    """

    win: float
    push: float
    lose: float
    stake: float = 1.0
    ev_override: float | None = None
    blackjack_win: float = 0.0
    blackjack_payout: float = 1.5

    @property
    def ev(self) -> float:
        if self.ev_override is not None:
            return self.ev_override
        return (self.win - self.lose + self.blackjack_win * self.blackjack_payout) * self.stake

    def scaled(self, factor: float) -> OutcomeOdds:
        return OutcomeOdds(
            self.win * factor,
            self.push * factor,
            self.lose * factor,
            self.stake,
            self.ev_override if self.ev_override is None else self.ev_override * factor,
            self.blackjack_win * factor,
            self.blackjack_payout,
        )

    def __add__(self, other: OutcomeOdds) -> OutcomeOdds:
        return OutcomeOdds(
            self.win + other.win,
            self.push + other.push,
            self.lose + other.lose,
            self.stake,
            None
            if self.ev_override is None and other.ev_override is None
            else (self.ev_override or 0.0) + (other.ev_override or 0.0),
            self.blackjack_win + other.blackjack_win,
            self.blackjack_payout,
        )


NO_ODDS = OutcomeOdds(0.0, 0.0, 0.0)
LOSS = OutcomeOdds(0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Analysis:
    """The odds for every available action at one decision point."""

    by_action: dict[Action, OutcomeOdds]

    def odds(self, action: Action) -> OutcomeOdds:
        return self.by_action[action]

    def offers(self, action: Action) -> bool:
        return action in self.by_action

    def best(self) -> Action:
        """Highest-EV action; STAND wins ties."""
        best = Action.STAND
        for action, o in self.by_action.items():
            if o.ev > self.by_action[best].ev + 1e-12:
                best = action
        return best


def _best_total(hard: int, has_ace: bool) -> int:
    return hard + 10 if has_ace and hard + 10 <= 21 else hard


class OddsCalculator:
    """Not thread-safe: analyze() reuses internal caches."""

    def __init__(self, rules: Rules) -> None:
        self._rules = rules
        self._up_value = 0
        self._up_is_ace = False
        self._dealer_cache: dict[tuple, list[float]] = {}
        self._player_cache: dict[tuple, OutcomeOdds] = {}
        # No split caches live here any more: the split EV is computed by
        # the C decomposition engine (see _split_odds), which keeps its own
        # tables. The pure-Python decomposition is archived at
        # blackjack/archive/split_sequential.py.

    def analyze(
        self,
        player: Hand,
        dealer_upcard: Card,
        unseen: tuple[int, ...],
        allow_split: bool = False,
    ) -> Analysis:
        """unseen: composition of unseen cards, indexed by blackjack value - 1
        (shoe contents plus the dealer's hole card).

        allow_split is set by the caller (the Round knows the split cap and whether
        the balance covers another stake); the SPLIT odds are the near-exact
        value from the C decomposition engine (_split_odds), not the old
        independent-hands approximation.
        """
        self._dealer_cache.clear()
        self._player_cache.clear()
        self._up_value = dealer_upcard.rank.blackjack_value
        self._up_is_ace = dealer_upcard.rank.is_ace

        hard, ace = player.hard_total, player.has_ace
        result = {
            Action.STAND: self._stand_odds(
                unseen, _best_total(hard, ace), is_blackjack=player.is_blackjack
            )
        }
        if player.is_blackjack:
            # A natural ends the round before any decision: no hit, double,
            # split or surrender is ever on the table for it.
            return Analysis(result)
        if sum(unseen) > 0:
            result[Action.HIT] = self._hit_odds(unseen, hard, ace)
            if len(player.cards) == 2 and (
                not player.from_split or self._rules.double_after_split
            ):
                # Doubling a split hand needs the double-after-split toggle.
                result[Action.DOUBLE] = self._double_odds(unseen, hard, ace)
                if self._rules.surrender_allowed and not player.from_split:
                    # Late surrender, first decision only: the hand is certainly
                    # lost (lose = 1.0, so the probabilities still sum to one),
                    # but only half the stake is lost, so the EV rides in
                    # ev_override as exactly -0.5 per unit staked. Split hands
                    # were created by a decision, so they can never surrender.
                    result[Action.SURRENDER] = OutcomeOdds(
                        0.0, 0.0, 1.0, ev_override=-0.5
                    )
            if allow_split and player.is_pair and sum(unseen) >= 2:
                # Two cards must remain: one to deal to each split hand.
                single = player.cards[0]
                result[Action.SPLIT] = self._split_odds(
                    unseen, single.rank.blackjack_value, single.rank.is_ace
                )
        return Analysis(result)

    # ---- insurance ---------------------------------------------------------------------

    def insurance_ev(self, unseen: tuple[int, ...]) -> float:
        """EV of the insurance side bet, per unit staked.

        The dealer shows an ace and has not peeked yet, so the hole card is
        still in the unseen pool: P(dealer blackjack) is the share of tens in
        it. Insurance pays 2:1 on a dealer blackjack and loses the stake
        otherwise, so EV = 2p - (1 - p) = 3p - 1.
        """
        n = sum(unseen)
        if n == 0:
            return -1.0
        return 3.0 * unseen[9] / n - 1.0

    def even_money_ev(self, unseen: tuple[int, ...]) -> float:
        """EV of DECLINING even money on a player blackjack vs a dealer ace.

        Taking even money is worth exactly +1. Declining risks the push: the
        natural pays the table payout when the dealer does not have blackjack
        (probability 1 - p, with p the share of tens in the unseen pool, as in
        insurance_ev) and pushes otherwise. Declining wins whenever the payout
        times (1 - p) beats 1 - on a full single deck at 3:2 that is barely true.
        """
        n = sum(unseen)
        if n == 0:
            return 0.0
        p = unseen[9] / n
        return self._rules.blackjack_payout_ratio * (1.0 - p)

    # ---- player side ---------------------------------------------------------------------

    def _peek_excluded_value(self) -> int | None:
        """The hole-card value the peek rules out, if the upcard can make blackjack."""
        if self._up_is_ace:
            return 10
        if self._up_value == 10:
            return 1
        return None

    def _stand_odds(
        self, counts: tuple[int, ...], player_total: int, *, is_blackjack: bool = False
    ) -> OutcomeOdds:
        d = self._dealer_distribution(
            counts, self._up_value, self._up_is_ace, exclude_first=self._peek_excluded_value()
        )
        if is_blackjack:
            # A natural beats every dealer outcome in this distribution: the
            # peek has already ruled out a dealer blackjack (the only way the
            # dealer ties a natural), and a dealer 21 in d is a drawn 21, not
            # a blackjack - it loses to a natural. So the whole probability
            # mass is a blackjack win, paying the table's payout.
            return OutcomeOdds(
                0.0,
                0.0,
                0.0,
                blackjack_win=1.0,
                blackjack_payout=self._rules.blackjack_payout_ratio,
            )
        win = d[_BUST] + sum(d[:player_total])
        push = d[player_total]
        lose = sum(d[player_total + 1 : _BUST])
        return OutcomeOdds(win, push, lose, blackjack_payout=self._rules.blackjack_payout_ratio)

    def _hit_odds(self, counts: tuple[int, ...], hard: int, ace: bool) -> OutcomeOdds:
        """Take exactly one card, then continue optimally. Caller guarantees cards remain."""
        n = sum(counts)
        acc = NO_ODDS
        for v in range(1, 11):
            c = counts[v - 1]
            if c == 0:
                continue
            nxt = counts[: v - 1] + (c - 1,) + counts[v:]
            new_hard = hard + v
            if new_hard > 21:
                outcome = LOSS
            else:
                outcome = self._optimal_play(nxt, new_hard, ace or v == 1)
            acc = acc + outcome.scaled(c / n)
        return acc

    def _double_odds(self, counts: tuple[int, ...], hard: int, ace: bool) -> OutcomeOdds:
        """Take exactly one card, then stand, with the stake doubled.

        The win/push/lose figures are the same probabilities as standing after one
        card; only the EV doubles, via stake=2.
        """
        n = sum(counts)
        acc = NO_ODDS
        for v in range(1, 11):
            c = counts[v - 1]
            if c == 0:
                continue
            nxt = counts[: v - 1] + (c - 1,) + counts[v:]
            new_hard = hard + v
            if new_hard > 21:
                outcome = LOSS
            else:
                outcome = self._stand_odds(nxt, _best_total(new_hard, ace or v == 1))
            acc = acc + outcome.scaled(c / n)
        return OutcomeOdds(acc.win, acc.push, acc.lose, stake=2.0)

    # ---- splitting (the C decomposition engine) ---------------------------
    #
    # The split EV comes from the C decomposition engine (split_exact.py):
    # the split is valued in the order the game actually plays the hands,
    # the hand being split maximising the TOTAL ev, the later hands each
    # maximising their own ev against the expected dealer distribution. The
    # pure-Python version of that algorithm is archived at
    # blackjack/archive/split_sequential.py; the C port is the live engine
    # because the Python one takes minutes a spot on a full deck. The C
    # joint recursion (split_exact.split_ev) is the exact check, not the
    # live engine. See the module docstring and the iteration 5 README.

    def _split_odds(
        self, counts: tuple[int, ...], single_hard: int, single_ace: bool
    ) -> OutcomeOdds:
        """EV of splitting a pair, per 1 unit bet on the original hand.

        Each split hand is dealt its second card first (casino order), then
        the decomposition plays them out -- all from the same unseen
        composition. Only the combined EV is returned, in ev_override: the
        hands' outcomes are correlated through the one dealer hand, so no
        per-hand win/push/lose split is tracked. Raises RuntimeError when
        the C helper is not built (run python build.py): a wrong split
        number is worse than no split number, so there is no silent
        fallback to the archived Python version.
        """
        n = sum(counts)
        if n < 2:
            return OutcomeOdds(0.0, 0.0, 0.0, ev_override=0.0)
        ev = split_exact.split_decomp_ev(
            counts,
            single_hard,
            single_ace,
            self._up_value,
            self._up_is_ace,
            self._peek_excluded_value(),
            self._rules,
        )
        return OutcomeOdds(0.0, 0.0, 0.0, ev_override=ev)

    def _optimal_play(self, counts: tuple[int, ...], hard: int, ace: bool) -> OutcomeOdds:
        """Best of standing or hitting again from this state."""
        key = (counts, hard, ace)
        cached = self._player_cache.get(key)
        if cached is not None:
            return cached
        best = _best_total(hard, ace)
        result = self._stand_odds(counts, best)
        if best < 21 and sum(counts) > 0:
            hit = self._hit_odds(counts, hard, ace)
            if hit.ev > result.ev:
                result = hit
        self._player_cache[key] = result
        return result

    # ---- dealer side ---------------------------------------------------------------------

    def _dealer_distribution(
        self,
        counts: tuple[int, ...],
        hard: int,
        ace: bool,
        exclude_first: int | None = None,
    ) -> list[float]:
        """Distribution of the dealer's final result. Returned lists are read-only.

        exclude_first: on this (outermost) call only, the first card drawn — which
        models the hole card — may not have this blackjack value, and the draw is
        renormalised without it. That is the peek condition. Deeper draws are
        unconditioned. If every unseen card has the excluded value there is nothing
        to condition on and the draw falls back to unconditioned.
        """
        out = [0.0] * (_BUST + 1)
        if hard > 21:
            out[_BUST] = 1.0
            return out
        best = _best_total(hard, ace)
        n = sum(counts)
        if n == 0 or not self._rules.dealer_must_hit(best, best != hard):
            out[best] = 1.0
            return out

        if exclude_first is not None and n - counts[exclude_first - 1] == 0:
            exclude_first = None
        if exclude_first is None:
            key = (counts, hard, ace)
            cached = self._dealer_cache.get(key)
            if cached is not None:
                return cached
        denom = n if exclude_first is None else n - counts[exclude_first - 1]
        for v in range(1, 11):
            if v == exclude_first:
                continue
            c = counts[v - 1]
            if c == 0:
                continue
            nxt = counts[: v - 1] + (c - 1,) + counts[v:]
            sub = self._dealer_distribution(nxt, hard + v, ace or v == 1)
            p = c / denom
            for i, s in enumerate(sub):
                if s:
                    out[i] += p * s
        if exclude_first is None:
            self._dealer_cache[key] = out
        return out
