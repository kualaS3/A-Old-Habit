"""Round moments: the EV and the variance of a whole round, per unit bet.

Iteration 6a. Kelly sizing needs two numbers for every round, the edge
and the variance, and until now this project only had the first. This
module supplies the second, for a whole round played optimally, including
everything that moves the money: doubles (the stake changes mid-hand),
splits (two hands settled against one dealer hand, so their payoffs are
correlated), insurance and even money (side bets decided before the peek).

The pieces, and where each comes from:
- A plain hand, stood, hit or doubled: the win, push and lose
  probabilities already in OutcomeOdds give the second moment directly,
  E[X^2] = (win + lose + blackjack_win * payout^2) * stake^2, and the
  variance is that minus the EV squared. See OutcomeOdds.second_moment.
- A split: the C joint moments engine (split_exact.split_moments), which
  carries E[X^2] through the same joint recursion that carries the EV.
  Only the joint recursion can see the split variance, because the cross
  term between the two hands lives in the one dealer hand they share.
  The live decomposition engine returns an EV and no second moment, so
  it is not used here: moments are measured with the joint, decisions at
  the table still use the decomposition.
- Surrender: a certain -0.5, so E[X^2] is 0.25 and the variance is 0.
- Insurance and even money, and the dealer-blackjack branch itself: the
  round starts before the peek, so fresh_round_moments conditions on it
  explicitly. The hand is only played when the peek says the dealer has
  no blackjack, and on the blackjack branch the hand simply loses its
  single stake (or pushes on a player blackjack). Insurance is taken
  only when its EV is positive, and even money only when taking it beats
  declining it, both by EV.

All values are per unit bet on the original hand. A split or a double
puts more than one unit at risk, so the variance of those rounds is
bigger than a plain hand, often much bigger: two split hands roughly
triple it, because each hand carries its own stake and the two payoffs
move together against the same dealer.
"""

from __future__ import annotations

from . import split_exact
from .cards import Card, Rank, Suit
from .hand import Hand
from .odds import OddsCalculator
from .rules import Rules

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


def _rank_for_value(value: int) -> Rank:
    """A rank worth this blackjack value, for building a pair by hand."""
    if value == 1:
        return Rank.ACE
    if value == 10:
        return Rank.TEN
    return _RANK_BY_VALUE[value]


def _split_premium_and_var(
    pair_key: tuple[int, int], rules: Rules, calc: OddsCalculator
) -> tuple[float, float]:
    """(Premium, variance) of splitting this pair, measured on a full shoe.

    pair_key is (pair rank value, dealer upcard value). The premium is
    how much splitting beats the best non-split action, both valued on a
    fresh full shoe of rules.num_decks decks. It is the part of the split
    value that barely moves with the shoe composition, which is what makes
    it safe to reuse across the sampled compositions of an edge curve
    (see the pair_cache argument of fresh_round_moments).
    """
    rank_value, upcard_value = pair_key
    counts = [4 * rules.num_decks if i < 9 else 16 * rules.num_decks for i in range(10)]
    counts[rank_value - 1] -= 2
    counts[upcard_value - 1] -= 1
    unseen = tuple(counts)
    rank = _rank_for_value(rank_value)
    upcard = Card(_rank_for_value(upcard_value), Suit.CLUBS)
    hand = Hand()
    hand.add(Card(rank, Suit.SPADES))
    hand.add(Card(rank, Suit.HEARTS))
    analysis = calc.analyze(hand, upcard, unseen, allow_split=False)
    best_nonsplit = max(o.ev for o in analysis.by_action.values())
    ev_split, var_split = split_exact.split_moments(
        unseen,
        rank_value,
        rank_value == 1,
        upcard_value,
        upcard_value == 1,
        _peek_excluded_value(upcard),
        rules,
    )
    return ev_split - best_nonsplit, var_split


def _peek_excluded_value(upcard: Card) -> int | None:
    """The hole-card value the peek rules out, if the upcard can make blackjack."""
    if upcard.rank.is_ace:
        return 10
    if upcard.rank.blackjack_value == 10:
        return 1
    return None


def _dealer_blackjack_probability(unseen: tuple[int, ...], upcard: Card) -> float:
    """P(dealer blackjack) from the unseen composition, before the peek."""
    n = sum(unseen)
    if n == 0:
        return 0.0
    if upcard.rank.is_ace:
        return unseen[9] / n
    if upcard.rank.blackjack_value == 10:
        return unseen[0] / n
    return 0.0


def fresh_round_moments(
    player: Hand,
    upcard: Card,
    unseen: tuple[int, ...],
    rules: Rules,
    calc: OddsCalculator | None = None,
    pair_cache: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """(EV, variance) of one round, per unit bet, under optimal play.

    player is the dealt hand (two cards), upcard the dealer's showing
    card, unseen the composition of everything the player has not seen
    (the shoe plus the dealer's hole card), indexed by blackjack value - 1.

    The split candidate is valued with the C joint moments engine, so the
    caller needs the C helper built when the hand is a pair (run python
    build.py). The measurement assumes the player can cover the split or
    the double: the edge of a round is a property of the game, not of
    the bankroll in front of it.

    pair_cache is a shortcut for measuring many rounds, used by the edge
    curve: when it holds a (pair, upcard) key, that pair is valued as the
    best non-split action on this composition plus the split premium for
    that pair and upcard, measured once on a full shoe and stored in the
    dict. The premium barely moves with the composition (a few hundredths
    at most, checked in the tests), so the shortcut trades a small bias,
    well under the sampling noise of a curve, for not calling the C engine
    on every dealt pair. A pair whose key is NOT in the cache is valued
    exactly, on this composition, every time: the caller pre-populates
    the cache with just the pairs that are too slow to value exactly
    (the low pairs, whose C moments state space explodes), and every
    other pair stays exact. Without a cache at all every split is exact.
    """
    n = sum(unseen)
    if n == 0:
        return 0.0, 0.0
    calc = calc or OddsCalculator(rules)
    p_bj = _dealer_blackjack_probability(unseen, upcard)
    payout = rules.blackjack_payout_ratio

    if player.is_blackjack:
        # A natural, before the peek. Against an ace the choice is even
        # money: take a certain +1, or decline and win the payout unless
        # the dealer has blackjack (then it pushes). Against anything
        # else there is no choice, and the same declining numbers apply.
        decline_ev = payout * (1.0 - p_bj)
        decline_e2 = payout * payout * (1.0 - p_bj)
        if upcard.rank.is_ace and 1.0 > decline_ev:
            return 1.0, 0.0
        ev = decline_ev
        return ev, max(0.0, decline_e2 - ev * ev)

    # The hand itself, played after the peek says no dealer blackjack.
    # The non-split actions come from the odds engine. The split
    # candidate comes from the C joint moments engine, and the best EV
    # of the two wins. The split EV used here is the joint one, which
    # agrees with the live decomposition to about a hundredth.
    analysis = calc.analyze(player, upcard, unseen, allow_split=False)
    best_ev: float | None = None
    best_e2: float | None = None
    for action in analysis.by_action:
        odds = analysis.odds(action)
        e2 = odds.second_moment
        if e2 is None:
            continue
        if best_ev is None or odds.ev > best_ev:
            best_ev = odds.ev
            best_e2 = e2
    if best_ev is None or best_e2 is None:
        return 0.0, 0.0
    if player.is_pair and n >= 2:
        single = player.cards[0]
        rank_value = single.rank.blackjack_value
        up_value = upcard.rank.blackjack_value
        if pair_cache is not None and (rank_value, up_value) in pair_cache:
            premium, var_split = pair_cache[(rank_value, up_value)]
            ev_split = best_ev + premium
            if ev_split > best_ev:
                best_ev = ev_split
                best_e2 = var_split + ev_split * ev_split
        else:
            ev_split, var_split = split_exact.split_moments(
                unseen,
                rank_value,
                single.rank.is_ace,
                up_value,
                upcard.rank.is_ace,
                _peek_excluded_value(upcard),
                rules,
            )
            if ev_split > best_ev:
                best_ev = ev_split
                best_e2 = var_split + ev_split * ev_split
    ev_hand = best_ev
    e2_hand = best_e2

    # Insurance, decided before the peek, taken only when its EV is
    # positive. The stake is half the bet: +1 when the dealer has
    # blackjack (2:1 on half a unit), -0.5 when he does not.
    ins_ev = calc.insurance_ev(unseen) if upcard.rank.is_ace else -1.0
    insured = upcard.rank.is_ace and ins_ev > 0.0
    if insured:
        ev_ins = p_bj * 1.0 + (1.0 - p_bj) * (-0.5)
        e2_ins = p_bj * 1.0 + (1.0 - p_bj) * 0.25
        # On the blackjack branch the insurance pays and the hand loses,
        # so I * H is -1 there. Off it the insurance loses half a unit
        # whatever the hand does, so I * H is -0.5 times the hand payoff.
        e_ins_hand = p_bj * (-1.0) + (1.0 - p_bj) * (-0.5) * ev_hand
    else:
        ev_ins = 0.0
        e2_ins = 0.0
        e_ins_hand = 0.0

    # Combine the hand and the side bet over the peek branch. On the
    # blackjack branch the hand loses its single stake before it is ever
    # played. Off it the hand pays what the odds engine says.
    ev_h = p_bj * (-1.0) + (1.0 - p_bj) * ev_hand
    e2_h = p_bj * 1.0 + (1.0 - p_bj) * e2_hand
    ev = ev_ins + ev_h
    e2 = e2_ins + 2.0 * e_ins_hand + e2_h
    return ev, max(0.0, e2 - ev * ev)
