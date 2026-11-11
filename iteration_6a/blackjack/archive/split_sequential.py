"""ARCHIVED: the split EV in pure Python (sequential decomposition).

This is the version of the split odds that iteration 5 shipped first, kept
here to show the attempt -- it is NOT imported anywhere and NOT used by the
game. The live engine is the C port of this same algorithm
(blackjack/split_exact.c, split_ev_decomp, loaded through split_exact.py):
same maths, compiled, so a full-deck spot takes seconds instead of minutes.
The C joint recursion in the same file (split_ev_exact) is the exact
verification tool, not the live engine.

What it does: it values a split sequentially, in the order the game plays
the hands. The hand being split is played to maximise the TOTAL ev of
every hand. The hands still to come are valued by _tail_ev, each to
maximise its own ev, returning the expected dealer distribution after it
-- so the first hand's decisions do account for what its cards do to the
second hand and to the dealer. The one thing the later hands do not
account for is the reverse: a later hand's draws also move the dealer
distribution the earlier hands are settled against, and its decisions are
not re-optimised for that. That effect is second-order (about a hundredth
at a full deck), so this version is near-exact -- but it is far too slow
to use: a full-deck spot did not finish in 280 seconds. On a 13-card pool
it takes about a second a spot, which is what the demo below runs.

Two honest limits, both carried over into the live C port of this file:
the SPLIT row carries only the combined EV (the hands' outcomes are
correlated through the one dealer hand, so no per-hand win/push/lose is
tracked), and a restricted pair of split aces is stood on rather than
resplit, although the game itself allows the resplit -- the C joint
(the verification tool) resplits them, which is why its ace-split value
is a touch higher. With the default split cap of one that never arises:
no resplit is possible at all.

Run the demo as a module from the iteration_5 folder:
    python -m blackjack.archive.split_sequential
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..odds import OutcomeOdds, _BUST, _best_total

if TYPE_CHECKING:
    from ..odds import OddsCalculator


# ---- splitting (sequential exact - see the module docstring) ------------
#
# A hand here is (values, restricted): values is the SORTED tuple of
# blackjack values drawn so far - sorted so drawing a 3 then a 5 is the
# same state as drawing a 5 then a 3 - and restricted means a split ace
# that takes one card only. Three functions share the work:
# - _play_split_first plays the hand being split to maximise the TOTAL ev
#   of every hand, calling _tail_ev for the hands still to come.
# - _hand_after values ONE hand played after others have finished: its
#   own ev, plus the expected payoff of a finished hand standing on each
#   total 2..21 against the dealer distribution after it. It is keyed by
#   (counts, hard, ace, cards, restricted) only - no other hand in the
#   key - so its states are shared across every hand that is played
#   first, which is what keeps the whole thing fast enough to run.
# - _tail_ev values a list of hands still to come (only ever needed after
#   a resplit), each to maximise its own ev, returning (per-hand EVs,
#   expected payoffs). It delegates to _hand_after for a single hand.
# A payoff is linear in the dealer distribution, so the expected payoff
# against the expected distribution is the expected payoff - that is the
# step that lets a finished hand be settled without knowing which cards
# the later hands will draw.

def _split_odds(
    calc, counts: tuple[int, ...], single_hard: int, single_ace: bool
) -> OutcomeOdds:
    """EV of splitting a pair, per 1 unit bet on the original hand.

    Each split hand is dealt its second card first (casino order), then the
    first hand is played to maximise the total ev of both hands, then the
    second, then the dealer - all from the same unseen composition. Only
    the combined EV is returned, in ev_override: the hands' outcomes are
    correlated through the one dealer hand, so no per-hand win/push/lose
    split is tracked.
    """
    # The live calculator no longer carries these caches (it uses the
    # C engine now), so this archived entry point sets them up itself.
    calc._split_cache = {}
    calc._remaining_cache = {}
    calc._hand_cache = {}
    calc._split_dealer_cache = {}
    n = sum(counts)
    if n < 2:
        return OutcomeOdds(0.0, 0.0, 0.0, ev_override=0.0)
    restricted = single_ace and not calc._rules.hit_split_aces
    ev = 0.0
    # The two split hands are symmetric - dealing v1 to hand 1 and v2 to
    # hand 2 is the same game as the other way round - so only unordered
    # pairs are enumerated, with their combined probability.
    for v1 in range(1, 11):
        c1 = counts[v1 - 1]
        if c1 == 0:
            continue
        mid = counts[: v1 - 1] + (c1 - 1,) + counts[v1:]
        for v2 in range(v1, 11):
            c2 = mid[v2 - 1]
            if c2 == 0:
                continue
            nxt = mid[: v2 - 1] + (c2 - 1,) + mid[v2:]
            h1 = (tuple(sorted((single_hard, v1))), restricted)
            h2 = (tuple(sorted((single_hard, v2))), restricted)
            p = (c1 / n) * (c2 / (n - 1))
            if v2 != v1:
                p *= 2.0
            ev += p * _play_split_first(calc, nxt, (h1, h2), 1)
    return OutcomeOdds(0.0, 0.0, 0.0, ev_override=ev)

def _split_dealer(calc, counts: tuple[int, ...]) -> tuple[float, ...]:
    """The dealer's final distribution from these counts, cached.

    The upcard and the peek condition are fixed for one analyze call, so
    the distribution depends only on the counts - and the split recursion
    reaches a great many leaves, so recomputing it at each one would be
    the difference between seconds and minutes.
    """
    d = calc._split_dealer_cache.get(counts)
    if d is None:
        d = tuple(
            calc._dealer_distribution(
                counts,
                calc._up_value,
                calc._up_is_ace,
                exclude_first=calc._peek_excluded_value(),
            )
        )
        calc._split_dealer_cache[counts] = d
    return d

def _payoffs_from_dist(dealer_dist: tuple[float, ...]) -> tuple[float, ...]:
    """Expected payoff of standing on each total 2..21 (index total - 2),
    per unit staked, against a dealer distribution."""
    out = []
    for total in range(2, 22):
        win = dealer_dist[_BUST] + sum(dealer_dist[:total])
        lose = sum(dealer_dist[total + 1 : _BUST])
        out.append(win - lose)
    return tuple(out)

def _hand_after(
    calc,
    counts: tuple[int, ...],
    hard: int,
    ace: bool,
    n_cards: int,
    restricted: bool,
) -> tuple[float, tuple[float, ...]]:
    """(Own EV, expected payoffs) for one hand played from here, optimally
    for its own ev. The payoffs are the expected payoff of a finished hand
    standing on each total 2..21 against the dealer distribution after
    this hand has played - which is what a hand that finished earlier
    needs, to be settled against a dealer it has not seen yet."""
    key = (counts, hard, ace, n_cards, restricted)
    cached = calc._hand_cache.get(key)
    if cached is not None:
        return cached
    best = _best_total(hard, ace)
    dealer_dist = _split_dealer(calc, counts)
    payoffs = _payoffs_from_dist(dealer_dist)
    candidates = [(payoffs[best - 2], payoffs)]  # STAND
    if not restricted and best < 21 and sum(counts) > 0:
        # HIT: average over the card drawn, then continue optimally.
        n = sum(counts)
        own_ev = 0.0
        payoffs_acc = [0.0] * 20
        for v in range(1, 11):
            c = counts[v - 1]
            if c == 0:
                continue
            nxt = counts[: v - 1] + (c - 1,) + counts[v:]
            new_hard = hard + v
            if new_hard > 21:
                own = -1.0
                pay = _payoffs_from_dist(_split_dealer(calc, nxt))
            else:
                own, pay = _hand_after(calc, 
                    nxt, new_hard, ace or v == 1, n_cards + 1, restricted
                )
            p = c / n
            own_ev += p * own
            for i, x in enumerate(pay):
                payoffs_acc[i] += p * x
        candidates.append((own_ev, tuple(payoffs_acc)))
        # DOUBLE: exactly one card, on a doubled stake - only when the
        # table allows doubling after a split.
        if n_cards == 2 and calc._rules.double_after_split:
            own_ev = 0.0
            payoffs_acc = [0.0] * 20
            for v in range(1, 11):
                c = counts[v - 1]
                if c == 0:
                    continue
                nxt = counts[: v - 1] + (c - 1,) + counts[v:]
                new_hard = hard + v
                pay = _payoffs_from_dist(_split_dealer(calc, nxt))
                if new_hard > 21:
                    own = -2.0
                else:
                    own = 2.0 * pay[_best_total(new_hard, ace or v == 1) - 2]
                p = c / n
                own_ev += p * own
                for i, x in enumerate(pay):
                    payoffs_acc[i] += p * x
            candidates.append((own_ev, tuple(payoffs_acc)))
    own_ev, payoffs = max(candidates, key=lambda c: c[0])
    result = (own_ev, payoffs)
    calc._hand_cache[key] = result
    return result

def _tail_ev(
    calc,
    counts: tuple[int, ...],
    hands: tuple[tuple, ...],
    splits_used: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(Per-hand EVs, expected payoffs) for the hands still to come.

    Each hand is played to maximise its own ev, and the payoffs are the
    expected payoff of a finished hand standing on each total 2..21
    against the dealer distribution after every hand has played. A single
    hand delegates to _hand_after. No hands left means the dealer plays
    now, from these counts.
    """
    key = (counts, hands, splits_used)
    cached = calc._remaining_cache.get(key)
    if cached is not None:
        return cached
    if not hands:
        result = ((), _payoffs_from_dist(_split_dealer(calc, counts)))
        calc._remaining_cache[key] = result
        return result
    if len(hands) == 1:
        values, restricted = hands[0]
        own, payoffs = _hand_after(calc, 
            counts, sum(values), 1 in values, len(values), restricted
        )
        result = ((own,), payoffs)
        calc._remaining_cache[key] = result
        return result
    values, restricted = hands[0]
    rest = hands[1:]
    hard = sum(values)
    ace = 1 in values
    best = _best_total(hard, ace)
    n = sum(counts)
    # Every candidate is (this hand's ev, per-hand EVs of the rest,
    # expected payoffs). The hand picks the action with the best ev for
    # itself. The rest are valued the same way either way.
    candidates = []
    # STAND: this hand finishes on its best total. The rest play out.
    evs_rest, payoffs_rest = _tail_ev(calc, counts, rest, splits_used)
    candidates.append((payoffs_rest[best - 2], evs_rest, payoffs_rest))
    if not restricted and best < 21 and n > 0:
        candidates.append(
            _draw_tail(calc, counts, hands, splits_used, double=False)
        )
        if len(values) == 2 and calc._rules.double_after_split:
            candidates.append(
                _draw_tail(calc, counts, hands, splits_used, double=True)
            )
        if (
            len(values) == 2
            and values[0] == values[1]
            and splits_used < calc._rules.max_splits
            and not (values[0] == 1 and not calc._rules.resplit_aces)
        ):
            resplit = _resplit_tail(calc, counts, hands, splits_used)
            if resplit is not None:
                candidates.append(resplit)
    own_ev, evs_rest, payoffs = max(candidates, key=lambda c: c[0])
    result = ((own_ev,) + tuple(evs_rest), payoffs)
    calc._remaining_cache[key] = result
    return result

def _draw_tail(
    calc,
    counts: tuple[int, ...],
    hands: tuple[tuple, ...],
    splits_used: int,
    *,
    double: bool,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """Draw one card to the first hand still to come: a hit (keep playing
    it) or a double (finish it on a doubled stake). Returns (this hand's
    ev, the rest's per-hand EVs, expected payoffs)."""
    n = sum(counts)
    values, restricted = hands[0]
    rest = hands[1:]
    own_ev = 0.0
    evs_rest_acc = [0.0] * len(rest)
    payoffs_acc = [0.0] * 20
    for v in range(1, 11):
        c = counts[v - 1]
        if c == 0:
            continue
        nxt = counts[: v - 1] + (c - 1,) + counts[v:]
        new_values = tuple(sorted(values + (v,)))
        new_hard = sum(new_values)
        if new_hard > 21:
            evs_rest, payoffs = _tail_ev(calc, nxt, rest, splits_used)
            own = -2.0 if double else -1.0
        elif double:
            total = _best_total(new_hard, 1 in new_values)
            evs_rest, payoffs = _tail_ev(calc, nxt, rest, splits_used)
            own = 2.0 * payoffs[total - 2]
        else:
            evs_child, payoffs = _tail_ev(calc, 
                nxt, ((new_values, restricted),) + rest, splits_used
            )
            own, evs_rest = evs_child[0], evs_child[1:]
        p = c / n
        own_ev += p * own
        for i, e in enumerate(evs_rest):
            evs_rest_acc[i] += p * e
        for i, x in enumerate(payoffs):
            payoffs_acc[i] += p * x
    return own_ev, tuple(evs_rest_acc), tuple(payoffs_acc)

def _resplit_tail(
    calc,
    counts: tuple[int, ...],
    hands: tuple[tuple, ...],
    splits_used: int,
) -> tuple[float, tuple[float, ...], tuple[float, ...]] | None:
    """Split the first hand still to come again, and value the two new
    hands as the hand's own ev (their EVs added). Returns like
    _draw_tail, or None when fewer than two cards remain."""
    values, _ = hands[0]
    single = values[0]
    rest = hands[1:]
    n = sum(counts)
    if n < 2:
        return None
    new_restricted = single == 1 and not calc._rules.hit_split_aces
    own_ev = 0.0
    evs_rest_acc = [0.0] * len(rest)
    payoffs_acc = [0.0] * 20
    # Unordered pairs again: the two new hands are symmetric.
    for v1 in range(1, 11):
        c1 = counts[v1 - 1]
        if c1 == 0:
            continue
        mid = counts[: v1 - 1] + (c1 - 1,) + counts[v1:]
        for v2 in range(v1, 11):
            c2 = mid[v2 - 1]
            if c2 == 0:
                continue
            nxt = mid[: v2 - 1] + (c2 - 1,) + mid[v2:]
            h1 = (tuple(sorted((single, v1))), new_restricted)
            h2 = (tuple(sorted((single, v2))), new_restricted)
            evs_new, payoffs = _tail_ev(calc, 
                nxt, (h1, h2) + rest, splits_used + 1
            )
            p = (c1 / n) * (c2 / (n - 1))
            if v2 != v1:
                p *= 2.0
            own_ev += p * (evs_new[0] + evs_new[1])
            for i, e in enumerate(evs_new[2:]):
                evs_rest_acc[i] += p * e
            for i, x in enumerate(payoffs):
                payoffs_acc[i] += p * x
    return own_ev, tuple(evs_rest_acc), tuple(payoffs_acc)

def _play_split_first(
    calc,
    counts: tuple[int, ...],
    hands: tuple[tuple, ...],
    splits_used: int,
) -> float:
    """Total EV of all the split hands, with hands[0] played to maximise it.

    hands[0] is the hand being split (or one of its resplits). Hands[1:]
    are the hands still to come, valued by _tail_ev. When hands[0]
    finishes, the rest play out under their own policies and the total is
    its ev plus theirs.
    """
    key = (counts, hands, splits_used)
    cached = calc._split_cache.get(key)
    if cached is not None:
        return cached
    values, restricted = hands[0]
    rest = hands[1:]
    hard = sum(values)
    ace = 1 in values
    best = _best_total(hard, ace)
    n = sum(counts)
    candidates = []
    # STAND: finish on the best total, then the rest play out.
    evs_rest, payoffs = _tail_ev(calc, counts, rest, splits_used)
    candidates.append(payoffs[best - 2] + sum(evs_rest))
    if not restricted and best < 21 and n > 0:
        # HIT: draw and keep playing this hand (or bust, and move on).
        candidates.append(
            _draw_split_first(calc, counts, hands, splits_used, double=False)
        )
        # DOUBLE: exactly one card, on a doubled stake - only when the
        # table allows doubling after a split.
        if len(values) == 2 and calc._rules.double_after_split:
            candidates.append(
                _draw_split_first(calc, counts, hands, splits_used, double=True)
            )
        # RESPLIT: a pair under the cap. Aces need the resplit_aces toggle.
        # any other pair resplits whenever the cap allows it.
        if (
            len(values) == 2
            and values[0] == values[1]
            and splits_used < calc._rules.max_splits
            and not (values[0] == 1 and not calc._rules.resplit_aces)
        ):
            resplit = _resplit_first(calc, counts, hands, splits_used)
            if resplit is not None:
                candidates.append(resplit)
    result = max(candidates)
    calc._split_cache[key] = result
    return result

def _draw_split_first(
    calc,
    counts: tuple[int, ...],
    hands: tuple[tuple, ...],
    splits_used: int,
    *,
    double: bool,
) -> float:
    """Draw one card to the hand being split: a hit (keep playing it) or a
    double (finish it on a doubled stake, then the rest play out)."""
    n = sum(counts)
    values, restricted = hands[0]
    rest = hands[1:]
    ev = 0.0
    for v in range(1, 11):
        c = counts[v - 1]
        if c == 0:
            continue
        nxt = counts[: v - 1] + (c - 1,) + counts[v:]
        new_values = tuple(sorted(values + (v,)))
        new_hard = sum(new_values)
        if new_hard > 21:
            evs_rest, _ = _tail_ev(calc, nxt, rest, splits_used)
            sub = (-2.0 if double else -1.0) + sum(evs_rest)
        elif double:
            total = _best_total(new_hard, 1 in new_values)
            evs_rest, payoffs = _tail_ev(calc, nxt, rest, splits_used)
            sub = 2.0 * payoffs[total - 2] + sum(evs_rest)
        else:
            sub = _play_split_first(calc, 
                nxt, ((new_values, restricted),) + rest, splits_used
            )
        ev += (c / n) * sub
    return ev

def _resplit_first(
    calc,
    counts: tuple[int, ...],
    hands: tuple[tuple, ...],
    splits_used: int,
) -> float | None:
    """Split the hand being split again: two new one-card hands, each dealt
    a second card, then play resumes at the first of them."""
    values, _ = hands[0]
    single = values[0]
    rest = hands[1:]
    n = sum(counts)
    if n < 2:
        return None
    new_restricted = single == 1 and not calc._rules.hit_split_aces
    ev = 0.0
    # Unordered pairs again: the two new hands are symmetric.
    for v1 in range(1, 11):
        c1 = counts[v1 - 1]
        if c1 == 0:
            continue
        mid = counts[: v1 - 1] + (c1 - 1,) + counts[v1:]
        for v2 in range(v1, 11):
            c2 = mid[v2 - 1]
            if c2 == 0:
                continue
            nxt = mid[: v2 - 1] + (c2 - 1,) + mid[v2:]
            h1 = (tuple(sorted((single, v1))), new_restricted)
            h2 = (tuple(sorted((single, v2))), new_restricted)
            p = (c1 / n) * (c2 / (n - 1))
            if v2 != v1:
                p *= 2.0
            ev += p * _play_split_first(calc, 
                nxt, (h1, h2) + rest, splits_used + 1
            )
    return ev



if __name__ == "__main__":
    import time

    from ..odds import OddsCalculator
    from ..rules import Rules

    pool = (1, 1, 1, 1, 1, 1, 1, 1, 1, 4)  # 13 unseen cards, for speed
    for single, up, label in [(8, 6, "8,8 vs 6"), (1, 5, "A,A vs 5")]:
        calc = OddsCalculator(Rules())
        calc._up_value = up
        calc._up_is_ace = up == 1
        t0 = time.time()
        odds = _split_odds(calc, pool, single, single == 1)
        print(f"{label}: {odds.ev:+.4f} ({time.time() - t0:.1f}s on a 13-card pool)")
    print("On a full deck this takes minutes per spot -- that is why it is archived.")
