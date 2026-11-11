"""Basic strategy: the fixed table of best moves, for the simulator.

Iteration 6b. The simulator cannot call the exact engine for every
decision: a split hand costs seconds, and a thousand-round session
would take days. So it plays basic strategy instead, a fixed table of
the best move for each (hand, upcard), and the exact engine is the
cross-check that proves the table is right, not the player.

The table below is the standard one for the game this project plays:
multi-deck, dealer stands on soft 17, double after split allowed, late
surrender allowed, blackjack pays 3:2. It is written out as data, not
computed, so a lookup is instant and the simulator spends its time
playing rounds. Each row is one hand, each column one upcard, in the
order ace, 2, 3, 4, 5, 6, 7, 8, 9, 10. H is hit, S is stand, D is double,
P is split, R is surrender. A D means double if the table allows it,
else hit. A P means split if it is allowed, else play the hand as its
total. An R means surrender if it is allowed, else hit. The simulator
driver applies those fallbacks, so the table never asks for a move the
round cannot make.

Two things to know. First, the table ignores the composition of the
hand: a 16 is a 16 whether it is 10+6 or 9+7, and a few of those cells
are within a hundredth of a tie, so the exact engine on a particular
composition can disagree with the table there. The verification script
(verify_basic_strategy.py) generates the table from the exact engine
and reports every cell where they differ, so the disagreements are on
the record instead of hidden. Second, the table only fits the standard
game. Give BasicStrategy any other rules and it builds its table from
the exact engine instead, one cell at a time as the simulator needs
them, which is slow but correct for that table.
"""

from __future__ import annotations

from .cards import Card, Rank, Suit
from .hand import Hand
from .odds import Action, OddsCalculator
from .rules import Rules

H = Action.HIT
S = Action.STAND
D = Action.DOUBLE
P = Action.SPLIT
R = Action.SURRENDER

# Upcard columns, in order: ace, 2, 3, 4, 5, 6, 7, 8, 9, 10.

# Hard totals 4 to 20. A 4 is only a pair of 2s that cannot split again,
# and it always hits. A 16 is surrender against 9, 10 and ace, and a 15
# against 10. The rest is the familiar stand on 13-16 against a weak
# upcard, hit against a strong one.
HARD: dict[int, list[Action]] = {
    4: [H, H, H, H, H, H, H, H, H, H],
    5: [H, H, H, H, H, H, H, H, H, H],
    6: [H, H, H, H, H, H, H, H, H, H],
    7: [H, H, H, H, H, H, H, H, H, H],
    8: [H, H, H, H, H, H, H, H, H, H],
    9: [H, H, D, D, D, D, H, H, H, H],
    10: [H, D, D, D, D, D, D, D, D, H],
    11: [H, D, D, D, D, D, D, D, D, D],
    12: [H, H, H, S, S, S, H, H, H, H],
    13: [H, S, S, S, S, S, H, H, H, H],
    14: [H, S, S, S, S, S, H, H, H, H],
    15: [H, S, S, S, S, S, H, H, H, R],
    16: [R, S, S, S, S, S, H, H, R, R],
    17: [S, S, S, S, S, S, S, S, S, S],
    18: [S, S, S, S, S, S, S, S, S, S],
    19: [S, S, S, S, S, S, S, S, S, S],
    20: [S, S, S, S, S, S, S, S, S, S],
}

# Soft totals 13 to 20. Soft 18 doubles against 3-6, stands against 2, 7
# and 8, and hits against 9, 10 and ace. Soft 19 always stands on this
# table: doubling it against a 6 is the H17 play (and a tie on one deck),
# but on a multi-deck S17 shoe standing beats it by about a hundredth,
# so the table stands. The verification script caught that one.
SOFT: dict[int, list[Action]] = {
    13: [H, H, H, H, D, D, H, H, H, H],
    14: [H, H, H, H, D, D, H, H, H, H],
    15: [H, H, H, D, D, D, H, H, H, H],
    16: [H, H, H, D, D, D, H, H, H, H],
    17: [H, H, D, D, D, D, H, H, H, H],
    18: [H, S, D, D, D, D, S, S, H, H],
    19: [S, S, S, S, S, S, S, S, S, S],
    20: [S, S, S, S, S, S, S, S, S, S],
}

# Pairs by rank value. Always split aces and 8s (8s surrender against an
# ace instead), never split 10s, and 5s are played as a hard 10.
PAIRS: dict[int, list[Action]] = {
    1: [P, P, P, P, P, P, P, P, P, P],
    2: [H, P, P, P, P, P, P, H, H, H],
    3: [H, P, P, P, P, P, P, H, H, H],
    4: [H, H, H, H, P, P, H, H, H, H],
    5: [H, D, D, D, D, D, D, D, D, H],
    6: [H, P, P, P, P, P, H, H, H, H],
    7: [H, P, P, P, P, P, P, H, H, H],
    8: [P, P, P, P, P, P, P, P, P, P],
    9: [S, P, P, P, P, P, S, P, P, S],
    10: [S, S, S, S, S, S, S, S, S, S],
}


def is_standard_game(rules: Rules) -> bool:
    """True when the hardcoded table fits these rules: multi-deck, S17,
    double after split, late surrender, 3:2, split aces take one card."""
    return (
        not rules.dealer_hits_soft_17
        and rules.double_after_split
        and rules.surrender_allowed
        and rules.blackjack_payout == (3, 2)
        and rules.num_decks >= 2
        and not rules.hit_split_aces
    )


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


def _representative_hand(kind: str, total: int) -> Hand:
    """A hand with this total and shape, for asking the exact engine.

    Hard totals use ten plus the remainder (ten plus jack for 20, so it
    is not a pair). Soft totals use ace plus the remainder. Pairs use
    two of the rank. The exact action can depend on the composition of
    the hand, so this is one representative, not every hand of that
    total: that is the approximation basic strategy makes.
    """
    hand = Hand()
    if kind == "pair":
        rank = Rank.ACE if total == 1 else Rank.TEN if total == 10 else _RANK_BY_VALUE[total]
        hand.add(Card(rank, Suit.SPADES))
        hand.add(Card(rank, Suit.HEARTS))
    elif kind == "soft":
        hand.add(Card(Rank.ACE, Suit.SPADES))
        hand.add(Card(_RANK_BY_VALUE[total - 11], Suit.HEARTS))
    else:
        if total >= 12:
            hand.add(Card(Rank.TEN, Suit.SPADES))
            rest = total - 10
            hand.add(Card(Rank.JACK if rest == 10 else _RANK_BY_VALUE[rest], Suit.HEARTS))
        else:
            hand.add(Card(_RANK_BY_VALUE[total - 2], Suit.SPADES))
            hand.add(Card(Rank.TWO, Suit.HEARTS))
    return hand


def _upcard(value: int) -> Card:
    rank = Rank.ACE if value == 1 else Rank.TEN if value == 10 else _RANK_BY_VALUE[value]
    return Card(rank, Suit.CLUBS)


def generate_table(
    rules: Rules, calc: OddsCalculator | None = None
) -> tuple[dict[int, list[Action]], dict[int, list[Action]], dict[int, list[Action]]]:
    """Build the (hard, soft, pairs) tables from the exact engine.

    Every cell is the engine's best action for a representative hand of
    that shape against that upcard, on a full shoe of the configured
    decks. Slow: about a hundred seconds on a six-deck shoe, most of it
    the dealer distribution. Used to verify the hardcoded table and to
    play a non-standard game. The calculator keeps its dealer cache
    across cells, which roughly halves the cost. It grows to on the
    order of a gigabyte on a six-deck shoe, so this is a batch tool, not
    something to run inside a game.
    """
    calc = calc or OddsCalculator(rules, keep_dealer_cache=True)
    full = [4 * rules.num_decks if i < 9 else 16 * rules.num_decks for i in range(10)]
    hard: dict[int, list[Action]] = {}
    soft: dict[int, list[Action]] = {}
    pairs: dict[int, list[Action]] = {}
    for kind, table, totals in (
        ("hard", hard, range(4, 21)),
        ("soft", soft, range(13, 21)),
        ("pair", pairs, range(1, 11)),
    ):
        for total in totals:
            hand = _representative_hand(kind, total)
            row: list[Action] = []
            for up_value in range(1, 11):
                upcard = _upcard(up_value)
                counts = list(full)
                for c in list(hand.cards) + [upcard]:
                    counts[c.rank.blackjack_value - 1] -= 1
                # allow_split: the pair rows need the SPLIT action, which
                # analyze() only computes when the caller asks for it (it
                # costs a C decomposition call). Without it every pair
                # cell comes out as the best non-split action instead.
                analysis = calc.analyze(hand, upcard, tuple(counts), allow_split=True)
                row.append(analysis.best())
            table[total] = row
    return hard, soft, pairs


def diff_tables(
    left: tuple[dict[int, list[Action]], dict[int, list[Action]], dict[int, list[Action]]],
    right: tuple[dict[int, list[Action]], dict[int, list[Action]], dict[int, list[Action]]],
) -> list[tuple[str, int, int, Action, Action]]:
    """Cells where two tables disagree: (section, total, upcard, left, right)."""
    out: list[tuple[str, int, int, Action, Action]] = []
    for name, a, b in zip(("hard", "soft", "pair"), left, right):
        for total, row_a in a.items():
            row_b = b[total]
            for i, (x, y) in enumerate(zip(row_a, row_b)):
                if x != y:
                    out.append((name, total, i + 1, x, y))
    return out


class BasicStrategy:
    """The best move for a hand, from the fixed table.

    For the standard game the hardcoded table is used, so a lookup is
    instant. For any other rules the table is built from the exact
    engine one cell at a time as the simulator needs them, and cached:
    slow on the first session, then free. decide() never returns a move
    the round cannot make -- it only reads the table, and the simulator
    driver applies the fallbacks for a double, split or surrender that
    is not available on this hand.
    """

    def __init__(self, rules: Rules) -> None:
        self._rules = rules
        self._standard = is_standard_game(rules)
        self._generated: dict[tuple[str, int, int], Action] = {}
        self._calc: OddsCalculator | None = None

    def decide(self, hand: Hand, upcard: Card) -> Action:
        """The table action for this hand against this upcard.

        Pairs use the pair table even when they came from a split: the
        driver falls back to the total when a further split is not
        available, so decide() reports what the table says and the
        driver reports what the round can do.
        """
        up_value = upcard.rank.blackjack_value
        if hand.is_pair and len(hand.cards) == 2:
            kind = "pair"
            total = hand.cards[0].rank.blackjack_value
        elif hand.is_soft:
            kind = "soft"
            total = hand.total
        else:
            kind = "hard"
            total = hand.total
        return self._lookup(kind, total, up_value)

    def decide_total(self, total: int, soft: bool, upcard: Card) -> Action:
        """The table action for a total, ignoring any pair: the fallback
        when the table says split and the round cannot split again."""
        return self._lookup("soft" if soft else "hard", total, upcard.rank.blackjack_value)

    def _lookup(self, kind: str, total: int, up_value: int) -> Action:
        if kind == "hard" and total > 20:
            return Action.STAND
        if kind == "soft" and total > 20:
            return Action.STAND
        if self._standard:
            table = {"hard": HARD, "soft": SOFT, "pair": PAIRS}[kind]
            return table[total][up_value - 1]
        key = (kind, total, up_value)
        if key not in self._generated:
            self._generated[key] = self._generate_cell(kind, total, up_value)
        return self._generated[key]

    def _generate_cell(self, kind: str, total: int, up_value: int) -> Action:
        if self._calc is None:
            self._calc = OddsCalculator(self._rules, keep_dealer_cache=True)
        hand = _representative_hand(kind, total)
        upcard = _upcard(up_value)
        full = [4 * self._rules.num_decks if i < 9 else 16 * self._rules.num_decks for i in range(10)]
        counts = list(full)
        for c in list(hand.cards) + [upcard]:
            counts[c.rank.blackjack_value - 1] -= 1
        analysis = self._calc.analyze(hand, upcard, tuple(counts), allow_split=True)
        return analysis.best()
