"""Command-line front end: play hands and see the odds at every decision.

Iteration 3: you choose each hand's bet, and pairs can be split - the player's
hands are then played one at a time against the same dealer hand. Split aces take
their one card and are done, unless the card is another ace and they can resplit.

Iteration 4: the table options are on the banner - S17/H17, the blackjack payout,
late surrender - and a surrender is a command of its own (r), offered on the
first decision of the dealt hand. Money is shown with halves when the balance
holds them ($101.50 after a 3:2 blackjack on an odd bet, say).

Iteration 5: the table comes from Rules - the shoe (decks, cut, burn card), the
money (starting balance, table min/max), and the rule toggles. Insurance is
offered before the peek when the dealer shows an ace, and even money on a player
blackjack against an ace; both show their EV before you answer. The shoe is
reshuffled when the cut card is reached, after the round in progress finishes.
"""
from __future__ import annotations

from .bankroll import Bankroll
from .counting import CardCounter, HiLo
from .hand import Hand
from .odds import Action, Analysis, OddsCalculator
from .player import Player
from .round import Round
from .rules import Rules
from .shoe import Shoe

RESHUFFLE_BELOW = 15  # safety net: never start a round with fewer cards than this


def _prompt() -> str | None:
    try:
        return input("> ").strip().lower()
    except EOFError:
        return None


def _prompt_yes_no(question: str) -> bool:
    """Ask a yes/no question; anything starting with y is yes, EOF is no."""
    try:
        raw = input(f"{question} (y/n): ").strip().lower()
    except EOFError:
        return False
    return raw.startswith("y")


def _prompt_bet(balance: float, rules: Rules) -> int | None:
    """Ask for this hand's bet: a whole number of dollars, within the table
    limits and the balance. None means quit. The balance can hold halves
    (3:2, surrender), but the bet itself is still whole dollars, so the top
    bet is the whole part of the smaller of the table max and the balance."""
    low = rules.min_bet
    top = min(rules.max_bet, int(balance))
    while True:
        try:
            raw = input(f"Bet (${low}-${top}, q to quit): ").strip().lower()
        except EOFError:
            return None
        if raw.startswith("q"):
            return None
        try:
            bet = int(raw)
        except ValueError:
            bet = -1
        if low <= bet <= top:
            return bet
        print(f"Please enter a whole-dollar bet between ${low} and ${top}.")


def _print_status(shoe: Shoe, counter: CardCounter) -> None:
    print(
        f"[{counter.system_name} running count: {counter.running_count:+d} | "
        f"true count: {counter.true_count(shoe.decks_remaining):+.1f} | "
        f"cards left: {shoe.cards_remaining}]"
    )


def _money(amount: float) -> str:
    """$100, or $101.50 when the amount holds halves (3:2, surrender)."""
    if amount == int(amount):
        return f"${int(amount)}"
    return f"${amount:.2f}"


def _print_analysis(analysis: Analysis) -> None:
    for action in (Action.STAND, Action.HIT, Action.DOUBLE, Action.SPLIT, Action.SURRENDER):
        if not analysis.offers(action):
            continue
        o = analysis.odds(action)
        if action is Action.SPLIT:
            # win/push/lose for a split are per hand; only the combined EV is shown.
            print(f"  {'SPLIT':<6} EV {o.ev:+.3f}  (both hands, near-exact)")
        elif action is Action.SURRENDER:
            # Surrender has no win/push/lose split: you always lose half the stake.
            print(f"  {'SURRENDER':<9} EV {o.ev:+.3f}  (lose half your bet)")
        elif o.blackjack_win:
            print(
                f"  {action.value:<6} blackjack {o.blackjack_win * 100:5.1f}%  "
                f"EV {o.ev:+.3f}  (natural pays {o.blackjack_payout:.1f} to 1)"
            )
        else:
            print(
                f"  {action.value:<6} win {o.win * 100:5.1f}%  push {o.push * 100:5.1f}%  "
                f"lose {o.lose * 100:5.1f}%  EV {o.ev:+.3f}"
            )
    print(f"  Best: {analysis.best().value}")


def _play_hand(rnd: Round, hand: Hand, shoe: Shoe, counter: CardCounter,
               calculator: OddsCalculator, label: str) -> None:
    """Run the decisions for one hand until it stands, busts, doubles or splits.

    Returns with the hand finished - or replaced in place by its two split hands,
    which the caller then plays in turn.
    """
    while True:
        if hand.restricted:
            # A split-ace hand is already finished - unless it paired up again
            # and a resplit is on the table.
            if not rnd.can_split(hand):
                return
            print(f"{label}: {hand} - split aces again?")
            cmd = _prompt()
            if cmd is not None and cmd.startswith("p"):
                first, second = rnd.split_hand(hand)
                print(f"  Resplit: {first} and {second}")
            return

        if not rnd.hand_has_decision(hand) and not rnd.can_split(hand):
            return

        _print_status(shoe, counter)
        print(f"{label}: {hand} (${hand.bet} bet)")
        # The live table never computes the split EV: the C decomposition
        # takes a few seconds on a full shoe, so it is on demand - press o.
        # (The pure-Python version took minutes, which is why it is archived.)
        analysis = calculator.analyze(
            hand, rnd.dealer_upcard, rnd.unseen_counts(), allow_split=False
        )
        _print_analysis(analysis)
        split_available = rnd.can_split(hand)
        if split_available:
            print(
                "  (SPLIT is available - the Best above ignores it. "
                "Press o for its EV; it takes a few seconds on a full shoe.)"
            )

        cmd = _prompt()
        if cmd is None or cmd.startswith("q"):
            raise KeyboardInterrupt  # treated as "leave the table" by main()
        if cmd.startswith("o") and split_available:
            print("  Computing the split EV (a few seconds)...")
            split_analysis = calculator.analyze(
                hand, rnd.dealer_upcard, rnd.unseen_counts(), allow_split=True
            )
            o = split_analysis.odds(Action.SPLIT)
            print(f"  SPLIT  EV {o.ev:+.3f}  (both hands, near-exact)")
            print(f"  Best with splitting: {split_analysis.best().value}")
            continue
        if cmd.startswith("h") and analysis.offers(Action.HIT):
            card = rnd.player_hit(hand)
            print(f"  You drew {card} -> {hand}")
        elif cmd.startswith("p") and split_available:
            first, second = rnd.split_hand(hand)
            print(f"  Split into: {first} and {second}")
            return  # the caller replays this position with the first split hand
        elif (
            cmd.startswith("d")
            and analysis.offers(Action.DOUBLE)
            and rnd.can_double(hand)
        ):
            card = rnd.player_double(hand)
            print(f"  You double down and drew {card} -> {hand}")
            return
        elif cmd.startswith("d") and analysis.offers(Action.DOUBLE):
            print("  Your balance does not cover a double here.")
        elif cmd.startswith("r") and analysis.offers(Action.SURRENDER) and rnd.can_surrender(hand):
            rnd.player_surrender(hand)
            print(f"  You surrender and get half your bet back ({_money(hand.bet / 2)}).")
            return
        elif cmd.startswith("s"):
            return
        else:
            print("Please type h, s, d, p, r (surrender), o (split odds) or q.")


def main() -> None:
    rules = Rules()
    shoe = Shoe(
        num_decks=rules.num_decks,
        cut_cards=rules.cut_cards,
        cut_fraction=rules.cut_fraction,
        burn_card=rules.burn_card,
    )
    counter = CardCounter(HiLo())
    calculator = OddsCalculator(rules)
    player = Player(Bankroll(rules.starting_balance))
    bankroll = player.bankroll

    print(
        f"Blackjack odds trainer - {rules.num_decks} deck(s), "
        f"cut at {shoe.cut_position} cards, burn card "
        f"{'on' if rules.burn_card else 'off'}, "
        f"{rules.dealer_rule_label}, blackjack pays {rules.blackjack_payout_label}, "
        f"late surrender {'on' if rules.surrender_allowed else 'off'}, "
        f"insurance {'on' if rules.insurance_allowed else 'off'}, "
        f"splits max {rules.max_splits}"
        f"{' (no resplits)' if rules.max_splits == 1 else ''}"
    )
    print(
        f"You start with {_money(bankroll.balance)}. "
        f"Table limits ${_money(rules.min_bet)}-${_money(rules.max_bet)}."
    )
    print("Commands at a decision: (h)it, (s)tand, (d)ouble, s(p)lit, su(r)render, (o)dds for a split, (q)uit\n")

    try:
        while True:
            if bankroll.balance < rules.min_bet:
                print(f"You're out of money - session over with {_money(bankroll.balance)}.")
                return

            if shoe.cut_reached or shoe.cards_remaining < RESHUFFLE_BELOW:
                shoe.reshuffle()
                counter.reset()
                print(
                    f"--- Shoe reshuffled (cut at {shoe.cut_position} cards), "
                    "count reset ---\n"
                )

            print(f"Balance: {_money(bankroll.balance)}")
            bet = _prompt_bet(bankroll.balance, rules)
            if bet is None:
                print("Bye.")
                return

            rnd = Round(shoe, counter, rules, player)
            rnd.deal(bet)
            print(f"Dealer shows: {rnd.dealer_upcard}")
            print(f"You:          {rnd.hands[0]}")

            # Insurance and even money come before the peek: the dealer shows
            # an ace, and the player bets on the hole card before it is checked.
            if rnd.can_take_even_money():
                decline_ev = calculator.even_money_ev(rnd.unseen_counts())
                print(
                    f"Even money? Take 1:1 now (EV +1.000), or decline "
                    f"(EV {decline_ev:+.3f} at {rules.blackjack_payout_label})."
                )
                if _prompt_yes_no("Take even money"):
                    rnd.take_even_money()
                    print("Even money - you take 1:1 and the hand is over.")
            elif rnd.can_insure():
                ins_ev = calculator.insurance_ev(rnd.unseen_counts())
                print(
                    f"Insurance? {_money(bet / 2)} to insure, pays 2:1 if the "
                    f"dealer has blackjack (EV {ins_ev:+.3f} per unit staked)."
                )
                if _prompt_yes_no("Take insurance"):
                    rnd.take_insurance()
                    print(f"Insured for {_money(bet / 2)}.")
                else:
                    rnd.decline_insurance()

            if rnd.peek():
                print("Dealer peeks... and has blackjack.")
            elif rnd.peeked:
                print("Dealer peeks... no blackjack.")

            index = 0
            while index < len(rnd.hands):
                hand = rnd.hands[index]
                label = (
                    f"Hand {index + 1} of {len(rnd.hands)}"
                    if len(rnd.hands) > 1
                    else "You"
                )
                _play_hand(rnd, hand, shoe, counter, calculator, label)
                if index < len(rnd.hands) and rnd.hands[index] is hand:
                    index += 1
                # else: the hand split and was replaced at this index - play the
                # first of the new hands next without advancing.

            rnd.dealer_play()
            print(f"Dealer:       {rnd.dealer}")
            for i, hand in enumerate(rnd.hands):
                net = rnd.hand_net(hand)
                tag = f"Hand {i + 1}" if len(rnd.hands) > 1 else "Result"
                print(f"{tag}: {hand} - {rnd.hand_outcome(hand).value} ({net:+.2f}$)")
            if rnd.insurance_bet:
                print(
                    f"Insurance: {_money(rnd.insurance_bet)} staked - "
                    f"({rnd.insurance_net():+.2f}$)"
                )
            balance = bankroll.settle(rnd.net_dollars())
            print(f"Balance: {_money(balance)}")
            _print_status(shoe, counter)
            print()
    except KeyboardInterrupt:
        print("Bye.")
        return
