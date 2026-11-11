"""Command-line front end: play hands and see the odds at every decision.

Iteration 3: you choose each hand's bet, and pairs can be split - the player's
hands are then played one at a time against the same dealer hand. Split aces take
their one card and are done, unless the card is another ace and they can resplit.
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

RESHUFFLE_BELOW = 15
MIN_BET = 1


def _prompt() -> str | None:
    try:
        return input("> ").strip().lower()
    except EOFError:
        return None


def _prompt_bet(balance: int) -> int | None:
    """Ask for this hand's bet: a whole number of dollars, MIN_BET..balance.
    None means quit."""
    while True:
        try:
            raw = input(f"Bet (${MIN_BET}-${balance}, q to quit): ").strip().lower()
        except EOFError:
            return None
        if raw.startswith("q"):
            return None
        try:
            bet = int(raw)
        except ValueError:
            bet = -1
        if MIN_BET <= bet <= balance:
            return bet
        print(f"Please enter a whole-dollar bet between ${MIN_BET} and ${balance}.")


def _print_status(shoe: Shoe, counter: CardCounter) -> None:
    print(
        f"[{counter.system_name} running count: {counter.running_count:+d} | "
        f"true count: {counter.true_count(shoe.decks_remaining):+.1f} | "
        f"cards left: {shoe.cards_remaining}]"
    )


def _print_analysis(analysis: Analysis) -> None:
    for action in (Action.STAND, Action.HIT, Action.DOUBLE, Action.SPLIT):
        if not analysis.offers(action):
            continue
        o = analysis.odds(action)
        if action is Action.SPLIT:
            # win/push/lose for a split are per hand. Only the combined EV is shown.
            print(f"  {'SPLIT':<6} EV {o.ev:+.3f}  (both hands, approximated)")
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
        analysis = calculator.analyze(
            hand, rnd.dealer_upcard, rnd.unseen_counts(), allow_split=rnd.can_split(hand)
        )
        _print_analysis(analysis)

        cmd = _prompt()
        if cmd is None or cmd.startswith("q"):
            raise KeyboardInterrupt  # treated as "leave the table" by main()
        if cmd.startswith("h") and analysis.offers(Action.HIT):
            card = rnd.player_hit(hand)
            print(f"  You drew {card} -> {hand}")
        elif cmd.startswith("p") and analysis.offers(Action.SPLIT):
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
        elif cmd.startswith("s"):
            return
        else:
            print("Please type h, s, d, p or q.")


def main() -> None:
    rules = Rules()
    shoe = Shoe(num_decks=1)
    counter = CardCounter(HiLo())
    calculator = OddsCalculator(rules)
    player = Player(Bankroll())
    bankroll = player.bankroll

    print("Blackjack odds trainer - 1 deck, dealer stands on all 17s")
    print(f"You start with ${bankroll.balance}. You choose each hand's bet.")
    print("Commands at a decision: (h)it, (s)tand, (d)ouble, s(p)lit, (q)uit\n")

    try:
        while True:
            if bankroll.balance < MIN_BET:
                print(f"You're out of money - session over with ${bankroll.balance}.")
                return

            if shoe.cards_remaining < RESHUFFLE_BELOW:
                shoe.reshuffle()
                counter.reset()
                print("--- Shoe reshuffled, count reset ---\n")

            print(f"Balance: ${bankroll.balance}")
            bet = _prompt_bet(bankroll.balance)
            if bet is None:
                print("Bye.")
                return

            rnd = Round(shoe, counter, rules, player)
            rnd.deal(bet)
            print(f"Dealer shows: {rnd.dealer_upcard}")
            print(f"You:          {rnd.hands[0]}")

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
                print(f"{tag}: {hand} - {rnd.hand_outcome(hand).value} ({net:+d}$)")
            balance = bankroll.settle(rnd.net_dollars())
            print(f"Balance: ${balance}")
            _print_status(shoe, counter)
            print()
    except KeyboardInterrupt:
        print("Bye.")
        return
