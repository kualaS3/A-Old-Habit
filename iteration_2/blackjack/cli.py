"""Command-line front end: play hands and see the odds at every decision."""
from __future__ import annotations

from .bankroll import Bankroll
from .counting import CardCounter, HiLo
from .odds import Action, Analysis, OddsCalculator
from .round import Round
from .rules import Rules
from .shoe import Shoe

RESHUFFLE_BELOW = 15


def _prompt() -> str | None:
    try:
        return input("> ").strip().lower()
    except EOFError:
        return None


def _print_status(shoe: Shoe, counter: CardCounter) -> None:
    print(
        f"[{counter.system_name} running count: {counter.running_count:+d} | "
        f"true count: {counter.true_count(shoe.decks_remaining):+.1f} | "
        f"cards left: {shoe.cards_remaining}]"
    )


def _print_analysis(analysis: Analysis) -> None:
    for action in Action:
        if not analysis.offers(action):
            continue
        o = analysis.odds(action)
        print(
            f"  {action.value:<6} win {o.win * 100:5.1f}%  push {o.push * 100:5.1f}%  "
            f"lose {o.lose * 100:5.1f}%  EV {o.ev:+.3f}"
        )
    print(f"  Best: {analysis.best().value}")


def main() -> None:
    rules = Rules()
    shoe = Shoe(num_decks=1)
    counter = CardCounter(HiLo())
    calculator = OddsCalculator(rules)
    bankroll = Bankroll()

    print("Blackjack odds trainer - 1 deck, dealer stands on all 17s")
    print(f"You start with ${bankroll.balance}. Every hand is a flat ${bankroll.bet} bet.")
    print("Commands at a decision: (h)it, (s)tand, (d)ouble, (q)uit\n")

    while True:
        if not bankroll.can_play():
            print(f"You're out of money - session over with ${bankroll.balance}.")
            return

        if shoe.cards_remaining < RESHUFFLE_BELOW:
            shoe.reshuffle()
            counter.reset()
            print("--- Shoe reshuffled, count reset ---\n")

        rnd = Round(shoe, counter, rules)
        rnd.deal()
        print(f"Balance: ${bankroll.balance}")
        print(f"Dealer shows: {rnd.dealer_upcard}")
        print(f"You:          {rnd.player}")

        if rnd.peek():
            print("Dealer peeks... and has blackjack.")
        elif rnd.peeked:
            print("Dealer peeks... no blackjack.")

        while rnd.player_has_decision:
            _print_status(shoe, counter)
            analysis = calculator.analyze(rnd.player, rnd.dealer_upcard, rnd.unseen_counts())
            _print_analysis(analysis)

            cmd = _prompt()
            if cmd is None or cmd.startswith("q"):
                print("Bye.")
                return
            if cmd.startswith("h") and analysis.offers(Action.HIT):
                card = rnd.player_hit()
                print(f"You drew {card} -> {rnd.player}")
            elif cmd.startswith("d") and analysis.offers(Action.DOUBLE) and bankroll.can_double():
                card = rnd.player_double()
                print(f"You double down and drew {card} -> {rnd.player}")
            elif cmd.startswith("s"):
                break
            else:
                print("Please type h, s, d or q.")

        rnd.dealer_play()
        net = rnd.net_units()
        balance = bankroll.settle(net)
        print(f"Dealer:       {rnd.dealer}")
        print(f"Result: {rnd.outcome.value} ({net * bankroll.bet:+d}$ -> balance ${balance})")
        _print_status(shoe, counter)
        print()
