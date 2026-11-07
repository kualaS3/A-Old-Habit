## Iteration 1

Iteration 1, how exciting. This is what I think should be the bare bones version and I'm hoping its not going to be too complex:

- One deck, one player vs dealer, actions: hit / stand
- Exact win / push / lose probabilities and EV for each action at every decision
- Hi-Lo running count (and true count) of all cards seen so far
- Runs as a command-line program

### How it works

You play hands against the dealer from a single shuffled deck. Before each decision the
program shows, for both hit and stand:

- the chance you win, push or lose
- the expected value (EV) per 1 unit bet, which is `win% - lose%`

It also shows the Hi-Lo running count of every card you can see (2-6 are +1, 7-9 are 0, 10/J/Q/K/A are -1)
and the true count (running count divided by decks left). The dealer's face-down card is only counted once it is turned over.

The odds are exact, not simulated. The program looks at the cards you haven't seen (the rest of the
deck plus the dealer's face-down card) and works through every possible way the hand could play out.

or more.
- **Hit**: you take one card, then play whichever is better from there, stand or hit again.

### Running it

Python 3.10 or newer. Nothing to install to play.

    python -m blackjack

In Google Colab, clone the repo, then run this in a cell:

    !git clone https://github.com/kualaS3/A-Old-Habit.git
    %cd A-Old-Habit/iteration_1

    from blackjack.cli import main
    main()

Type `h` to hit, `s` to stand, `q` to quit. Example:

    Dealer shows: 6C
    You:          8H 5C (13)
    [Hi-Lo running count: +2 | true count: +2.4 | cards left: 43]
      HIT   win  35.9%  push   4.4%  lose  59.8%  EV -0.239
      STAND win  44.5%  push   0.0%  lose  55.5%  EV -0.110
      Best: STAND
    >

Tests: `pip install pytest`, then `python -m pytest`.

### Project layout

    iteration_1/
      blackjack/
        cards.py      Rank, Suit, Card
        shoe.py       Shoe: the deck(s)
        hand.py       Hand: totals, soft/hard, bust, blackjack
        rules.py      Rules: when the dealer must draw
        counting.py   HiLo and CardCounter: the running and true count
        odds.py       OddsCalculator: the win/push/lose odds for hit and stand
        round.py      Round: one hand from deal to result
        cli.py        The command-line loop
        __main__.py   Lets you start it with `python -m blackjack`
      tests/          pytest tests

### Limits of this version

- Hit and stand only: no doubling, splitting, surrender or insurance, and blackjack is not paid 3:2.
- The dealer checking for blackjack is not factored into the odds, so they are very slightly off
  when the dealer shows an ace or a ten.
