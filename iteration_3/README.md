## Iteration 3

Everything from iteration 2, plus three changes:

- **Variable bet sizing**: you choose each hand's bet before the deal (whole dollars, $1 up to your balance). Doubles and splits stake against that bet.
- **Splitting**: an equal-rank pair (J+J yes, J+Q no) can split into two hands, each with its own stake, played one at a time against the same dealer hand. Resplitting is allowed up to 3 splits (4 hands max, `Rules.max_splits`), and doubling after a split is allowed. Split aces take exactly one card each and no decisions — unless that card is another ace, which may resplit. A 21 on a split hand is just 21, not a blackjack: only a dealt hand can be a natural. Underneath, the money moved to where it belongs — a `Player` owns the bankroll and the hands; each `Hand` carries its own stake.
- **Split odds — the one approximation in the project**: once split hands exist they get the same exact odds as any hand. But the SPLIT row in the decision table values the two hands independently and adds them, ignoring that they share one shrinking shoe and one dealer outcome, and it does not value resplitting. Everywhere else in this project the numbers are exact; this one is labelled because it isn't. Measured against a simulation of the true game: against a 6 it reads about 0.05 low, splitting aces against a 5 about 0.08 low — but against a ten it overstates splitting by about 0.18, because both hands losing to the same strong dealer hand is exactly what independence throws away. Exact split odds are iteration 4's problem.

Still not in: no surrender or insurance, blackjack still pays 1:1 (3:2 not yet), one player only, one deck.
