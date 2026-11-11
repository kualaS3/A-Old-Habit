## Iteration 4

Everything from iteration 3, plus three table options, all of them on `Rules`, because that is the config:

- **Late surrender, on by default** (`Rules.surrender_allowed`): on the first decision of the dealt hand you can give it up (command `r`) and get half your bet back. Its EV is exactly −0.5 per unit staked, shown on its own row, and it is the best action in the classic spot: 16 against a ten, where standing loses ~0.543 and hitting ~0.507. Split hands can never surrender, and neither can a hand after a hit. A surrendered hand never makes the dealer play.
- **Blackjack pays 3:2 by default** (`Rules.blackjack_payout`, the other option is 2:1): a natural now pays its payout on the stake, in the game and in the EV. Standing on a natural is worth +1.5 (or +2.0 on the 2:1 table), not +1. A 21 on a split hand is still an ordinary 21 and still pays even money. This is why money is no longer whole dollars: a 3:2 blackjack on an odd bet pays $1.50, and a surrender gives back half a stake, so the bankroll is a float and bets stay whole dollars.
- **S17 or H17** (`Rules.dealer_hits_soft_17`, default stand): the dealer rule was already a field. Now the game, the odds and the CLI banner all answer to it. Hitting soft 17 costs the player. Standing on 18 against an ace is −0.082 under S17 and −0.211 under H17, because a dealer who stands on 17 is a certain win for the 18, and a dealer who hits can improve.

The split approximation from iteration 3 is unchanged and still labelled: the SPLIT row values the two hands independently, about 0.05 low against a 6, 0.08 low splitting aces against a 5, and overstating splitting by about 0.18 against a ten. Exact split odds are still a future iteration's problem.

Still not in: no insurance, one player only, one deck.
