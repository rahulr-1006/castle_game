# castle_game

Choosing allocations for two castle (Colonel Blotto) games by modeling who the other
applicants are, not by solving the game in a vacuum. I put an explicit prior over what the
field looks like, simulate every world that prior allows, and pick the allocation that holds
up best across them.

## 1. Rules

100 soldiers, five castles worth 1, 2, 3, 4, 5 points (15 in total). An allocation is five
non-negative integers summing to 100.

- **Game 1 (head to head).** My allocation plays every other applicant's. Per castle, more
  soldiers takes its points and a tie splits them. Score = average castle points per matchup.
- **Game 2 (four-way).** Random groups of four; per castle the most soldiers takes it (ties
  split). Players are ranked by castle points and paid 9 / 4 / 2 / 0, with tied players
  splitting the payouts of the places they share. Score = average payout over reshuffles.

## 2. Structural facts, with evidence

**Fact 1: Game 1 is separable.** Against a field, sending `s` soldiers to castle `i` is worth
`w[i,s] = v_i * (P(opponent sends < s) + 0.5 * P(opponent sends s))`, and an allocation
scores `sum_i w[i, a_i]`. Only each castle's marginal distribution in the field matters. So:

- The exact best response to any field is a knapsack DP over 5 castles x 101 soldier counts
  (0.2 ms). The tests check it against brute force on small boards and against exhaustive
  search over all 4,598,126 legal allocations on the real board.
- Swapping the DP's `max` for `log-sum-exp` gives *exact* softmax sampling over all 4.6M
  allocations, which is how every noisy player in this repo is drawn.
- Even the prior-weighted score is separable, so the Bayes-optimal allocation is exact too,
  and exhaustive search over all allocations in all 180 worlds takes about 10 seconds.

**Fact 2: Game 1 is a symmetric constant-sum game.** Every matchup's two scores sum to 15
(tested), so the value is 7.5 and fictitious play must converge (Robinson 1951). It does:
exploitability falls as about `1/sqrt(rounds)`, to 0.039 points after 100,000 rounds. The
equilibrium also matches the known continuous solution (Gross and Wagner 1950: castle `i`
gets soldiers uniform on `[0, 2*100*v_i/15]`) to within 0.05 in CDF, with the same supports
(13 / 27 / 41 / 54 / 68 vs 13.3 / 26.7 / 40 / 53.3 / 66.7). Its mean allocation is
6-13-20-27-34: proportional to value.

![equilibrium](results/equilibrium.png)

**Fact 3: more than 7.5 points wins Game 2 outright.** Castle points always total 15, so a
player above 7.5 cannot be caught. The handoff's "8 or more" is the integer version of this.
Exactly 7.5 is *not* enough: `test_exactly_half_the_points_can_still_tie` builds a group where
two players tie for 1st on 7.5 each. The minimal castle sets worth 8 or more are
{5,3}, {5,4}, {5,2,1}, {4,3,1} and {4,3,2} (tested).

**Fact 4: integers create focal points.** Under the prior, the value of one more soldier
spikes just above round numbers, because a crowd sits exactly on them and one soldier
turns a tie into a win.

![focal points](results/focal_points.png)

## 3. Player types and worlds

- **Level 0 (habits).** The "focal" flavor is a mix: 15% equal split, 15% proportional
  (7-13-20-27-33), 45% round-number splits tilted toward the big castles, 15% "stack the 5"
  (40/50/60 on castle 5) and 10% uniform random. The "random" flavor is the textbook
  uniform-over-allocations level 0.
- **Cognitive hierarchy (Camerer, Ho and Chong 2004).** Thinking levels are Poisson(tau).
  Level k noisily best-responds to levels 0..k-1 in their Poisson proportions. tau is the
  "how smart is the field" knob.
- **Equilibrium players** sample the mixed equilibrium (fictitious play in both games).
- **Meta exploiters** noisily best-respond to the actual crowd of their world, which is what
  this repo does. They punish naive play and are themselves a cluster that can be punished.
- **Noise.** Every strategic player plays a softmax ("logit") best response. Game 1: exact
  over all 4.6M allocations, lambda = 8 (low) or 3 (high) per point. Game 2: over the plans
  the search considered, lambda = 4 or 1 per payout point. These cost a strategic player
  facing the round-number crowd about 3% and 11-12% of the best achievable score
  (`results/noise_calibration.csv`).

In Game 2 there is no DP: the ranking couples castles. A best response is found by scoring
DP-built "sweep" plans (one per castle subset, each maximizing the chance of winning every
castle in the subset against the max of three opponents) by simulation, then polishing the
best by local search (moves of 1-5 soldiers).

A **world** is one setting of: tau (five values, set by the prior), equilibrium share in
{0, 0.1, 0.25}, exploiter share in {0, 0.1, 0.25}, level-0 flavor, and noise. That is 180
worlds per prior. Thinking levels run to 8; the Poisson tail beyond that is pooled.

## 4. The prior over the field

There are two priors, both in `PRIORS` in `experiments.py` (each run also writes its prior to
`prior.csv`). Axes are treated as independent. Every weight carries its reason; edit and rerun.

**Main prior: "sharp" (results in `results/`).** The sponsors (Sequoia, OpenAI, Citadel), the
selective nature of the event and priority registration for top scores all point to a sharp
pool. My read is that most thinkers sit at levels 2-3, with a real minority at 4 and up. In a
Poisson hierarchy that is tau of about 2.5: at tau = 2.5, 47% of players are at levels 2-3,
24% at 4+, and 28% still at 0-1.

| Axis | Value: weight | Reason |
|---|---|---|
| tau | 1.5: 0.10 | Even a sharp pool has people who answer a 3-minute optional question quickly. |
| | 2.0: 0.20 | Sharp applicants who stop at "beat the people who beat the obvious split". |
| | **2.5: 0.35** | Core belief: this pool mostly thinks 2-3 levels deep. |
| | 3.0: 0.25 | Priority registration rewards top scores, so many will think harder than usual. |
| | 4.0: 0.10 | Tail: a pool where level 4+ is common. |
| equilibrium share | 0: 0.25 / 0.1: 0.45 / 0.25: 0.30 | Quant applicants who know Blotto randomize around proportional; few recall it exactly under time pressure. |
| exploiter share | 0: 0.15 / 0.1: 0.45 / 0.25: 0.40 | In a pool this competitive, many will model the crowd in some form. |
| level 0 | focal: 0.80 / random: 0.20 | Whoever doesn't think still piles onto round numbers; uniform kept as a stress test. |
| noise | low: 0.60 / high: 0.40 | Sharp players mostly get the arithmetic right; three minutes still makes some sloppy. |

Prior mean tau: about 2.6.

**Comparison prior: "baseline" (results in `results/baseline/`).** My first, more skeptical
prior (tau in {0.5, 1, 1.5, 2, 3} weighted 0.15 / 0.30 / 0.30 / 0.15 / 0.10, mean about 1.4,
fewer equilibrium players and exploiters, noisier). It reflects lab estimates of tau, which
typically land around 1-2 even for selected subjects, and a 3-minute optional format that
pulls depth down. Section 6 shows what changes between the two.

## 5. Results under the sharp prior

### Game 1 (exact: no sampling error in the scoring)

From `results/game1_candidates.csv`. Regret is the gap to the exact best allocation for that
world.

| Candidate | Allocation | Prior score | Worst world | Prior regret | Max regret |
|---|---|---|---|---|---|
| **Bayes (best prior score, exhaustive)** | **8-17-30-44-1** | **8.689** | 7.454 | **0.199** | 0.798 |
| Minimax regret (exhaustive) | 5-19-31-45-0 | 8.630 | 7.484 | 0.258 | **0.752** |
| Concede 5, focal+1 | 7-21-31-41-0 | 8.553 | 7.444 | 0.335 | 1.081 |
| Concede 5 | 10-20-30-40-0 | 8.476 | 7.381 | 0.412 | 1.210 |
| Maximin (exhaustive) | 5-5-28-44-18 | 7.949 | **7.741** | 0.939 | 1.754 |
| Equal split | 20-20-20-20-20 | 6.903 | 5.255 | 1.985 | 4.589 |
| Proportional | 7-13-20-27-33 | 6.742 | 5.509 | 2.146 | 4.336 |

The leaders all **concede castle 5**. In a sharp field, castle 5 is where the thinking players
fight: level-k best responses stack 41-46 there, exploiters go higher, and equilibrium players
spread up to 68. Winning it costs 60+ soldiers. Conceding it funds a win on castles 1-4
(10 points) against most of the field.

**Crossovers** (tau swept in steps of 0.125 from 0.25 to 4.5, the other axes averaged by the
prior). The Bayes pick is the best candidate for **every tau above 0.99**. It is within 0.07
points of the best possible allocation for that exact tau everywhere from 1.75 to 4.0. It loses
0.13-0.28 if the field is really at tau 1-1.5, and more only below that.

| tau | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 |
|---|---|---|---|---|---|---|---|
| 8-17-30-44-1 | 8.72 | 8.79 | 8.76 | 8.71 | 8.66 | 8.57 | 8.45 |
| best possible at that tau | 9.00 | 8.91 | 8.83 | 8.77 | 8.69 | 8.61 | 8.52 |

![crossover](results/crossover.png)

**Testing "aim one level above the field, not three"** (`results/game1_levels.csv`; levels
built at tau = 1.5, so these are "too shallow" players under this prior):

| Level | CH best response | prior score | level-k chain | prior score |
|---|---|---|---|---|
| 1 | 1-6-21-31-41 | 6.54 | 1-6-21-31-41 | 6.54 |
| 3 | 1-0-23-33-43 | 6.68 | 0-1-23-33-43 | 6.69 |
| 4 | 3-16-1-36-44 | 6.85 | 20-2-0-34-44 | **5.91** |
| 6 | 3-11-2-38-46 | 7.06 | 12-4-2-36-46 | 6.73 |

Overshooting is real: in the level-k chain, where each level only beats the level below,
level 4 drops 0.8 points below level 3. But every single-level answer loses 1.6-2.8 points to
the Bayes pick. The field is a mixture, and the best move is to respond to the whole mixture
rather than to one imagined opponent.

### Game 2 (simulated: 20,000 groups per candidate per world, common random numbers)

From `results/game2_candidates.csv`. The CI is the 95% interval on the prior payout. The gap
CI is the paired 95% interval on the difference from the leader, which is much tighter
because every candidate faces the same groups.

| Candidate | Allocation | Prior payout (+-95%) | Gap to leader | P(outright 1st) | Worst world | Max regret |
|---|---|---|---|---|---|---|
| **Sweep {5,3,2,1}, locally searched** | **18-8-2-2-70** | **6.567 +- 0.004** | 0 | 0.47 | **5.74** | **0.96** |
| Expected points vs 3, locally searched | 2-8-7-13-70 | 6.449 +- 0.004 | 0.118 +- 0.003 | 0.46 | 5.50 | 1.04 |
| Sweep {5,4,3,2}, locally searched | 3-8-8-11-70 | 6.444 +- 0.004 | 0.123 +- 0.003 | 0.46 | 5.53 | 1.02 |
| Sweep {5,3} (an 8-point set) | 0-0-30-0-70 | 6.244 | 0.323 | 0.45 | | |
| Sweep {4,5} (an 8-point set) | 0-0-0-31-69 | 5.926 | 0.641 | 0.41 | | |
| Game 1's pick | 8-17-30-44-1 | 3.920 | 2.647 | 0.26 | | |
| Equal split | 20-20-20-20-20 | 3.059 | 3.508 | 0.17 | | |

By tau, the leader is best at 1.5, 2.0, 2.5 and 4.0. At tau = 3.0 the expected-points plan
2-8-7-13-70 edges it (6.64 vs 6.59). It also gets *better* as exploiters grow (6.53 / 6.55 /
6.61 at exploiter share 0 / 0.1 / 0.25), because exploiters in this game cluster on castle 5
just below 70.

**Testing the 8-point family (fact 3).** It is good but not best. Every leader puts 70 on
castle 5, which beats the max of three opponents most of the time, and then spreads the rest
thinly to pick up cheap castles from players who abandoned them. Committing the remaining 30
to castle 3 (the {5,3} sweep, guaranteed 1st when it works) is worth 0.32 less.

**Game 2 equilibrium.** Replicator dynamics on a fixed pool of about 200 plans did not
converge: a best response still earned 3.3 above the 3.75 baseline. The mix piled onto a few
plans with exact soldier counts, which one extra soldier beats. Fictitious play with the
search-based best response did converge: exploitability fell 5.25 -> 1.03 -> 0.37 -> 0.40
over 10 / 175 / 733 / 1500 rounds (`results/game2_equilibrium_trace.csv`). The search oracle
is approximate, so these are lower bounds.

## 6. Final picks

**Game 1: 8-17-30-44-1.** It has the best prior score of all 4.6M allocations (8.69) and the
lowest prior regret (0.20). Its max regret, 0.80, is close to the minimax-regret optimum of
0.75. It is the best candidate at every tau above 1, and within 0.07 of that tau's optimum
across the whole 1.75-4.0 range. Give up castle 5, the one the sharp crowd fights over, and
win the other four.

**Game 2: 18-8-2-2-70.** It has the best prior payout (6.567 +- 0.004), and its lead is clear
(0.118 +- 0.003, paired). It also has the best worst case (5.74) and the lowest max regret
(0.96) on the shortlist, and it improves as exploiters grow. Take castle 5 decisively and
scatter the rest where others left gaps.

**What the prior changed, and why it doesn't matter much for Game 1.** Under the skeptical
baseline prior the Game 1 pick is 3-11-27-3-56: stack castle 5 and concede castle 4, the
opposite bet. Scoring both picks under both priors (same worlds, exact):

| | Under the sharp prior | Under the baseline prior |
|---|---|---|
| 8-17-30-44-1 (sharp pick) | **8.691**, max regret 0.80, worst 7.46 | 8.631 (0.08 behind), max regret **1.58**, worst **7.66** |
| 3-11-27-3-56 (baseline pick) | 8.092 (0.60 behind), max regret 2.64, worst 6.97 | **8.711**, max regret 2.30, worst 7.24 |

The bet is asymmetric. If the field is sharp and you played the shallow pick, you lose 0.60
points. If the field is shallow and you played the sharp pick, you lose 0.08. And even under
the skeptical prior, the sharp pick has the better worst case and the lower max regret. So
8-17-30-44-1 is the right Game 1 answer *whichever prior you hold*, unless you are confident the
field is shallow.

Game 2 barely moves between priors: about 70 on castle 5 is right under both (baseline
leader: 1-28-1-1-69).

## 7. Did the strategy change between games?

Yes, completely. Game 1 rewards the *average* over matchups, so it pays to win many castles by
small margins: four of five contested. Game 2 pays 9 / 4 / 2 / 0, so winning matters far more
than placing, and beating the *max of three* opponents on a castle costs far more than
beating one. That pushes toward concentration. The two games even disagree about castle 5:
the Game 1 pick concedes it, and the Game 2 pick spends 70 to own it outright. Game 1's pick
scores 3.92 in Game 2, barely above the 3.75 a random seat earns.

## 8. Limitations, and how to reproduce

- **The prior is a judgment call.** Under the skeptical prior the Game 1 pick changes (section
  6). The sharp pick is the more robust of the two, but if you think the field is shallow,
  read `results/baseline/`.
- **Level 0 is modeled, not measured.** I have no real data from this contest. The Riddler
  Blotto contests are the closest public data, but they use 10 castles.
- **Game 2 types assume strategic players re-optimize for Game 2.** In reality many will paste
  their Game 1 answer. That would make the field less concentrated on castle 5, but it is
  untested.
- **Game 2 best responses are approximate** (DP-seeded plans plus local search), so Game 2
  regrets are against the best allocation found, not the true optimum.
- **Axes are independent in each prior.** A deeper field probably also means more equilibrium
  players and exploiters. The sharp prior raises all three, but not jointly per world.
- Noise is calibrated against one field (the round-number crowd) and held fixed elsewhere.

Reproduce (Python 3.12, about 3 minutes per prior on an M-series Mac):

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                        # 59 tests, about 10 s
.venv/bin/python experiments.py                      # sharp prior -> results/
.venv/bin/python experiments.py --prior baseline     # skeptical prior -> results/baseline/
.venv/bin/python experiments.py --quick              # 25 s smoke test
.venv/bin/python experiments.py --plots-only         # redraw plots from results/
```

Seeds are fixed, so a rerun reproduces the same numbers.

## 9. Draft form answer (to rewrite in my own words)

> Game 1: 8-17-30-44-1. Game 2: 18-8-2-2-70. I assumed a sharp field: most players two to
> three levels deep, plus equilibrium players and people modeling the crowd. I weighted 180
> such worlds by a prior and searched all 4.6M allocations. In Game 1 the smart crowd fights
> over castle 5, so it is overpriced. I concede it and win the other four, which stays best at
> every depth from one level up. Game 2 pays for winning, not placing, and you must beat the
> max of three, so I take castle 5 decisively with 70 and scatter the rest where others left
> gaps.
