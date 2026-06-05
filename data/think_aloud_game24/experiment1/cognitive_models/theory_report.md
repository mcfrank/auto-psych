# Theory Report — Experiment 1

## random_walk
**Motivation:** Establishes a difficulty baseline with no cognitive assumptions. If humans perform worse than random walk on a puzzle, it indicates active mischoices; if they perform better, it confirms strategy use. This model anchors the comparison space.
**Mechanism:** Uniformly samples a successor state at each step (random pair selection, random operation). No heuristic, no lookahead. Stochasticity is maximal — every legal move is equally likely.

## dfs_sum_heuristic
**Motivation:** Participants verbalize intermediate numbers and appear to gravitate toward states that "feel closer" to the target. The sum-distance heuristic (sum of |num − 24| for all current numbers) captures this intuition. DFS matches the common human strategy of committing to a path and backtracking only when stuck.
**Mechanism:** Stochastic DFS with softmax-weighted child selection: children with lower sum-distance to the target are more likely to be explored next, but selection is probabilistic (temperature = 5). Depth-limited at 12 steps to model bounded working memory.

## factor_biased_search
**Motivation:** Think-aloud protocols in Game of 24 frequently reveal participants naming intermediate "subgoals" such as "I need to get 8" or "I want a 6". These subgoals are almost always factors of 24. This model captures the hypothesis that humans preferentially choose operations that produce factors of the target.
**Mechanism:** At each step, greedily samples a successor state using softmax weights over a factor-proximity heuristic: sum of minimum distances from each current number to the nearest factor of 24 (factors: 1, 2, 3, 4, 6, 8, 12, 24). States where numbers land on or near factors receive higher weight (temperature = 4). This is a one-step greedy lookahead with stochastic selection, making it intermediate between random walk and exhaustive DFS.
