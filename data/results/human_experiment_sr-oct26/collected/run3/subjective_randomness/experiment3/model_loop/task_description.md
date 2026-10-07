# The task

Each participant takes part in one experiment. On every trial they see two
sequences of coin flips, written as strings of H (heads) and T (tails), side by
side. The two sequences always have the same length, between 2 and 8 flips;
lengths vary from trial to trial. The participant chooses the one sequence that
looks more random to them. There is no correct answer and no feedback.

Before the first trial, participants read these instructions:

> In this study, you will look at sequences of coin flips and judge how
> random they look.
>
> Imagine flipping a fair coin over and over. Each flip is equally likely to
> come up Heads (H) or Tails (T), and every flip is independent — the coin has
> no memory, so what came before does not change what comes next.
>
> On each trial you will see two sequences of coin flips, side by side. The two
> sequences will have the same length. Your task is to pick the one sequence
> that looks more random to you — the one that looks more like it was produced
> by genuinely random coin flipping.
>
> Different people have different impressions of what makes a sequence look
> random, and there are no right or wrong answers. We are interested in your
> own honest impression, so go with your gut.

Each trial asks "Which sequence looks more random?" and the participant clicks
`Left` or `Right`. Which sequence of a pair is shown on the left is randomised.

## The data

Each row of `responses.csv` is one choice:

| Column           | Meaning                                                         |
| ---------------- | --------------------------------------------------------------- |
| `sequence_a`     | the sequence shown on the left                                  |
| `sequence_b`     | the sequence shown on the right                                 |
| `chose_left`     | 1 if the participant chose the left sequence as more random, else 0 |
| `participant_id` | the participant; unique across experiments                      |
| `trial_index`    | the trial's position in that participant's session              |

Every participant in an experiment sees the same set of pairs.
