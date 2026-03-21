# Subjective randomness: problem definition

The domain of inquiry is the subjective perception of randomness by humans judges. We seek an explantory theory of which sequences humans find more or less random. To explore this we use an forced-choice experimental paradigm, in which participants are asked to choose the more random sequence from a pair.

## Task

On each trial, the participant sees **two sequences of coin flips** (H and T) and chooses which sequence looks **more random**. This is a classic paradigm for studying representativeness and alternation biases (e.g. Griffiths, "The Rational Basis of Representativeness").

## Stimulus design schema

- **Stimulus**: A pair of two sequences of H and T. Each sequence is a string (e.g. `"HHT"`, `"HTHTHT"`, `"HHTHTTHT"`). The two sequences in a pair **may have different lengths** (e.g. length 4 vs length 6).
- **Response**: Which sequence was chosen (e.g. "left" / "right").

## Experiment design constraints

- **Total trials per experiment: 30.** Target duration is about 5 minutes at ~5 seconds per trial (consistent with Prolific).
- **Maximum sequence length: 8.**

## Models

All models are implemented as functions that, given a stimulus (pair of sequences) and response options, return a probability distribution over responses. The experiment designer uses these to compute expected information gain and select stimuli.
