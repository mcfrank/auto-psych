# Instruction wording: literature basis

Verbatim instruction wordings from prior subjective-randomness experiments, used
to ground the participant instructions in `problem_definition.md` (the
"Experiment presentation" section). Each entry was extracted from the primary
source and independently verified against the source PDF. The goal: word the
task clearly without **coaching** participants toward the structural cues under
study (run length / streakiness, alternation / switching rate, H–T balance,
independence).

## Forced-choice / judgment paradigms ("which looks more random")

- **Griffiths & Tenenbaum (2003), "From Algorithmic to Subjective Randomness"
  (NIPS).** The canonical pair `HHTHTHTT` vs `HHHHHHHH` with the question which
  "seems more random." *Caveat (flagged in verification):* in this paper that
  two-sequence comparison is described as "a common demonstration conducted in
  introductory psychology classes" — a pedagogical illustration. Their actual
  experiment was single-sequence classification of all 128 length-8 sequences
  (fair-coin vs. other source). So "which looks more random" is the canonical
  *framing*, not a verbatim experiment instruction from this paper.

- **Griffiths, Daniels, Austerweil & Tenenbaum (2018), "Subjective randomness as
  statistical inference" (Cognitive Psychology).** Deliberately **avoid
  specifying the alternative hypothesis** — "non-random" is left for participants
  to supply themselves rather than defined for them. Canonical 2AFC asks which of
  two equally probable length-8 sequences (e.g. `HHHHHHHH` vs `HTHTTHTT`) is
  "more likely to occur"; most people pick the more alternating one. Falk &
  Konold's rating paradigm = 1–10 randomness scale on length-21 sequences.

- **Reimers, Donkin & Le Pelley (2018), "Perceptions of randomness in binary
  sequences" (Cognition 172:11–25).** 2AFC framed via a **"fair unbiased coin"**;
  Exp. 2 asked which string is "more likely to occur at least once in a sequence
  of 20 coin flips," on-screen buttons labelled "THIS ONE", options left/right,
  a "NEXT" button. Exp. 1 rating task used a slider from "impossible … will
  appear" to "certain … will appear". **Explicitly identifies the cues a neutral
  instruction must NOT mention: alternation rate, equal H/T proportion,
  incompressibility/complexity.**

- **Williams & Griffiths (2013), JEP:General.** Model of a **minimal, non-coaching**
  instruction (Exp. 3 nested condition): participants were told only that half the
  sequences came from a random process and half from a non-random process,
  **without being told what defines random vs. non-random**, so they relied on
  their own intuitions. Other (leading) conditions explicitly named the cue (a coin
  that "tends to repeat its flips" vs. one that "tends to change its flips").
  Fair-coin / 50%-heads framing used throughout.

- **Gronchi & Sloman (2021).** 2AFC over pairs of 8-char X/O sequences: "Your task
  is to decide as fast as possible which one has been produced by the random
  process" — note this *tells* the participant one is random and one regular (a
  different framing from "which looks more random").

- **Bar-Hillel & Wagenaar (1991), "The Perception of Randomness" (Adv. Appl.
  Math. 12:428–454).** Foundational review. Canonical judgment-task phrasings:
  "is this series like a coin?" / "which of these series is most like a coin?";
  production tasks "produce a series like a coin." Documents the over-alternation
  bias (preferred alternation rate ≈ .60 vs. .50 for a fair coin).

- **Nickerson (2002), Psychological Review 109(2):330–357.** Review. Argues task
  instructions are critically important and results are hard to interpret when
  instructions are vague/ambiguous — frames the core tension between
  clear-but-leading and minimal-but-ambiguous wording.

## Production paradigms (for contrast — NOT our 2AFC task)

- **Kleinberg, Liang & Mullainathan (2017), arXiv:1706.06974.** Verbatim: "We are
  researchers interested in how well humans can produce randomness. A coin flip …
  is about as random as it gets. Your job is to mimic a coin. We will ask you to
  generate 8 flips of a coin. … just like what we would get if we flipped a coin."
  Deliberately avoids naming alternation/runs/balance; only a detection-algorithm
  incentive. Anti-randomizer instruction: "it is important to us that you not
  actually flip a coin or use some other randomizing device."
- **Guseva, Bogler, Allefeld & Haynes (2023), Frontiers in Psychology
  14:1113654.** Five between-subjects instruction conditions for binary
  production. Wording materially changed output randomness. E.g. Explicit
  Randomness = "You have to choose the sides of the coin randomly."; Mental Coin
  Toss = "You have to simulate a coin toss in your head … not different from the
  results of a real fair coin toss."; Irregularity (a *leading* instruction) =
  "… maximally irregular and chaotic … should not be able to see any pattern or
  regularity." Demonstrates how strongly wording can coach.
- **Warren et al. (2018); Fudenberg, Kleinberg, Liang & Mullainathan (2019/2022).**
  Minimal production framing ("representative of flipping a fair coin"); no
  cue-coaching.

## Theory context (no verbatim instructions)

- **Hahn & Warren (2009), Psychological Review 116(2):454–461,**
  "Perceptions of randomness: Why three heads are better than four." Argues the
  preference against long runs is an apt reflection of finite-sequence statistics
  under limited memory, not an error.
- **Falk & Konold (1997), Psychological Review.** Single-sequence 1–10 randomness
  rating; normative baseline tradition.

## What this implies for our instruction (applied in `problem_definition.md`)

- **Keep** the fair-coin frame (50/50, independent, "no memory") — standard across
  the literature (Reimers et al.; Williams & Griffiths).
- **Keep** the question "which … looks more random" (Griffiths & Tenenbaum;
  Bar-Hillel & Wagenaar).
- **Leave** the meaning of "random-looking" to the participant; state explicitly
  that people differ and there are no right/wrong answers (Williams & Griffiths
  2013 Exp. 3; Griffiths et al. 2018).
- **Removed** the earlier draft's coaching examples ("too orderly, too streaky, or
  too regular in their pattern of switching between H and T") — these named the
  exact run-length and alternation cues being measured.

---
*Source: literature-search agents that read and verified each primary source.
Six over-reaching framing claims were caught and discarded in verification (e.g.
mis-attributing the G&T classroom demonstration as that paper's experimental
method; calling the Guseva et al. ER wording "the most minimal" — unsupported).
Verbatim quoted wordings above survived verification.*
