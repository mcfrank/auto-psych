# Theory Report — Experiment 1

## alternation_bias
**Motivation:** This is a foundational model in subjective randomness research. People systematically expect more alternations in random sequences than probability theory predicts (the "gambler's fallacy" in production tasks). A model with a monotonic preference for alternation provides a baseline for this bias.
**Mechanism:** Computes the fraction of adjacent pairs that differ (alternation rate) for each sequence. The sequence with the higher alternation rate is judged more random. Uses a softmax with temperature β=5 to convert the score difference to a probability.

## balance_heuristic
**Motivation:** A distinct prediction from alternation bias: people also expect roughly equal numbers of H and T in a random sequence. This model captures only the balance dimension, allowing the experiment to tease apart alternation vs. balance as drivers of perceived randomness.
**Mechanism:** Scores each sequence by how close its proportion of H is to 0.5. A perfectly balanced sequence (e.g., HHTT) scores 1.0; an all-heads sequence scores 0.0. The model is silent on the order of outcomes and responds only to the aggregate count.

## griffiths_representativeness
**Motivation:** Griffiths & Tenenbaum's rational analysis of representativeness predicts that a sequence is judged random when its observable statistics jointly match the expected output of a random process. Neither alternation alone nor balance alone is sufficient — both must be near their expected value of 0.5.
**Mechanism:** Computes the sum of absolute deviations from 0.5 on two dimensions: H/T balance and alternation rate. Sequences with smaller total deviation are assigned higher representativeness. This model makes different predictions than alternation_bias when sequences have very high alternation rates (e.g., HTHTHTHT alternates perfectly but is not balanced in the same way as expected).
