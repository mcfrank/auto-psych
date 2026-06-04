- **rational_representativeness**: A rational model of representativeness (Griffiths & Tenenbaum, 2001). 
Observers compare the likelihood of the sequence under a random model (fair coin) 
versus a regular model (mixture of a biased coin and a first-order Markov model).
- **alternation_heuristic**: A simple heuristic model where observers judge randomness based on the proportion of alternations (H↔T transitions). Sequences with a higher alternation rate are perceived as more random.
- **proportion_heuristic**: A representativeness heuristic that prefers sequences whose head proportion is closer to 0.5. The model computes the absolute difference between the sequence's head proportion and 0.5, and uses a softmax decision rule to choose the sequence with the smaller difference.