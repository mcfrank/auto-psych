# Theory Report — Experiment 3

## logarithmic_heads_penalty
**Hypothesis:** People judge the randomness of a sequence strictly by the absolute number of heads it contains, but their sensitivity to this count diminishes logarithmically, such that the difference between small head counts matters more than between large head counts.
**Motivation:** In the experiment 2 model loop, the `inner_loop_model` (linear absolute head penalty) had the highest posterior mass. A logarithmic penalty (`iter1_candidate0`) was also competitive but slightly worse. Testing it formally in the outer loop allows us to verify if a diminishing sensitivity better captures human responses across more diverse data points, representing a refinement of the "fewer heads = more random" mechanism.
**Mechanism:** The model penalizes sequences using the logarithm of their absolute head count (`-tau * log(h + 1)`), instead of a linear penalty, meaning the perceived randomness difference shrinks as head counts grow.

## quadratic_heads_penalty
**Hypothesis:** People judge the randomness of a sequence strictly by the absolute number of heads it contains, but the penalty for heads grows quadratically, such that each additional head decreases perceived randomness more than the previous one.
**Motivation:** Similarly to the logarithmic penalty, this model was identified in the experiment 2 inner loop (`iter1_candidate2`) as a competitive variation of the absolute head count penalty. We add it to the outer loop to formally distinguish whether the penalty function for absolute heads is accelerating (quadratic), linear (inner_loop_model), or decelerating (logarithmic).
**Mechanism:** The model penalizes sequences using the square of their absolute head count (`-tau * h^2`), meaning the marginal penalty for each additional head is increasing, which is a distinct functional form of the absolute head count mechanism.
