# Tried before

3 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### tenenbaum_rules_similarity — dropped (experiment1)

**Outcome:** MCMC fit failed on CV fold 3 (AssertionError: Apply node that caused the error: Mul(Assert{msg='Could not broadcast dimensi...'}.0, Shape_i{2}.0) Toposort index: 54 Inputs types: [TensorType(int64, shape=()), TensorType(int64, shape=())] HINT: Use a linker other than the C linker to print the inputs' shapes and strides. HINT: Re-running with most PyTensor optimizations disabled could provide a back-trace showing when this node was created. This can be done by setting the PyTensor flag 'optimizer=fast_compile'. If that does not work, PyTensor optimizations can be disabled with 'optimizer=None'. HINT: Use the PyTensor flag `exception_verbosity=high` for a debug print-out and storage map footprint of this Apply node.)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### erlang_interval_prior_concepts_2 — rejected (experiment1 round 1 candidate 4 refine incumbent bigelow_piantadosi_independent)

**Outcome:** predicts like existing model 'erlang_interval_prior_concepts' (p_yes RMSE 0.00067 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_prior_concepts, not a new hypothesis.

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a prior weight inferred for each rule), but they do not consider every interval equally likely a priori; instead their prior over intervals follows an Erlang density on the interval's size with a fitted scale, so very narrow intervals hugging the examples are a priori implausible and intervals of a characteristic, moderate breadth are preferred. This one change to the interval prior sets how far beyond and between the examples people extend a magnitude-based concept.

### decade_aligned_interval_concepts — rejected (experiment1 round 2 candidate 4 refine incumbent erlang_interval_prior_concepts)

**Outcome:** predicts like existing model 'erlang_interval_prior_concepts' (p_yes RMSE 0.00154 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_prior_concepts, not a new hypothesis.

**Hypothesis:** Refinement of erlang_interval_prior_concepts: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a free prior weight per rule, an Erlang prior on interval size), but when they think of the program as producing a numerical range they favour ranges bounded by round numbers — decade boundaries such as "1 to 10", "20 to 29", "between 10 and 50", "51 to 100" — over ranges with arbitrary endpoints. The one change is a fitted prior boost for decade-aligned intervals, so people generalise from examples out to the nearest decade boundaries (e.g. from 24 and 27 to the whole of the twenties) rather than smoothly beyond the examples' span.
