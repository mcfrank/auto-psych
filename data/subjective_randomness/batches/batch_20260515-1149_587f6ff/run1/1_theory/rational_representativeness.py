# file: rational_representativeness.py
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))

    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    # Softmax temperature for the decision rule
    tau = pm.HalfNormal("tau", sigma=5.0)

    def log_p_rand(n):
        """Log-likelihood under a fair coin model."""
        return n * pt.log(0.5)

    def log_p_bias(n, h):
        """Log-likelihood under a biased coin model with uniform prior on p_H."""
        return pt.gammaln(h + 1.0) + pt.gammaln(n - h + 1.0) - pt.gammaln(n + 2.0)

    def log_p_markov(n, alts):
        """Log-likelihood under a Markov model with uniform prior on transition probability."""
        return (
            pt.log(0.5)
            + pt.gammaln(alts + 1.0)
            + pt.gammaln(n - alts)
            - pt.gammaln(n + 1.0)
        )

    def log_p_reg(n, h, alts):
        """Log-likelihood under the regular model (50/50 mixture of bias and markov)."""
        l_bias = log_p_bias(n, h)
        l_markov = log_p_markov(n, alts)
        # Using log(exp(a) + exp(b)) since values are well within float64 range for n <= 8
        return pt.log(0.5) + pt.log(pt.exp(l_bias) + pt.exp(l_markov))

    # Evidence for the random model (log posterior odds assuming equal priors)
    V_a = log_p_rand(n_a) - log_p_reg(n_a, h_a, alts_a)
    V_b = log_p_rand(n_b) - log_p_reg(n_b, h_b, alts_b)

    # Probability of choosing sequence A (left) as more random
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (V_a - V_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
