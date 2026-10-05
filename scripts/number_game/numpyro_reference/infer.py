import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.ops.indexing import Vindex


def prior_eb_numbergame(n_rules=42):
    # Lambdas from Bigelow + Piantadosi (2016)    -- Gamma(2,1)  hyper-prior
    p_mix   = numpyro.sample('p_mix',   dist.Beta(4, 4))
    p_rules = numpyro.sample('p_rules', dist.Gamma(2, 1).expand([n_rules]).to_event(1))
    p_rules = p_rules / p_rules.sum(axis=-1, keepdims=True)     # normalize
    # p_rules = numpyro.sample('p_rules', dist.Dirichlet(jnp.ones(n_rules) * 2))

    # Organize priors into a flat array
    h_prior = jnp.zeros(shape=(n_rules + 5050))
    h_prior = h_prior.at[:n_rules].set(p_mix * p_rules)
    h_prior = h_prior.at[n_rules:].set((1 - p_mix) * (1 / 5050))

    h_prior = h_prior / h_prior.sum(axis=-1, keepdims=True)
    h_prior = numpyro.deterministic('h_prior', h_prior)
    return h_prior


def prior_eb_numbergame_noise(n_rules=42):
    epsilon   = numpyro.sample('epsilon', dist.Beta(1, 7))

    # Lambdas from Bigelow + Piantadosi (2016)    -- Gamma(2,1)  hyper-prior
    p_mix   = numpyro.sample('p_mix',   dist.Beta(4, 4))
    p_rules = numpyro.sample('p_rules', dist.Gamma(2, 1).expand([n_rules]).to_event(1))
    p_rules = p_rules / p_rules.sum(axis=-1, keepdims=True)     # normalize
    # p_rules = numpyro.sample('p_rules', dist.Dirichlet(jnp.ones(n_rules) * 2))

    # Organize priors into a flat array
    h_prior = jnp.zeros(shape=(n_rules + 5050 + 1))
    h_prior = h_prior.at[:n_rules].set((1 - epsilon) * p_mix * p_rules)
    h_prior = h_prior.at[n_rules:-1].set((1 - epsilon) * (1 - p_mix) * (1 / 5050))
    h_prior = h_prior.at[-1].set(epsilon)

    h_prior = h_prior / h_prior.sum(axis=-1, keepdims=True)
    h_prior = numpyro.deterministic('h_prior', h_prior)
    return h_prior


def prior_eb_numbergame_noise2(n_rules=42):
    epsilon_mix   = numpyro.sample('epsilon_mix', dist.Beta(1, 7))
    epsilon_prior = numpyro.sample('epsilon_prior', dist.Uniform(0, 1))

    # Lambdas from Bigelow + Piantadosi (2016)    -- Gamma(2,1)  hyper-prior
    p_mix   = numpyro.sample('p_mix',   dist.Beta(4, 4))
    p_rules = numpyro.sample('p_rules', dist.Gamma(2, 1).expand([n_rules]).to_event(1))
    p_rules = p_rules / p_rules.sum(axis=-1, keepdims=True)     # normalize
    # p_rules = numpyro.sample('p_rules', dist.Dirichlet(jnp.ones(n_rules) * 2))

    # Organize priors into a flat array
    h_prior = jnp.zeros(shape=(n_rules + 5050 + 2))
    h_prior = h_prior.at[:n_rules].set((1 - epsilon_mix) * p_mix * p_rules)
    h_prior = h_prior.at[n_rules:-2].set((1 - epsilon_mix) * (1 - p_mix) * (1 / 5050))
    h_prior = h_prior.at[-2].set(epsilon_mix * (1 - epsilon_prior))
    h_prior = h_prior.at[-1].set(epsilon_mix * epsilon_prior)

    h_prior = h_prior / h_prior.sum(axis=-1, keepdims=True)
    h_prior = numpyro.deterministic('h_prior', h_prior)
    return h_prior



interval_bounds = None
interval_means = None
mean_range = None


def prior_eb_numbergame_interval(n_rules=42):    
    # Lambdas from Bigelow + Piantadosi (2016)    -- Gamma(2,1)  hyper-prior
    p_mix   = numpyro.sample('p_mix',   dist.Beta(2, 2))
    p_rules = numpyro.sample('p_rules', dist.Gamma(2, 1).expand([n_rules]).to_event(1))
    p_rules = p_rules / p_rules.sum(axis=-1, keepdims=True)     # normalize

    # Set up mean and length values for interval hypotheses
    global interval_bounds, interval_means, mean_range       # idk an easy way to do this w/o globals
    if interval_bounds is None:
        #### TODO: instead round means to nearest .05, or even .1, to reduce number of mean buckets
        interval_bounds = [(i, j+1) for i in range(1, 101) for j in range(1, 101) if i <= j]
        interval_means = jnp.array([(i + j) for i, j in interval_bounds], dtype=jnp.int16)   # this is actually 2*mean so we can index by .5 increments
        mean_range = (jnp.arange(200) / 199).clip(min=1e-5, max=1 - 1e-5)

    # Sample hyper-priors for interval (1) means and (2) lengths
    p_mean_hyper = numpyro.sample('p_mean_hyper', dist.Gamma(2, 1).expand([2]))  # First, sample 2 hyper-parameters
    mean_dist = dist.Beta(*p_mean_hyper)                                         # Hyper-prior over interval means is a beta dist with these values
    p_mean = jnp.exp(mean_dist.log_prob(mean_range))                             # Compute the probability of each possible interval, rescaling [1, 100] to [0, 1]                      # note: 200 here instead of 100 to index by .5 increments of `interval_means`
    p_mean /= p_mean.sum()                                                       # Normalize
    p_mean = numpyro.deterministic('p_mean', p_mean)                             # Record value so we can visualize the dist of sampled hyper-priors

    # Compute final interval weights and normalize so they sum to 1
    p_intervals = p_mean[interval_means]
    p_intervals /= p_intervals.sum()

    # Organize priors into a flat array
    h_prior = jnp.zeros(shape=(n_rules + 5050))
    h_prior = h_prior.at[:n_rules].set(p_mix * p_rules)
    h_prior = h_prior.at[n_rules:].set((1 - p_mix) * p_intervals)

    h_prior = h_prior / h_prior.sum(axis=-1, keepdims=True)
    h_prior = numpyro.deterministic('h_prior', h_prior)
    return h_prior


def ApproxEqualProb(vals, variance=.05):
    """Used for observing LLM logit probabilities, since those are basically a delta function"""
    # Soft delta function for value in [0, 1] -- truncated normal with tiny variance
    return dist.TruncatedNormal(vals, scale=variance, low=0, high=1)


#### version with model averaging instead of discrete h sampling
def number_bpl_model(
        y_predictives,         # size: (n_hypotheses, n_support)     <--  p(X | C)  pre-computed likelihood function
        X_likelihoods,         # size: (n_data, n_hypotheses)
        # Xk_likelihoods=None,

        obs_y=None,            # size: (n_data, n_support, 2)   <-- human y/n observations for each category  ;    TODO use -1 for missing values, then use `val == -1` as a mask
        obs_py=None,           # size: (n_data, n_support)      <-- LLM logit probs for  p = yes / (yes + no)

        prior_type='basic',

        n_support=100,
        **kwargs
    ):
    # if Xk_likelihoods is not None:
    #     epsilon = numpyro.sample('epsilon', dist.Beta(1, 5))
    #     X_likelihoods = jnp.log(Xk_likelihoods * (1 - epsilon) + (epsilon / 100)).sum(axis=-2)
    n_data, n_hypotheses = X_likelihoods.shape

    if prior_type == 'basic':
        prior_h = prior_eb_numbergame()
    if prior_type == 'noise':
        prior_h = prior_eb_numbergame_noise()
    if prior_type == 'noise2':
        prior_h = prior_eb_numbergame_noise2()
    if prior_type == 'interval':
        prior_h = prior_eb_numbergame_interval()  # prior_eb_numbergame()   # prior_eb_numbergame_interval

    # posterior  = prior_h[None, :] * X_likelihoods
    # posterior /= jnp.sum(posterior, axis=-1, keepdims=True)
    log_prior      = jnp.log(prior_h)
    log_posterior  = log_prior[None, :] + X_likelihoods
    log_posterior -= jax.nn.logsumexp(log_posterior, axis=-1, keepdims=True)

    # Compute predictive by model averaging over H
    # post_predictive = posterior @ y_predictives
    post_predictive = jnp.exp(log_posterior) @ y_predictives

    # if obs_py is not None:
    #     py_var = numpyro.sample('py_var', dist.Exponential(5))

    post_predictive = post_predictive.reshape(n_data * n_support)
    if obs_y is not None:
        obs_y = obs_y.reshape(n_data * n_support, 2)
    if obs_py is not None:
        obs_py = obs_py.reshape(n_data * n_support)

    with numpyro.plate('data', n_data * n_support, dim=-1) as n:
        # n_ = n[..., None]
        y_pred = post_predictive  #[n]

        if obs_y is not None:
            y_yes, y_total = obs_y[n, 0], obs_y[n, 1]
            numpyro.sample('y', dist.Binomial(y_total, y_pred).mask(y_yes != -1), obs=y_yes)            # observe binomial samples  p(y | X)

        if obs_py is not None:
            numpyro.sample('py', ApproxEqualProb(y_pred, variance=.1).mask(obs_py != -1), obs=obs_py)   # observe probabilities with LLMs

    '''
    with numpyro.plate('data', n_data, dim=-2) as n:
        n_ = n[..., None]

        # Observed output data for human (+ LLM, soon) predictions
        with numpyro.plate('y_support', n_support, dim=-1) as s:
            y_pred = post_predictive[n_, s]

            if obs_y is not None:
                y_yes, y_total = obs_y[n_, s, 0], obs_y[n_, s, 1]
                numpyro.sample('y', dist.Binomial(y_total, y_pred).mask(y_yes != -1), obs=y_yes)            # observe binomial samples  p(y | X)

            if obs_py is not None:
                numpyro.sample('py', ApproxEqualProb(y_pred, variance=.1).mask(obs_py != -1), obs=obs_py)   # observe probabilities with LLMs
    '''

