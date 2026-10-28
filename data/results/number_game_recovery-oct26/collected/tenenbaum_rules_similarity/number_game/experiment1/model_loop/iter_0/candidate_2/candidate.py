"""Arithmetic step lattice.

People read the spacing between the examples as a fixed counting step (the
greatest common divisor of the gaps between examples) and accept a target to
the extent that it can be reached from an example by repeatedly adding or
subtracting that step. Off-lattice targets get only a baseline "yes" rate;
acceptance of on-lattice targets falls off with the number of steps needed
from the nearest example.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm


def compute_features(number_set, target):
    examples = [int(x) for x in number_set.split(",")]
    t = int(target)
    gaps = [abs(e - examples[0]) for e in examples[1:]]
    step = reduce(math.gcd, gaps, 0)
    if step == 0:
        # A single example (or repeats): no step can be read off the set.
        on = 1.0 if t in examples else 0.0
        steps = 0.0
    else:
        on = 1.0 if (t - examples[0]) % step == 0 else 0.0
        steps = min(abs(t - e) for e in examples) / step if on else 0.0
    return {"on_lattice": on, "log_steps": math.log1p(steps)}


with pm.Model() as model:
    on_lattice = pm.Data("on_lattice", np.zeros(1, dtype="float64"))
    log_steps = pm.Data("log_steps", np.zeros(1, dtype="float64"))

    # Baseline log-odds of "yes" for a target off the lattice.
    base = pm.Normal("base", mu=0.0, sigma=1.5)
    # Boost for a target exactly on the lattice (at an example).
    gain = pm.Normal("gain", mu=1.0, sigma=1.5)
    # How quickly the boost fades with each additional (log) step.
    decay = pm.HalfNormal("decay", sigma=1.0)

    logit = base + on_lattice * (gain - decay * log_steps)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(logit))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
