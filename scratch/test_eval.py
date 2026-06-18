import numpy as np
import pandas as pd
import pymc as pm
from candidate import model

df = pd.read_csv("responses.csv")
with model:
    pm.set_data({
        "n_a": df["n_a"].values,
        "max_run_a": df["max_run_a"].values,
        "n_b": df["n_b"].values,
        "max_run_b": df["max_run_b"].values,
        "chose_left": df["chose_left"].values
    })
    idata = pm.sample(100, tune=100, chains=2, cores=1, progressbar=False)
    pm.compute_log_likelihood(idata)
    import arviz as az
    loo = az.loo(idata)
    print("LOO:", loo.elpd_loo)
