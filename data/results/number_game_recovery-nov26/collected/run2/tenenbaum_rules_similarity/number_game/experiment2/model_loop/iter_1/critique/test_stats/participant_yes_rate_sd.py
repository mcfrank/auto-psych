# name: participant_yes_rate_sd
# description: SD across participants (participant = run of trials starting at trial_index 0) of the overall yes rate; observed > null means the model (one shared parameter set, no individual differences) underproduces between-person heterogeneity in yes-bias/generalization breadth.
def test_statistic(df):
    pid = (df["trial_index"].to_numpy() == 0).cumsum()
    r = df["rating"].to_numpy().astype(float)
    means = pd.Series(r).groupby(pid).mean()
    return float(means.std(ddof=0))
