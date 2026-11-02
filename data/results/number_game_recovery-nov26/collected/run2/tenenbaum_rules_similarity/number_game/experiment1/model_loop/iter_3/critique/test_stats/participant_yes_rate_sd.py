# name: participant_yes_rate_sd
# description: Standard deviation across participants (identified as the k-th row within each trial_index) of their overall yes-rate; the model has one shared parameter set, so observed above null means real individual differences in response bias, below null means participants are more alike than binomial sampling allows.
def test_statistic(df):
    pid = df.groupby("trial_index").cumcount().to_numpy()
    r = df["rating"].astype(float).to_numpy()
    m = pd.Series(r).groupby(pid).mean()
    if len(m) < 2:
        return 0.0
    return float(m.std(ddof=1))
