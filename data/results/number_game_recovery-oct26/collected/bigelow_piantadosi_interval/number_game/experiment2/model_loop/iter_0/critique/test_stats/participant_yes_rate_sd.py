# name: participant_yes_rate_sd
# description: SD across participants (rows are in session order; a new participant starts where trial_index resets to 0) of each participant's overall yes rate; observed > null means people differ in response bias/generalization breadth more than the single shared-parameter model produces.
def test_statistic(df):
    pid = (df["trial_index"].values == 0).cumsum()
    r = df["rating"].values.astype(float)
    rates = pd.Series(r).groupby(pid).mean()
    if len(rates) < 2:
        return 0.0
    return float(rates.std(ddof=0))
