# name: participant_yes_rate_sd
# description: SD across sessions (a new session starts at each trial_index==0) of the per-session yes rate; observed above null means individual differences in overall yes-bias that the single-parameter-set model underproduces.
def test_statistic(df):
    pid = (df["trial_index"].to_numpy() == 0).cumsum()
    rates = pd.Series(df["rating"].astype(float).to_numpy()).groupby(pid).mean()
    if len(rates) < 2:
        return 0.0
    return float(rates.std(ddof=1))
