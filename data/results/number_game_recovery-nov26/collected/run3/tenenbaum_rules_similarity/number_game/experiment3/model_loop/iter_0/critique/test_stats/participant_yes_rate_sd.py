# name: participant_yes_rate_sd
# description: SD across participants (sessions delimited by trial_index resets to 0) of each participant's overall yes rate; observed above null means the single-population model under-produces individual differences in response bias.
def test_statistic(df):
    pid = (df["trial_index"].to_numpy() == 0).cumsum()
    r = df["rating"].to_numpy().astype(float)
    rates = pd.Series(r).groupby(pid).mean()
    return float(rates.std(ddof=0))
