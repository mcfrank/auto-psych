# name: participant_yes_rate_sd
# description: SD across participants (rows in blocks; a new participant starts whenever trial_index == 0) of the overall yes rate; observed > null means people differ in their yes-bias / generalisation breadth more than the single shared-parameter model produces.
def test_statistic(df):
    pid = (df["trial_index"].to_numpy() == 0).cumsum()
    r = df["rating"].to_numpy().astype(float)
    rates = pd.Series(r).groupby(pid).mean()
    return float(rates.std(ddof=0))
