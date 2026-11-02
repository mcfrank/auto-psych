# name: participant_yes_rate_sd
# description: SD across participants (participant = run of rows starting at trial_index 0) of the overall yes rate; observed > model means the model (no individual differences) under-produces between-person variation in response bias.
def test_statistic(df):
    pid = (df["trial_index"].to_numpy() == 0).cumsum()
    r = df["rating"].astype(float).to_numpy()
    m = pd.Series(r).groupby(pid).mean()
    return float(m.std(ddof=0)) if len(m) > 1 else 0.0
