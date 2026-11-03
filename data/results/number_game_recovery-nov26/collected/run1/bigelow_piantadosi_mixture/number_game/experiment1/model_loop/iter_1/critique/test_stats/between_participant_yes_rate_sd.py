# name: between_participant_yes_rate_sd
# description: Standard deviation across participants (contiguous row blocks, each starting where trial_index resets to 0) of the overall yes rate; observed above the null means individuals differ in response bias far more than the single shared-parameter model produces.
def test_statistic(df):
    t = df["trial_index"].astype(int).values
    pid = np.cumsum(np.r_[1, (np.diff(t) < 0).astype(int)])
    r = df["rating"].astype(float).values
    rates = pd.Series(r).groupby(pid).mean().values
    if len(rates) < 2:
        return 0.0
    return float(np.std(rates))
