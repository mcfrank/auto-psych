# name: participant_yes_rate_sd
# description: SD across participants (blocks inferred from trial_index resets; all see the same 450 items) of the overall yes-rate; observed > model means people differ in baseline yes-bias/generalisation breadth that the single shared-parameter model does not produce.
def test_statistic(df):
    t = df["trial_index"].to_numpy()
    pid = np.concatenate([[0], np.cumsum(np.diff(t) < 0)])
    r = df["rating"].to_numpy().astype(float)
    rates = pd.Series(r).groupby(pid).mean().to_numpy()
    if len(rates) < 2:
        return 0.0
    return float(np.std(rates))
