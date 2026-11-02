# name: between_participant_yes_rate_sd
# description: Standard deviation across participants (a new participant starts each time trial_index resets to 0 in row order) of each participant's overall yes-rate; observed above null_mean means people differ in yes-bias more than the single-population model allows.
def test_statistic(df):
    ti = df["trial_index"].astype(float).to_numpy()
    pid = np.cumsum(np.r_[True, ti[1:] <= ti[:-1]])
    r = df["rating"].astype(float).to_numpy()
    means = pd.Series(r).groupby(pid).mean().to_numpy()
    return float(np.std(means)) if len(means) > 1 else 0.0
