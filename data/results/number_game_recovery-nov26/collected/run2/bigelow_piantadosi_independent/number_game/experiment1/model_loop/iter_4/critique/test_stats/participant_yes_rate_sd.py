# name: participant_yes_rate_sd
# description: SD across participants (identified by trial_index restarting at 0 in row order) of each participant's overall yes rate; the model has no individual differences, so t_observed above null_mean means people differ in generalization breadth more than the model allows.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.cumsum(np.r_[1, (np.diff(ti) < 0).astype(int)])
    r = pd.Series(df["rating"].to_numpy(dtype=float)).groupby(pid).mean()
    return float(r.std(ddof=0)) if len(r) > 1 else 0.0
