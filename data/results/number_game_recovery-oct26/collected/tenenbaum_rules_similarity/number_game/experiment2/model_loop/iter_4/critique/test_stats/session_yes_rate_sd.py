# name: session_yes_rate_sd
# description: Standard deviation across sessions (inferred from trial_index resetting to a lower value) of each session's overall yes-rate; observed above null means participants differ in their overall willingness to say "yes" far more than the single-population model (no individual differences) produces.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    sess = np.concatenate([[0], np.cumsum(np.diff(ti) < 0)])
    r = df["rating"].to_numpy().astype(float)
    counts = np.bincount(sess)
    means = np.bincount(sess, weights=r) / np.maximum(counts, 1)
    means = means[counts > 0]
    if len(means) < 2:
        return 0.0
    return float(np.std(means))
