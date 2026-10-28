# name: session_example_yes_sd
# description: Standard deviation across sessions (inferred from trial_index resets) of the yes-rate on trials whose target is one of the shown examples; observed above null means some participants reject trivially consistent targets much more than others (participant-specific noise/lapse heterogeneity the model lacks).
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    sess = np.concatenate([[0], np.cumsum(np.diff(ti) < 0)])
    t = df["target"].astype(int).astype(str)
    s = "," + df["number_set"].astype(str).str.replace(" ", "") + ","
    is_ex = np.array([("," + a + ",") in b for a, b in zip(t, s)])
    r = df["rating"].to_numpy().astype(float)
    if is_ex.sum() == 0:
        return 0.0
    counts = np.bincount(sess[is_ex], minlength=sess.max() + 1)
    sums = np.bincount(sess[is_ex], weights=r[is_ex], minlength=sess.max() + 1)
    ok = counts >= 3
    if ok.sum() < 2:
        return 0.0
    return float(np.std(sums[ok] / counts[ok]))
