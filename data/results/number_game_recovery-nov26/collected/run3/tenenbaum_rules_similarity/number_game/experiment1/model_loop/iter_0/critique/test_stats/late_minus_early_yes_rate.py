# name: late_minus_early_yes_rate
# description: Yes rate in the second half of trials (trial_index >= median) minus the first half; a value different from the model mean (which has no time dependence) means responding drifts over the session (e.g. becoming more conservative).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = np.full((len(uniq), 4), np.nan)
    for i, s in enumerate(uniq):
        v = [int(x) for x in s.split(",")][:4]
        E[i, :len(v)] = v
    X = E[codes]  # (rows, 4) examples, NaN padded
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(X), axis=1)
    ti = df["trial_index"].to_numpy()
    late = ti >= np.median(ti)
    if late.all() or (~late).all():
        return 0.0
    return float(r[late].mean() - r[~late].mean())
