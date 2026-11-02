# name: partial_last_digit_match_bonus
# description: Among multi-example sets whose examples do NOT all share a final digit, yes-rate for non-example targets ending in the same digit as at least one example minus yes-rate for targets matching no example's final digit; observed above null means the model under-produces feature-wise (partial) final-digit similarity.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    ok = ~np.isnan(ex)
    n = ok.sum(axis=1)
    exl = np.nan_to_num(ex) % 10
    first = exl[:, :1]
    allsame = np.where(ok, exl == first, True).all(axis=1)
    match = np.where(ok, exl == (t % 10)[:, None], False).any(axis=1)
    inset = np.where(ok, ex == t[:, None], False).any(axis=1)
    m = (n >= 2) & ~allsame & ~inset
    a, b = m & match, m & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
