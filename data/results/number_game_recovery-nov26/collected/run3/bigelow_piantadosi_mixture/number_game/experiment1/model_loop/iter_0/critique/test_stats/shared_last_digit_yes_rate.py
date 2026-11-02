# name: shared_last_digit_yes_rate
# description: Among sets of 2+ examples that all end in the same digit, yes rate for non-example targets ending in that digit; observed < null means the model over-produces "ends in d" rule generalization (people endorse last-digit rules less than the model infers), observed > null the reverse.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    k = valid.sum(axis=1)
    last = np.mod(ex, 10)
    first_last = last[:, 0]
    alls = np.where(valid, last == first_last[:, None], True).all(axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    notex = ~(ex == t[:, None]).any(axis=1)
    m = (k >= 2) & alls & notex & (np.mod(t, 10) == first_last)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
