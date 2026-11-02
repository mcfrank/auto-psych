# name: yes_rate_last_digit_violation
# description: Among sets whose examples all end in the same digit, the yes rate for targets ending in a different digit; observed above the model mean means the model is too strongly committed to ends-in-d rules.
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
    ld = np.where(np.isnan(X), np.nan, X % 10)
    lo = np.nanmin(ld, axis=1); hi = np.nanmax(ld, axis=1)
    m = (lo == hi) & ((t % 10) != lo) & (n >= 2)
    return float(r[m].mean()) if m.any() else 0.0
