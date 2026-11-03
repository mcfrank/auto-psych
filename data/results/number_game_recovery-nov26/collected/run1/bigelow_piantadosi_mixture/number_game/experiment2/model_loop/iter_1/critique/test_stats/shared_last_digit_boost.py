# name: shared_last_digit_boost
# description: For sets of >=2 examples all ending in the same digit, yes rate for targets outside the examples' range that end in that digit minus those that do not; observed above null means people generalise an "ends in d" rule more than the model (with its digit-rule discount) predicts.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = s.notna().sum(axis=1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    ld = s % 10
    same = (ld.nunique(axis=1).to_numpy() == 1) & (n >= 2)
    d0 = ld.iloc[:, 0].to_numpy()
    mn = s.min(axis=1).to_numpy(); mx = s.max(axis=1).to_numpy()
    out = (t < mn) | (t > mx)
    match = (t % 10) == d0
    r = df["rating"].astype(float).to_numpy()
    a = same & out & match; b = same & out & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
