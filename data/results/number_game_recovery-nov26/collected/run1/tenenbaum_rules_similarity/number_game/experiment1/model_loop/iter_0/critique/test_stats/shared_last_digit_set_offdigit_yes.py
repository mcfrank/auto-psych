# name: shared_last_digit_set_offdigit_yes
# description: In sets of >=2 examples that all end in the same digit, the yes rate for targets that do NOT end in that digit; observed above the null mean means the model's "ends in d" rule is too sharp and under-produces yes for non-matching targets.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    return ex, t, r

def test_statistic(df):
    ex, t, r = _parse(df)
    ld = np.where(np.isnan(ex), np.nan, ex % 10)
    mx = np.nanmax(ld, axis=1); mn = np.nanmin(ld, axis=1)
    uni = (mx == mn) & ((~np.isnan(ex)).sum(axis=1) >= 2)
    m = uni & ((t % 10) != mn)
    return float(r[m].mean()) if m.any() else 0.0
