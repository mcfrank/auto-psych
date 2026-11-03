# name: shared_last_digit_yes_rate
# description: Yes rate for targets whose final digit matches that of every example, in sets of >=2 examples that all share a final digit (target not an example); observed > null means the model under-weights the "ends in d" rule, observed < null means it over-weights it.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(s)
    n = valid.sum(axis=1)
    d = np.where(valid, s % 10, np.nan)
    dmin = np.nanmin(d, axis=1); dmax = np.nanmax(d, axis=1)
    t = df["target"].astype(int).to_numpy()
    is_ex = np.where(valid, s == t[:, None], False).any(axis=1)
    m = (n >= 2) & (dmin == dmax) & ((t % 10) == dmin) & ~is_ex
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
