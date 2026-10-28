# name: mixed_parity_set_yes_rate
# description: Yes rate for non-example targets on multi-example sets containing both odd and even numbers and no shared last digit (no obvious rule); observed above the replicate mean means people generalize more broadly from rule-less sets than the model, below means more conservatively.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    n = np.sum(~np.isnan(ex), axis=1)
    par = np.mod(ex, 2)
    has_odd = np.nansum(par == 1, axis=1) > 0
    has_even = np.nansum(par == 0, axis=1) > 0
    d = np.mod(ex, 10)
    same_digit = np.all(np.isnan(d) | (d == d[:, [0]]), axis=1)
    isex = (ex == t[:, None]).any(axis=1)
    m = (n >= 2) & has_odd & has_even & ~same_digit & ~isex
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
