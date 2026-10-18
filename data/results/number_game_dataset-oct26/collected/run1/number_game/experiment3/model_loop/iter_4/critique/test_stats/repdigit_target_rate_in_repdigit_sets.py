# name: repdigit_target_rate_in_repdigit_sets
# description: In sets containing at least one repdigit example (11,22,...,99), yes-rate on non-example repdigit targets; observed > null means people hold a "same-digit numbers" concept the model lacks (it under-produces repdigit generalisation); observed < null that it over-credits them.
def test_statistic(df):
    X = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy(float)
    valid = ~np.isnan(X)
    Xi = np.nan_to_num(X).astype(int)
    rep_ex = (valid & (Xi % 11 == 0) & (Xi > 0) & (Xi < 100)).any(1)
    rep_t = (t % 11 == 0) & (t < 100)
    is_ex = (valid & (Xi == t[:, None])).any(1)
    m = rep_ex & rep_t & ~is_ex
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
