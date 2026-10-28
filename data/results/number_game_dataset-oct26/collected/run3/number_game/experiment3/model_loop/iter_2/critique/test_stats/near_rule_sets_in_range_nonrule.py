# name: near_rule_sets_in_range_nonrule
# description: For 4-example sets in which exactly 3 examples share a last digit (one exception, e.g. 92,82,27,42), yes-rate for non-example targets strictly inside the [min,max] range that do NOT have the majority last digit; observed > null means people treat exception-containing sets as range/interval concepts more than the model's exception-tolerant digit rule allows (observed < null: they keep the rule more strictly).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[codes]
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    inset = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    ld = np.mod(E, 10)
    cnt = (ld[:, :, None] == np.arange(10)[None, None, :]).sum(axis=1)  # (rows, 10)
    maj = np.argmax(cnt, axis=1)
    near = (n == 4) & (cnt.max(axis=1) == 3)
    m = near & ~inset & (t > lo) & (t < hi) & (np.mod(t, 10) != maj)
    return float(y[m].mean()) if m.any() else 0.0
