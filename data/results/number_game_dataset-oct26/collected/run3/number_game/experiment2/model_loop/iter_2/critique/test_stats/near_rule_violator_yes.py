# name: near_rule_violator_yes
# description: For sets of 2+ examples all divisible by 3, 4, 5 or 10 (or all sharing a last digit), yes-rate on non-example targets within 5 of the nearest example that violate every such shared rule; observed > model means numeric proximity pulls people to "yes" despite a rule violation more than the model predicts, observed < model means people reject nearby rule violators more sharply.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    tf = t.astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    valid = ~np.isnan(ex)
    has_rule = np.zeros(len(t), dtype=bool)
    satisfies = np.zeros(len(t), dtype=bool)
    for k in (3, 4, 5, 10):
        r = np.all(~valid | (np.mod(ex, k) == 0), axis=1)
        has_rule |= r
        satisfies |= r & (t % k == 0)
    ld = np.mod(ex, 10)
    same = np.nanmax(ld, axis=1) == np.nanmin(ld, axis=1)
    has_rule |= same
    satisfies |= same & (t % 10 == np.mod(ex[:, 0], 10))
    near = np.nanmin(np.abs(ex - tf[:, None]), axis=1)
    is_ex = np.any(ex == tf[:, None], axis=1)
    m = (n >= 2) & has_rule & ~satisfies & ~is_ex & (near <= 5)
    return float(y[m].mean()) if m.any() else 0.0
