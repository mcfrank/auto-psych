# name: off_by_one_power_member_yes
# description: For sets of >=2 examples all in {2^k+1} or all in {2^k-1} (e.g. 33,17,5,9), yes rate on non-example targets in that same off-by-one sequence; observed > null means people learn the "power of 2 plus/minus one" concept that the model's rule space lacks, observed < null means they generalise to its members even less than the model's ranges/rules imply.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    valid = ~np.isnan(E)
    n = valid.sum(axis=1)
    in_set = np.any(E == t[:, None], axis=1)
    hit = np.zeros(len(t), bool)
    for S in (np.array([2 ** k + 1 for k in range(0, 7)]), np.array([2 ** k - 1 for k in range(1, 7)])):
        hit |= np.all(np.isin(E, S) | ~valid, axis=1) & np.isin(t, S)
    m = (n >= 2) & hit & ~in_set
    return float(r[m].mean()) if m.sum() else 0.0
