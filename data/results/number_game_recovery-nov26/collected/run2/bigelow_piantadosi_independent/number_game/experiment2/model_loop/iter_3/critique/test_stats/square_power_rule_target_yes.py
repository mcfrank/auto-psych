# name: square_power_rule_target_yes
# description: In sets of >=2 examples that are all perfect squares or all powers of 2, yes rate for non-example targets belonging to that same rule; observed below null means people endorse square/power-of-2 rule members less than the model predicts (model over-commits to these rules), above means more.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    pad = np.isnan(E)
    n = (~pad).sum(axis=1)
    in_set = (E == t[:, None]).any(axis=1)
    def rate(mask):
        return float(y[mask].mean()) if mask.any() else 0.0
    sq = np.array([k * k for k in range(1, 11)], dtype=float)
    p2 = np.array([2 ** k for k in range(0, 7)], dtype=float)
    all_sq = (np.isin(E, sq) | pad).all(axis=1) & (n >= 2)
    all_p2 = (np.isin(E, p2) | pad).all(axis=1) & (n >= 2)
    member = (all_sq & np.isin(t, sq)) | (all_p2 & np.isin(t, p2))
    return rate(member & ~in_set)
