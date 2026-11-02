# name: strong_units_rule_nonmember_yes
# description: Yes rate for targets whose last digit differs from the shared last digit of a >=3-example set whose examples all end in the same digit; observed below the null means the model (via lapse guessing and intervals) over-produces "yes" to clear rule violators, above means humans accept violators more than predicted.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    u = np.where(valid, np.nan_to_num(ex) % 10, np.nan)
    u0 = u[:, 0]
    same = np.all(~valid | (u == u0[:, None]), axis=1)
    m = same & (n >= 3) & (t % 10 != u0)
    return float(y[m].mean()) if m.any() else 0.0
