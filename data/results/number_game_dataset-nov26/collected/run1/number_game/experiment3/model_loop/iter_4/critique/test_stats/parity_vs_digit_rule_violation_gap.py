# name: parity_vs_digit_rule_violation_gap
# description: Yes-rate for wrong-parity targets in 2-4-example same-parity sets that do not share a final digit and are not all multiples of 5, minus yes-rate for wrong-final-digit targets in same-final-digit sets; observed > null means people tolerate breaking a weak parity rule more (and/or a digit rule less) than the model's single shared violation cost implies, observed < null the reverse.
def test_statistic(df):
    s = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    n = s.notna().sum(axis=1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    e0 = s[0].to_numpy().astype(int)
    dig = s % 10
    same_dig = (dig.max(axis=1) == dig.min(axis=1)).to_numpy() & (n >= 2)
    f5 = ((s % 5 == 0) | s.isna()).all(axis=1).to_numpy()
    par = s % 2
    same_par = (par.max(axis=1) == par.min(axis=1)).to_numpy() & (n >= 2)
    pset = same_par & ~same_dig & ~f5
    r = df["rating"].to_numpy(dtype=float)
    a = pset & (t % 2 != e0 % 2)
    b = same_dig & (t % 10 != e0 % 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
