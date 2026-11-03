# name: digit_rule_set_offdigit_neighbour_yes
# description: In sets of >= 3 examples that all end in the same digit, yes rate for targets with a different final digit lying 1-3 from some example; observed > null means people keep similarity-based (neighbour) generalization even when a sharp final-digit rule fits, more than modelled; observed < null means the digit rule suppresses neighbours more than modelled.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    d = dist.min(axis=1)
    fd = np.where(valid, np.mod(ex, 10), -1)
    fdmax = fd.max(axis=1)
    fdmin = np.where(valid, fd, 99).min(axis=1)
    same = (n >= 3) & (fdmax == fdmin)
    m = same & (np.mod(t, 10) != fdmax) & (d >= 1) & (d <= 3)
    return float(y[m].mean()) if m.sum() > 0 else 0.0
