# name: rule_set_near_nonmember_yes
# description: For sets of >=2 examples that all fit a small arithmetic rule (powers of 2-10, squares, cubes, multiples of 5-12; <=20 members), yes rate for targets within 1-5 of an example that fit none of those rules; observed > null means people blend similarity with the rule (accept rule-breaking neighbours) more than the model's single-guess predicts, observed < null that they reject neighbours more sharply.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    absd = np.abs(E - t[:, None])
    d = np.nanmin(absd, axis=1)
    ne = E[np.arange(len(t)), np.nanargmin(absd, axis=1)]
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    N = np.arange(1, 101)
    rules = [[m ** k for k in range(1, 8) if m ** k <= 100] for m in range(2, 11)]
    rules += [[k * k for k in range(1, 11)], [k ** 3 for k in range(1, 5)]]
    rules += [list(range(m, 101, m)) for m in range(5, 13)]
    R = np.array([np.isin(N, r) for r in rules], dtype=bool)  # (rules, 100)
    Ei = np.nan_to_num(E, nan=0).astype(int)
    valid = ~np.isnan(E)
    memb = R[:, np.clip(Ei - 1, 0, 99)]  # (rules, rows, 4)
    fits = np.all(memb | ~valid[None, :, :], axis=2)  # (rules, rows)
    tin = R[:, t - 1]  # (rules, rows)
    rule_set = fits.any(axis=0) & (n >= 2)
    tfits = (fits & tin).any(axis=0)
    m = rule_set & ~tfits & (d >= 1) & (d <= 5)
    if m.sum() == 0:
        return 0.0
    return float(y[m].mean())
