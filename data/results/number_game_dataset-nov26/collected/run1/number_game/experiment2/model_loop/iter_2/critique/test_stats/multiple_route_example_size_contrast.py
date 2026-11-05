# name: multiple_route_example_size_contrast
# description: Among targets >=10 from every example and sharing no example's final digit, yes-rate for targets that are multiples of a two-digit example (>=10, e.g. 94 after 47, 62 after 31) minus yes-rate for targets that are multiples only of a one-digit example (2-9, e.g. 49 after 7); observed > null means multiples of large examples are endorsed relatively more than the model's size-independent route predicts, observed < null means the route is really driven by small examples (times-table knowledge).
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)[:, None]
    pres = ~np.isnan(ex)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    far = (np.abs(t - e) >= 10).all(1)
    nodig = ~((np.mod(t, 10) == np.mod(e, 10)) & pres).any(1)
    m = pres & (e > 1) & (np.mod(t, e1) == 0) & (t != e)
    big = (m & (e >= 10)).any(1)
    small = (m & (e < 10)).any(1)
    base = far & nodig
    r = df["rating"].to_numpy(dtype=float)
    a, b = base & big, base & small & ~big
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
