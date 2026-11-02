# name: ap_lattice_minus_offlattice_yes
# description: Among sets whose sorted examples are evenly spaced with step >= 3, yes rate for non-example targets within two steps of the span lying on the progression lattice minus those off it; observed above null means people extrapolate arithmetic progressions more than the model predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    dist = x.sub(t, axis=0).abs()
    nd = dist.min(axis=1)
    s = np.sort(x.values, axis=1)
    dif = np.diff(s, axis=1)
    with np.errstate(all="ignore"):
        step = np.where(n.values >= 2, np.nanmin(np.where(np.isnan(dif), np.inf, dif), axis=1), 0.0)
        big = np.where(n.values >= 2, np.nanmax(np.where(np.isnan(dif), -np.inf, dif), axis=1), -1.0)
    ap = (n.values >= 2) & (big == step) & (step >= 3)
    st = np.where(ap, step, 1.0)
    lat = ((t.values - mn.values) % st) == 0
    w = ap & ~isin.values & (t.values >= mn.values - 2 * st) & (t.values <= mx.values + 2 * st)
    a = r.values[w & lat].mean() if (w & lat).any() else np.nan
    b = r.values[w & ~lat].mean() if (w & ~lat).any() else np.nan
    v = a - b
    return float(0.0 if (v is None or np.isnan(v)) else v)
