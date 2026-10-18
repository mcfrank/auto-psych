# name: single_example_magnitude_slope
# description: Among single-example sets (targets other than the example), OLS slope of rating on log10(example value); t_observed below null_mean means people generalize more broadly from small examples (e.g. 5, 8) than from large ones (e.g. 91, 100) beyond what digit/band features predict; above means the reverse.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    single = ~ns.str.contains(",")
    sub = df.loc[single]
    x = pd.to_numeric(sub["number_set"], errors="coerce").to_numpy().astype(float)
    t = sub["target"].astype(int).to_numpy()
    keep = (x != t) & np.isfinite(x)
    if keep.sum() < 3:
        return 0.0
    lx = np.log10(x[keep])
    y = sub["rating"].astype(float).to_numpy()[keep]
    xc = lx - lx.mean()
    den = float((xc ** 2).sum())
    if den <= 0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / den)
