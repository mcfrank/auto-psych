# name: top_decade_targets_low_sets
# description: Yes rate for targets 91-100 that are not examples, in sets whose largest example is at most 80; observed above null means people say yes to the top of the number line more than the model's ranges/thresholds/rules predict (an end-of-scale anchor), below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    member = np.any(ex == t[:, None], axis=1)
    sel = (t >= 91) & (hi <= 80) & ~member
    if sel.sum() == 0:
        return 0.0
    return float(r[sel].mean())
