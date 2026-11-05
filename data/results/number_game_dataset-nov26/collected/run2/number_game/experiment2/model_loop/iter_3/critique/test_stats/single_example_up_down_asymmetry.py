# name: single_example_up_down_asymmetry
# description: Single-example sets, targets within 30 of the example: yes-rate on targets above the example minus yes-rate on targets below it; the model's band uses one width for both directions, so observed above null means people extend a single example upward more readily than the model (beyond seed multiples), below means downward.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    isex = np.any(E == t[:, None], axis=1)
    e = E[:, 0]
    ok = (n == 1) & ~isex & (np.abs(t - e) <= 30)
    a = ok & (t > e); b = ok & (t < e)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
