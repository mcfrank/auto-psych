# name: single_example_near_yes
# description: For single-example sets, yes rate for non-member targets within 1-10 of the example; observed > null means people generalise from one example to its neighbours more than the model's width prior/rule mixture predicts, observed < null less.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    absd = np.abs(E - t[:, None])
    d = np.nanmin(absd, axis=1)
    ne = E[np.arange(len(t)), np.nanargmin(absd, axis=1)]
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    m = (n == 1) & (d >= 1) & (d <= 10)
    if m.sum() == 0:
        return 0.0
    return float(y[m].mean())
