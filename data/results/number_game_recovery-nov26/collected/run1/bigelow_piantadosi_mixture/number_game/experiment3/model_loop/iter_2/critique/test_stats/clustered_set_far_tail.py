# name: clustered_set_far_tail
# description: For sets of >=2 examples spanning <=20, yes rate for targets more than 20 from every example; observed > null means people extend tightly clustered examples out to distant numbers more than the model (too thin a long tail), observed < null means the model's broad-range tail is too heavy.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    absd = np.abs(E - t[:, None])
    d = np.nanmin(absd, axis=1)
    ne = E[np.arange(len(t)), np.nanargmin(absd, axis=1)]
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    m = (n >= 2) & ((hi - lo) <= 20) & (d > 20)
    if m.sum() == 0:
        return 0.0
    return float(y[m].mean())
