# name: single_example_far_generalization
# description: Yes-rate on targets 20+ away from every example (and outside the span) for one-example sets minus the same for multi-example sets; observed > null means people generalize far more broadly from a single example (size principle) than the model produces.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    exv = ex.to_numpy(dtype=float)
    k = ex.notna().sum(axis=1).to_numpy()
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    near = np.nanmin(np.abs(exv - t[:, None]), axis=1)
    out = (t < np.nanmin(exv, axis=1)) | (t > np.nanmax(exv, axis=1))
    far = (near >= 20) & out
    a = far & (k == 1)
    b = far & (k >= 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
