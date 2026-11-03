# name: far_target_yes_rate
# description: Yes rate for targets more than 32 away from every example; observed > null means people generalise far beyond the examples more than the model (its far-target floor is too low), observed < null the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = d > 32
    return float(y[m].mean()) if m.sum() else 0.0
