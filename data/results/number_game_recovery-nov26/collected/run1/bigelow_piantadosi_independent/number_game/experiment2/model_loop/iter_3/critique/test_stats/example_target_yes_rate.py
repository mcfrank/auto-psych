# name: example_target_yes_rate
# description: Yes rate when the target is itself one of the shown examples; observed > null means people endorse the given examples more reliably than the model (its exception/noise mass leaves too much doubt), observed < null less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = d == 0
    return float(y[m].mean()) if m.sum() else 0.0
