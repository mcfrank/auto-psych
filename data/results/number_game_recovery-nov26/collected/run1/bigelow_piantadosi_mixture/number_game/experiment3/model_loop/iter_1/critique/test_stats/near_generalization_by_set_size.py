# name: near_generalization_by_set_size
# description: Yes rate for non-member targets within 10 of the nearest example in 4-example sets minus the same in 1-example sets; observed < null means people's near generalization tightens with more examples more strongly than the model predicts (observed > null: less tightening).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4))
    E = ex.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)  # (rows, 4), NaN padded
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    in_set = np.any(E == t[:, None], axis=1)
    d = np.nanmin(np.abs(E - t[:, None]), axis=1)
    near = ~in_set & (d <= 10)
    a = near & (n == 4); b = near & (n == 1)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
