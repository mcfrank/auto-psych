# name: example_target_yes_rate
# description: Yes rate when the target is itself one of the set's examples; observed below null means the model over-endorses the shown examples (people say yes to them only ~70% of the time, implying a response floor/lapse the model lacks).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    m = is_ex
    return float(y[m].mean()) if m.any() else 0.0
