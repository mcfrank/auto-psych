# name: example_target_yes_rate
# description: Yes rate when the target is itself one of the shown examples; observed below null means people reject shown examples more often than the model (which predicts near-certain yes), i.e. the model lacks response noise on trivially-consistent targets.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    mx = np.nanmax(ex, axis=1)
    mn = np.nanmin(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    if is_ex.sum() == 0:
        return 0.0
    return float(y[is_ex].mean())
