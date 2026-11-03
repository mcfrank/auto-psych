# name: far_target_yes_rate
# description: Yes-rate on non-member targets lying more than 20 from every example; observed above null means humans say yes to remote numbers more than the model (unmodelled response noise/broad generalisation), below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None].astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    member = np.any(valid & (ex == t), axis=1)
    dist = np.nanmin(np.where(valid, np.abs(ex - t), np.nan), axis=1)
    m = (~member) & (dist > 20)
    return float(r[m].mean()) if m.any() else 0.0
