# name: far_target_early_minus_late
# description: Among non-example targets more than 10 from every example, yes rate on trials in the first half of the session (trial_index below its median) minus the second half; the model has no time dependence, so observed above null_mean means people's broad generalization shrinks over the session, below means it grows.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n_ex = valid.sum(axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    ti = df["trial_index"].astype(float).to_numpy()
    med = np.median(ti)
    base = (~member) & (dist > 10)
    a = base & (ti < med); b = base & (ti >= med)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
