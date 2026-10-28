# name: same_half_far_generalization
# description: For sets whose examples all lie in one half (1-50 or 51-100), among targets more than 10 from every example, yes rate for targets in the same half minus the other half; the model's round-boundary prior favours 1-50/51-100 ranges, so observed below null_mean means people generalize less by half-range than the model predicts, above means more.
def test_statistic(df):
    ex = pd.DataFrame(df["number_set"].astype(str).str.split(",").tolist(), index=df.index).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n_ex = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    def diff(a, b):
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return float(y[a].mean() - y[b].mean())
    all_low = np.nanmax(ex, axis=1) <= 50
    all_high = np.nanmin(ex, axis=1) >= 51
    base = (all_low | all_high) & (dist > 10)
    same = np.where(all_low, t <= 50, t >= 51)
    return diff(base & same, base & ~same)
