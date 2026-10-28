# name: same_decade_outside_span_boost
# description: Among non-example targets outside the examples' span and 1-9 away from the nearest example, yes rate when the target lies in the same decade (1-10, 11-20, ...) as that nearest example minus when it lies across a decade boundary; the incumbent's decade-bounded prior predicts a positive gap, so observed below null_mean means people's generalization does not stop at decade edges as much as the model says (above means an even sharper decade edge).
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
    near = np.nanargmin(np.abs(ex - t[:, None]), axis=1)
    nx = ex[np.arange(len(t)), near]
    same = np.floor((nx - 1) / 10) == np.floor((t - 1) / 10)
    base = (~member) & ((t < lo) | (t > hi)) & (dist <= 9)
    return diff(base & same, base & ~same)
