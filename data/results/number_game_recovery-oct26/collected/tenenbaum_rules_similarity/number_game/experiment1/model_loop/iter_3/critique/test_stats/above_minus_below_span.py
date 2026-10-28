# name: above_minus_below_span
# description: Among non-example targets 1-15 outside the examples' span, yes rate for targets above the largest example minus targets below the smallest; the model's interval prior is roughly symmetric, so observed below null_mean means people extend generalization downward more (upward less) than the model predicts, above means the reverse.
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
    out_d = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    base = (~member) & (out_d >= 1) & (out_d <= 15)
    return diff(base & (t > hi), base & (t < lo))
