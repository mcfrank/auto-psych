# name: round_number_target_contrast
# description: For sets with no multiple-of-10 example, among out-of-span targets, yes-rate on multiples of 10 minus other targets; observed > null means people are attracted to round numbers beyond what final-digit similarity predicts, observed < null means they avoid them more than the model predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    mn = ex.min(axis=1).to_numpy(); mx = ex.max(axis=1).to_numpy()
    k = ex.notna().sum(axis=1).to_numpy()
    first = ex[0].to_numpy(); last = ex.ffill(axis=1).iloc[:, -1].to_numpy()
    isex = ex.eq(pd.Series(t, index=ex.index), axis=0).any(axis=1).to_numpy()
    nearest = np.nanmin(np.abs(ex.to_numpy() - t[:, None]), axis=1)
    gap = np.maximum(np.maximum(mn - t, t - mx), 0)
    def m(mask):
        return float(y[mask].mean()) if mask.sum() > 0 else 0.0
    has10 = (ex.to_numpy() % 10 == 0).any(axis=1)
    sel = ~has10 & (gap > 0)
    return m(sel & (t % 10 == 0)) - m(sel & (t % 10 != 0))
