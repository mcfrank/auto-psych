# name: drift_by_distance_interaction
# description: Late-minus-early (trial_index >=225 vs <225) yes-rate change among far targets (gap >30 outside span) minus the same change among in-span non-example targets; observed > null means the session-long relaxation of criterion concentrates on far/unlikely targets (broadening generalization) more than the model additive logit drift predicts.
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
    late = df["trial_index"].astype(float).to_numpy() >= 225
    far = gap > 30
    near = (gap == 0) & ~isex & (k >= 2)
    return (m(far & late) - m(far & ~late)) - (m(near & late) - m(near & ~late))
