# name: session_midsession_plateau
# description: Yes-rate on trials 100-274 minus the average of the yes-rates on trials 0-99 and 275-449; observed > null means the relaxation of the yes criterion happens early then plateaus (concave), which the model linear-in-trial criterion drift cannot produce.
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
    ti = df["trial_index"].astype(float).to_numpy()
    return m((ti >= 100) & (ti < 275)) - 0.5 * (m(ti < 100) + m(ti >= 275))
