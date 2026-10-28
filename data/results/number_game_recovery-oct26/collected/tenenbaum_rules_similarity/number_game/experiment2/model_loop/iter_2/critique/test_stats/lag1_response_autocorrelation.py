# name: lag1_response_autocorrelation
# description: Pearson correlation between the response on a trial and on the immediately following trial (consecutive rows whose trial_index increases by 1); observed above null means people perseverate on their previous answer (a sequential dependency the trial-independent model lacks), below null means they alternate.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    mx = np.nanmax(ex, axis=1)
    mn = np.nanmin(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    ti = df["trial_index"].to_numpy()
    nxt = np.flatnonzero(np.diff(ti) == 1)
    a = y[nxt]; b = y[nxt + 1]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
