# name: mid_session_yes
# description: Yes rate over trials with trial_index in [180,360) (the middle-to-late part of the 450-trial session); observed above null means people say yes more in mid-to-late session than the stationary model predicts (drift toward leniency).
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = np.sum(~np.isnan(X), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(X, axis=1)
    mx = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    ti = df["trial_index"].astype(int).to_numpy()
    m = (ti >= 180) & (ti < 360)
    return float(y[m].mean()) if m.sum() > 0 else 0.0
