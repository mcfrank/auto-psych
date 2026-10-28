# name: late_session_yes_shift
# description: Yes rate in the second half of the session (trial_index above median) minus the first half; observed below null means people become more conservative over the session than the stationary model predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    ti = df["trial_index"].astype(float).to_numpy()
    late = ti > np.median(ti)
    return float(r[late].mean() - r[~late].mean())
