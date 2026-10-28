# name: between_participant_breadth_sd
# description: SD across participants (session blocks delimited by trial_index resets) of the yes rate for non-example targets outside the examples' range; the model has no individual differences, so observed above null_mean means people differ in how broadly they generalize more than the model's single shared observer produces.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",")
    ex = pd.DataFrame(sets.tolist(), index=df.index).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    n_ex = np.sum(~np.isnan(ex), axis=1)
    outside = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    pid = np.cumsum(np.r_[1, (np.diff(df["trial_index"].to_numpy()) < 0).astype(int)])
    m = (~member) & (outside > 0)
    s = pd.Series(y[m]).groupby(pid[m]).mean()
    return float(s.std(ddof=0)) if len(s) > 1 else 0.0
