# name: far_unrelated_yes_rate
# description: Yes rate for targets >15 from every example that share neither parity-with-all-examples nor last digit with any example; observed > null means the model under-produces baseline/noise "yes" answers (missing lapse/broad-generalization).
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None].astype(float)
    valid = ~np.isnan(s)
    dist = np.where(valid, np.abs(s - t), np.inf).min(axis=1)
    par_all = np.where(valid, (s % 2) == (t % 2), True).all(axis=1)
    ld_any = np.where(valid, (s % 10) == (t % 10), False).any(axis=1)
    m = (dist > 15) & ~par_all & ~ld_any
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
