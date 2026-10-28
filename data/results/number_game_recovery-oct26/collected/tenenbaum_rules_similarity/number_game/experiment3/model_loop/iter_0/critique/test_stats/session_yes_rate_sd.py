# name: session_yes_rate_sd
# description: SD across sessions (split where trial_index resets) of the per-session yes-rate; observed above null means between-participant differences in overall yes-bias exceed what the single-population model produces.
def test_statistic(df):
    ti = df["trial_index"].astype(int).values
    sess = np.concatenate([[0], np.cumsum(np.diff(ti) < 0)])
    r = df["rating"].astype(float).values
    m = pd.Series(r).groupby(sess).mean()
    return float(m.std(ddof=0)) if len(m) > 1 else 0.0
