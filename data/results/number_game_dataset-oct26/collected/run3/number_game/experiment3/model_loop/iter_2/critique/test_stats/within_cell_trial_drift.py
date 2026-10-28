# name: within_cell_trial_drift
# description: Correlation between rating and trial_index after removing each (set, target) cell's mean from both; observed > null (positive) means people say "yes" more often later in the session for the same stimulus, a drift toward liberal responding the stationary model cannot produce (observed < null: more conservative over time).
def test_statistic(df):
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    y = df["rating"].astype(float)
    ti = df["trial_index"].astype(float)
    ry = (y - y.groupby(key).transform("mean")).to_numpy()
    rt = (ti - ti.groupby(key).transform("mean")).to_numpy()
    den = np.sqrt((ry ** 2).sum() * (rt ** 2).sum())
    return float((ry * rt).sum() / den) if den > 0 else 0.0
