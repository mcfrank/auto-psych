# name: within_item_trial_drift
# description: Correlation between trial_index and the rating after subtracting each (set, target) cell's mean rating; observed > null means people become more liberal (more yes) later in the session, observed < null more conservative/fatigued, a time trend the stationary model cannot produce.
def test_statistic(df):
    r = df["rating"].to_numpy(dtype=float)
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    resid = r - pd.Series(r).groupby(key.to_numpy()).transform("mean").to_numpy()
    ti = df["trial_index"].to_numpy(dtype=float)
    if np.std(resid) == 0 or np.std(ti) == 0:
        return 0.0
    return float(np.corrcoef(ti, resid)[0, 1])
