# name: shared_digit_violator_yes_rate
# description: In multi-example sets whose examples all share one final digit, the yes-rate on non-example targets with a different final digit AND opposite parity to the examples (clear violators); observed above null means people keep a floor of yes responses (guessing/lapse) the model drives too close to zero, below means the model is too lenient on violators.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    D = np.mod(E, 10)
    dmin = np.nanmin(D, axis=1); dmax = np.nanmax(D, axis=1)
    samed = (dmin == dmax) & (n >= 2)
    m = samed & (np.mod(t, 10) != dmin) & (np.mod(t, 2) != np.mod(dmin, 2))
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
