# name: upward_minus_downward_extension
# description: Among multi-example sets, yes-rate for targets above the largest example minus yes-rate for targets below the smallest example; observed different from null_mean means people extend the concept asymmetrically in magnitude (above = more upward generalization) beyond what the model's symmetric intervals and rules imply.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    mx = np.nanmax(ex, axis=1)
    mn = np.nanmin(ex, axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    up = (n >= 2) & (t > mx)
    down = (n >= 2) & (t < mn)
    if not up.any() or not down.any():
        return 0.0
    return float(r[up].mean() - r[down].mean())
