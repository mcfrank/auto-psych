# name: far_target_yes_rate
# description: Yes rate for targets outside the examples' range and more than 20 from every example; observed > null means the model under-produces broad/diffuse "yes" generalisation to remote numbers.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float).values
    a = sets.values
    mind = np.nanmin(np.abs(a - t[:, None]), axis=1)
    lo, hi = np.nanmin(a, axis=1), np.nanmax(a, axis=1)
    m = (mind > 20) & ((t < lo) | (t > hi))
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
