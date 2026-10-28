# name: far_outside_range_yes_rate
# description: Yes rate for targets outside the examples' [min,max] range and more than 25 from the nearest example; observed above the replicate mean means the model under-produces far-outside yeses (floor too low), below means it over-produces them.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = ((t < lo) | (t > hi)) & (dist > 25)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
