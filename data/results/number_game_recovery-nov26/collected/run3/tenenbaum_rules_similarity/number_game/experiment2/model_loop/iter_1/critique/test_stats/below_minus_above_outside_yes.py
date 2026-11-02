# name: below_minus_above_outside_yes
# description: Among non-example targets 1-15 outside the examples' [min,max], yes rate for targets below the minimum minus targets above the maximum; observed > null means people extend generalisation toward smaller numbers more than toward larger ones (a log/Weber-like magnitude asymmetry) beyond what the model's linear intervals and stimulus design produce, observed < null the reverse.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    lo, hi = np.nanmin(E, axis=1), np.nanmax(E, axis=1)
    below = (lo - t >= 1) & (lo - t <= 15)
    above = (t - hi >= 1) & (t - hi <= 15)
    if below.sum() == 0 or above.sum() == 0:
        return 0.0
    return float(r[below].mean() - r[above].mean())
