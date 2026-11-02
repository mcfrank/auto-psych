# name: digit_set_inrange_nonmatching_yes_rate
# description: Yes-rate for targets inside [min-5,max+5] of sets whose >=2 examples all share a final digit, when the target does NOT end in that digit; observed < null means the model over-produces interval/parity generalization and under-weights the "ends in d" rule people adopt for such sets.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(int).values
    n = (~np.isnan(ex)).sum(axis=1)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    ld = np.where(np.isnan(ex), np.nan, ex % 10)
    same = (np.nanmin(ld, axis=1) == np.nanmax(ld, axis=1)) & (n >= 2)
    mask = same & ((t % 10) != np.nanmax(ld, axis=1)) & (t >= lo - 5) & (t <= hi + 5)
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].values[mask].mean())
