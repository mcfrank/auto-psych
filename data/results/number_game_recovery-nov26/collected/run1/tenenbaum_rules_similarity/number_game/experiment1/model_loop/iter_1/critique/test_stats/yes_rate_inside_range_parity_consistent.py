# name: yes_rate_inside_range_parity_consistent
# description: Yes-rate for targets strictly inside [min,max] of an all-same-parity example set whose examples do NOT all share a final digit, with the target sharing that parity; observed < null means the model over-generalizes within the example span (too confident in interval/parity concepts), observed > null means it under-generalizes.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(int).values
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    par = np.where(np.isnan(ex), np.nan, ex % 2)
    ld = np.where(np.isnan(ex), np.nan, ex % 10)
    digit_same = np.nanmin(ld, axis=1) == np.nanmax(ld, axis=1)
    allsame = np.nanmin(par, axis=1) == np.nanmax(par, axis=1)
    mask = allsame & ~digit_same & ((t % 2) == np.nanmax(par, axis=1)) & (t > lo) & (t < hi)
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].values[mask].mean())
