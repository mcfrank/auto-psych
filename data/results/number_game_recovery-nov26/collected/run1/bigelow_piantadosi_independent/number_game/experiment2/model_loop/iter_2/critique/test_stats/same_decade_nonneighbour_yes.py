# name: same_decade_nonneighbour_yes
# description: For sets of >= 2 examples, yes rate for non-example targets sharing the tens digit (floor(t/10), 100 counted as 10) with some example but 3+ away from every example; observed > null means people use a same-decade concept the model lacks, observed < null means the model over-generalises within decades.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n = np.sum(~np.isnan(ex), axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    nonex = dist > 0
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    dec = np.floor(np.minimum(t, 99) / 10)
    exdec = np.floor(np.minimum(ex, 99) / 10)
    same = np.any(exdec == dec[:, None], axis=1)
    return rate((n >= 2) & same & (dist >= 3))
