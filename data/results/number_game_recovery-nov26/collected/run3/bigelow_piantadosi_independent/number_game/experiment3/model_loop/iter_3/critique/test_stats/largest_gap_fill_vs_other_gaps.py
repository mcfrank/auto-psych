# name: largest_gap_fill_vs_other_gaps
# description: On 3-4 example sets, yes rate for non-example targets lying inside the largest gap between sorted consecutive examples minus those inside the other gaps; observed below null means people fill big gaps between examples less than the single-interval model predicts (clustered/multi-interval generalisation), above null means more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    n = (~np.isnan(ex)).sum(1)
    srt = np.sort(np.where(np.isnan(ex), np.inf, ex), 1)
    t = df["target"].astype(float).values
    K = srt.shape[1]
    lo = srt[:, :K - 1]; hi = srt[:, 1:]
    with np.errstate(invalid="ignore"):
        gap = np.where(np.isfinite(hi), hi - lo, -1.0)
    big = np.argmax(gap, 1)
    ing = (t[:, None] > lo) & (t[:, None] < hi) & np.isfinite(hi)
    in_any = ing.any(1)
    which = np.argmax(ing, 1)
    r = df["rating"].astype(float).values
    m = (n >= 3) & in_any
    a = r[m & (which == big)]; b = r[m & (which != big)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
