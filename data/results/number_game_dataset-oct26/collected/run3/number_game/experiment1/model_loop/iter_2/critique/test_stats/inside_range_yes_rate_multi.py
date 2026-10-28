# name: inside_range_yes_rate_multi
# description: Yes-rate for targets strictly inside the [min, max] range of the examples (and not an example) on sets of 2+ examples; observed below null means the model over-fills the gap between examples (people do not treat the range as a contiguous interval as strongly), above means it under-fills.
def test_statistic(df):
    ns = df["number_set"].astype(str)
    parts = ns.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = parts.notna().sum(axis=1).to_numpy()
    lo = parts.min(axis=1).to_numpy(float)
    hi = parts.max(axis=1).to_numpy(float)
    tg = df["target"].to_numpy(float)
    inset = parts.eq(df["target"].astype(float), axis=0).any(axis=1).to_numpy()
    ok = (n >= 2) & (tg > lo) & (tg < hi) & ~inset
    if not ok.any():
        return 0.5
    return float(df["rating"].to_numpy(float)[ok].mean())
