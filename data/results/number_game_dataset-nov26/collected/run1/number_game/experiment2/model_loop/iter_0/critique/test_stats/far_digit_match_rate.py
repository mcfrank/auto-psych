# name: far_digit_match_rate
# description: In sets of >=2 examples that all share one final digit, the yes-rate for targets that share that digit but lie >=15 from every example; observed > null means people apply the final-digit (e.g. multiples-of-10) rule beyond magnitude neighbourhood more than the model's distance-decaying similarity allows, observed < null means they generalize the digit less far.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    n = (~np.isnan(ex)).sum(1)
    dig = np.where(np.isnan(ex), np.nan, np.mod(ex, 10))
    same = (np.nanmax(dig, 1) == np.nanmin(dig, 1)) & (n >= 2)
    tmatch = np.mod(t, 10) == np.nanmin(dig, 1)
    dmin = np.nanmin(np.abs(ex - t[:, None]), 1)
    m = same & tmatch & (dmin >= 15)
    r = df["rating"].to_numpy()
    return float(r[m].mean()) if m.sum() else 0.0
