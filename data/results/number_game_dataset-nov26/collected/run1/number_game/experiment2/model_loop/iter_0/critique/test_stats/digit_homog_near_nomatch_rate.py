# name: digit_homog_near_nomatch_rate
# description: In sets of >=2 examples all sharing one final digit, the yes-rate for targets within 10 of some example that do NOT share that digit; observed < null means people reject magnitude-near digit violators more strictly than the model's coincidence-sharpened similarity predicts, observed > null means magnitude still wins more than predicted.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    n = (~np.isnan(ex)).sum(1)
    dig = np.mod(ex, 10)
    same = (np.nanmax(dig, 1) == np.nanmin(dig, 1)) & (n >= 2)
    tmatch = np.mod(t, 10) == np.nanmin(dig, 1)
    dmin = np.nanmin(np.abs(ex - t[:, None]), 1)
    m = same & ~tmatch & (dmin <= 10)
    r = df["rating"].to_numpy()
    return float(r[m].mean()) if m.sum() else 0.0
